"""Evidence-first ALF innovation study.

This script does not change the controller.  It separates four questions:

1. How much of the existing ALF/CRMF result is radius-cap operation?
2. Do the implemented fixed-probe and current-radius maps have the predicted
   piecewise local sensitivity on frozen geometry?
3. Do those map differences persist when both controllers read exactly the
   same position trajectory?
4. Do they remain visible in fresh-seed autonomous closed-loop episodes?

Agent-step rows are diagnostics.  Statistical summaries and paired bootstrap
intervals use independent episodes (seeds) as the sampling unit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Sequence

from ..core import (
    DensityAdaptiveNeighborhood,
    EALMFGSimulator,
    PopulationState,
    Rollout,
    SimulationConfig,
    _clip,
    run_rollout,
)
from ..metrics import fragmentation_score, scc_summary
from .neighborhood_bench import (
    AnalyticalAlignmentActor,
    CurrentRadiusMultiplicativeNeighborhood,
    FixedRadiusNeighborhood,
    _base_config,
    _fixed_config,
    run_cell,
)


METHODS = ("alf", "crmf")
SOURCE_METHODS = ("alf", "crmf", "fixed_r6")
FORMAL_WIDTHS = (11.0, 14.0, 18.0)
FORMAL_REPLAY_SEEDS = tuple(range(23_000, 23_050))
FORMAL_CLOSED_LOOP_SEEDS = tuple(range(24_000, 24_050))
DEVELOPMENT_SEEDS = tuple(range(20_000, 20_050))
BOOTSTRAP_SEED = 2_026_090_901
EPSILON = 1e-12


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_interval(
    values: Sequence[float], replicates: int, rng: random.Random
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("bootstrap requires at least one episode")
    estimates = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    ]
    return (
        statistics.fmean(values),
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    )


def reaggregate_existing(root: Path) -> list[dict[str, Any]]:
    """Reaggregate immutable development evidence by method and density."""

    source = root / "experiments/adaptive_feedback_matched/crdf_matched_raw.csv"
    rows = _read_csv(source)
    metrics = (
        "radius_max_saturation_fraction",
        "radius_temporal_variation",
        "radius_command_temporal_variation",
        "contact_break_rate",
        "mean_interaction_out_degree",
        "mean_interaction_area",
        "mean_largest_scc_fraction",
        "mean_fragmentation",
    )
    output: list[dict[str, Any]] = []
    for method in ("alf", "crmf", "fixed_r8_retained"):
        method_rows = [row for row in rows if row["method"] == method]
        for width in sorted({float(row["half_width"]) for row in method_rows}):
            cell = [row for row in method_rows if float(row["half_width"]) == width]
            seeds = {int(row["seed"]) for row in cell}
            if not cell:
                continue
            item: dict[str, Any] = {
                "method": method,
                "half_width": width,
                "density": float(cell[0]["density"]),
                "n_episodes": len(cell),
                "n_unique_seeds": len(seeds),
            }
            for metric in metrics:
                item[metric] = statistics.fmean(float(row[metric]) for row in cell)
            output.append(item)
    return output


def fixed_geometry_rows() -> list[dict[str, Any]]:
    """Evaluate both implemented maps on one declared frozen geometry."""

    positions = (
        (0.0, 0.0),
        (1.5, 0.0),
        (2.8, 0.0),
        (3.8, 0.0),
        (4.4, 0.0),
        (5.2, 0.0),
        (6.1, 0.0),
        (7.1, 0.0),
        (9.0, 0.0),
    )
    cfg = SimulationConfig(
        dim=2,
        space_half_width=20.0,
        dt=0.5,
        velocity_limit=1.0,
        acceleration_limit=1.0,
        noise_std=0.0,
        beta=0.5,
        base_radius=4.0,
        target_degree=6.0,
        min_radius=1.0,
        max_radius=8.0,
        radius_exponent=0.5,
        radius_smoothing=0.9,
        contact_retention_margin=0.25,
        seed=0,
    )
    alf = DensityAdaptiveNeighborhood(cfg)
    crmf = CurrentRadiusMultiplicativeNeighborhood(cfg)
    fixed_count = alf.base_degrees(positions)[0]
    output: list[dict[str, Any]] = []
    for index in range(141):
        previous_radius = 1.0 + 0.05 * index
        current_count = crmf.current_degrees(
            positions, (previous_radius,) * len(positions)
        )[0]
        for method, degree, base in (
            ("alf", fixed_count, cfg.base_radius),
            ("crmf", current_count, previous_radius),
        ):
            ratio = (cfg.target_degree / max(float(degree), 1.0)) ** cfg.radius_exponent
            command_unclipped = base * ratio
            command_clipped = _clip(command_unclipped, cfg.min_radius, cfg.max_radius)
            realized = _clip(
                cfg.radius_smoothing * previous_radius
                + (1.0 - cfg.radius_smoothing) * command_clipped,
                cfg.min_radius,
                cfg.max_radius,
            )
            expected_gain = (
                cfg.radius_smoothing
                if method == "alf"
                else cfg.radius_smoothing + (1.0 - cfg.radius_smoothing) * ratio
            )
            output.append(
                {
                    "geometry": "declared_radial_threshold_geometry",
                    "focal_agent": 0,
                    "method": method,
                    "previous_radius": previous_radius,
                    "measurement_degree": degree,
                    "command_unclipped": command_unclipped,
                    "command_clipped": command_clipped,
                    "realized_radius": realized,
                    "expected_piecewise_local_gain": expected_gain,
                    "command_lower_clipped": int(command_unclipped <= cfg.min_radius),
                    "command_upper_clipped": int(command_unclipped >= cfg.max_radius),
                }
            )
    return output


def _adaptive_config(width: float, seed: int) -> SimulationConfig:
    return _base_config(width, seed)


def _source_rollout(method: str, width: float, seed: int, steps: int) -> Rollout:
    if method == "fixed_r6":
        cfg = _fixed_config(
            width, seed, 6.0, contact_retention_margin=0.25
        )
        simulator = EALMFGSimulator(cfg)
        simulator.neighborhood = FixedRadiusNeighborhood(cfg)
    else:
        cfg = _adaptive_config(width, seed)
        simulator = EALMFGSimulator(cfg)
        if method == "crmf":
            simulator.neighborhood = CurrentRadiusMultiplicativeNeighborhood(cfg)
    simulator.reset(n_agents=16)
    return run_rollout(simulator, AnalyticalAlignmentActor(), steps=steps)


def replay_feedback(
    states: Sequence[PopulationState], method: str, width: float, seed: int,
) -> list[dict[str, Any]]:
    """Replay one radius controller on a fixed sequence of population states."""

    if method not in METHODS:
        raise ValueError(f"unsupported replay method: {method}")
    cfg = _adaptive_config(width, seed)
    neighborhood: DensityAdaptiveNeighborhood
    if method == "crmf":
        neighborhood = CurrentRadiusMultiplicativeNeighborhood(cfg)
    else:
        neighborhood = DensityAdaptiveNeighborhood(cfg)
    previous_radii: tuple[float, ...] | None = None
    previous_graph = None
    rows: list[dict[str, Any]] = []
    for step, state in enumerate(states):
        fixed_degrees = neighborhood.base_degrees(state.positions)
        current_degrees = (
            fixed_degrees
            if previous_radii is None
            else CurrentRadiusMultiplicativeNeighborhood(cfg).current_degrees(
                state.positions, previous_radii
            )
        )
        measurement = fixed_degrees if method == "alf" else current_degrees
        bases = (
            (cfg.base_radius,) * state.n_agents
            if method == "alf" or previous_radii is None
            else previous_radii
        )
        commands_unclipped = tuple(
            bases[i]
            * (cfg.target_degree / max(float(measurement[i]), 1.0))
            ** cfg.radius_exponent
            for i in range(state.n_agents)
        )
        commands_clipped = tuple(
            _clip(value, cfg.min_radius, cfg.max_radius)
            for value in commands_unclipped
        )
        radii = neighborhood.radii(state.positions, previous_radii)
        graph = neighborhood.neighbors(state.positions, radii, previous_graph)
        weak = fragmentation_score(graph)
        directed = scc_summary(graph)
        for agent in range(state.n_agents):
            before = set() if previous_graph is None else set(previous_graph[agent])
            after = set(graph[agent])
            ratio = (
                cfg.target_degree / max(float(measurement[agent]), 1.0)
            ) ** cfg.radius_exponent
            local_gain = (
                cfg.radius_smoothing
                if method == "alf"
                else cfg.radius_smoothing + (1.0 - cfg.radius_smoothing) * ratio
            )
            rows.append(
                {
                    "method": method,
                    "step": step,
                    "agent_id": agent,
                    "x": state.positions[agent][0],
                    "y": state.positions[agent][1],
                    "vx": state.velocities[agent][0],
                    "vy": state.velocities[agent][1],
                    "fixed_probe_degree": fixed_degrees[agent],
                    "current_radius_degree": current_degrees[agent],
                    "measurement_degree": measurement[agent],
                    "command_base": bases[agent],
                    "command_unclipped": commands_unclipped[agent],
                    "command_clipped": commands_clipped[agent],
                    "realized_radius": radii[agent],
                    "effective_radius_cap": cfg.max_radius,
                    "command_upper_clipped": int(commands_unclipped[agent] >= cfg.max_radius),
                    "realized_upper_saturated": int(radii[agent] >= cfg.max_radius - EPSILON),
                    "interaction_degree": len(graph[agent]),
                    "retained_contact_count": len(after.difference(
                        j for j in range(state.n_agents)
                        if j != agent
                        and neighborhood.pair_distance_sq(state.positions, agent, j)
                        <= radii[agent] ** 2
                    )),
                    "contact_break": int(bool(before.difference(after))),
                    "degree_loss": int(
                        previous_graph is not None
                        and len(after) < len(previous_graph[agent])
                    ),
                    "fragmentation": weak["fragmentation"],
                    "largest_scc_fraction": directed["largest_scc_fraction"],
                    "expected_piecewise_local_gain": local_gain,
                    "joint_unsaturated": 0,
                }
            )
        previous_radii = tuple(float(value) for value in radii)
        previous_graph = graph
    return rows


def _mark_joint_unsaturated(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> None:
    if len(left) != len(right):
        raise ValueError("paired replay traces differ in length")
    for a, b in zip(left, right):
        if (a["step"], a["agent_id"]) != (b["step"], b["agent_id"]):
            raise ValueError("paired replay traces are misaligned")
        joint = all(
            1.0 + EPSILON < float(row[metric]) < float(row["effective_radius_cap"]) - EPSILON
            for row in (a, b)
            for metric in ("command_clipped", "realized_radius")
        )
        a["joint_unsaturated"] = int(joint)
        b["joint_unsaturated"] = int(joint)


def summarize_replay_trace(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty replay trace")
    by_agent: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_agent.setdefault(int(row["agent_id"]), []).append(row)
    command_updates: list[float] = []
    radius_updates: list[float] = []
    joint_command_updates: list[float] = []
    joint_radius_updates: list[float] = []
    for agent_rows in by_agent.values():
        ordered = sorted(agent_rows, key=lambda row: int(row["step"]))
        for before, after in zip(ordered, ordered[1:]):
            command_updates.append(abs(float(after["command_clipped"]) - float(before["command_clipped"])))
            radius_updates.append(abs(float(after["realized_radius"]) - float(before["realized_radius"])))
            if int(before["joint_unsaturated"]) and int(after["joint_unsaturated"]):
                joint_command_updates.append(
                    abs(float(after["command_clipped"]) - float(before["command_clipped"]))
                )
                joint_radius_updates.append(
                    abs(float(after["realized_radius"]) - float(before["realized_radius"]))
                )
    step_agent_count = len(rows)
    steps = sorted({int(row["step"]) for row in rows})
    step_scc = [
        float(next(row["largest_scc_fraction"] for row in rows if int(row["step"]) == step))
        for step in steps
    ]
    return {
        "mean_abs_command_update": statistics.fmean(command_updates),
        "mean_abs_realized_radius_update": statistics.fmean(radius_updates),
        "joint_unsaturated_mean_abs_command_update": (
            statistics.fmean(joint_command_updates) if joint_command_updates else math.nan
        ),
        "joint_unsaturated_mean_abs_realized_radius_update": (
            statistics.fmean(joint_radius_updates) if joint_radius_updates else math.nan
        ),
        "joint_unsaturated_transition_count": len(joint_radius_updates),
        "joint_unsaturated_transition_fraction": len(joint_radius_updates) / max(len(radius_updates), 1),
        "command_upper_clipped_fraction": sum(int(row["command_upper_clipped"]) for row in rows) / step_agent_count,
        "realized_upper_saturation_fraction": sum(int(row["realized_upper_saturated"]) for row in rows) / step_agent_count,
        "contact_break_rate": sum(int(row["contact_break"]) for row in rows if int(row["step"]) > 0)
        / max(sum(int(row["step"]) > 0 for row in rows), 1),
        "degree_loss_rate": sum(int(row["degree_loss"]) for row in rows if int(row["step"]) > 0)
        / max(sum(int(row["step"]) > 0 for row in rows), 1),
        "mean_interaction_degree": statistics.fmean(float(row["interaction_degree"]) for row in rows),
        "mean_largest_scc_fraction": statistics.fmean(step_scc),
        "scc_deficit_integral": sum(1.0 - value for value in step_scc),
        "disconnected_step_fraction": statistics.fmean(value < 1.0 for value in step_scc),
        "mean_expected_piecewise_local_gain": statistics.fmean(
            float(row["expected_piecewise_local_gain"]) for row in rows
        ),
    }


def run_replay_study(
    widths: Sequence[float], seeds: Sequence[int], steps: int, trace_path: Path
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] | None = None
    with gzip.open(trace_path, "wt", newline="", encoding="utf-8") as stream:
        writer: csv.DictWriter | None = None
        for source_method in SOURCE_METHODS:
            for width in widths:
                for seed in seeds:
                    rollout = _source_rollout(source_method, width, int(seed), steps)
                    traces = {
                        method: replay_feedback(rollout.states, method, width, int(seed))
                        for method in METHODS
                    }
                    _mark_joint_unsaturated(traces["alf"], traces["crmf"])
                    for method in METHODS:
                        summary = summarize_replay_trace(traces[method])
                        summaries.append(
                            {
                                "trajectory_source": source_method,
                                "method": method,
                                "half_width": width,
                                "density": 16 / (2.0 * width) ** 2,
                                "seed": int(seed),
                                "steps": steps,
                                **summary,
                            }
                        )
                        for row in traces[method]:
                            output_row = {
                                "trajectory_source": source_method,
                                "half_width": width,
                                "density": 16 / (2.0 * width) ** 2,
                                "seed": int(seed),
                                **row,
                            }
                            if writer is None:
                                fields = list(output_row)
                                writer = csv.DictWriter(stream, fieldnames=fields)
                                writer.writeheader()
                            writer.writerow(output_row)
    return summaries


REPLAY_METRICS = (
    "mean_abs_command_update",
    "mean_abs_realized_radius_update",
    "joint_unsaturated_mean_abs_command_update",
    "joint_unsaturated_mean_abs_realized_radius_update",
    "joint_unsaturated_transition_fraction",
    "command_upper_clipped_fraction",
    "realized_upper_saturation_fraction",
    "contact_break_rate",
    "mean_interaction_degree",
    "mean_largest_scc_fraction",
    "scc_deficit_integral",
)


def paired_replay_summary(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["trajectory_source"]), float(row["half_width"])), []
        ).append(row)
    rng = random.Random(BOOTSTRAP_SEED)
    output: list[dict[str, Any]] = []
    for (source, width), cell in sorted(grouped.items()):
        by_key = {(int(row["seed"]), str(row["method"])): row for row in cell}
        seeds = sorted({int(row["seed"]) for row in cell})
        if any((seed, method) not in by_key for seed in seeds for method in METHODS):
            raise ValueError(f"incomplete replay pair for {source}, L={width}")
        for metric in REPLAY_METRICS:
            pairs = []
            for seed in seeds:
                alf_value = float(by_key[(seed, "alf")][metric])
                crmf_value = float(by_key[(seed, "crmf")][metric])
                if math.isfinite(alf_value) and math.isfinite(crmf_value):
                    pairs.append(crmf_value - alf_value)
            if not pairs:
                continue
            mean, low, high = _bootstrap_mean_interval(pairs, replicates, rng)
            output.append(
                {
                    "comparison": "CRMF - ALF",
                    "trajectory_source": source,
                    "half_width": width,
                    "density": 16 / (2.0 * width) ** 2,
                    "metric": metric,
                    "n_pairs": len(pairs),
                    "n_bootstrap": replicates,
                    "paired_mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_unit": "shared replay trajectory seed",
                }
            )
    return output


def run_fresh_closed_loop(
    widths: Sequence[float], seeds: Sequence[int], steps: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for width in widths:
            for seed in seeds:
                rows.append(run_cell(method, 0.0, width, int(seed), steps))
    return rows


CLOSED_LOOP_METRICS = (
    "final_fragmentation",
    "mean_fragmentation",
    "final_largest_scc_fraction",
    "mean_largest_scc_fraction",
    "mean_interaction_out_degree",
    "mean_interaction_area",
    "radius_temporal_variation",
    "radius_command_temporal_variation",
    "radius_max_saturation_fraction",
    "neighbor_loss_rate",
    "contact_break_rate",
    "velocity_dispersion",
)


def paired_closed_loop_summary(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED + 1)
    output: list[dict[str, Any]] = []
    for width in sorted({float(row["half_width"]) for row in rows}):
        cell = [row for row in rows if float(row["half_width"]) == width]
        by_key = {(int(row["seed"]), str(row["method"])): row for row in cell}
        seeds = sorted({int(row["seed"]) for row in cell})
        if any((seed, method) not in by_key for seed in seeds for method in METHODS):
            raise ValueError(f"incomplete closed-loop pair for L={width}")
        for metric in CLOSED_LOOP_METRICS:
            differences = [
                float(by_key[(seed, "crmf")][metric])
                - float(by_key[(seed, "alf")][metric])
                for seed in seeds
            ]
            mean, low, high = _bootstrap_mean_interval(
                differences, replicates, rng
            )
            output.append(
                {
                    "comparison": "CRMF - ALF",
                    "half_width": width,
                    "density": 16 / (2.0 * width) ** 2,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "n_bootstrap": replicates,
                    "paired_mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_unit": "shared fresh evaluation seed",
                }
            )
    return output


def make_figures(out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for mechanism figures") from exc
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    existing = _read_csv(out / "existing_saturation_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    panels = (
        ("radius_max_saturation_fraction", "Realized-radius cap fraction"),
        ("radius_temporal_variation", "Mean absolute radius update"),
        ("contact_break_rate", "Identity contact-break rate"),
        ("mean_largest_scc_fraction", "Mean largest-SCC fraction"),
    )
    styles = {"alf": ("ALF", "#0072B2", "o"), "crmf": ("CRMF", "#D55E00", "s")}
    for axis, (metric, label) in zip(axes.flat, panels):
        for method, (method_label, color, marker) in styles.items():
            rows = sorted(
                (row for row in existing if row["method"] == method),
                key=lambda row: float(row["density"]),
            )
            axis.plot(
                [float(row["density"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker=marker, color=color, label=method_label,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Density")
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_existing_saturation_context.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_existing_saturation_context.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    frozen = _read_csv(out / "fixed_geometry_response.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for method, (method_label, color, _) in styles.items():
        rows = [row for row in frozen if row["method"] == method]
        x = [float(row["previous_radius"]) for row in rows]
        axes[0].plot(x, [float(row["realized_radius"]) for row in rows], label=method_label, color=color)
        axes[1].plot(x, [float(row["measurement_degree"]) for row in rows], label=method_label, color=color)
    axes[0].plot([1, 8], [1, 8], color="#777777", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Next realized radius")
    axes[1].set_ylabel("Integer measurement degree")
    for axis in axes:
        axis.set_xlabel("Previous radius")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.4)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_fixed_geometry_response.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_fixed_geometry_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    paired = _read_csv(out / "replay_paired_bootstrap.csv")
    chosen = [
        row for row in paired
        if row["metric"] == "mean_abs_realized_radius_update"
    ]
    fig, axis = plt.subplots(figsize=(7.0, 2.9))
    source_order = list(SOURCE_METHODS)
    width_order = list(FORMAL_WIDTHS)
    x_values = []
    labels = []
    means = []
    lows = []
    highs = []
    for source_index, source in enumerate(source_order):
        for width_index, width in enumerate(width_order):
            matches = [
                row for row in chosen
                if row["trajectory_source"] == source
                and abs(float(row["half_width"]) - width) < EPSILON
            ]
            if not matches:
                continue
            row = matches[0]
            x_values.append(source_index * 4 + width_index)
            labels.append(f"{source}\nL={width:g}")
            mean = float(row["paired_mean_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
    axis.errorbar(x_values, means, yerr=[lows, highs], fmt="o", color="#333333", capsize=2)
    axis.axhline(0.0, color="#999999", linewidth=0.8)
    axis.set_xticks(x_values, labels, rotation=28, ha="right")
    axis.set_ylabel("CRMF - ALF radius update")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "fig_common_trajectory_replay.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_common_trajectory_replay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _manifest(root: Path, out: Path, args: argparse.Namespace, counts: dict[str, int]) -> dict[str, Any]:
    sources = (
        "eal_mfg_icra/core.py",
        "eal_mfg_icra/scripts/neighborhood_bench.py",
        "eal_mfg_icra/scripts/innovation_mechanism_study.py",
        "experiments/adaptive_feedback_matched/crdf_matched_raw.csv",
    )
    return {
        "schema_version": "alf-innovation-mechanism/v1",
        "status": "completed",
        "scope": "P0 existing-evidence reaggregation and P1 fixed-geometry, common-trajectory, and fresh-seed closed-loop tests",
        "methods": list(METHODS),
        "trajectory_sources": list(SOURCE_METHODS),
        "half_widths": list(args.half_widths),
        "replay_seeds": list(args.replay_seeds),
        "closed_loop_seeds": list(args.closed_loop_seeds),
        "steps": args.steps,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "unit": "episode/shared seed",
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
        "metric_boundaries": {
            "joint_unsaturated": "Both replayed methods have clipped command and realized radius strictly inside shared bounds at the agent-step; transition subset requires both endpoints.",
            "existing_radius_max_saturation_fraction": "Existing run_cell fraction of realized agent-state radii at the effective upper cap, including initialization.",
            "causal_scope": "Replay isolates radius/graph response to a common state trajectory; autonomous closed-loop rows include trajectory divergence.",
            "statistical_unit": "Agent and time rows are diagnostics, not independent replicates.",
        },
        "counts": counts,
        "sources": [
            {"path": relative, "sha256": _sha256(root / relative), "bytes": (root / relative).stat().st_size}
            for relative in sources
        ],
        "outputs": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--out", type=Path, default=Path("experiments/alf_innovation_mechanism"))
    parser.add_argument("--half-widths", type=float, nargs="+", default=None)
    parser.add_argument("--replay-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--closed-loop-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.mode == "smoke":
        args.half_widths = tuple(args.half_widths or (14.0,))
        args.replay_seeds = tuple(args.replay_seeds or (23_000,))
        args.closed_loop_seeds = tuple(args.closed_loop_seeds or (24_000,))
        args.steps = args.steps or 12
    else:
        args.half_widths = tuple(args.half_widths or FORMAL_WIDTHS)
        args.replay_seeds = tuple(args.replay_seeds or FORMAL_REPLAY_SEEDS)
        args.closed_loop_seeds = tuple(args.closed_loop_seeds or FORMAL_CLOSED_LOOP_SEEDS)
        args.steps = args.steps or 200
    if args.bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if set(args.replay_seeds).intersection(args.closed_loop_seeds):
        raise ValueError("replay and closed-loop seed banks must be disjoint")

    root = Path(__file__).resolve().parents[2]
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    existing = reaggregate_existing(root)
    frozen = fixed_geometry_rows()
    _write_csv(out / "existing_saturation_summary.csv", existing)
    _write_csv(out / "fixed_geometry_response.csv", frozen)

    trace_path = out / "replay_feedback_chain.csv.gz"
    replay = run_replay_study(args.half_widths, args.replay_seeds, args.steps, trace_path)
    replay_paired = paired_replay_summary(replay, args.bootstrap_replicates)
    _write_csv(out / "replay_episode_summary.csv", replay)
    _write_csv(out / "replay_paired_bootstrap.csv", replay_paired)

    closed = run_fresh_closed_loop(args.half_widths, args.closed_loop_seeds, args.steps)
    closed_paired = paired_closed_loop_summary(closed, args.bootstrap_replicates)
    _write_csv(out / "fresh_closed_loop_episodes.csv", closed)
    _write_csv(out / "fresh_closed_loop_paired_bootstrap.csv", closed_paired)
    make_figures(out)

    counts = {
        "existing_summary_rows": len(existing),
        "fixed_geometry_rows": len(frozen),
        "replay_episode_rows": len(replay),
        "replay_trace_rows": len(SOURCE_METHODS) * len(args.half_widths)
        * len(args.replay_seeds) * len(METHODS) * (args.steps + 1) * 16,
        "replay_paired_rows": len(replay_paired),
        "fresh_closed_loop_episode_rows": len(closed),
        "fresh_closed_loop_paired_rows": len(closed_paired),
    }
    manifest = _manifest(root, out, args, counts)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
