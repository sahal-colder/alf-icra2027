"""Cross-geometry, selected-edge-load matching for ALF operating regimes.

Canonical ALF (k=6, cap=8) defines a development-bank target mean realized
out-degree in each density/geometry condition.  CRMF, range-limited KNN, and a
retention-matched fixed-radius controller receive separate, equal-size
candidate grids and the same number of development episodes per candidate.
The selected controls are frozen before evaluation on a disjoint seed bank.

The matching variable includes retained edges.  It is an interaction-graph
descriptor, not measured radio traffic, computation, or energy consumption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Sequence

from ..core import EALMFGSimulator, PopulationState, Rollout, run_rollout
from ..metrics import compute_connectivity_metrics, compute_metrics, fragmentation_score, scc_summary
from .innovation_mechanism_study import BOOTSTRAP_SEED, _bootstrap_mean_interval
from .neighborhood_bench import (
    AnalyticalAlignmentActor,
    CurrentRadiusMultiplicativeNeighborhood,
    FixedRadiusNeighborhood,
    KNNSimulator,
    _base_config,
    _fixed_config,
    _knn_config,
)


GEOMETRIES = ("uniform", "clustered", "elongated")
METHODS = ("alf", "crmf", "knn", "fixed_retained")
WIDTHS = (11.0, 14.0, 18.0)
TUNING_SEEDS = tuple(range(25_000, 25_020))
EVALUATION_SEEDS = tuple(range(26_000, 26_050))
CANDIDATES = {
    "alf": (8.0,),
    "crmf": (7.0, 7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875, 8.0),
    "knn": (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    "fixed_retained": (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0),
}
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(value: float, half_width: float) -> float:
    return ((float(value) + half_width) % (2.0 * half_width)) - half_width


def initial_state(geometry: str, half_width: float, seed: int, n_agents: int = 16) -> PopulationState:
    """Generate one deterministic paired initial state for a declared geometry."""

    if geometry not in GEOMETRIES:
        raise ValueError(f"unsupported geometry: {geometry}")
    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    offsets = {"uniform": 0, "clustered": 10_000_000, "elongated": 20_000_000}
    rng = random.Random(int(seed) + offsets[geometry])
    if geometry == "uniform":
        positions = tuple(
            (rng.uniform(-half_width, half_width), rng.uniform(-half_width, half_width))
            for _ in range(n_agents)
        )
    elif geometry == "clustered":
        sigma = 0.12 * half_width
        centers = (-0.32 * half_width, 0.32 * half_width)
        positions = tuple(
            (
                _wrap(rng.gauss(centers[i % 2], sigma), half_width),
                _wrap(rng.gauss(0.0, sigma), half_width),
            )
            for i in range(n_agents)
        )
    else:
        positions = tuple(
            (
                rng.uniform(-0.82 * half_width, 0.82 * half_width),
                _wrap(rng.gauss(0.0, 0.07 * half_width), half_width),
            )
            for _ in range(n_agents)
        )
    velocities = tuple(
        (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        for _ in range(n_agents)
    )
    return PopulationState(positions, velocities)


def _make_simulator(
    method: str, candidate: float, half_width: float, seed: int, state: PopulationState
) -> EALMFGSimulator:
    if method == "alf":
        cfg = _base_config(half_width, seed, max_radius_cap=candidate)
        return EALMFGSimulator(cfg, initial_state=state)
    if method == "crmf":
        cfg = _base_config(half_width, seed, max_radius_cap=candidate)
        simulator = EALMFGSimulator(cfg, initial_state=state)
        simulator.neighborhood = CurrentRadiusMultiplicativeNeighborhood(cfg)
        return simulator
    if method == "knn":
        cfg = _knn_config(half_width, seed, int(candidate))
        simulator = KNNSimulator(cfg, k=int(candidate))
        simulator.reset(state=state)
        return simulator
    if method == "fixed_retained":
        cfg = _fixed_config(
            half_width,
            seed,
            candidate,
            contact_retention_margin=0.25,
        )
        simulator = EALMFGSimulator(cfg, initial_state=state)
        simulator.neighborhood = FixedRadiusNeighborhood(cfg)
        return simulator
    raise ValueError(f"unsupported method: {method}")


def _adaptive_commands(
    rollout: Rollout, simulator: EALMFGSimulator, method: str
) -> list[tuple[float, ...]]:
    cfg = simulator.config
    history: list[tuple[float, ...]] = []
    for step, state in enumerate(rollout.states):
        if method == "crmf" and step > 0:
            degrees = simulator.neighborhood.current_degrees(
                state.positions, rollout.radius_history[step - 1]
            )
            bases = rollout.radius_history[step - 1]
        elif method in {"alf", "crmf"}:
            degrees = simulator.neighborhood.base_degrees(state.positions)
            bases = (cfg.base_radius,) * state.n_agents
        else:
            history.append(tuple(float(value) for value in rollout.radius_history[step]))
            continue
        history.append(
            tuple(
                max(
                    cfg.min_radius,
                    min(
                        cfg.max_radius,
                        float(bases[i])
                        * (cfg.target_degree / max(float(degrees[i]), 1.0))
                        ** cfg.radius_exponent,
                    ),
                )
                for i in range(state.n_agents)
            )
        )
    return history


def run_episode(
    method: str,
    candidate: float,
    geometry: str,
    half_width: float,
    seed: int,
    steps: int,
    stage: str,
) -> dict[str, Any]:
    state = initial_state(geometry, half_width, seed)
    simulator = _make_simulator(method, candidate, half_width, seed, state)
    rollout = run_rollout(simulator, AnalyticalAlignmentActor(), steps=steps)
    metrics = compute_metrics(rollout)
    connectivity = compute_connectivity_metrics(rollout)
    commands = _adaptive_commands(rollout, simulator, method)
    radius_updates = [
        abs(float(after[i]) - float(before[i]))
        for before, after in zip(rollout.radius_history, rollout.radius_history[1:])
        for i in range(state.n_agents)
    ]
    command_updates = [
        abs(float(after[i]) - float(before[i]))
        for before, after in zip(commands, commands[1:])
        for i in range(state.n_agents)
    ]
    max_radius = float(simulator.config.max_radius)
    radius_values = [float(value) for snapshot in rollout.radius_history for value in snapshot]
    scc = [scc_summary(graph)["largest_scc_fraction"] for graph in rollout.neighbor_history]
    weak = [fragmentation_score(graph)["fragmentation"] for graph in rollout.neighbor_history]
    control_energy = sum(
        value * value
        for actions in rollout.actions
        for action in actions
        for value in action
    ) / max(state.n_agents * steps, 1)
    return {
        "stage": stage,
        "method": method,
        "candidate": float(candidate),
        "geometry": geometry,
        "half_width": float(half_width),
        "density": state.n_agents / (2.0 * half_width) ** 2,
        "seed": int(seed),
        "steps": int(steps),
        "effective_radius_cap": max_radius,
        "radius_temporal_variation": statistics.fmean(radius_updates) if radius_updates else 0.0,
        "command_temporal_variation": statistics.fmean(command_updates) if command_updates else 0.0,
        "radius_max_saturation_fraction": sum(value >= max_radius - EPSILON for value in radius_values)
        / max(len(radius_values), 1),
        "scc_deficit_integral": sum(1.0 - value for value in scc),
        "disconnected_step_fraction": statistics.fmean(value > 0.0 for value in weak),
        "mean_interaction_area": math.pi * float(connectivity["mean_radius_sq"]),
        "control_energy": control_energy,
        **metrics,
        **connectivity,
    }


def _run_task(task: tuple[str, float, str, float, int, int, str]) -> dict[str, Any]:
    return run_episode(*task)


def _run_tasks(
    tasks: Sequence[tuple[str, float, str, float, int, int, str]], workers: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            output.append(_run_task(task))
            if index % 100 == 0:
                print(f"completed {index}/{len(tasks)} episodes", flush=True)
        return output
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            output.append(future.result())
            if index % 100 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} episodes", flush=True)
    return output


def select_candidates(tuning_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for geometry in GEOMETRIES:
        for width in WIDTHS:
            condition = [
                row for row in tuning_rows
                if row["geometry"] == geometry
                and abs(float(row["half_width"]) - width) < EPSILON
            ]
            alf = [row for row in condition if row["method"] == "alf"]
            if not alf:
                continue
            target = statistics.fmean(float(row["mean_interaction_out_degree"]) for row in alf)
            for method in METHODS:
                method_rows = [row for row in condition if row["method"] == method]
                candidate_rows: list[dict[str, Any]] = []
                for candidate in sorted({float(row["candidate"]) for row in method_rows}):
                    cell = [row for row in method_rows if float(row["candidate"]) == candidate]
                    mean_degree = statistics.fmean(
                        float(row["mean_interaction_out_degree"]) for row in cell
                    )
                    candidate_rows.append(
                        {
                            "method": method,
                            "candidate": candidate,
                            "mean_interaction_out_degree": mean_degree,
                            "absolute_degree_gap": abs(mean_degree - target),
                            "n_tuning_episodes": len(cell),
                        }
                    )
                selected = min(
                    candidate_rows,
                    key=lambda row: (
                        float(row["absolute_degree_gap"]),
                        float(row["candidate"]),
                    ),
                )
                output.append(
                    {
                        "geometry": geometry,
                        "half_width": width,
                        "density": 16 / (2.0 * width) ** 2,
                        "method": method,
                        "selected_candidate": selected["candidate"],
                        "target_alf_mean_out_degree": target,
                        "selected_tuning_mean_out_degree": selected["mean_interaction_out_degree"],
                        "signed_tuning_degree_gap": selected["mean_interaction_out_degree"] - target,
                        "absolute_tuning_degree_gap": selected["absolute_degree_gap"],
                        "n_tuning_episodes_per_candidate": selected["n_tuning_episodes"],
                        "n_candidates": len(candidate_rows),
                        "selection_rule": "minimum absolute development-bank mean realized out-degree gap; lower candidate breaks ties",
                    }
                )
    return output


PAIRED_METRICS = (
    "final_fragmentation",
    "mean_fragmentation",
    "final_largest_scc_fraction",
    "mean_largest_scc_fraction",
    "mean_interaction_out_degree",
    "mean_edge_count",
    "mean_interaction_area",
    "radius_temporal_variation",
    "command_temporal_variation",
    "radius_max_saturation_fraction",
    "neighbor_loss_rate",
    "contact_break_rate",
    "velocity_dispersion",
    "control_energy",
    "scc_deficit_integral",
    "disconnected_step_fraction",
)


def paired_evaluation_summary(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED + 2)
    output: list[dict[str, Any]] = []
    for geometry in GEOMETRIES:
        for width in WIDTHS:
            cell = [
                row for row in rows
                if row["geometry"] == geometry
                and abs(float(row["half_width"]) - width) < EPSILON
            ]
            by_key = {(int(row["seed"]), str(row["method"])): row for row in cell}
            seeds = sorted({int(row["seed"]) for row in cell})
            if any((seed, method) not in by_key for seed in seeds for method in METHODS):
                raise ValueError(f"incomplete evaluation pair for {geometry}, L={width}")
            for method in METHODS[1:]:
                for metric in PAIRED_METRICS:
                    differences = [
                        float(by_key[(seed, method)][metric])
                        - float(by_key[(seed, "alf")][metric])
                        for seed in seeds
                    ]
                    mean, low, high = _bootstrap_mean_interval(
                        differences, replicates, rng
                    )
                    output.append(
                        {
                            "comparison": f"{method} - ALF",
                            "method": method,
                            "geometry": geometry,
                            "half_width": width,
                            "density": 16 / (2.0 * width) ** 2,
                            "metric": metric,
                            "n_pairs": len(differences),
                            "n_bootstrap": replicates,
                            "paired_mean_difference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "bootstrap_unit": "shared held-out evaluation seed",
                        }
                    )
    return output


def make_figure(summary_path: Path, figure_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for the regime figure") from exc
    with summary_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    methods = ("crmf", "knn", "fixed_retained")
    colors = {"crmf": "#D55E00", "knn": "#009E73", "fixed_retained": "#E69F00"}
    markers = {"crmf": "s", "knn": "D", "fixed_retained": "^"}
    labels = {"crmf": "CRMF", "knn": "range-limited KNN", "fixed_retained": "fixed radius + retention"}
    panels = (
        ("mean_interaction_out_degree", "Selected out-degree difference"),
        ("mean_largest_scc_fraction", "Mean largest-SCC difference"),
        ("radius_temporal_variation", "Radius-update difference"),
    )
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(GEOMETRIES), len(panels), figsize=(7.1, 6.1), sharex=True)
    for row_index, geometry in enumerate(GEOMETRIES):
        for column_index, (metric, ylabel) in enumerate(panels):
            axis = axes[row_index, column_index]
            for method in methods:
                points = sorted(
                    (row for row in rows if row["geometry"] == geometry
                     and row["method"] == method and row["metric"] == metric),
                    key=lambda row: float(row["density"]),
                )
                x = [float(row["density"]) for row in points]
                mean = [float(row["paired_mean_difference"]) for row in points]
                lower = [m - float(row["ci95_low"]) for m, row in zip(mean, points)]
                upper = [float(row["ci95_high"]) - m for m, row in zip(mean, points)]
                axis.errorbar(x, mean, yerr=[lower, upper], marker=markers[method],
                              color=colors[method], label=labels[method], capsize=2)
            axis.axhline(0.0, color="#999999", linewidth=0.7)
            axis.set_xscale("log")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.4)
            if row_index == len(GEOMETRIES) - 1:
                axis.set_xlabel("Density")
            if column_index == 0:
                axis.set_ylabel(f"{geometry}\n{ylabel}")
            elif row_index == 0:
                axis.set_title(ylabel)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    global GEOMETRIES, WIDTHS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--out", type=Path, default=Path("experiments/matched_load_regimes"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if args.mode == "smoke":
        geometries = ("uniform",)
        widths = (14.0,)
        tuning_seeds = (25_000,)
        evaluation_seeds = (26_000,)
        steps = args.steps or 12
        candidates = {method: values[:2] if len(values) > 1 else values for method, values in CANDIDATES.items()}
    else:
        geometries = GEOMETRIES
        widths = WIDTHS
        tuning_seeds = TUNING_SEEDS
        evaluation_seeds = EVALUATION_SEEDS
        steps = args.steps or 200
        candidates = CANDIDATES
    if set(tuning_seeds).intersection(evaluation_seeds):
        raise ValueError("tuning and evaluation seeds must be disjoint")

    tuning_tasks = [
        (method, candidate, geometry, width, seed, steps, "tuning")
        for geometry in geometries
        for width in widths
        for method in METHODS
        for candidate in candidates[method]
        for seed in tuning_seeds
    ]
    tuning_rows = _run_tasks(tuning_tasks, args.workers)
    tuning_rows.sort(key=lambda row: (row["geometry"], row["half_width"], row["method"], row["candidate"], row["seed"]))

    original_geometries, original_widths = GEOMETRIES, WIDTHS
    GEOMETRIES, WIDTHS = tuple(geometries), tuple(widths)
    try:
        selection = select_candidates(tuning_rows)
        selected = {
            (row["geometry"], float(row["half_width"]), row["method"]): float(row["selected_candidate"])
            for row in selection
        }
        evaluation_tasks = [
            (method, selected[(geometry, width, method)], geometry, width, seed, steps, "evaluation")
            for geometry in geometries
            for width in widths
            for method in METHODS
            for seed in evaluation_seeds
        ]
        evaluation_rows = _run_tasks(evaluation_tasks, args.workers)
        evaluation_rows.sort(key=lambda row: (row["geometry"], row["half_width"], row["method"], row["seed"]))
        paired = paired_evaluation_summary(evaluation_rows, args.bootstrap_replicates)
    finally:
        GEOMETRIES, WIDTHS = original_geometries, original_widths

    root = Path(__file__).resolve().parents[2]
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "tuning_episodes.csv", tuning_rows)
    _write_csv(out / "selection.csv", selection)
    _write_csv(out / "evaluation_episodes.csv", evaluation_rows)
    _write_csv(out / "paired_bootstrap.csv", paired)
    make_figure(out / "paired_bootstrap.csv", out / "fig_matched_load_regimes")
    manifest = {
        "schema_version": "alf-matched-load-regimes/v1",
        "status": "completed",
        "mode": args.mode,
        "geometries": list(geometries),
        "geometry_definitions": {
            "uniform": "i.i.d. uniform positions on the periodic square",
            "clustered": "two Gaussian clusters at x=+-0.32L with sigma=0.12L",
            "elongated": "x uniform on +-0.82L and y Gaussian with sigma=0.07L",
        },
        "half_widths": list(widths),
        "tuning_seeds": list(tuning_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "steps": steps,
        "methods": list(METHODS),
        "candidate_grids": {method: list(values) for method, values in candidates.items()},
        "selection": {
            "target": "canonical ALF development-bank mean realized out-degree, including retained edges",
            "rule": "minimum absolute mean-out-degree gap; lower candidate breaks ties",
            "boundary": "A close scalar match does not match all graph properties or prove equivalence.",
        },
        "bootstrap": {"replicates": args.bootstrap_replicates, "unit": "shared held-out evaluation seed",
                      "seed": BOOTSTRAP_SEED + 2, "interval": "percentile 95%"},
        "counts": {"tuning_episodes": len(tuning_rows), "selection_rows": len(selection),
                   "evaluation_episodes": len(evaluation_rows), "paired_rows": len(paired)},
        "source_hashes": {
            str(path.relative_to(root)): _sha256(path)
            for path in (
                root / "eal_mfg_icra/core.py",
                root / "eal_mfg_icra/scripts/neighborhood_bench.py",
                root / "eal_mfg_icra/scripts/matched_load_regime_study.py",
            )
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
