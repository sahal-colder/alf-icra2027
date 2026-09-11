"""Held-out 2x2 aperture--command-base study for finite-sensing ALF.

The four controllers differ only in two binary factors: the occupancy
measurement aperture (fixed probe or previous realized radius) and the
multiplicative command base (fixed probe radius or previous realized radius).
Each output cell has a compressed per-agent/per-step trace and an episode CSV;
all inferential summaries use paired held-out episode seeds, never trace rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

from ..core import DensityAdaptiveNeighborhood, EALMFGSimulator, PopulationState, _clip, run_rollout
from ..metrics import (
    compute_connectivity_metrics,
    compute_metrics,
    contact_break_rate,
    fragmentation_score,
    neighbor_loss_rate,
    scc_summary,
)
from .matched_load_regime_study import GEOMETRIES, initial_state
from .neighborhood_bench import (
    AnalyticalAlignmentActor,
    CurrentRadiusDegreeNeighborhood,
    CurrentRadiusMultiplicativeNeighborhood,
    FixedProbeCurrentBaseNeighborhood,
    _base_config,
    matched_half_width,
)


METHODS = ("alf", "crdf", "fpcm", "crmf")
METHOD_FACTORS = {
    "alf": {"aperture": "fixed_probe", "command_base": "fixed_probe"},
    "crdf": {"aperture": "current_radius", "command_base": "fixed_probe"},
    "fpcm": {"aperture": "fixed_probe", "command_base": "current_radius"},
    "crmf": {"aperture": "current_radius", "command_base": "current_radius"},
}
DENSITIES = (0.0123, 0.0204, 0.0331)
POPULATIONS = (16, 32)
EVALUATION_SEEDS = tuple(range(26_000, 26_050))
DEVELOPMENT_SEEDS = tuple(range(25_000, 25_020))
PRIMARY_METRICS = (
    "mean_abs_radius_update",
    "contact_break_rate",
    "mean_largest_scc_fraction",
)
EPSILON = 1e-12
BOOTSTRAP_SEED = 20260909


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def _mean_interval(
    values: Sequence[float], replicates: int, rng: random.Random
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("bootstrap requires at least one value")
    estimates = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    ]
    return statistics.fmean(values), _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _sign_flip_pvalue(values: Sequence[float], replicates: int, rng: random.Random) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return math.nan
    observed = abs(statistics.fmean(values))
    extreme = 0
    for _ in range(replicates):
        estimate = abs(statistics.fmean(value if rng.randrange(2) else -value for value in values))
        extreme += int(estimate >= observed - EPSILON)
    return (extreme + 1) / (replicates + 1)


def _safe_density(density: float) -> str:
    return f"rho_{density:.4f}".replace(".", "p")


def _neighborhood(method: str, simulator: EALMFGSimulator) -> DensityAdaptiveNeighborhood:
    if method == "alf":
        return simulator.neighborhood
    if method == "crdf":
        return CurrentRadiusDegreeNeighborhood(simulator.config)
    if method == "fpcm":
        return FixedProbeCurrentBaseNeighborhood(simulator.config)
    if method == "crmf":
        return CurrentRadiusMultiplicativeNeighborhood(simulator.config)
    raise ValueError(f"unsupported factorial method: {method}")


def _make_simulator(
    method: str, half_width: float, seed: int, state: PopulationState
) -> EALMFGSimulator:
    cfg = _base_config(half_width, seed)
    simulator = EALMFGSimulator(cfg, initial_state=state)
    simulator.neighborhood = _neighborhood(method, simulator)
    return simulator


def _trace_episode(
    method: str,
    geometry: str,
    density: float,
    n_agents: int,
    seed: int,
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    half_width = matched_half_width(n_agents, density)
    start_state = initial_state(geometry, half_width, seed, n_agents=n_agents)
    simulator = _make_simulator(method, half_width, seed, start_state)
    rollout_start = time.perf_counter()
    rollout = run_rollout(simulator, AnalyticalAlignmentActor(), steps=steps)
    rollout_runtime = time.perf_counter() - rollout_start
    metrics = compute_metrics(rollout)
    connectivity = compute_connectivity_metrics(rollout)
    cfg = simulator.config
    audit_current = CurrentRadiusDegreeNeighborhood(cfg)
    previous_graph = None
    previous_measurement: tuple[int, ...] | None = None
    controller_times: list[float] = []
    trace_rows: list[dict[str, Any]] = []
    fixed_degrees_all: list[int] = []
    current_degrees_all: list[int] = []
    measurement_degrees_all: list[int] = []
    command_values: list[float] = []
    command_updates: list[float] = []
    radius_updates: list[float] = []
    threshold_crossings = 0
    retained_total = 0
    fresh_total = 0

    for step, state in enumerate(rollout.states):
        radii = tuple(float(value) for value in rollout.radius_history[step])
        prior_radii = None if step == 0 else rollout.radius_history[step - 1]
        fixed_degrees = simulator.neighborhood.base_degrees(state.positions)
        current_reference = radii if prior_radii is None else prior_radii
        current_degrees = audit_current.current_degrees(state.positions, current_reference)
        measurement = fixed_degrees if method in {"alf", "fpcm"} or step == 0 else current_degrees
        bases = (
            (cfg.base_radius,) * n_agents
            if method in {"alf", "crdf"} or step == 0
            else tuple(float(value) for value in prior_radii)
        )
        command_unclipped = tuple(
            float(bases[i])
            * (cfg.target_degree / max(float(measurement[i]), 1.0)) ** cfg.radius_exponent
            for i in range(n_agents)
        )
        command_clipped = tuple(
            _clip(value, cfg.min_radius, cfg.max_radius) for value in command_unclipped
        )
        controller_start = time.perf_counter()
        audited_radii = simulator.neighborhood.radii(state.positions, prior_radii)
        audited_graph = simulator.neighborhood.neighbors(state.positions, audited_radii, previous_graph)
        controller_elapsed = time.perf_counter() - controller_start
        if tuple(float(value) for value in audited_radii) != radii or audited_graph != rollout.neighbor_history[step]:
            raise RuntimeError("controller trace does not reproduce rollout neighborhood")
        controller_times.append(controller_elapsed)
        graph = audited_graph
        weak = fragmentation_score(graph)
        directed = scc_summary(graph)
        in_degrees = [0] * n_agents
        for local in graph:
            for neighbor in local:
                in_degrees[neighbor] += 1
        for agent in range(n_agents):
            fresh = {
                other
                for other in range(n_agents)
                if other != agent
                and simulator.neighborhood.pair_distance_sq(state.positions, agent, other)
                <= radii[agent] * radii[agent]
            }
            selected = set(graph[agent])
            retained = selected.difference(fresh)
            retained_total += len(retained)
            fresh_total += len(fresh)
            if prior_radii is None:
                signed_update = 0.0
                previous_command = command_clipped[agent]
                contact_break = 0
                degree_loss = 0
                crossing = 0
            else:
                signed_update = radii[agent] - float(prior_radii[agent])
                previous_command = float(trace_rows[-n_agents + agent]["command_clipped"])
                contact_break = int(bool(set(previous_graph[agent]).difference(selected)))
                degree_loss = int(len(selected) < len(previous_graph[agent]))
                was_above = previous_measurement[agent] >= cfg.target_degree
                is_above = measurement[agent] >= cfg.target_degree
                crossing = 0 if was_above == is_above else (1 if is_above else -1)
                threshold_crossings += int(crossing != 0)
            fixed_degrees_all.append(int(fixed_degrees[agent]))
            current_degrees_all.append(int(current_degrees[agent]))
            measurement_degrees_all.append(int(measurement[agent]))
            command_values.append(float(command_clipped[agent]))
            if prior_radii is not None:
                command_updates.append(float(command_clipped[agent]) - previous_command)
                radius_updates.append(signed_update)
            trace_rows.append(
                {
                    "method": method,
                    "aperture": METHOD_FACTORS[method]["aperture"],
                    "command_base_factor": METHOD_FACTORS[method]["command_base"],
                    "geometry": geometry,
                    "density": density,
                    "n_agents": n_agents,
                    "half_width": half_width,
                    "seed": seed,
                    "step": step,
                    "agent_id": agent,
                    "fixed_probe_degree": fixed_degrees[agent],
                    "current_radius_degree": current_degrees[agent],
                    "measurement_degree": measurement[agent],
                    "count_threshold_crossing": crossing,
                    "command_base": bases[agent],
                    "command_unclipped": command_unclipped[agent],
                    "command_clipped": command_clipped[agent],
                    "command_was_clipped": int(abs(command_unclipped[agent] - command_clipped[agent]) > EPSILON),
                    "realized_radius": radii[agent],
                    "realized_min_saturated": int(radii[agent] <= cfg.min_radius + EPSILON),
                    "realized_max_saturated": int(radii[agent] >= cfg.max_radius - EPSILON),
                    "signed_radius_update": signed_update,
                    "absolute_radius_update": abs(signed_update),
                    "signed_command_update": float(command_clipped[agent]) - previous_command,
                    "absolute_command_update": abs(float(command_clipped[agent]) - previous_command),
                    "fresh_neighbor_count": len(fresh),
                    "retained_neighbor_count": len(retained),
                    "realized_out_degree": len(selected),
                    "realized_in_degree": in_degrees[agent],
                    "contact_break": contact_break,
                    "degree_loss": degree_loss,
                    "weak_fragmentation": weak["fragmentation"],
                    "largest_scc_fraction": directed["largest_scc_fraction"],
                    "controller_wall_s_per_agent_step": controller_elapsed / n_agents,
                }
            )
        previous_graph = graph
        previous_measurement = tuple(int(value) for value in measurement)

    radius_values = [float(value) for item in rollout.radius_history for value in item]
    episode: dict[str, Any] = {
        "method": method,
        "aperture": METHOD_FACTORS[method]["aperture"],
        "command_base_factor": METHOD_FACTORS[method]["command_base"],
        "geometry": geometry,
        "density": density,
        "n_agents": n_agents,
        "half_width": half_width,
        "seed": seed,
        "steps": steps,
        "physical_sensing_envelope": 8.25,
        "mean_fixed_probe_degree": statistics.fmean(fixed_degrees_all),
        "mean_current_radius_degree": statistics.fmean(current_degrees_all),
        "mean_measurement_degree": statistics.fmean(measurement_degrees_all),
        "mean_clipped_command": statistics.fmean(command_values),
        "mean_abs_radius_update": statistics.fmean(abs(value) for value in radius_updates) if radius_updates else 0.0,
        "mean_abs_command_update": statistics.fmean(abs(value) for value in command_updates) if command_updates else 0.0,
        "radius_max_saturation_fraction": sum(value >= cfg.max_radius - EPSILON for value in radius_values) / len(radius_values),
        "radius_min_saturation_fraction": sum(value <= cfg.min_radius + EPSILON for value in radius_values) / len(radius_values),
        "command_clip_fraction": sum(abs(float(row["command_unclipped"]) - float(row["command_clipped"])) > EPSILON for row in trace_rows) / len(trace_rows),
        "count_threshold_crossing_rate": threshold_crossings / max(n_agents * steps, 1),
        "mean_fresh_neighbor_count": fresh_total / len(trace_rows),
        "mean_retained_neighbor_count": retained_total / len(trace_rows),
        "neighbor_loss_rate": neighbor_loss_rate(rollout.neighbor_history),
        "contact_break_rate": contact_break_rate(rollout.neighbor_history),
        "controller_wall_s_per_agent_step": sum(controller_times) / max(len(controller_times) * n_agents, 1),
        "rollout_wall_s": rollout_runtime,
        **metrics,
        **connectivity,
    }
    return episode, trace_rows


def _cell_paths(out: Path, method: str, n_agents: int, density: float, geometry: str) -> tuple[Path, Path, Path]:
    cell = out / "cells" / method / f"n_{n_agents}" / _safe_density(density) / geometry
    return cell / "episodes.csv", cell / "agent_steps.csv.gz", cell / "cell_metadata.json"


def _run_cell_task(task: tuple[str, str, float, int, tuple[int, ...], int, str]) -> dict[str, Any]:
    method, geometry, density, n_agents, seeds, steps, out_text = task
    out = Path(out_text)
    episode_path, trace_path, metadata_path = _cell_paths(out, method, n_agents, density, geometry)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    trace_temp = trace_path.with_suffix(trace_path.suffix + ".tmp")
    with gzip.open(trace_temp, "wt", newline="", encoding="utf-8") as trace_stream:
        writer: csv.DictWriter | None = None
        for seed in seeds:
            episode, trace = _trace_episode(method, geometry, density, n_agents, seed, steps)
            rows.append(episode)
            if writer is None:
                writer = csv.DictWriter(trace_stream, fieldnames=list(trace[0]))
                writer.writeheader()
            writer.writerows(trace)
    episode_temp = episode_path.with_suffix(".csv.tmp")
    _write_csv(episode_temp, rows)
    trace_temp.replace(trace_path)
    episode_temp.replace(episode_path)
    metadata = {
        "status": "completed",
        "method": method,
        "geometry": geometry,
        "density": density,
        "n_agents": n_agents,
        "seeds": list(seeds),
        "steps": steps,
        "episode_rows": len(rows),
        "trace_rows": len(rows) * (steps + 1) * n_agents,
        "episodes_sha256": _sha256(episode_path),
        "trace_sha256": _sha256(trace_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _cell_records(rows: Sequence[dict[str, Any]]) -> Iterable[tuple[tuple[int, float, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["n_agents"]), float(row["density"]), str(row["geometry"])), []).append(row)
    yield from sorted(grouped.items())


def factorial_contrasts(rows: Sequence[dict[str, Any]], metric: str) -> list[dict[str, float]]:
    """Return paired 2x2 contrasts for one complete condition and metric."""

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = row
    output: list[dict[str, float]] = []
    for seed, methods in sorted(by_seed.items()):
        if set(methods) != set(METHODS):
            raise ValueError(f"incomplete paired factorial for seed {seed}")
        alf = float(methods["alf"][metric])
        crdf = float(methods["crdf"][metric])
        fpcm = float(methods["fpcm"][metric])
        crmf = float(methods["crmf"][metric])
        output.append(
            {
                "seed": float(seed),
                "aperture_main_effect": 0.5 * ((crdf - alf) + (crmf - fpcm)),
                "base_main_effect": 0.5 * ((fpcm - alf) + (crmf - crdf)),
                "aperture_base_interaction": crmf - fpcm - crdf + alf,
                # The task-packet's two requested difference-in-differences
                # expressions are algebraically the same interaction.
                "requested_measurement_did": (crmf - fpcm) - (crdf - alf),
                "requested_base_did": (crmf - crdf) - (fpcm - alf),
            }
        )
    return output


def _summarize_effect(
    values: Sequence[float], replicates: int, permutations: int, rng: random.Random
) -> dict[str, float]:
    mean, low, high = _mean_interval(values, replicates, rng)
    return {
        "paired_mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "median_difference": statistics.median(values),
        "positive_fraction": statistics.fmean(value > 0.0 for value in values),
        "sign_consistency": max(
            statistics.fmean(value >= 0.0 for value in values),
            statistics.fmean(value <= 0.0 for value in values),
        ),
        "sign_flip_pvalue": _sign_flip_pvalue(values, permutations, rng),
    }


def _aggregate_stratified(
    groups: Sequence[list[dict[str, float]]], effect: str, replicates: int, rng: random.Random
) -> tuple[float, float, float, list[float]]:
    per_episode = [float(item[effect]) for group in groups for item in group]
    point = statistics.fmean(statistics.fmean(float(item[effect]) for item in group) for group in groups)
    estimates = []
    for _ in range(replicates):
        estimates.append(
            statistics.fmean(
                statistics.fmean(float(group[rng.randrange(len(group))][effect]) for _ in group)
                for group in groups
            )
        )
    return point, _percentile(estimates, 0.025), _percentile(estimates, 0.975), per_episode


def _holm_step_down(rows: Sequence[dict[str, Any]]) -> None:
    """Apply a monotone Holm adjustment in place to one declared test family."""

    running = 0.0
    for index, row in enumerate(sorted(rows, key=lambda item: float(item["sign_flip_pvalue"]))):
        running = max(
            running,
            min(1.0, (len(rows) - index) * float(row["sign_flip_pvalue"])),
        )
        row["holm_adjusted_pvalue"] = running


def factorial_statistics(
    rows: Sequence[dict[str, Any]], bootstrap_replicates: int, permutations: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    rng = random.Random(BOOTSTRAP_SEED)
    cells = list(_cell_records(rows))
    for (n_agents, density, geometry), cell in cells:
        for metric in PRIMARY_METRICS:
            contrasts = factorial_contrasts(cell, metric)
            for effect in (
                "aperture_main_effect",
                "base_main_effect",
                "aperture_base_interaction",
                "requested_measurement_did",
                "requested_base_did",
            ):
                values = [float(row[effect]) for row in contrasts]
                results.append(
                    {
                        "scope": "cell",
                        "n_agents": n_agents,
                        "density": density,
                        "geometry": geometry,
                        "metric": metric,
                        "effect": effect,
                        "n_pairs": len(values),
                        "bootstrap_unit": "shared held-out evaluation seed",
                        "holm_family": "none_cellwise",
                        **_summarize_effect(values, bootstrap_replicates, permutations, rng),
                    }
                )
    for n_agents in sorted({int(row["n_agents"]) for row in rows}):
        selected_cells = [cell for key, cell in cells if key[0] == n_agents]
        for metric in PRIMARY_METRICS:
            groups = [factorial_contrasts(cell, metric) for cell in selected_cells]
            for effect in ("aperture_main_effect", "base_main_effect", "aperture_base_interaction"):
                mean, low, high, values = _aggregate_stratified(groups, effect, bootstrap_replicates, rng)
                results.append(
                    {
                        "scope": "n_agents_stratified_aggregate",
                        "n_agents": n_agents,
                        "density": "all",
                        "geometry": "all",
                        "metric": metric,
                        "effect": effect,
                        "n_pairs": len(values),
                        "bootstrap_unit": "within-cell shared held-out seed; equal-weighted density-geometry cells",
                        "holm_family": (
                            f"primary_interaction_n{n_agents}"
                            if effect == "aperture_base_interaction"
                            else f"secondary_main_effect_n{n_agents}"
                        ),
                        "paired_mean_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "median_difference": statistics.median(values),
                        "positive_fraction": statistics.fmean(value > 0.0 for value in values),
                        "sign_consistency": max(statistics.fmean(value >= 0.0 for value in values), statistics.fmean(value <= 0.0 for value in values)),
                        "sign_flip_pvalue": _sign_flip_pvalue(values, permutations, rng),
                    }
                )
    primary_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        family = str(row["holm_family"])
        if family.startswith("primary_interaction_n"):
            primary_by_family.setdefault(family, []).append(row)
    for family_rows in primary_by_family.values():
        _holm_step_down(family_rows)
    # The main effects are declared as a *secondary* family per population size.
    # They were previously labelled but left unadjusted; adjusting them here
    # keeps the artifact and the manuscript on one convention.  The primary
    # interaction rows are untouched, so their published values cannot move.
    secondary_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        family = str(row["holm_family"])
        if family.startswith("secondary_main_effect_n"):
            secondary_by_family.setdefault(family, []).append(row)
    for family_rows in secondary_by_family.values():
        _holm_step_down(family_rows)
    for row in results:
        row.setdefault("holm_adjusted_pvalue", math.nan)
    return results


def _cell_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = (
        "mean_abs_radius_update", "mean_abs_command_update", "contact_break_rate",
        "neighbor_loss_rate", "mean_largest_scc_fraction", "mean_fragmentation",
        "radius_max_saturation_fraction", "command_clip_fraction",
        "count_threshold_crossing_rate", "controller_wall_s_per_agent_step",
    )
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["n_agents"]), float(row["density"]), str(row["geometry"])), []).append(row)
    output = []
    for (method, n_agents, density, geometry), cell in sorted(grouped.items()):
        item: dict[str, Any] = {
            "method": method, "n_agents": n_agents, "density": density, "geometry": geometry,
            "n_episodes": len(cell),
        }
        item.update({f"{metric}_mean": statistics.fmean(float(row[metric]) for row in cell) for metric in numeric})
        output.append(item)
    return output


def _manifest(root: Path, out: Path, args: argparse.Namespace, cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sources = (
        "eal_mfg_icra/core.py",
        "eal_mfg_icra/metrics.py",
        "eal_mfg_icra/scripts/neighborhood_bench.py",
        "eal_mfg_icra/scripts/matched_load_regime_study.py",
        "eal_mfg_icra/scripts/aperture_base_factorial.py",
    )
    outputs = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs.append({"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return {
        "schema_version": "alf-aperture-base-factorial/v1",
        "status": "completed",
        "methods": METHOD_FACTORS,
        "population_sizes": list(args.populations),
        "densities": list(args.densities),
        "geometries": list(GEOMETRIES),
        "evaluation_seeds": list(args.seeds),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "seed_policy": "Canonical parameters are frozen; no candidate is tuned on development or evaluation seeds. Development and evaluation banks are disjoint by construction.",
        "protocol": {
            "steps": args.steps,
            "base_radius": 4.0,
            "target_degree": 6.0,
            "radius_exponent": 0.5,
            "radius_smoothing": 0.9,
            "contact_retention_margin": 0.25,
            "interaction_radius_cap": 8.0,
            "physical_sensing_envelope": 8.25,
            "dynamics": "shared analytical local velocity alignment",
        },
        "primary_metrics": list(PRIMARY_METRICS),
        "contrast_coding": {
            "aperture_main_effect": "0.5*((CRDF-ALF)+(CRMF-FPCM))",
            "base_main_effect": "0.5*((FPCM-ALF)+(CRMF-CRDF))",
            "interaction": "CRMF-FPCM-CRDF+ALF",
            "requested_dids": "Both difference-in-differences expressions in the task packet equal the same 2x2 interaction under this coding.",
        },
        "statistics": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "permutation_replicates": args.permutation_replicates,
            "interval": "percentile 95% paired bootstrap",
            "aggregate": "resample evaluation seeds within every density-geometry cell, then equal-weight cell means",
            "holm_family": "Separate predeclared three-metric aggregate aperture-base interaction families for N=16 and N=32; monotone Holm step-down adjustment within each population family.",
        },
        "cell_completion": list(cells),
        "exclusions": [],
        "runtime": {"python": sys.version, "pid": os.getpid()},
        "sources": [{"path": path, "sha256": _sha256(root / path), "bytes": (root / path).stat().st_size} for path in sources],
        "outputs": outputs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--out", type=Path, default=Path("experiments/aperture_base_factorial"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--populations", type=int, nargs="+", default=None)
    parser.add_argument("--densities", type=float, nargs="+", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--permutation-replicates", type=int, default=5000)
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="Skip every rollout and recompute factorial_statistics.csv from the frozen episode_summary.csv.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.bootstrap_replicates < 100 or args.permutation_replicates < 100:
        raise ValueError("workers must be positive and statistical replicates must be at least 100")
    if args.mode == "smoke":
        args.steps = args.steps or 6
        args.seeds = tuple(args.seeds or (26_000, 26_001))
        args.populations = tuple(args.populations or (16,))
        args.densities = tuple(args.densities or (0.0204,))
        geometries = ("uniform",)
    else:
        args.steps = args.steps or 200
        args.seeds = tuple(args.seeds or EVALUATION_SEEDS)
        args.populations = tuple(args.populations or POPULATIONS)
        args.densities = tuple(args.densities or DENSITIES)
        geometries = GEOMETRIES
    if set(args.seeds).intersection(DEVELOPMENT_SEEDS):
        raise ValueError("evaluation and development seed banks must be disjoint")
    root = Path(__file__).resolve().parents[2]
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.statistics_only:
        all_rows = _read_csv(out / "episode_summary.csv")
        all_rows.sort(
            key=lambda row: (
                int(row["n_agents"]),
                float(row["density"]),
                str(row["geometry"]),
                int(row["seed"]),
                str(row["method"]),
            )
        )
        stats = factorial_statistics(all_rows, args.bootstrap_replicates, args.permutation_replicates)
        _write_csv(out / "factorial_statistics.csv", stats)
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statistics"]["holm_family"] = (
                "Separate predeclared aggregate families per population size: "
                "primary_interaction_n{16,32} (three metrics) and "
                "secondary_main_effect_n{16,32} (three metrics x two main effects); "
                "monotone Holm step-down adjustment is applied within every family."
            )
            manifest["reanalysis"] = {
                "mode": "statistics-only",
                "episode_source": "episode_summary.csv",
                "episodes": len(all_rows),
                "bootstrap_replicates": args.bootstrap_replicates,
                "permutation_replicates": args.permutation_replicates,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"episodes": len(all_rows), "statistics": len(stats), "mode": "statistics-only"}, indent=2))
        return 0
    tasks = [
        (method, geometry, float(density), int(n_agents), tuple(args.seeds), int(args.steps), str(out))
        for method in METHODS for n_agents in args.populations for density in args.densities for geometry in geometries
    ]
    completed = []
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            completed.append(_run_cell_task(task))
            print(f"completed {index}/{len(tasks)} cells", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_cell_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                completed.append(future.result())
                print(f"completed {index}/{len(tasks)} cells", flush=True)
    all_rows = []
    for method in METHODS:
        for n_agents in args.populations:
            for density in args.densities:
                for geometry in geometries:
                    episode_path, _, _ = _cell_paths(out, method, int(n_agents), float(density), geometry)
                    all_rows.extend(_read_csv(episode_path))
    all_rows.sort(key=lambda row: (int(row["n_agents"]), float(row["density"]), str(row["geometry"]), int(row["seed"]), str(row["method"])))
    _write_csv(out / "episode_summary.csv", all_rows)
    _write_csv(out / "cell_summary.csv", _cell_summary(all_rows))
    stats = factorial_statistics(all_rows, args.bootstrap_replicates, args.permutation_replicates)
    _write_csv(out / "factorial_statistics.csv", stats)
    manifest = _manifest(root, out, args, completed)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(all_rows), "cells": len(completed), "statistics": len(stats)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
