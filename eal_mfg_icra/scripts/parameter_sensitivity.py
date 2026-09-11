"""One-at-a-time parameter sensitivity of ALF around the canonical setting.

Motivation
----------
Round-4 review (minor m10) noted that the paper reports a single gamma = 0
ablation and no scan over the remaining controller parameters, so a reader cannot
tell which choices the conclusions actually depend on.  This script runs a
predeclared one-at-a-time (OAT) grid over the five free parameters of ALF --
target degree ``k``, smoothing ``alpha``, retention margin ``delta_m``,
alignment exponent ``beta`` and probe radius ``r_b`` -- against the canonical
setting, on the same paired seed bank and the same densities.  Every cell is
paired with the canonical cell on the same seed, so the estimand is a paired
difference and the baseline enters once per density.

The sweep is deliberately OAT rather than a full grid: with five parameters and
three values a full grid would be 243 configurations per density, and the point
here is direction and magnitude of the leading sensitivities, not an optimum.

Usage
-----
    python -m eal_mfg_icra.scripts.parameter_sensitivity \
        --out experiments/parameter_sensitivity_20260910
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from ..core import EALMFGSimulator, SimulationConfig, run_rollout
from ..metrics import (
    compute_connectivity_metrics,
    compute_metrics,
    contact_break_rate,
    neighbor_loss_rate,
)
from .matched_load_regime_study import initial_state
from .neighborhood_bench import AnalyticalAlignmentActor, matched_half_width

N_AGENTS = 16
STEPS = 200
DENSITIES = (0.012345679012345678, 0.03305785123966942, 0.08163265306122448)
EVALUATION_SEEDS = tuple(range(20_000, 20_050))
DEVELOPMENT_SEEDS = tuple(range(21_000, 21_020))
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_910

CANONICAL = {
    "target_degree": 6.0,
    "radius_smoothing": 0.9,
    "contact_retention_margin": 0.25,
    "beta": 0.5,
    "base_radius": 4.0,
}
GRID = (
    ("target_degree", (4.0, 6.0, 9.0), "k"),
    ("radius_smoothing", (0.0, 0.5, 0.9), "alpha"),
    ("contact_retention_margin", (0.0, 0.25, 0.75), "delta_m"),
    ("beta", (0.0, 0.5, 1.5), "beta"),
    ("base_radius", (2.0, 4.0, 7.0), "r_b"),
)
REPORT_METRICS = (
    "final_fragmentation",
    "mean_largest_scc_fraction",
    "polar_order",
    "mean_degree",
    "radius_temporal_variation_direct",
    "radius_max_saturation_fraction",
    "contact_break_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    frac = position - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _config(half_width: float, seed: int, overrides: dict[str, float]) -> SimulationConfig:
    settings = dict(CANONICAL)
    settings.update(overrides)
    # The margin has to stay inside the domain (core.py enforces
    # max_radius + margin <= L), so a delta_m variant also moves the cap at the
    # densest cell.  That coupling is reported with the result rather than hidden.
    max_radius = min(8.0, half_width - settings["contact_retention_margin"])
    return SimulationConfig(
        dim=2,
        space_half_width=half_width,
        dt=0.5,
        velocity_limit=1.0,
        acceleration_limit=1.0,
        noise_std=0.0,
        beta=settings["beta"],
        base_radius=settings["base_radius"],
        target_degree=settings["target_degree"],
        min_radius=1.0,
        max_radius=max_radius,
        radius_exponent=0.5,
        radius_smoothing=settings["radius_smoothing"],
        contact_retention_margin=settings["contact_retention_margin"],
        flocking_normalization="population",
        seed=seed,
    )


def _task(task: tuple[str, float, float, int]) -> dict[str, Any]:
    parameter, value, density, seed = task
    overrides: dict[str, float] = {}
    if parameter != "canonical":
        overrides[parameter] = value
    half_width = matched_half_width(N_AGENTS, density)
    config = _config(half_width, seed, overrides)
    state = initial_state("uniform", half_width, seed, n_agents=N_AGENTS)
    simulator = EALMFGSimulator(config, initial_state=state)
    # AnalyticalAlignmentActor carries its own ``beta`` (it reads the weight
    # exponent from the actor, not from SimulationConfig), so it has to be passed
    # explicitly or the beta manipulation is silently inert.
    actor = AnalyticalAlignmentActor(beta=config.beta)
    start = time.perf_counter()
    rollout = run_rollout(simulator, actor, steps=STEPS)
    runtime = time.perf_counter() - start
    radii = [float(value) for item in rollout.radius_history for value in item]
    row: dict[str, Any] = {
        "parameter": parameter,
        "value": value,
        "density": density,
        "half_width": half_width,
        "n_agents": N_AGENTS,
        "seed": seed,
        "steps": STEPS,
        "max_radius": config.max_radius,
        "contact_break_rate": contact_break_rate(rollout.neighbor_history),
        "neighbor_loss_rate": neighbor_loss_rate(rollout.neighbor_history),
        "radius_max_saturation_fraction": sum(
            item >= config.max_radius - 1e-12 for item in radii
        ) / len(radii),
        "radius_temporal_variation_direct": statistics.fmean(
            abs(float(rollout.radius_history[t][i]) - float(rollout.radius_history[t - 1][i]))
            for t in range(1, len(rollout.radius_history))
            for i in range(N_AGENTS)
        ),
        "rollout_wall_s": runtime,
        **compute_metrics(rollout),
        **compute_connectivity_metrics(rollout),
    }
    return row


def _paired(values: Sequence[float], replicates: int, rng: random.Random) -> dict[str, float]:
    values = tuple(float(value) for value in values)
    draws = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(replicates)]
    return {
        "paired_mean_difference": statistics.fmean(values),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "positive_fraction": statistics.fmean(value > 0.0 for value in values),
        "n_pairs": len(values),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--out", type=Path, default=Path("experiments/parameter_sensitivity"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVALUATION_SEEDS))
    parser.add_argument("--densities", type=float, nargs="+", default=list(DENSITIES))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)

    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    if set(args.seeds).intersection(DEVELOPMENT_SEEDS):
        raise ValueError("evaluation and development seed banks must be disjoint")

    tasks = [
        (parameter, float(value), float(density), int(seed))
        for parameter, values, _ in GRID
        for value in values
        for density in args.densities
        for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} episodes", flush=True)
    _write_csv(out / "episodes.csv", rows)

    index: dict[tuple[float, str, float, int], dict[str, Any]] = {
        (float(r["density"]), str(r["parameter"]), float(r["value"]), int(r["seed"])): r for r in rows
    }
    rng = random.Random(BOOTSTRAP_SEED)
    comparisons: list[dict[str, Any]] = []
    for parameter, values, label in GRID:
        for value in values:
            if value == CANONICAL[parameter]:
                continue
            for density in args.densities:
                baseline_seeds = sorted(s for (d, p, v, s) in index if d == density and p == parameter and v == CANONICAL[parameter])
                variant_seeds = sorted(s for (d, p, v, s) in index if d == density and p == parameter and v == value)
                if baseline_seeds != variant_seeds:
                    raise ValueError(f"unpaired seeds for {parameter}={value} at rho={density}")
                for metric in REPORT_METRICS:
                    diffs = [
                        float(index[(density, parameter, value, seed)][metric])
                        - float(index[(density, parameter, CANONICAL[parameter], seed)][metric])
                        for seed in variant_seeds
                    ]
                    comparisons.append(
                        {
                            "parameter": parameter,
                            "label": label,
                            "value": value,
                            "canonical_value": CANONICAL[parameter],
                            "density": density,
                            "comparison": f"{label}={value:g} minus {label}={CANONICAL[parameter]:g}",
                            "metric": metric,
                            **_paired(diffs, args.bootstrap_replicates, rng),
                        }
                    )
    _write_csv(out / "paired_vs_canonical.csv", comparisons)

    summary: list[dict[str, Any]] = []
    for parameter, values, label in GRID:
        for metric in REPORT_METRICS:
            blocks = [c for c in comparisons if c["parameter"] == parameter and c["metric"] == metric]
            resolved = sum(1 for c in blocks if c["ci95_low"] > 0.0 or c["ci95_high"] < 0.0)
            magnitude = max((abs(c["paired_mean_difference"]) for c in blocks), default=0.0)
            summary.append(
                {
                    "parameter": parameter,
                    "label": label,
                    "metric": metric,
                    "n_cells": len(blocks),
                    "n_intervals_excluding_zero": resolved,
                    "max_abs_paired_difference": magnitude,
                }
            )
    _write_csv(out / "sensitivity_summary.csv", summary)

    manifest = {
        "schema_version": "alf-parameter-sensitivity/v1",
        "status": "completed",
        "question": (
            "Which ALF parameters do the reported conclusions actually depend on, "
            "holding everything else at the canonical setting?"
        ),
        "design": "one-at-a-time grid, paired with the canonical cell on the same seed bank",
        "canonical": CANONICAL,
        "grid": {name: {"values": list(values), "label": label} for name, values, label in GRID},
        "protocol": {
            "n_agents": N_AGENTS,
            "steps": STEPS,
            "densities": list(args.densities),
            "radius_exponent": 0.5,
            "min_radius": 1.0,
            "interaction_radius_cap": 8.0,
            "physical_sensing_envelope": 8.25,
            "estimand": "paired difference against the canonical cell at the same density and seed",
        },
        "evaluation_seeds": list(args.seeds),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "episodes": len(rows),
        "sources": [
            {"path": path, "sha256": _sha256(root / path)}
            for path in (
                "eal_mfg_icra/core.py",
                "eal_mfg_icra/metrics.py",
                "eal_mfg_icra/scripts/neighborhood_bench.py",
                "eal_mfg_icra/scripts/matched_load_regime_study.py",
                "eal_mfg_icra/scripts/parameter_sensitivity.py",
            )
        ],
        "outputs": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\n=== leading sensitivities (|paired mean| over the tested cells) ===")
    ranked = sorted(
        [row for row in summary],
        key=lambda row: (-row["n_intervals_excluding_zero"], -row["max_abs_paired_difference"]),
    )
    for row in ranked[:14]:
        print(
            f"  {row['label']:<8}{row['metric']:<30}"
            f"cells_excluding_zero={row['n_intervals_excluding_zero']}/{row['n_cells']}"
            f"  max|diff|={row['max_abs_paired_difference']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
