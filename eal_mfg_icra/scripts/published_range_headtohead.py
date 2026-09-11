"""Head-to-head against faithful implementations of two published range laws.

Motivation
----------
Round-4 review (M4) objected that every non-ALF comparator in the manuscript is
either a fixed radius, a local-KNN cap, a maximum radius, or a controller the
authors constructed for the 2x2 factorial.  The only literature-derived method
was StarDisplay, and the manuscript itself called it a "transplant".  This
script closes that gap with two *published* adaptive-range laws, implemented
from their stated equations and dropped into the manuscript's shared motor and
sensing contract:

``stardisplay``
    Hildenbrandt, Carere & Hemelrijk, Behavioral Ecology 21(6):1349-1359, 2010,
    Eq. 2:
        R_next = (1 - s) R + s R_max (1 - |N(R)| / n_c)
    ``s = 1 - radius_smoothing``, ``n_c`` the published target count.

``lint``
    Ramanathan & Rosales-Hain, IEEE INFOCOM 2000, Sec. III (LINT: Local
    Information No Topology).  Each node keeps a locally measured neighbour
    count inside its own current range and steps that range by a fixed increment
    whenever the count leaves a tolerance band ``[k_lo, k_hi]``:
        d < k_lo  -> R <- R + delta
        d > k_hi  -> R <- R - delta
        else         R <- R
    LINT has no exponential smoothing: the step *is* the update.  The tolerance
    band is the published mechanism for avoiding chatter around the target.

Both are evaluated against ALF with identical dynamics, identical strictly
local sensing envelope, identical retention margin, identical radius bounds and
the same shared seed bank.  The estimand is the paired ALF-minus-comparator
difference per cell, with a seed bootstrap interval.

Usage
-----
    python -m eal_mfg_icra.scripts.published_range_headtohead --out experiments/published_range_headtohead_20260910
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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

from ..core import DensityAdaptiveNeighborhood, EALMFGSimulator, _clip, run_rollout
from ..metrics import (
    compute_connectivity_metrics,
    compute_metrics,
    contact_break_rate,
    neighbor_loss_rate,
)
from .matched_load_regime_study import initial_state
from .neighborhood_bench import (
    AnalyticalAlignmentActor,
    CurrentRadiusDegreeNeighborhood,
    StarDisplayNeighborhood,
    _base_config,
    matched_half_width,
)

METHODS = ("alf", "stardisplay", "lint")
INTEGRATED_NEIGHBORHOODS = (CurrentRadiusDegreeNeighborhood, StarDisplayNeighborhood)

DENSITIES = (0.012345679012345678, 0.02040816326530612, 0.03305785123966942,
             0.04938271604938271, 0.08163265306122448)
EVALUATION_SEEDS = tuple(range(20_000, 20_050))
DEVELOPMENT_SEEDS = tuple(range(21_000, 21_020))
N_AGENTS = 16
STEPS = 200
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260910
STARDISPLAY_NC = 15.0
LINT_LOW_DEGREE = 4
LINT_HIGH_DEGREE = 8
LINT_STEP = 0.5
PRIMARY_METRICS = (
    "final_fragmentation",
    "mean_fragmentation",
    "final_largest_scc_fraction",
    "mean_largest_scc_fraction",
    "mean_degree",
    "mean_interaction_out_degree",
    "mean_radius",
    "mean_radius_footprint",
    "mean_radius_sq",
    "mean_normalized_comm_cost",
    "radius_max_saturation_fraction",
    "contact_break_rate",
    "neighbor_loss_rate",
    "polar_order",
    "velocity_dispersion",
)


class LintStepNeighborhood(CurrentRadiusDegreeNeighborhood):
    """Published step-based range adaptation (LINT, Ramanathan & Rosales-Hain 2000).

    Every agent already knows its neighbour count inside its own current range,
    because that is the graph it used for control.  LINT applies a fixed
    increment to the transmit range whenever that count leaves a tolerance band
    and leaves the range unchanged otherwise.  The band replaces the
    multiplicative correction used by ALF; there is no smoothing because the
    step itself is the (already damped) update.
    """

    def __init__(
        self,
        config,
        *,
        low_degree: int = LINT_LOW_DEGREE,
        high_degree: int = LINT_HIGH_DEGREE,
        step: float = LINT_STEP,
    ) -> None:
        super().__init__(config)
        if not 0 <= low_degree <= high_degree:
            raise ValueError("LINT tolerance band must satisfy 0 <= low <= high")
        if step <= 0.0:
            raise ValueError("LINT step must be positive")
        self.low_degree = int(low_degree)
        self.high_degree = int(high_degree)
        self.step = float(step)

    def radii(
        self,
        positions: Sequence[Sequence[float]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        cfg = self.config
        if previous_radii is None:
            # Match the shared t=0 graph across every adaptive controller.
            return DensityAdaptiveNeighborhood.radii(self, positions, None)
        previous = self.validate_radii(positions, previous_radii)
        degrees = self.current_degrees(positions, previous)
        output: list[float] = []
        for radius, degree in zip(previous, degrees):
            if degree < self.low_degree:
                candidate = radius + self.step
            elif degree > self.high_degree:
                candidate = radius - self.step
            else:
                candidate = radius
            output.append(_clip(candidate, cfg.min_radius, cfg.max_radius))
        return tuple(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _task(task: tuple[str, float, int, float]) -> dict[str, Any]:
    method, density, seed, stardisplay_nc = task
    half_width = matched_half_width(N_AGENTS, density)
    state = initial_state("uniform", half_width, seed, n_agents=N_AGENTS)
    cfg = _base_config(half_width, seed)
    simulator = EALMFGSimulator(cfg, initial_state=state)
    if method == "alf":
        pass  # default neighborhood is ALF
    elif method == "stardisplay":
        simulator.neighborhood = StarDisplayNeighborhood(cfg)
    elif method == "lint":
        simulator.neighborhood = LintStepNeighborhood(cfg)
    else:
        raise ValueError(f"unsupported method: {method}")
    start = time.perf_counter()
    rollout = run_rollout(simulator, AnalyticalAlignmentActor(), steps=STEPS)
    runtime = time.perf_counter() - start
    metrics = compute_metrics(rollout)
    connectivity = compute_connectivity_metrics(rollout)
    radii = [float(value) for item in rollout.radius_history for value in item]
    row: dict[str, Any] = {
        "method": method,
        "density": density,
        "half_width": half_width,
        "n_agents": N_AGENTS,
        "seed": seed,
        "steps": STEPS,
        "stardisplay_nc": stardisplay_nc,
        "contact_break_rate": contact_break_rate(rollout.neighbor_history),
        "neighbor_loss_rate": neighbor_loss_rate(rollout.neighbor_history),
        "radius_max_saturation_fraction": sum(
            value >= cfg.max_radius - 1e-12 for value in radii
        )
        / len(radii),
        "radius_temporal_variation": statistics.fmean(
            abs(float(rollout.radius_history[t][i]) - float(rollout.radius_history[t - 1][i]))
            for t in range(1, len(rollout.radius_history))
            for i in range(N_AGENTS)
        ),
        "rollout_wall_s": runtime,
        **metrics,
        **connectivity,
    }
    return row


def _paired_bootstrap(values: Sequence[float], replicates: int, rng: random.Random) -> dict[str, float]:
    values = tuple(float(value) for value in values)
    estimates = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    ]
    return {
        "paired_mean_difference": statistics.fmean(values),
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
        "positive_fraction": statistics.fmean(value > 0.0 for value in values),
        "n_pairs": len(values),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--out", type=Path, default=Path("experiments/published_range_headtohead"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVALUATION_SEEDS))
    parser.add_argument("--densities", type=float, nargs="+", default=list(DENSITIES))
    parser.add_argument("--stardisplay-nc", type=float, default=STARDISPLAY_NC)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)

    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    if set(args.seeds).intersection(DEVELOPMENT_SEEDS):
        raise ValueError("evaluation and development seed banks must be disjoint")

    tasks = [
        (method, float(density), int(seed), float(args.stardisplay_nc))
        for method in METHODS
        for density in args.densities
        for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} episodes", flush=True)
    rows.sort(key=lambda row: (float(row["density"]), str(row["method"]), int(row["seed"])))
    _write_csv(out / "episodes.csv", rows)

    rng = random.Random(BOOTSTRAP_SEED)
    by_cell: dict[tuple[float, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((float(row["density"]), str(row["method"])), {})[int(row["seed"])] = row
    comparisons: list[dict[str, Any]] = []
    for density in sorted({float(row["density"]) for row in rows}):
        alf = by_cell[(density, "alf")]
        for comparator in ("stardisplay", "lint"):
            other = by_cell[(density, comparator)]
            if set(alf) != set(other):
                raise ValueError(f"unpaired seeds in cell {density}/{comparator}")
            for metric in PRIMARY_METRICS:
                differences = [float(alf[seed][metric]) - float(other[seed][metric]) for seed in sorted(alf)]
                comparisons.append(
                    {
                        "density": density,
                        "comparison": f"alf - {comparator}",
                        "metric": metric,
                        **_paired_bootstrap(differences, args.bootstrap_replicates, rng),
                    }
                )
    for metric in PRIMARY_METRICS:
        for comparator in ("stardisplay", "lint"):
            values = [
                float(row["paired_mean_difference"])
                for row in comparisons
                if row["metric"] == metric and row["comparison"] == f"alf - {comparator}"
            ]
            comparisons.append(
                {
                    "density": "all",
                    "comparison": f"alf - {comparator}",
                    "metric": metric,
                    "paired_mean_difference": statistics.fmean(values),
                    "ci95_low": math.nan,
                    "ci95_high": math.nan,
                    "positive_fraction": statistics.fmean(value > 0.0 for value in values),
                    "n_pairs": len(values),
                }
            )
    _write_csv(out / "paired_comparisons.csv", comparisons)

    manifest = {
        "schema_version": "alf-published-range-headtohead/v1",
        "status": "completed",
        "question": (
            "How do faithful implementations of two published adaptive-range laws compare with ALF "
            "under one strictly local sensing contract and one shared seed bank?"
        ),
        "methods": {
            "alf": "fixed-probe count, fixed multiplicative base, bounded radius (this paper)",
            "stardisplay": "R_next=(1-s)R+s*Rmax*(1-|N(R)|/nc), Hildenbrandt et al. 2010 Eq. 2",
            "lint": "step/bang-bang range update over a neighbour tolerance band, Ramanathan & Rosales-Hain 2000 Sec. III",
        },
        "protocol": {
            "n_agents": N_AGENTS,
            "steps": STEPS,
            "densities": list(args.densities),
            "stardisplay_nc": args.stardisplay_nc,
            "lint_band": [LINT_LOW_DEGREE, LINT_HIGH_DEGREE],
            "lint_step": LINT_STEP,
            "base_radius": 4.0,
            "target_degree": 6.0,
            "radius_exponent": 0.5,
            "radius_smoothing": 0.9,
            "contact_retention_margin": 0.25,
            "interaction_radius_cap": 8.0,
            "physical_sensing_envelope": 8.25,
            "dynamics": "shared analytical local velocity alignment",
            "estimand": "paired ALF-minus-comparator difference per density cell, seed bootstrap",
        },
        "evaluation_seeds": list(args.seeds),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "episodes": len(rows),
        "sources": [
            {
                "path": path,
                "sha256": _sha256(root / path),
            }
            for path in (
                "eal_mfg_icra/core.py",
                "eal_mfg_icra/metrics.py",
                "eal_mfg_icra/scripts/neighborhood_bench.py",
                "eal_mfg_icra/scripts/matched_load_regime_study.py",
                "eal_mfg_icra/scripts/published_range_headtohead.py",
            )
        ],
        "outputs": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\n=== paired ALF - comparator (density cells) ===")
    header = f"{'rho':>8} {'comparison':<16}{'metric':<30}{'diff':>11}{'ci_low':>11}{'ci_high':>11}"
    print(header)
    for row in comparisons:
        if row["density"] == "all":
            continue
        if row["metric"] not in ("final_fragmentation", "mean_largest_scc_fraction", "mean_degree", "radius_temporal_variation"):
            continue
        print(
            f"{float(row['density']):>8.4f} {row['comparison']:<16}{row['metric']:<30}"
            f"{row['paired_mean_difference']:>11.4f}{row['ci95_low']:>11.4f}{row['ci95_high']:>11.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
