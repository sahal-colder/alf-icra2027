"""Connectivity-margin and local-density audit for the canonical density sweep.

This is an **additive, read-only** analysis.  It never re-runs a simulation and never
rewrites a frozen experiment artifact; it reads the per-episode table that
``crdf_comparison.py`` already wrote (``crdf_matched_raw.csv``) and reports three
things the earlier reporting pass omitted:

1. **Connectivity margin.**  ``Frag`` and ``largest-SCC`` are nearly binary: a graph
   held together by a single bridge reads the same as a heavily redundant one.  The
   per-episode table already stores the symmetric-graph Fiedler value
   (``mean_sym_lambda2`` / ``final_sym_lambda2``), the minimum in-degree
   (``mean_min_in_degree`` / ``final_min_in_degree``) and the SCC count.  This script
   turns those into paired, seed-matched contrasts against ALF with the same
   percentile-bootstrap convention as the main pipeline.
2. **Disconnection frequency.**  Fraction of episodes with ``lambda2 == 0`` (i.e. the
   realized symmetric graph is disconnected) and with more than one weak component.
3. **Local-density validation.**  The regime analysis assumes "approximately uniform"
   density.  ``mean_probe_degree`` measures the realized occupancy actually seen by
   the controller, so comparing it with the nominal homogeneous mean
   ``lambda_b = rho * pi * r_b^2`` tests that assumption directly instead of asserting
   it.

Usage
-----
    python -m eal_mfg_icra.scripts.connectivity_margin_audit \
        --raw experiments/adaptive_feedback_matched/crdf_matched_raw.csv \
        --out paper/evidence/connectivity_margin_paired.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Sequence

from eal_mfg_icra.scripts.crdf_comparison import (
    DEFAULT_BOOTSTRAP,
    METHOD_LABELS,
    N_AGENTS,
    RNG_SEED,
    _bootstrap_mean,
)

# Probes used by the manuscript: r_b = 4 (probe aperture) and r_max = 8.
PROBE_RADIUS = 4.0
PRIMARY_BAND_MIN_DENSITY = 0.01
TUNED_BAND_MAX_HALF_WIDTH = 11.0  # L <= 11 <=> rho >= 0.0331 (> rho_k)

# Metrics reported here that the earlier reporting pass did not carry.
MARGIN_METRICS = (
    "mean_sym_lambda2",
    "final_sym_lambda2",
    "mean_min_in_degree",
    "final_min_in_degree",
    "mean_n_scc",
)

COMPARATORS = (
    ("fixed_r8_retained", "maxR+retention - ALF"),
    ("knn_k6", "Local-KNN - ALF"),
    ("crdf", "CRDF - ALF"),
    ("crmf", "CRMF - ALF"),
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("density", "half_width"):
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])
        for key in MARGIN_METRICS + ("mean_probe_degree", "agent_collision_rate", "components"):
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
    return rows


def _lookup(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, float, int], dict[str, Any]]:
    return {(str(r["method"]), float(r["half_width"]), int(r["seed"])): r for r in rows}


def paired_rows(
    rows: Sequence[dict[str, Any]],
    replicates: int,
    band: str,
) -> list[dict[str, Any]]:
    """Seed-matched contrasts against ALF, cell-stratified and equal-weighted."""

    rng = __import__("random").Random(RNG_SEED + 1)
    index = _lookup(rows)
    widths = sorted({float(r["half_width"]) for r in rows})
    if band == "primary":
        widths = [
            w
            for w in widths
            if N_AGENTS / (2.0 * w) ** 2 >= PRIMARY_BAND_MIN_DENSITY
        ]
    elif band == "tuned":
        widths = [w for w in widths if w <= TUNED_BAND_MAX_HALF_WIDTH]
    output: list[dict[str, Any]] = []
    for comparator, label in COMPARATORS:
        for metric in MARGIN_METRICS:
            # Cell-level paired means, then equal weight across cells -> the same
            # stratified aggregate the manuscript uses elsewhere.
            cell_means: list[float] = []
            for width in widths:
                seeds = sorted(
                    r["seed"]
                    for r in rows
                    if float(r["half_width"]) == width and r["method"] == "alf"
                )
                diffs = [
                    float(index[(comparator, width, s)][metric])
                    - float(index[("alf", width, s)][metric])
                    for s in seeds
                    if (comparator, width, s) in index
                ]
                if diffs:
                    cell_means.append(statistics.fmean(diffs))
            if not cell_means:
                continue
            mean, low, high = _bootstrap_mean(cell_means, replicates, rng)
            output.append(
                {
                    "band": band,
                    "comparison": label,
                    "metric": metric,
                    "paired_mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_cells": len(cell_means),
                    "n_bootstrap": replicates,
                    "bootstrap_unit": "cell-stratified, seed-paired",
                }
            )
    return output


def disconnection_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per method/L: the frequency of genuinely disconnected realized graphs."""

    output: list[dict[str, Any]] = []
    keys = sorted({(str(r["method"]), float(r["half_width"])) for r in rows})
    for method, width in keys:
        block = [
            r
            for r in rows
            if str(r["method"]) == method and float(r["half_width"]) == width
        ]
        n = len(block)
        if not n:
            continue
        lam0 = sum(1 for r in block if float(r["mean_sym_lambda2"]) == 0.0)
        multi = sum(1 for r in block if float(r["components"]) > 1.0)
        output.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "half_width": width,
                "density": N_AGENTS / (2.0 * width) ** 2,
                "n_seeds": n,
                "disconnected_fraction": lam0 / n,
                "multi_component_fraction": multi / n,
                "mean_lambda2": statistics.fmean(
                    float(r["mean_sym_lambda2"]) for r in block
                ),
                "mean_min_in_degree": statistics.fmean(
                    float(r["mean_min_in_degree"]) for r in block
                ),
            }
        )
    return output


def density_validation_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare the realized probe occupancy with the nominal homogeneous mean."""

    output: list[dict[str, Any]] = []
    widths = sorted({float(r["half_width"]) for r in rows})
    for width in widths:
        density = N_AGENTS / (2.0 * width) ** 2
        nominal = density * 3.141592653589793 * PROBE_RADIUS**2
        for method in ("alf", "fixed_r8_retained"):
            block = [
                float(r["mean_probe_degree"])
                for r in rows
                if str(r["method"]) == method and float(r["half_width"]) == width
            ]
            if not block:
                continue
            output.append(
                {
                    "method": method,
                    "half_width": width,
                    "density": density,
                    "nominal_probe_mean": nominal,
                    "realized_probe_mean": statistics.fmean(block),
                    "ratio_realized_over_nominal": (
                        statistics.fmean(block) / nominal if nominal else float("nan")
                    ),
                    "n_seeds": len(block),
                }
            )
    return output


def collision_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(r["method"]) for r in rows}):
        block = [
            float(r["agent_collision_rate"])
            for r in rows
            if str(r["method"]) == method
            and r.get("agent_collision_rate") not in ("", None)
        ]
        if block:
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "mean_agent_collision_rate": statistics.fmean(block),
                    "max_agent_collision_rate": max(block),
                    "n_episodes": len(block),
                }
            )
    return output


def cell_summary_rows(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    """Per (method, cell) mean and episode-bootstrap CI, in the schema of
    ``crdf_matched_bootstrap_summary.csv`` so the figure code can consume either."""

    rng = __import__("random").Random(RNG_SEED)
    output: list[dict[str, Any]] = []
    keys = sorted({(str(r["method"]), float(r["half_width"])) for r in rows})
    for method, width in keys:
        block = [
            r
            for r in rows
            if str(r["method"]) == method and float(r["half_width"]) == width
        ]
        for metric in ("mean_sym_lambda2", "mean_min_in_degree"):
            values = [float(r[metric]) for r in block]
            if not values:
                continue
            mean, low, high = _bootstrap_mean(values, replicates, rng)
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "half_width": width,
                    "density": N_AGENTS / (2.0 * width) ** 2,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": len(values),
                    "n_bootstrap": replicates,
                    "bootstrap_unit": "shared evaluation seed",
                }
            )
    return output


def cell_interaction_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell 2x2 interaction for the connectivity-margin metrics.

    Sec. VI-F quotes a range of per-cell interactions; the table backed only the
    pooled-ratio view, so this emits the individual cell values for traceability.
    """

    output: list[dict[str, Any]] = []
    index = _lookup(rows)
    cells = sorted(
        {(float(r["half_width"]), int(r["seed"])) for r in rows}
    )
    widths = sorted({w for w, _ in cells})
    for width in widths:
        seeds = sorted({s for w, s in cells if w == width})
        for metric in MARGIN_METRICS:
            values = []
            for seed in seeds:
                try:
                    values.append(
                        float(index[("crmf", width, seed)][metric])
                        - float(index[("fpcm", width, seed)][metric])
                        - float(index[("crdf", width, seed)][metric])
                        + float(index[("alf", width, seed)][metric])
                    )
                except KeyError:
                    continue
            if not values:
                continue
            output.append(
                {
                    "half_width": width,
                    "density": N_AGENTS / (2.0 * width) ** 2,
                    "metric": metric,
                    "interaction": statistics.fmean(values),
                    "n_seeds": len(values),
                }
            )
    return output


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("experiments/adaptive_feedback_matched/crdf_matched_raw.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper/evidence/connectivity_margin_paired.csv"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args(argv)

    rows = _read_rows(args.raw)

    paired = (
        paired_rows(rows, args.bootstrap_replicates, "primary")
        + paired_rows(rows, args.bootstrap_replicates, "tuned")
        + paired_rows(rows, args.bootstrap_replicates, "all")
    )
    _write(args.out, paired)
    _write(
        args.out.with_name("connectivity_margin_disconnection.csv"),
        disconnection_rows(rows),
    )
    _write(
        args.out.with_name("local_density_validation.csv"),
        density_validation_rows(rows),
    )
    _write(
        args.out.with_name("agent_collision.csv"),
        collision_rows(rows),
    )
    _write(
        args.out.with_name("connectivity_margin_cell_summary.csv"),
        cell_summary_rows(rows, args.bootstrap_replicates),
    )
    _write(
        args.out.with_name("connectivity_margin_interaction.csv"),
        cell_interaction_rows(rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
