"""Re-analysis of the 2x2 aperture--base factorial from frozen artifacts.

Purpose
-------
Three things that the published `aperture_base_factorial_statistics.csv` does
not make explicit, and which reviewers have repeatedly mis-read.

1. Contrast coding.  Write the four cell means as

       ALF  = mu
       CRDF = mu + a
       FPCM = mu + b
       CRMF = mu + a + b + ab

   Under the paper's +/-1/2 coding the three reported effects are

       aperture main  A = 0.5*((CRDF-ALF)+(CRMF-FPCM)) = a + ab/2
       base main      B = 0.5*((FPCM-ALF)+(CRMF-CRDF)) = b + ab/2
       interaction    I = CRMF-FPCM-CRDF+ALF          = ab

   Consequences that the manuscript must not get wrong:

       A + B = a + b + ab = CRMF - ALF          (exact identity)
       A + B + I  is NOT CRMF - ALF             (I would be double counted)
       a = A - I/2,  b = B - I/2,  ab = I       (raw factor-level components)

   Reporting the share of the CRMF - ALF contrast carried by "aperture",
   "base", and "interaction" therefore requires a stated convention.  This
   script emits both:

       * coded-share  = (A, B, I) / (A + B + I)          (the convention the
         round-4 reviewer used: 87% interaction, 19% base, -6% aperture)
       * cell-share   = (a, b, ab) / (CRMF - ALF)        (the exact
         cell-mean decomposition)

   and it never prints A + B + I as if it were the CRMF - ALF contrast.

2. Whether the four corners are actually distinguishable in a cell.  The
   identity  I = (ALF-CRDF) + (CRMF-FPCM)  is exact, so when two corners
   coincide the "interaction" degenerates onto a single pairwise contrast.  We
   report, per cell, the four corner means, the ratio |CRMF-FPCM| / |I|, the
   max-radius saturation fraction of each corner, and an explicit
   `interaction_is_separable` flag.

   Saturation is the mechanism that makes FPCM and CRMF coincide: above the
   reference density rho_k = k/(pi r_max^2) the cap can supply the target
   degree and the corners stay apart, below it the cap binds and they collapse.

3. Provenance.  Every coded effect (A, B, I) and its interval is read
   *verbatim* from the frozen `factorial_statistics.csv` produced by the run,
   so the figures quoted in the manuscript and the artifact cannot drift.
   Only the new quantities -- the CRMF - ALF total and the raw factor
   components a, b -- are re-estimated here, by the same stratified paired
   bootstrap (2000 percentile replicates, seed 20260909).

Outputs (relative to the repository root, or --out):
    <out>/aperture_base_decomposition_<tag>.csv   aggregate decomposition
    <out>/aperture_base_cellwise_<tag>.csv        per-cell separability
    <out>/aperture_base_validation_<tag>.json     identity + provenance check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from ..scripts.aperture_base_factorial import (
    PRIMARY_METRICS,
    _cell_records,
    _percentile,
    _read_csv,
)

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260909
METHODS = ("alf", "crdf", "fpcm", "crmf")
CODED_EFFECTS = ("aperture_main_effect", "base_main_effect", "aperture_base_interaction")
# A cell counts as "degenerate onto the aperture contrast" when the FPCM/CRMF
# gap carries less than this fraction of the reported interaction.
SEPARABILITY_RATIO_THRESHOLD = 0.10
# A corner counts as saturated when its radius sits at the cap for more than
# this fraction of agent-steps.
SATURATION_THRESHOLD = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratified_bootstrap(
    groups: Sequence[list[dict[str, float]]],
    key: str,
    replicates: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    """Resample evaluation seeds inside every cell, then equal-weight cells."""

    point = statistics.fmean(statistics.fmean(float(item[key]) for item in group) for group in groups)
    estimates = [
        statistics.fmean(
            statistics.fmean(float(group[rng.randrange(len(group))][key]) for _ in group)
            for group in groups
        )
        for _ in range(replicates)
    ]
    return point, _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _reference_effects(stats_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    """Load the frozen aggregate effects that the manuscript quotes."""

    index: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in _read_csv(stats_path):
        if str(row["scope"]) != "n_agents_stratified_aggregate":
            continue
        index[(int(row["n_agents"]), str(row["metric"]), str(row["effect"]))] = row
    missing = [
        (n, metric, effect)
        for n in sorted({key[0] for key in index})
        for metric in PRIMARY_METRICS
        for effect in CODED_EFFECTS
        if (n, metric, effect) not in index
    ]
    if missing:
        raise ValueError(f"reference statistics artifact is incomplete: {missing[:5]}")
    return index


def _enriched_cells(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, float, str], list[dict[str, float]]]:
    """Per-cell, per-seed cell means plus the two paired quantities we add."""

    grouped: dict[tuple[int, float, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (int(row["n_agents"]), float(row["density"]), str(row["geometry"]))
        grouped.setdefault(key, {}).setdefault(str(row["method"]), []).append(row)
    out: dict[tuple[int, float, str], list[dict[str, float]]] = {}
    for key, per_method in grouped.items():
        if set(per_method) != set(METHODS):
            continue
        counts = {m: len(per_method[m]) for m in METHODS}
        if len(set(counts.values())) != 1:
            continue
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for method in METHODS:
            for row in per_method[method]:
                by_seed.setdefault(int(row["seed"]), {})[method] = row
        if any(set(methods) != set(METHODS) for methods in by_seed.values()):
            raise ValueError(f"incomplete paired factorial in cell {key}")
        enriched: list[dict[str, float]] = []
        for seed in sorted(by_seed):
            methods = by_seed[seed]
            item: dict[str, float] = {"seed": float(seed)}
            for metric in PRIMARY_METRICS:
                values = {m: float(methods[m][metric]) for m in METHODS}
                item[f"{metric}__alf"] = values["alf"]
                item[f"{metric}__crdf"] = values["crdf"]
                item[f"{metric}__fpcm"] = values["fpcm"]
                item[f"{metric}__crmf"] = values["crmf"]
                item[f"{metric}__total"] = values["crmf"] - values["alf"]
                item[f"{metric}__aperture_level"] = values["crdf"] - values["alf"]
                item[f"{metric}__base_level"] = values["fpcm"] - values["alf"]
            enriched.append(item)
        out[key] = enriched
    return out


def _aggregate_rows(
    cells: dict[tuple[int, float, str], list[dict[str, float]]],
    reference: dict[tuple[int, str, str], dict[str, Any]],
    replicates: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = sorted({key[0] for key in cells})
    for n_agents in populations:
        selected = [cell for key, cell in cells.items() if key[0] == n_agents]
        for metric in PRIMARY_METRICS:
            ref = {e: reference[(n_agents, metric, e)] for e in CODED_EFFECTS}
            aperture = float(ref["aperture_main_effect"]["paired_mean_difference"])
            base = float(ref["base_main_effect"]["paired_mean_difference"])
            interaction = float(ref["aperture_base_interaction"]["paired_mean_difference"])
            total, total_low, total_high = _stratified_bootstrap(
                selected, f"{metric}__total", replicates, rng
            )
            # Raw factor-level components of the CRMF - ALF cell contrast.
            raw_a = aperture - 0.5 * interaction
            raw_b = base - 0.5 * interaction
            coded_sum = aperture + base + interaction
            coded_share = (
                {
                    "coded_share_aperture": aperture / coded_sum,
                    "coded_share_base": base / coded_sum,
                    "coded_share_interaction": interaction / coded_sum,
                }
                if abs(coded_sum) > 0.0
                else {"coded_share_aperture": math.nan, "coded_share_base": math.nan, "coded_share_interaction": math.nan}
            )
            cell_share = (
                {
                    "cell_share_a": raw_a / total,
                    "cell_share_b": raw_b / total,
                    "cell_share_ab": interaction / total,
                }
                if abs(total) > 0.0
                else {"cell_share_a": math.nan, "cell_share_b": math.nan, "cell_share_ab": math.nan}
            )
            rows.append(
                {
                    "n_agents": n_agents,
                    "metric": metric,
                    "A_aperture_main": aperture,
                    "A_ci_low": float(ref["aperture_main_effect"]["ci95_low"]),
                    "A_ci_high": float(ref["aperture_main_effect"]["ci95_high"]),
                    "B_base_main": base,
                    "B_ci_low": float(ref["base_main_effect"]["ci95_low"]),
                    "B_ci_high": float(ref["base_main_effect"]["ci95_high"]),
                    "I_interaction": interaction,
                    "I_ci_low": float(ref["aperture_base_interaction"]["ci95_low"]),
                    "I_ci_high": float(ref["aperture_base_interaction"]["ci95_high"]),
                    "I_holm_adjusted_pvalue": ref["aperture_base_interaction"]["holm_adjusted_pvalue"],
                    "A_holm_family": ref["aperture_main_effect"]["holm_family"],
                    "A_holm_adjusted_pvalue": ref["aperture_main_effect"]["holm_adjusted_pvalue"],
                    "B_holm_family": ref["base_main_effect"]["holm_family"],
                    "B_holm_adjusted_pvalue": ref["base_main_effect"]["holm_adjusted_pvalue"],
                    "raw_a_probe_level": raw_a,
                    "raw_b_base_level": raw_b,
                    "raw_ab_interaction": interaction,
                    "raw_sum_equals_total_error": raw_a + raw_b + interaction - total,
                    "total_CRMF_minus_ALF": total,
                    "total_ci_low": total_low,
                    "total_ci_high": total_high,
                    "A_plus_B_equals_total_error": aperture + base - total,
                    "abs_I_over_abs_B": abs(interaction) / abs(base) if abs(base) > 0.0 else math.nan,
                    "abs_I_over_abs_total": abs(interaction) / abs(total) if abs(total) > 0.0 else math.nan,
                    "coded_effect_sum_A_plus_B_plus_I": coded_sum,
                    **coded_share,
                    **cell_share,
                }
            )
    return rows


def _cell_rows(cells: dict[tuple[int, float, str], list[dict[str, float]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (n_agents, density, geometry), cell in sorted(cells.items()):
        item: dict[str, Any] = {"n_agents": n_agents, "density": density, "geometry": geometry}
        for metric in PRIMARY_METRICS:
            means = {m: statistics.fmean(float(row[f"{metric}__{m}"]) for row in cell) for m in METHODS}
            for m in METHODS:
                item[f"{metric}__{m}"] = means[m]
            aperture = 0.5 * ((means["crdf"] - means["alf"]) + (means["crmf"] - means["fpcm"]))
            base = 0.5 * ((means["fpcm"] - means["alf"]) + (means["crmf"] - means["crdf"]))
            interaction = means["crmf"] - means["fpcm"] - means["crdf"] + means["alf"]
            total = means["crmf"] - means["alf"]
            gap = means["crmf"] - means["fpcm"]
            item[f"{metric}__A_aperture_main"] = aperture
            item[f"{metric}__B_base_main"] = base
            item[f"{metric}__I_interaction"] = interaction
            item[f"{metric}__total_CRMF_minus_ALF"] = total
            item[f"{metric}__raw_a"] = aperture - 0.5 * interaction
            item[f"{metric}__raw_b"] = base - 0.5 * interaction
            item[f"{metric}__identity_error"] = aperture + base - total
            item[f"{metric}__abs_crmf_minus_fpcm_over_abs_I"] = (
                abs(gap) / abs(interaction) if abs(interaction) > 0.0 else math.nan
            )
            item[f"{metric}__interaction_is_separable"] = int(
                abs(interaction) > 0.0 and abs(gap) / abs(interaction) >= SEPARABILITY_RATIO_THRESHOLD
            )
        out.append(item)
    return out


def _saturation_rows(
    rows: Sequence[dict[str, Any]],
    cells: dict[tuple[int, float, str], list[dict[str, float]]],
) -> None:
    """Attach per-corner cap-saturation and realized-degree means to cells."""

    grouped: dict[tuple[int, float, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (int(row["n_agents"]), float(row["density"]), str(row["geometry"]))
        grouped.setdefault(key, {}).setdefault(str(row["method"]), []).append(row)
    for key in cells:
        saturation: dict[str, float] = {}
        degree: dict[str, float] = {}
        for method in METHODS:
            bucket = grouped[key][method]
            saturation[method] = statistics.fmean(float(r["radius_max_saturation_fraction"]) for r in bucket)
            degree[method] = statistics.fmean(float(r["mean_current_radius_degree"]) for r in bucket)
        for method in METHODS:
            cells[key][0][f"saturation__{method}"] = saturation[method]
            cells[key][0][f"realized_degree__{method}"] = degree[method]
        cells[key][0]["saturation__min_over_corners"] = min(saturation.values())
        cells[key][0]["saturation__max_over_corners"] = max(saturation.values())
        cells[key][0]["saturation__spread"] = max(saturation.values()) - min(saturation.values())
        cells[key][0]["saturated_corner_count"] = sum(
            1 for method in METHODS if saturation[method] >= SATURATION_THRESHOLD
        )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--run", type=Path, default=Path("experiments/aperture_base_factorial_20260909"))
    parser.add_argument("--stats", type=Path, default=None, help="defaults to <run>/factorial_statistics.csv")
    parser.add_argument("--out", type=Path, default=Path("paper/evidence"))
    parser.add_argument("--tag", type=str, default="sparse")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)

    run = args.run if args.run.is_absolute() else root / args.run
    stats = args.stats if args.stats is not None else run / "factorial_statistics.csv"
    stats = stats if stats.is_absolute() else root / stats
    out = args.out if args.out.is_absolute() else root / args.out
    rows = _read_csv(run / "episode_summary.csv")
    print(f"loaded {len(rows)} episodes from {run}", file=sys.stderr)

    reference = _reference_effects(stats)
    cells = _enriched_cells(rows)
    _saturation_rows(rows, cells)
    rng = random.Random(BOOTSTRAP_SEED)
    aggregate = _aggregate_rows(cells, reference, args.bootstrap_replicates, rng)
    cellwise = _cell_rows(cells)
    # re-attach the saturation diagnostics to the flat cell table
    key_lookup = {(c["n_agents"], c["density"], c["geometry"]): c for c in cellwise}
    for key, cell in cells.items():
        for field, value in cell[0].items():
            if field.startswith(("saturation__", "realized_degree__")) or field == "saturated_corner_count":
                key_lookup[key][field] = value

    identity_error = max(
        abs(float(row["A_plus_B_equals_total_error"])) for row in aggregate
    )
    raw_identity_error = max(abs(float(row["raw_sum_equals_total_error"])) for row in aggregate)
    cell_identity_error = max(
        abs(float(c[f"{metric}__identity_error"])) for c in cellwise for metric in PRIMARY_METRICS
    )

    _write_csv(out / f"aperture_base_decomposition_{args.tag}.csv", aggregate)
    _write_csv(out / f"aperture_base_cellwise_{args.tag}.csv", cellwise)
    validation = {
        "tag": args.tag,
        "run": str(run.relative_to(root)),
        "episodes": len(rows),
        "cells": len(cellwise),
        "coded_effect_source": {
            "path": str(stats.relative_to(root)),
            "sha256": _sha256(stats),
            "policy": "A, B and I are read verbatim from the frozen statistics artifact; no re-derivation.",
        },
        "episode_source": {
            "path": str((run / 'episode_summary.csv').relative_to(root)),
            "sha256": _sha256(run / "episode_summary.csv"),
        },
        "identity_checks": {
            "A_plus_B_equals_CRMF_minus_ALF_max_abs_error": identity_error,
            "raw_a_plus_b_plus_ab_equals_total_max_abs_error": raw_identity_error,
            "per_cell_identity_max_abs_error": cell_identity_error,
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "separability_ratio_threshold": SEPARABILITY_RATIO_THRESHOLD,
        "saturation_threshold": SATURATION_THRESHOLD,
        "degenerate_cells_mean_scc": sum(
            c["mean_largest_scc_fraction__interaction_is_separable"] == 0 for c in cellwise
        ),
    }
    (out / f"aperture_base_validation_{args.tag}.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    _report(aggregate, cellwise, validation)
    return 0


def _report(
    aggregate: Sequence[dict[str, Any]],
    cellwise: Sequence[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    print("\n=== CRMF - ALF decomposition (coded effects verbatim from the frozen artifact) ===")
    print(
        f"{'N':>3} {'metric':<26}{'A aperture':>12}{'B base':>12}{'I inter':>12}"
        f"{'CRMF-ALF':>12}{'|I|/|B|':>9}{'coded I%':>10}{'cell ab%':>10}"
    )
    for row in aggregate:
        print(
            f"{row['n_agents']:>3} {row['metric']:<26}"
            f"{_fmt(row['A_aperture_main']):>12}{_fmt(row['B_base_main']):>12}"
            f"{_fmt(row['I_interaction']):>12}{_fmt(row['total_CRMF_minus_ALF']):>12}"
            f"{row['abs_I_over_abs_B']:>9.2f}"
            f"{100.0 * row['coded_share_interaction']:>9.1f}%"
            f"{100.0 * row['cell_share_ab']:>9.1f}%"
        )
    print("\n=== intervals ===")
    for row in aggregate:
        print(
            f"  N={row['n_agents']:>3} {row['metric']:<26} "
            f"A[{_fmt(row['A_ci_low'])},{_fmt(row['A_ci_high'])}] "
            f"B[{_fmt(row['B_ci_low'])},{_fmt(row['B_ci_high'])}] "
            f"I[{_fmt(row['I_ci_low'])},{_fmt(row['I_ci_high'])}] "
            f"total[{_fmt(row['total_ci_low'])},{_fmt(row['total_ci_high'])}]"
        )
    verdict = validation["identity_checks"]
    print(
        f"\nidentity max|A+B-(CRMF-ALF)|={verdict['A_plus_B_equals_CRMF_minus_ALF_max_abs_error']:.2e}  "
        f"max|a+b+ab-total|={verdict['raw_a_plus_b_plus_ab_equals_total_max_abs_error']:.2e}  "
        f"max per-cell={verdict['per_cell_identity_max_abs_error']:.2e}"
    )
    degenerate = validation["degenerate_cells_mean_scc"]
    print(
        f"\n=== per-cell separability (mean SCC): {degenerate}/{len(cellwise)} cells have "
        f"|CRMF-FPCM|/|I| < {validation['separability_ratio_threshold']} ==="
    )
    print(
        f"{'N':>3} {'rho':>7} {'geom':<10}{'ratio':>7}{'I':>9}{'CRMF-FPCM':>11}"
        f"{'sat alf':>9}{'crdf':>7}{'fpcm':>7}{'crmf':>7}"
    )
    for c in cellwise:
        metric = "mean_largest_scc_fraction"
        # CRMF - FPCM = (CRDF - ALF) + I  by the exact 2x2 identity
        crmf_minus_fpcm = c[f"{metric}__raw_a"] + c[f"{metric}__I_interaction"]
        print(
            f"{c['n_agents']:>3} {c['density']:>7} {c['geometry']:<10}"
            f"{c[f'{metric}__abs_crmf_minus_fpcm_over_abs_I']:>7.2f}"
            f"{_fmt(c[f'{metric}__I_interaction']):>9}"
            f"{_fmt(crmf_minus_fpcm):>11}"
            f"{c['saturation__alf']:>9.2f}{c['saturation__crdf']:>7.2f}"
            f"{c['saturation__fpcm']:>7.2f}{c['saturation__crmf']:>7.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
