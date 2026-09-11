"""Task-level metric table for the canonical density sweep.

Review action M-1 asked for task-level flocking metrics (polar order and
velocity dispersion) to be reported alongside the graph-level metrics.  This
script reads the frozen per-episode artifact of the canonical density sweep
and emits

    experiments/task_metric_table/task_metric_summary.csv   (cell means + CI)
    experiments/task_metric_table/task_metric_paired.csv    (ALF vs others)
    experiments/task_metric_table/table_task_metrics.tex    (LaTeX snippet)

No simulation is re-run; every number is derived from
``experiments/adaptive_feedback_matched/crdf_matched_raw.csv``, which was
produced by ``eal_mfg_icra.scripts.crdf_comparison`` with the shared
50-seed bank (20000-20049).

Usage
-----
    python -m eal_mfg_icra.scripts.task_metric_table
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

METRICS = ("polar_order", "velocity_dispersion")
REFERENCE = "alf"
COMPARATORS = ("fixed_r4", "fixed_r8_retained", "knn_k6", "crdf", "crmf")
LABELS = {
    "fixed_r4": "Fixed $r=4$",
    "fixed_r8_retained": "Max radius + retention",
    "knn_k6": "Local-KNN $K=6$",
    "crdf": "CRDF",
    "crmf": "CRMF",
    "alf": "ALF",
}
PRIMARY_BAND = 0.01
BOOTSTRAP = 2000
RNG_SEED = 20260910


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: Sequence[float], rng: random.Random, replicates: int) -> tuple[float, float, float]:
    count = len(values)
    estimates = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    ]
    return statistics.fmean(values), _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _paired_ci(differences: Sequence[float], rng: random.Random, replicates: int) -> tuple[float, float, float]:
    count = len(differences)
    estimates = [
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    ]
    return statistics.fmean(differences), _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("experiments/adaptive_feedback_matched/crdf_matched_raw.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/task_metric_table"))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)

    with args.raw.open(newline="", encoding="utf-8") as handle:
        episodes = [row for row in csv.DictReader(handle)]
    if not episodes:
        raise ValueError(f"empty raw artifact: {args.raw}")

    by_seed: dict[tuple[str, float, int], dict[str, float]] = {}
    for row in episodes:
        key = (str(row["method"]), float(row["density"]), int(row["seed"]))
        by_seed[key] = {metric: float(row[metric]) for metric in METRICS}

    densities = sorted({float(r["density"]) for r in episodes})
    methods = sorted({str(r["method"]) for r in episodes})
    seeds = sorted({int(r["seed"]) for r in episodes})

    rng = random.Random(RNG_SEED)
    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        for method in methods:
            for density in densities:
                values = [
                    by_seed[(method, density, seed)][metric]
                    for seed in seeds
                    if (method, density, seed) in by_seed
                ]
                if not values:
                    continue
                mean, low, high = _bootstrap_ci(values, rng, args.bootstrap_replicates)
                summary.append(
                    {
                        "metric": metric,
                        "method": method,
                        "density": density,
                        "half_width": math.sqrt(0.0) if False else "",
                        "n_seeds": len(values),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "in_primary_band": int(density >= PRIMARY_BAND),
                    }
                )

    paired: list[dict[str, Any]] = []
    for metric in METRICS:
        for comparator in COMPARATORS:
            for density in densities:
                differences = []
                for seed in seeds:
                    ref = by_seed.get((REFERENCE, density, seed))
                    cmp_ = by_seed.get((comparator, density, seed))
                    if ref is None or cmp_ is None:
                        continue
                    differences.append(ref[metric] - cmp_[metric])
                if not differences:
                    continue
                mean, low, high = _paired_ci(differences, rng, args.bootstrap_replicates)
                paired.append(
                    {
                        "metric": metric,
                        "comparison": f"ALF - {comparator}",
                        "density": density,
                        "in_primary_band": int(density >= PRIMARY_BAND),
                        "n_pairs": len(differences),
                        "paired_mean_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "positive_fraction": sum(1 for d in differences if d > 0) / len(differences),
                    }
                )

    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "task_metric_summary.csv", summary)
    _write(args.out / "task_metric_paired.csv", paired)
    _write_tex(args.out / "table_task_metrics.tex", summary, densities)
    _write_tex_paired(args.out / "table_task_metrics_paired.tex", paired, densities)

    print(f"wrote {args.out / 'task_metric_summary.csv'}")
    print(f"wrote {args.out / 'task_metric_paired.csv'}")
    print(f"wrote {args.out / 'table_task_metrics.tex'}")
    print(f"wrote {args.out / 'table_task_metrics_paired.tex'}")
    return 0


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _primary_densities(densities: Sequence[float]) -> list[float]:
    return [d for d in densities if d >= PRIMARY_BAND]


def _write_tex(path: Path, summary: Sequence[dict[str, Any]], densities: Sequence[float]) -> None:
    lookup = {(r["metric"], r["method"], float(r["density"])): r for r in summary}
    band = _primary_densities(densities)
    rows_order = ("alf", "fixed_r8_retained", "knn_k6", "fixed_r4")
    lines = [
        "% Auto-generated by eal_mfg_icra/scripts/task_metric_table.py",
        r"\begin{tabular}{l" + "c" * len(band) + "}",
        r"\toprule",
        r"Controller & " + " & ".join(rf"$\rho={d:.4f}$" for d in band) + r"\\",
        r"\midrule",
    ]
    for metric in METRICS:
        lines.append(rf"\multicolumn{{{len(band)+1}}}{{l}}{{\textit{{{metric.replace('_',' ')}}}}}\\")
        for method in rows_order:
            cells = []
            for d in band:
                row = lookup.get((metric, method, d))
                if row is None:
                    cells.append("---")
                else:
                    cells.append(f"{row['mean']:.3f}")
            lines.append(f"{LABELS[method]} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex_paired(path: Path, paired: Sequence[dict[str, Any]], densities: Sequence[float]) -> None:
    band = _primary_densities(densities)
    lookup = {(r["metric"], r["comparison"], float(r["density"])): r for r in paired}
    lines = [
        "% Auto-generated by eal_mfg_icra/scripts/task_metric_table.py",
        r"\begin{tabular}{ll" + "c" * len(band) + "}",
        r"\toprule",
        r"Metric & Comparison & " + " & ".join(rf"$\rho={d:.4f}$" for d in band) + r"\\",
        r"\midrule",
    ]
    for metric in METRICS:
        for comparator in COMPARATORS:
            comp = f"ALF - {comparator}"
            cells = []
            for d in band:
                row = lookup.get((metric, comp, d))
                if row is None:
                    cells.append("---")
                else:
                    stars = "" if (row["ci95_low"] <= 0 <= row["ci95_high"]) else r"$^{*}$"
                    cells.append(f"{row['paired_mean_difference']:+.3f}{stars}")
            lines.append(f"{metric.replace('_',' ')} & ALF$-${LABELS[comparator]} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
