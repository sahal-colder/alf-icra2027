"""Run the matched ALF/CRDF analytical comparison.

This entry point is deliberately separate from the historical analytical
benchmarks.  It evaluates the matched methods required for the canonical
ICRA density sweep, stores one row per (density, method, seed), and computes
episode-bootstrap and paired-bootstrap intervals without changing any archived
ALF evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

from .neighborhood_bench import (
    DEFAULT_HALF_WIDTHS,
    AnalyticalAlignmentActor,
    CurrentRadiusDegreeNeighborhood,
    CurrentRadiusMultiplicativeNeighborhood,
    EALMFGSimulator,
    N_AGENTS,
    _base_config,
    _fixed_config,
    _knn_config,
    run_cell,
)
from ..core import run_rollout


METHODS = ("fixed_r4", "fixed_r8", "fixed_r8_retained", "knn_k6", "crdf", "crmf", "alf")
FIGURE_METHODS = ("fixed_r4", "fixed_r8_retained", "knn_k6", "crdf", "crmf", "alf")
METHOD_LABELS = {
    "fixed_r4": "Fixed r=4",
    "fixed_r8": "Maximum radius",
    "fixed_r8_retained": "Maximum radius + retention",
    "knn_k6": "Local-KNN K=6",
    "crdf": "Current-radius feedback",
    "crmf": "Current-radius multiplicative",
    "alf": "ALF",
}
L_VALUES = (7.0, 9.0, 11.0, 14.0, 18.0, 23.0, 30.0)
FULL_SEEDS = tuple(range(20000, 20050))
DEFAULT_BOOTSTRAP = 2000
RNG_SEED = 20260901

REPORT_METRICS = (
    "final_fragmentation",
    "mean_fragmentation",
    "final_largest_scc_fraction",
    "mean_largest_scc_fraction",
    "mean_degree",
    "mean_radius",
    "mean_interaction_area",
    "neighbor_loss_rate",
    "polar_order",
    "velocity_dispersion",
    "control_energy",
    "radius_temporal_variation",
    "radius_command_temporal_variation",
    "radius_min_saturation_fraction",
    "radius_max_saturation_fraction",
    "mean_degree_error",
    "mean_degree_error_sq",
    "radius_oscillation_frequency",
)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean(values: Sequence[float], replicates: int, rng: random.Random) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    count = len(values)
    estimates = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    ]
    return (
        statistics.fmean(values),
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    )


def _method_args(method: str) -> tuple[str, float | int]:
    if method == "fixed_r4":
        return "fixed", 4
    if method == "fixed_r8":
        return "fixed", 8
    if method == "fixed_r8_retained":
        return "fixed_retained", 8
    if method == "knn_k6":
        return "knn", 6
    if method == "crdf":
        return "crdf", 0.0
    if method == "crmf":
        return "crmf", 0.0
    if method == "alf":
        return "alf", 0.0
    raise ValueError(f"unknown method: {method}")


def _run_task(task: tuple[str, float, int, int]) -> dict[str, Any]:
    method, half_width, seed, steps = task
    mechanism, parameter = _method_args(method)
    return run_cell(
        mechanism,
        parameter,
        half_width,
        seed,
        steps,
        n_agents=N_AGENTS,
    )


def _trace_task(task: tuple[str, float, int, int]) -> list[dict[str, float | int | str]]:
    """Return one seed's radius trace for the compact mechanism diagnostic."""

    method, half_width, seed, steps = task
    mechanism, parameter = _method_args(method)
    if mechanism == "fixed":
        cfg = _fixed_config(half_width, seed, float(parameter))
        simulator = EALMFGSimulator(cfg)
    elif mechanism == "knn":
        cfg = _knn_config(half_width, seed, int(parameter))
        from .neighborhood_bench import KNNSimulator

        simulator = KNNSimulator(cfg, k=int(parameter))
    else:
        cfg = _base_config(half_width, seed)
        simulator = EALMFGSimulator(cfg)
        if mechanism == "crdf":
            simulator.neighborhood = CurrentRadiusDegreeNeighborhood(cfg)
        elif mechanism == "crmf":
            simulator.neighborhood = CurrentRadiusMultiplicativeNeighborhood(cfg)
    simulator.reset(n_agents=N_AGENTS)
    rollout = run_rollout(simulator, AnalyticalAlignmentActor(), steps=steps)
    rows: list[dict[str, float | int | str]] = []
    for step, radii in enumerate(rollout.radius_history):
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "half_width": half_width,
                "density": N_AGENTS / (2.0 * half_width) ** 2,
                "seed": seed,
                "step": step,
                "mean_radius": statistics.fmean(float(value) for value in radii),
            }
        )
    return rows


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(records: Sequence[dict[str, Any]], replicates: int) -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["method"]), float(row["half_width"])), []).append(row)
    output = []
    for (method, half_width), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "half_width": half_width,
            "density": float(rows[0]["density"]),
            "n_seeds": len(rows),
        }
        for metric in REPORT_METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            # The bootstrap CSV is the inferential artifact; these normal CIs
            # are retained only as a compact descriptive summary.
            item[f"{metric}_ci95_low"] = item[f"{metric}_mean"] - 1.96 * item[f"{metric}_std"] / math.sqrt(len(values))
            item[f"{metric}_ci95_high"] = item[f"{metric}_mean"] + 1.96 * item[f"{metric}_std"] / math.sqrt(len(values))
            mean, low, high = _bootstrap_mean(values, replicates, rng)
            item[f"{metric}_bootstrap_ci95_low"] = low
            item[f"{metric}_bootstrap_ci95_high"] = high
        output.append(item)
    return output


def _bootstrap_rows(records: Sequence[dict[str, Any]], replicates: int) -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["method"]), float(row["half_width"])), []).append(row)
    output = []
    for (method, half_width), rows in sorted(grouped.items()):
        density = float(rows[0]["density"])
        for metric in REPORT_METRICS:
            values = [float(row[metric]) for row in rows]
            mean, low, high = _bootstrap_mean(values, replicates, rng)
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "half_width": half_width,
                    "density": density,
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


def _paired_rows(records: Sequence[dict[str, Any]], replicates: int) -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED + 1)
    lookup = {
        (str(row["method"]), float(row["half_width"]), int(row["seed"])): row
        for row in records
    }
    cells = sorted({(float(row["half_width"]), int(row["seed"])) for row in records})
    widths = sorted({width for width, _ in cells})
    output = []
    for half_width in widths:
        seeds = sorted({seed for width, seed in cells if width == half_width})
        for comparator, label in (("crdf", "CRDF - ALF"), ("crmf", "CRMF - ALF")):
            for metric in REPORT_METRICS:
                differences = []
                for seed in seeds:
                    alf = lookup[("alf", half_width, seed)]
                    baseline = lookup[(comparator, half_width, seed)]
                    differences.append(float(baseline[metric]) - float(alf[metric]))
                mean, low, high = _bootstrap_mean(differences, replicates, rng)
                output.append(
                    {
                        "comparison": label,
                        "half_width": half_width,
                        "density": N_AGENTS / (2.0 * half_width) ** 2,
                        "metric": metric,
                        "paired_mean_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_pairs": len(differences),
                        "n_bootstrap": replicates,
                        "bootstrap_unit": "shared evaluation seed",
                    }
                )
    return output


def _trace_summary(
    traces: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED + 2)
    grouped: dict[tuple[str, float, int], list[float]] = {}
    for row in traces:
        grouped.setdefault((str(row["method"]), float(row["density"]), int(row["step"])), []).append(float(row["mean_radius"]))
    output = []
    for (method, density, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        mean, low, high = _bootstrap_mean(values, replicates, rng)
        output.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "density": density,
                "step": step,
                "mean_radius": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_seeds": len(values),
                "n_bootstrap": replicates,
            }
        )
    return output


def _make_figures(bootstrap_path: Path, trace_path: Path, figure_out: Path, diagnostic_out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to render CRDF figures") from exc

    with bootstrap_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = {
        (row["method"], float(row["density"]), row["metric"]): row
        for row in rows
    }
    styles = {
        "fixed_r4": ("Fixed r=4", "#666666", "s", "--"),
        "fixed_r8": ("Maximum radius (no retention)", "#D55E00", "^", "-."),
        "fixed_r8_retained": ("Maximum radius + retention", "#E69F00", "v", "--"),
        "knn_k6": ("Local-KNN K=6", "#009E73", "D", ":"),
        "crdf": ("Current-radius feedback", "#CC79A7", "P", "--"),
        "crmf": ("CRMF", "#56B4E9", "X", "-."),
        "alf": ("ALF", "#0072B2", "o", "-"),
    }
    panels = (
        ("final_fragmentation", "Final fragmentation", (0.0, 0.75)),
        ("mean_degree", "Mean interaction out-degree", (0.0, 13.0)),
        ("mean_radius", "Mean realized interaction radius", (3.0, 8.4)),
        ("final_largest_scc_fraction", "Largest SCC fraction", (0.35, 1.02)),
    )
    densities = sorted({float(row["density"]) for row in rows}, reverse=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.85), sharex=True)
    for ax, (metric, ylabel, ylim) in zip(axes.flat, panels):
        ax.axvspan(0.0, 0.01, color="#E5E5E5", alpha=0.65, zorder=0)
        ax.axvline(0.01, color="#777777", linewidth=0.7, linestyle=":", zorder=1)
        for method in FIGURE_METHODS:
            # Local-KNN has a common candidate-sensing cap and a selected
            # neighbor set, but no realized metric interaction radius that is
            # comparable to ALF/CRMF.  It is therefore omitted from panel (c).
            if metric == "mean_radius" and method == "knn_k6":
                continue
            points = [data[(method, density, metric)] for density in densities if (method, density, metric) in data]
            if not points:
                continue
            x = [float(row["density"]) for row in points]
            y = [float(row["mean"]) for row in points]
            low = [value - float(row["ci95_low"]) for value, row in zip(y, points)]
            high = [float(row["ci95_high"]) - value for value, row in zip(y, points)]
            label, color, marker, linestyle = styles[method]
            ax.errorbar(
                x, y, yerr=[low, high], label=label, color=color,
                marker=marker, linestyle=linestyle,
                linewidth=1.35 if method == "alf" else 0.95,
                markersize=5.3 if method == "alf" else 4.1,
                markerfacecolor="white" if method in {"fixed_r4", "crdf"} else color,
                markeredgecolor=color, capsize=1.4,
            )
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        ax.tick_params(direction="out", length=3, width=0.6)
    axes[0, 0].set_title("(a) Final fragmentation", loc="left", pad=2)
    axes[0, 1].set_title("(b) Interaction load", loc="left", pad=2)
    axes[1, 0].set_title("(c) Adaptive scale", loc="left", pad=2)
    axes[1, 1].set_title("(d) Directed connectivity", loc="left", pad=2)
    axes[1, 0].set_xlabel(r"Density $\rho=N/(2L)^2$")
    axes[1, 1].set_xlabel(r"Density $\rho=N/(2L)^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=0.75, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=0.4, w_pad=1.0, h_pad=0.8)
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figure_out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    with trace_path.open(newline="", encoding="utf-8") as handle:
        traces = list(csv.DictReader(handle))
    diagnostic = {
        (row["method"], int(row["step"])): row
        for row in traces
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for method in ("alf", "crdf", "crmf"):
        points = [diagnostic[(method, step)] for step in sorted({int(row["step"]) for row in traces if row["method"] == method})]
        x = [int(row["step"]) for row in points]
        y = [float(row["mean_radius"]) for row in points]
        low = [value - float(row["ci95_low"]) for value, row in zip(y, points)]
        high = [float(row["ci95_high"]) - value for value, row in zip(y, points)]
        label, color, marker, linestyle = styles[method]
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.2)
        ax.fill_between(x, [a - b for a, b in zip(y, low)], [a + b for a, b in zip(y, high)], color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean interaction radius")
    ax.set_xlim(0, max(int(row["step"]) for row in traces))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    ax.legend(frameon=False, loc="best", handletextpad=0.3)
    fig.tight_layout(pad=0.35)
    diagnostic_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(diagnostic_out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(diagnostic_out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/crdf_matched"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--half-widths", type=float, nargs="+", default=list(L_VALUES))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="rerun requested cells and keep their latest checkpoint records",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    seeds = tuple(args.seeds) if args.seeds is not None else FULL_SEEDS
    half_widths = tuple(float(value) for value in args.half_widths)
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if any(value <= 0.0 for value in half_widths):
        raise ValueError("half-widths must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "crdf_matched_checkpoint.jsonl"
    record_by_key = {
        (str(row["method"]), float(row["half_width"]), int(row["seed"])): row
        for row in _load_checkpoint(checkpoint)
    }
    completed = set(record_by_key)
    tasks = [
        (method, half_width, int(seed), args.steps)
        for method in METHODS
        for half_width in half_widths
        for seed in seeds
        if args.force_rerun or (method, half_width, int(seed)) not in completed
    ]
    print(f"pending episode cells: {len(tasks)}; workers: {args.workers}", flush=True)
    with checkpoint.open("a", encoding="utf-8") as handle:
        if args.workers == 1:
            for task in tasks:
                row = _run_task(task)
                key = (str(row["method"]), float(row["half_width"]), int(row["seed"]))
                record_by_key[key] = row
                handle.write(json.dumps(row) + "\n")
                print(f"{row['method']} L={row['half_width']:g} seed={row['seed']} frag={row['final_fragmentation']:.4f}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_task, task): task for task in tasks}
                for future in as_completed(futures):
                    row = future.result()
                    key = (str(row["method"]), float(row["half_width"]), int(row["seed"]))
                    record_by_key[key] = row
                    handle.write(json.dumps(row) + "\n")
                    print(f"{row['method']} L={row['half_width']:g} seed={row['seed']} frag={row['final_fragmentation']:.4f}", flush=True)

    records = sorted(
        record_by_key.values(),
        key=lambda row: (str(row["method"]), float(row["half_width"]), int(row["seed"])),
    )
    raw_path = args.out / "crdf_matched_raw.csv"
    summary_path = args.out / "crdf_matched_summary.csv"
    bootstrap_path = args.out / "crdf_matched_bootstrap_summary.csv"
    paired_path = args.out / "crdf_matched_paired_comparison.csv"
    _write_csv(raw_path, records)
    _write_csv(summary_path, _summary(records, args.bootstrap_replicates))
    _write_csv(bootstrap_path, _bootstrap_rows(records, args.bootstrap_replicates))
    _write_csv(paired_path, _paired_rows(records, args.bootstrap_replicates))

    trace_tasks = [(method, 18.0, int(seed), args.steps) for method in ("alf", "crdf", "crmf") for seed in seeds]
    trace_rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in trace_tasks:
            trace_rows.extend(_trace_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_trace_task, task) for task in trace_tasks]
            for future in as_completed(futures):
                trace_rows.extend(future.result())
    trace_rows.sort(key=lambda row: (str(row["method"]), int(row["seed"]), int(row["step"])))
    trace_raw_path = args.out / "crdf_radius_timeseries_raw.csv"
    trace_summary_path = args.out / "crdf_radius_timeseries.csv"
    _write_csv(trace_raw_path, trace_rows)
    _write_csv(trace_summary_path, _trace_summary(trace_rows, args.bootstrap_replicates))

    figure_out = args.out / "crdf_density_comparison"
    diagnostic_out = args.out / "crdf_radius_diagnostic"
    if not args.skip_figures:
        _make_figures(bootstrap_path, trace_summary_path, figure_out, diagnostic_out)

    manifest = {
        "schema_version": "alf-crdf-matched/v1",
        "status": "completed",
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "n_agents": N_AGENTS,
        "dimension": 2,
        "dt": 0.5,
        "steps": args.steps,
        "half_widths": list(half_widths),
        "densities": [N_AGENTS / (2.0 * value) ** 2 for value in half_widths],
        "seeds": list(seeds),
        "parameters": {
            "r_b": 4.0,
            "r_min": 1.0,
            "r_max": 8.0,
            "k": 6.0,
            "gamma": 0.5,
            "alpha": 0.9,
            "delta_m": 0.25,
            "g": 1.0,
            "beta": 0.5,
            "noise_std": 0.0,
        },
        "crdf_definition": "degree at t uses positions at t and realized radius at t-1; command remains anchored to base radius; t=0 uses fixed probe initialization",
        "crmf_definition": "degree at t uses positions at t and realized radius at t-1; the next command multiplies that radius by (k/max(d,1))^gamma; t=0 uses fixed probe initialization",
        "physical_sensing_envelope": 8.25,
        "nominal_interaction_radius_cap": 8.0,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_interval": "percentile 95%",
        "paired_difference": "CRDF - ALF and CRMF - ALF within shared seed and density",
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "bootstrap_summary_csv": str(bootstrap_path),
        "paired_comparison_csv": str(paired_path),
        "radius_timeseries_raw_csv": str(trace_raw_path),
        "radius_timeseries_csv": str(trace_summary_path),
        "figure_pdf": str(figure_out.with_suffix(".pdf")),
        "figure_png": str(figure_out.with_suffix(".png")),
        "diagnostic_pdf": str(diagnostic_out.with_suffix(".pdf")),
        "diagnostic_png": str(diagnostic_out.with_suffix(".png")),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"raw CSV: {raw_path}")
    print(f"bootstrap summary: {bootstrap_path}")
    print(f"paired comparison: {paired_path}")
    print(f"figure: {figure_out.with_suffix('.pdf')}")
    print(f"diagnostic: {diagnostic_out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
