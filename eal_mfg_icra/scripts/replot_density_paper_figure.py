"""Re-plot the canonical density-comparison figure from the frozen bootstrap summary.

Presentation-only re-render.  No simulation is (re)run: this script reads the
bootstrap artifact written by ``crdf_comparison.py`` and produces the four-panel
figure used in the manuscript.

Panels (R7 revision -- task-level redundancy removed, regime partition made explicit)
-----------------------------------------------------------------------------------
(a) final fragmentation          -- connectivity
(b) mean interaction out-degree  -- selected load (the operating saving)
(c) mean realized radius         -- the mechanism
(d) trajectory-mean polar order  -- task-level flocking quality

Removed from the manuscript figure in R7:
* velocity dispersion -- §VI-B shows every comparator other than fixed r=4 differs
  from ALF by less than 0.003, so the panel carried almost no information;
* largest-SCC fraction -- redundant with final fragmentation for the connectivity
  story at this density range; its numbers are quoted in the text and Table I.

Both the retired metrics remain available in the frozen artifact and are still
reported in the manuscript body.

The critical density rho_k = k / (pi r_max^2) = 0.0298 is drawn as a vertical
dashed rule and labelled on the first panel.  Only the *saturated* side carries
a regime name: the paper's second regime boundary is the separability threshold
rho* ~ 2.6 rho_k (Sec. VI-A), which is grid-dependent and offline-only, so it is
not drawn.  Labelling the whole rho >= rho_k half-plane "actively tuned" would
contradict Table I, where the uniform cells at rho <= 1.11 rho_k remain
cap-saturated and degenerate; the first panel therefore marks the boundary
neutrally.

Usage
-----
    python -m eal_mfg_icra.scripts.replot_density_paper_figure \
        --bootstrap experiments/adaptive_feedback_matched/crdf_matched_bootstrap_summary.csv \
        --out paper/figures/fig_crdf_density_comparison
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

FIGURE_METHODS = ("fixed_r4", "fixed_r8_retained", "knn_k6", "crdf", "crmf", "alf")

# Review action M-1: label CRDF explicitly instead of the ambiguous
# "Current-radius feedback" that was confusable with CRMF.
STYLES = {
    "fixed_r4": ("Fixed $r=4$", "#666666", "s", "--"),
    "fixed_r8_retained": ("Max radius $+$ retention", "#E69F00", "v", "--"),
    "knn_k6": ("Local-KNN $K=6$", "#009E73", "D", ":"),
    "crdf": ("CRDF", "#CC79A7", "P", "--"),
    "crmf": ("CRMF", "#56B4E9", "X", "-."),
    "alf": ("ALF", "#0072B2", "o", "-"),
}

# R7: four panels, each roughly (3.3 x 1.55) in, included in the paper at
# 0.97\textwidth, so an 8.5 pt label renders at ~8.2 pt instead of the ~3.5 pt
# produced by the previous 0.42\textwidth inclusion.  Do NOT drop the inclusion
# width: at 0.76\textwidth the same source renders labels at 6.7 pt and tick
# labels at 6.3 pt, both below the 7 pt legibility floor.
# Short axis labels: at a 7 in source width each panel is ~1 in tall, and a
# longer rotated label (e.g. the full "Final fragmentation") overlaps the label
# of the panel below.  Panel titles and the caption carry the full names.
PANELS = (
    ("final_fragmentation", "Fragmentation", (0.0, 0.60), "(a) Fragmentation"),
    ("mean_sym_lambda2", "Connectivity", (0.0, 20.0), "(b) Alg. connectivity"),
    ("mean_radius", "Radius", (3.0, 8.4), "(c) Realized radius"),
    ("polar_order", "Polar order", (0.35, 1.02), "(d) Flocking order"),
)

# The Fiedler value spans three decades and can be exactly zero when the realized
# graph is disconnected, so that panel needs a symmetric-log axis.
SYMLOG_PANELS = {"mean_sym_lambda2"}

# Local-KNN has no realized metric interaction radius comparable to
# ALF/CRDF/CRMF, so it is omitted from the radius panel only.
OMIT = {("mean_radius", "knn_k6")}

# Four plain-decimal ticks cover the plotted range without decade clutter.
XTICKS = (0.005, 0.01, 0.02, 0.05)
XTICK_LABELS = ("0.005", "0.01", "0.02", "0.05")

# k / (pi r_max^2) with k = 6, r_max = 8 (manuscript Sec. VI-A).
RHO_K = 6.0 / (3.141592653589793 * 64.0)  # 0.029841...
RHO_SENSING = 0.01
# 1 / (pi r_max^2): the density at which the cap holds exactly one expected
# neighbour (manuscript Sec. V-A).  Drawn explicitly so rho_k and rho_1 are both
# demarcated in the figure rather than only in the caption.
RHO_1 = 1.0 / (3.141592653589793 * 64.0)  # 0.004974...


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=Path(
            "experiments/adaptive_feedback_matched/crdf_matched_bootstrap_summary.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper/figures/fig_crdf_density_comparison"),
    )
    parser.add_argument(
        "--connectivity",
        type=Path,
        default=Path("paper/evidence/connectivity_margin_cell_summary.csv"),
        help=(
            "Cell-level algebraic-connectivity summary emitted by "
            "connectivity_margin_audit.py.  Fragmentation and largest-SCC are nearly "
            "binary, so panel (b) reports the Fiedler value instead of the "
            "near-definitional selected out-degree."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)

    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to re-render this figure") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.bootstrap.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty bootstrap summary: {args.bootstrap}")

    data = {(r["method"], float(r["density"]), r["metric"]): r for r in rows}
    density_values = {float(r["density"]) for r in rows}
    if args.connectivity.exists():
        with args.connectivity.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                key = (record["method"], float(record["density"]), record["metric"])
                data.setdefault(key, record)
                density_values.add(float(record["density"]))
    densities = sorted(density_values)

    # 8.5 pt body so that, after a ~1.0 inclusion scale, labels stay >= 8 pt.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.4), sharex=True)
    for ax, (metric, ylabel, ylim, title) in zip(axes.flat, PANELS):
        # Sensing-limited tail, unchanged from the previous render.
        ax.axvspan(0.0, RHO_SENSING, color="#E5E5E5", alpha=0.65, zorder=0)
        ax.axvline(
            RHO_SENSING,
            color="#777777",
            linewidth=0.7,
            linestyle=":",
            zorder=1,
        )
        # Analytic reference rho_1: the cap holds one expected neighbour here.
        ax.axvline(
            RHO_1,
            color="#4D4D4D",
            linewidth=0.8,
            linestyle="-.",
            zorder=1,
        )
        # Critical density: the regime boundary the paper's two-stage claim uses.
        ax.axvline(
            RHO_K,
            color="#B2182B",
            linewidth=1.1,
            linestyle="--",
            zorder=2,
        )
        for method in FIGURE_METHODS:
            if (metric, method) in OMIT:
                continue
            points = [
                data[(method, d, metric)]
                for d in densities
                if (method, d, metric) in data
            ]
            if not points:
                continue
            x = [float(r["density"]) for r in points]
            y = [float(r["mean"]) for r in points]
            low = [v - float(r["ci95_low"]) for v, r in zip(y, points)]
            high = [float(r["ci95_high"]) - v for v, r in zip(y, points)]
            label, color, marker, linestyle = STYLES[method]
            ax.errorbar(
                x,
                y,
                yerr=[low, high],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.45 if method == "alf" else 1.0,
                markersize=5.4 if method == "alf" else 4.0,
                markerfacecolor="white" if method in {"fixed_r4", "crdf"} else color,
                markeredgecolor=color,
                capsize=1.4,
            )
        ax.set_xscale("log")
        if metric in SYMLOG_PANELS:
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_yticks([0.0, 0.01, 0.1, 1.0, 10.0])
            ax.set_yticklabels(["0", "0.01", "0.1", "1", "10"])
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=2)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XTICK_LABELS)
        ax.set_xlim(0.0037, 0.105)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        ax.tick_params(direction="out", length=3, width=0.6)

    # Regime annotation: on the first panel only, so the other three stay clean.
    axes[0, 0].text(
        0.0039,
        PANELS[0][2][1] * 0.965,
        r"Saturated ($\rho<\rho_k$)",
        fontsize=7.4,
        color="#444444",
        ha="left",
        va="top",
    )
    # Boundary tag only.  The previous "Actively tuned (rho >= rho_k)" label was
    # stale: Sec. VI-C shows the uniform factorial cells at rho <= 1.11 rho_k are
    # still cap-saturated (0.51-0.89) and degenerate, and separation needs
    # rho > rho* ~ 2.6 rho_k.  The label must not assert a regime the data deny.
    axes[0, 0].text(
        0.0315,
        PANELS[0][2][1] * 0.965,
        r"$\rho\geq\rho_k$",
        fontsize=7.4,
        color="#B2182B",
        ha="left",
        va="top",
    )
    # Explicit tags for the two analytic references.  Panel (c) is used because
    # its lower band is empty at both rules, so neither tag can collide with a
    # series or with the regime annotations.
    axes[1, 0].text(
        RHO_1 * 1.06,
        PANELS[2][2][0] + 0.12,
        r"$\rho_1$",
        fontsize=7.6,
        color="#4D4D4D",
        ha="left",
        va="bottom",
    )
    axes[1, 0].text(
        RHO_K * 1.05,
        PANELS[2][2][0] + 0.12,
        r"$\rho_k$",
        fontsize=7.6,
        color="#B2182B",
        ha="left",
        va="bottom",
    )

    axes[1, 0].set_xlabel(r"Density $\rho=N/(2L)^2$")
    axes[1, 1].set_xlabel(r"Density $\rho=N/(2L)^2$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # A single legend row keeps the plot area tall enough for legible panel
    # labels; the aperture/base meaning of CRDF, CRMF and ALF is defined in the
    # caption instead of eating a second legend row.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
        columnspacing=0.75,
        handletextpad=0.3,
    )
    # A dashed red rule in the legend key would be misread as a series, so the
    # rho_k and rho_1 rules are explained in the caption instead.
    fig.tight_layout(rect=(0, 0, 1, 0.905), pad=0.5, w_pad=1.6, h_pad=1.9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        args.out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.03
    )
    plt.close(fig)
    print(f"wrote {args.out.with_suffix('.pdf')}")
    print(f"wrote {args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
