#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wsl_3d_paired_stats.py
======================
Paired analysis of the 3D quadrotor validation episodes produced by
`scripts/run_wsl_3d_quadrotor_validation.py`.

Reads paper/output/wsl_3d_metrics.csv, pairs ALF against the retention-matched
max-radius control on (density_ratio, seed) -- the two policies share the same
initial condition and the same noise realisation (common random numbers) -- and
reports paired differences with 95% percentile-bootstrap confidence intervals.

Metrics
-------
  dFrag          = Frag_ALF - Frag_Ctrl          (negative => ALF fragments less)
  dMCC_weak      = (1-Frag_ALF) - (1-Frag_Ctrl)  (largest weakly connected fraction)
  dMCC_strong    = SCC_ALF - SCC_Ctrl            (largest strongly connected fraction)
  dJu_pct        = (Ju_ALF - Ju_Ctrl)/Ju_Ctrl * 100   (mean squared control effort)
  dJerk_pct      = (jerk_ALF - jerk_Ctrl)/jerk_Ctrl * 100  (step-to-step control chatter)
  dOrder         = Order_ALF - Order_Ctrl
  dDisp          = Disp_ALF - Disp_Ctrl
Collisions are counted, not averaged.

Column semantics in the CSV (be explicit, two of these names are ambiguous):
  frag_final / frag_mean          -> 1 - largest WEAK component fraction
  largest_scc_final               -> largest STRONG component fraction
  scc_final_frac                  -> NUMBER of strong components (a count)
  n_components_final              -> NUMBER of weak components (a count)

Usage:
  python3 scripts/wsl_3d_paired_stats.py [--csv paper/output/wsl_3d_metrics.csv]
                                         [--bootstrap 10000] [--band 1.0,2.0,4.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

ALF, CTRL = "alf", "ctrl"


def load(csv_path):
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "ratio": float(r["density_ratio"]),
                    "rho": float(r["rho"]),
                    "seed": int(r["seed"]),
                    "policy": r["policy"].strip(),
                    "frag": float(r["frag_final"]),
                    "frag_mean": float(r["frag_mean"]),
                    "mcc_strong": float(r["largest_scc_final"]),
                    "n_scc": float(r["scc_final_frac"]),
                    "n_weak": float(r["n_components_final"]),
                    "lam2": float(r["lambda2_final"]),
                    "min_deg": float(r["min_degree_final"]),
                    "deg": float(r["mean_degree"]),
                    "radius": float(r["mean_radius"]),
                    "ju": float(r["ju_applied"]),
                    "jerk": float(r["ju_jerk"]),
                    "sat": float(r["sat_fraction"]),
                    "order": float(r["order_tail"]),
                    "disp": float(r["dispersion_tail"]),
                    "min_pair": float(r["min_pair_dist"]),
                    "coll_events": float(r["collision_events"]),
                    "coll_pairs": float(r["collision_pairs"]),
                    "finite": float(r["finite"]),
                    "wall": float(r["wall_time_s"]),
                })
            except (TypeError, ValueError):
                continue
    return rows


def pair_up(rows, ratios):
    """Return list of paired dicts {(ratio,seed): (alf_row, ctrl_row)}."""
    idx = {}
    for r in rows:
        if ratios and r["ratio"] not in ratios:
            continue
        idx.setdefault((r["ratio"], r["seed"]), {})[r["policy"]] = r
    pairs = []
    for key, d in sorted(idx.items()):
        if ALF in d and CTRL in d:
            pairs.append((key[0], key[1], d[ALF], d[CTRL]))
    return pairs


def boot_ci(x, n_boot=10000, seed=12345, alpha=0.05):
    """Percentile bootstrap CI of the mean of a 1-D array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi), int(x.size)


def rel_saving(a, c):
    """Relative saving of a vs c, in percent; positive => a is cheaper."""
    return (c - a) / c * 100.0 if c != 0 else float("nan")


def summarize(pairs, label, n_boot):
    d_frag = np.array([a["frag"] - c["frag"] for _, _, a, c in pairs])
    d_mccw = -d_frag
    d_mccs = np.array([a["mcc_strong"] - c["mcc_strong"] for _, _, a, c in pairs])
    d_lam2 = np.array([a["lam2"] - c["lam2"] for _, _, a, c in pairs])
    d_deg = np.array([a["deg"] - c["deg"] for _, _, a, c in pairs])
    d_rad = np.array([a["radius"] - c["radius"] for _, _, a, c in pairs])
    d_ju = np.array([(a["ju"] - c["ju"]) / c["ju"] * 100.0
                     for _, _, a, c in pairs])
    d_jerk = np.array([(a["jerk"] - c["jerk"]) / c["jerk"] * 100.0
                       for _, _, a, c in pairs])
    d_ord = np.array([a["order"] - c["order"] for _, _, a, c in pairs])
    d_disp = np.array([a["disp"] - c["disp"] for _, _, a, c in pairs])

    out = {"label": label, "n_pairs": len(pairs)}
    spec = [
        ("dFrag", d_frag, "Frag_ALF - Frag_Ctrl"),
        ("dMCC_weak", d_mccw, "(1-Frag)_ALF - (1-Frag)_Ctrl"),
        ("dMCC_strong", d_mccs, "SCCfrac_ALF - SCCfrac_Ctrl"),
        ("dLambda2", d_lam2, "lambda2_ALF - lambda2_Ctrl"),
        ("dMeanDegree", d_deg, "deg_ALF - deg_Ctrl"),
        ("dMeanRadius", d_rad, "r_ALF - r_Ctrl"),
        ("dJu_pct", d_ju, "Ju saving of ALF vs Ctrl, % (negative = ALF cheaper)"),
        ("dJerk_pct", d_jerk, "chatter saving of ALF vs Ctrl, %"),
        ("dOrder", d_ord, "Order_ALF - Order_Ctrl"),
        ("dDispersion", d_disp, "dispersion_ALF - dispersion_Ctrl"),
    ]
    for name, arr, desc in spec:
        m, lo, hi, n = boot_ci(arr, n_boot)
        out[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": n, "desc": desc,
                     "mean_abs": float(np.nanmean(np.abs(arr)))}
    out["ju_saving_pct"] = -out["dJu_pct"]["mean"]
    out["ju_saving_ci"] = [-out["dJu_pct"]["ci_hi"], -out["dJu_pct"]["ci_lo"]]
    out["jerk_saving_pct"] = -out["dJerk_pct"]["mean"]

    # level quantities
    for pol in (ALF, CTRL):
        sub = [r for _, _, a, c in pairs for r in ((a,) if pol == ALF else (c,))]
        key = f"{pol}"
        out[key] = {
            "frag_mean": float(np.mean([r["frag"] for r in sub])),
            "frag_max": float(np.max([r["frag"] for r in sub])),
            "n_fragmented_episodes": int(sum(1 for r in sub if r["frag"] > 1e-12)),
            "mcc_strong_mean": float(np.mean([r["mcc_strong"] for r in sub])),
            "lambda2_mean": float(np.mean([r["lam2"] for r in sub])),
            "degree_mean": float(np.mean([r["deg"] for r in sub])),
            "radius_mean": float(np.mean([r["radius"] for r in sub])),
            "ju_mean": float(np.mean([r["ju"] for r in sub])),
            "jerk_mean": float(np.mean([r["jerk"] for r in sub])),
            "order_mean": float(np.mean([r["order"] for r in sub])),
            "dispersion_mean": float(np.mean([r["disp"] for r in sub])),
            "min_pair_min": float(np.min([r["min_pair"] for r in sub])),
            "collision_events": int(sum(r["coll_events"] for r in sub)),
            "collision_pairs": int(sum(r["coll_pairs"] for r in sub)),
            "episodes_with_collision": int(sum(1 for r in sub if r["coll_events"] > 0)),
            "episodes": len(sub),
            "all_finite": bool(all(r["finite"] == 1 for r in sub)),
            "mean_wall_s": float(np.mean([r["wall"] for r in sub])),
        }
    return out


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if np.isfinite(v) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="paper/output/wsl_3d_metrics.csv")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--band", default="1.0,2.0,4.0",
                    help="density ratios pooled for the headline statistic")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 2

    rows = load(args.csv)
    ratios_all = sorted({r["ratio"] for r in rows})
    band = [float(x) for x in args.band.split(",") if x.strip()]
    band = [b for b in band if b in ratios_all] or ratios_all

    print("=" * 78)
    print("ALF / ICRA 2027 -- 3D quadrotor validation: paired statistics")
    print("=" * 78)
    print(f"input        : {args.csv}")
    print(f"rows         : {len(rows)}   densities: {ratios_all}")
    print(f"pooled band  : {band}")
    print(f"bootstrap    : {args.bootstrap} resamples (percentile, 95%)")
    print("design       : common random numbers -- same IC + same noise per seed")
    print("-" * 78)

    results = {}
    for ratio in ratios_all:
        pairs = pair_up(rows, [ratio])
        if not pairs:
            continue
        s = summarize(pairs, f"ratio={ratio}", args.bootstrap)
        results[f"ratio_{ratio}"] = s
        print(f"\n### density ratio {ratio}  (n_pairs={s['n_pairs']})")
        print(f"  ALF  : frag={fmt(s['alf']['frag_mean'])} "
              f"(max {fmt(s['alf']['frag_max'])}, "
              f"fragmented eps {s['alf']['n_fragmented_episodes']}/{s['alf']['episodes']})  "
              f"deg={fmt(s['alf']['degree_mean'],2)}  r={fmt(s['alf']['radius_mean'],2)}  "
              f"Ju={fmt(s['alf']['ju_mean'])}  jerk={fmt(s['alf']['jerk_mean'],4)}  "
              f"ord={fmt(s['alf']['order_mean'],3)}")
        print(f"  Ctrl : frag={fmt(s['ctrl']['frag_mean'])} "
              f"(max {fmt(s['ctrl']['frag_max'])}, "
              f"fragmented eps {s['ctrl']['n_fragmented_episodes']}/{s['ctrl']['episodes']})  "
              f"deg={fmt(s['ctrl']['degree_mean'],2)}  r={fmt(s['ctrl']['radius_mean'],2)}  "
              f"Ju={fmt(s['ctrl']['ju_mean'])}  jerk={fmt(s['ctrl']['jerk_mean'],4)}  "
              f"ord={fmt(s['ctrl']['order_mean'],3)}")
        for k in ("dFrag", "dMCC_weak", "dMCC_strong", "dMeanDegree",
                  "dJu_pct", "dJerk_pct", "dOrder", "dDispersion"):
            e = s[k]
            print(f"    {k:<13} = {e['mean']:+.4f}  95% CI "
                  f"[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]")
        print(f"    => Ju saving ALF vs Ctrl = {s['ju_saving_pct']:+.2f}% "
              f"(95% CI [{s['ju_saving_ci'][0]:+.2f}%, {s['ju_saving_ci'][1]:+.2f}%])")
        print(f"    collisions: ALF {s['alf']['collision_events']} events / "
              f"{s['alf']['episodes_with_collision']} eps, "
              f"Ctrl {s['ctrl']['collision_events']} events / "
              f"{s['ctrl']['episodes_with_collision']} eps; "
              f"min pair dist ALF {fmt(s['alf']['min_pair_min'],3)} m, "
              f"Ctrl {fmt(s['ctrl']['min_pair_min'],3)} m")

    band_pairs = pair_up(rows, band)
    if band_pairs:
        s = summarize(band_pairs, f"band={band}", args.bootstrap)
        results["band"] = s
        print("\n" + "=" * 78)
        print(f"HEADLINE (pooled over rho/rho_k^3D in {band}, n_pairs={s['n_pairs']})")
        print("=" * 78)
        print(f"  ALF  mean degree {fmt(s['alf']['degree_mean'],2)} "
              f"(radius {fmt(s['alf']['radius_mean'],2)} m) vs "
              f"Ctrl {fmt(s['ctrl']['degree_mean'],2)} "
              f"(radius {fmt(s['ctrl']['radius_mean'],2)} m)")
        print(f"  Fragmentation: ALF {fmt(s['alf']['frag_mean'])} "
              f"({s['alf']['n_fragmented_episodes']}/{s['alf']['episodes']} fragmented), "
              f"Ctrl {fmt(s['ctrl']['frag_mean'])} "
              f"({s['ctrl']['n_fragmented_episodes']}/{s['ctrl']['episodes']} fragmented)")
        print(f"  dFrag        = {s['dFrag']['mean']:+.4f}  "
              f"95% CI [{s['dFrag']['ci_lo']:+.4f}, {s['dFrag']['ci_hi']:+.4f}]")
        print(f"  dMCC_weak    = {s['dMCC_weak']['mean']:+.4f}  "
              f"95% CI [{s['dMCC_weak']['ci_lo']:+.4f}, {s['dMCC_weak']['ci_hi']:+.4f}]")
        print(f"  dMCC_strong  = {s['dMCC_strong']['mean']:+.4f}  "
              f"95% CI [{s['dMCC_strong']['ci_lo']:+.4f}, {s['dMCC_strong']['ci_hi']:+.4f}]")
        print(f"  >>> dJu      = {s['dJu_pct']['mean']:+.2f}%  "
              f"95% CI [{s['dJu_pct']['ci_lo']:+.2f}%, {s['dJu_pct']['ci_hi']:+.2f}%]")
        print(f"  >>> Ju saving (Ctrl->ALF) = {s['ju_saving_pct']:+.2f}%  "
              f"95% CI [{s['ju_saving_ci'][0]:+.2f}%, {s['ju_saving_ci'][1]:+.2f}%]")
        print(f"  dJerk        = {s['dJerk_pct']['mean']:+.2f}%  "
              f"95% CI [{s['dJerk_pct']['ci_lo']:+.2f}%, {s['dJerk_pct']['ci_hi']:+.2f}%]  "
              f"(chatter saving {s['jerk_saving_pct']:+.2f}%)")
        print(f"  dOrder       = {s['dOrder']['mean']:+.4f}  "
              f"95% CI [{s['dOrder']['ci_lo']:+.4f}, {s['dOrder']['ci_hi']:+.4f}]")
        print(f"  dDispersion  = {s['dDispersion']['mean']:+.4f} m  "
              f"95% CI [{s['dDispersion']['ci_lo']:+.4f}, {s['dDispersion']['ci_hi']:+.4f}]")
        print(f"  collisions   : ALF {s['alf']['collision_events']} events, "
              f"Ctrl {s['ctrl']['collision_events']} events "
              f"(body radius D_BODY in meta); all episodes finite = "
              f"{s['alf']['all_finite'] and s['ctrl']['all_finite']}")
        print(f"  saturation   : ALF {fmt(sum(a['sat'] for _,_,a,_ in band_pairs)/len(band_pairs),5)}, "
              f"Ctrl {fmt(sum(c['sat'] for _,_,_,c in band_pairs)/len(band_pairs),5)}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
