#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_wsl_3d_quadrotor_validation.py
==================================
ICRA 2027 / ALF -- 3D continuous-time dynamics robustness validation.

Purpose
-------
Answer two reviewer objections with one experiment:
  (R-a) "The evidence is idealised 2D, planar, noiseless."
  (R-b) "ALF's algebraic-connectivity (Fiedler) deficit means it is NOT robust."

Design
------
Backend: pure NumPy/SciPy headless physics kernel (no GUI, no ROS, no Gazebo).
  The WSL box has Gazebo Classic 11 binaries but NO ROS / gazebo_ros bridge, so a
  Gazebo route cannot command the agents; the NumPy/SciPy second-order 3D
  quadrotor-like dynamics kernel is used instead (explicitly allowed by spec).

Agents      : N = 16, state p_i, v_i in R^3, acceleration command u_i in R^3
              saturated componentwise to [-a_max, a_max].
Control     : bounded pairwise second-order flocking law (Tanner-style):
                  u_i = sum_{j in N_i} w_ij [ c_coh g(d_ij) e_ji + c_vel (v_j - v_i) ]
              with g(d) = tanh((d - d_eq)/s) + bounded short-range repulsion, and
              w_ij = (d_wmin / max(d_ij, d_wmin))^beta  (inverse-distance, beta=0.5).
              All per-link gains are IDENTICAL for both policies: the effort
              difference is a consequence of the realised interaction graph only.
Disturbances: (1) Gaussian white acceleration noise (air turbulence / actuator
                  jitter), Ito-scaled by sqrt(dt);
              (2) aerodynamic drag  -c_d ||v_i|| v_i;
              (3) rotor downwash: an agent above another inside a cone pushes the
                  lower agent downward and radially outward.
Policies    : A = ALF   -> fixed probe r_b=4, adaptive local interaction radius
                           r_i^t with count dead-band delta_count=1 and contact
                           retention delta_m=0.25 (EMA alpha=0.9, gamma=0.5, k=6);
              B = Ctrl  -> retention-matched max-radius, interaction radius fixed
                           at r_max=8.

Run spec    : dt = 0.05 s, T = 100 s (2000 steps), 20 paired seeds per density
              (common random numbers -> identical ICs and identical noise for both
              policies), streamed to paper/output/wsl_3d_metrics.csv.

Headless: no display, no GUI, no plotting.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --------------------------------------------------------------------------
# Fixed configuration (matches the manuscript's canonical setting)
# --------------------------------------------------------------------------
N_AGENTS = 16
DT = 0.05
T_END = 100.0
N_STEPS = int(round(T_END / DT))          # 2000

R_PROBE = 4.0                              # r_b, fixed 3D probe radius
R_MAX = 8.0                                # r_max
R_MIN = 1.0
K_TARGET = 6.0                             # k (target expected degree at r_b)
GAMMA = 0.5                                # r = r_b (k/max(d_b,1))^gamma
ALPHA_EMA = 0.9
DELTA_COUNT = 1                            # hysteresis dead-band on counted degree
DELTA_M = 0.25                             # contact-retention margin

# 3D reference density: k / ((4/3) pi r_max^3)
RHO_K_3D = K_TARGET / ((4.0 / 3.0) * np.pi * R_MAX ** 3)     # ~0.0027976

# -- physics constants (all pairwise forces are bounded -> no stiff blow-up) --
A_MAX = 10.0                               # per-axis acceleration saturation
C_COH = 0.70                               # distance-regulation gain (per link)
C_VEL = 0.50                               # velocity-alignment gain (per link)
D_EQ = 4.0                                 # equilibrium spacing
D_SCALE = 2.0                              # tanh softness of distance regulation
D_SEP = 3.0                                # short-range repulsion onset
K_REP = 4.0                                # repulsion strength
REP_CLAMP = 10.0                           # bounded repulsion (m/s^2)
D_FLOOR = 0.5                              # denominator floor for 1/d
D_WMIN = 1.0                               # weight floor
BETA_W = 0.5                               # inverse-distance weighting exponent
C_DRAG = 0.10                              # aerodynamic drag coefficient
SIGMA_TURB = 0.20                          # turbulence noise intensity
# NOTE: white acceleration noise of intensity sigma produces a velocity random
# walk with std sigma*sqrt(T) = 2 m/s over T=100 s, i.e. below the drag-limited
# terminal speed (~|u|/c_d)^(1/2) ~ 6 m/s: the swarm is perturbed, not blown apart.
C_WALL = 2.0                               # soft containment (geofence) stiffness
WALL_MARGIN = 0.80                         # containment acts beyond 0.8*ball_radius
A_DW = 0.80                                # downwash strength
H_DW = 5.0                                 # downwash vertical decay length
R_DW = 3.5                                 # downwash cone horizontal radius
K_SPREAD = 0.5                             # radial-spreading fraction of downwash
D_BODY = 0.7                               # physical collision radius

BASE_SEED = 71000                          # distinct from the paper's seed banks

POLICIES = ("alf", "ctrl")

CSV_FIELDS = [
    "density_ratio", "rho", "domain_side", "policy", "seed",
    "frag_final", "frag_mean", "largest_scc_final", "scc_final_frac",
    "n_components_final", "lambda2_final", "min_degree_final",
    "mean_degree", "mean_radius", "min_radius", "max_radius",
    "ju_applied", "ju_commanded", "ju_jerk", "sat_fraction",
    "order_tail", "dispersion_tail", "min_pair_dist",
    "collision_events", "collision_pairs", "finite", "wall_time_s",
]


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------
def weak_components(adj: np.ndarray):
    """Number and size of weakly connected components of a directed adjacency."""
    n_comp, labels = connected_components(csr_matrix(adj), directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    return int(n_comp), int(sizes.max())


def strong_components(adj: np.ndarray):
    """Number and size of strongly connected components."""
    n_comp, labels = connected_components(
        csr_matrix(adj), directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=n_comp)
    return int(n_comp), int(sizes.max())


def fiedler_value(adj_sym: np.ndarray) -> float:
    """Second-smallest Laplacian eigenvalue of the symmetrised graph."""
    deg = adj_sym.sum(axis=1)
    lap = np.diag(deg) - adj_sym.astype(float)
    ev = np.linalg.eigvalsh(lap)
    return float(ev[1]) if ev.size > 1 else 0.0


# --------------------------------------------------------------------------
# ALF radius law (3D)
# --------------------------------------------------------------------------
def alf_radius_update(dist, r_state, r_ema, last_count):
    """One step of the ALF adaptive-radius state machine.

    probe count -> count dead-band -> raw law -> EMA -> retention -> clip.
    """
    d_b = (dist <= R_PROBE).sum(axis=1).astype(float)     # dist diagonal = inf
    moved = np.abs(d_b - last_count) > DELTA_COUNT        # +/-1 count dead-band
    count = np.where(moved, d_b, last_count)

    r_raw = np.clip(
        R_PROBE * (K_TARGET / np.maximum(count, 1.0)) ** GAMMA, R_MIN, R_MAX)

    r_ema = ALPHA_EMA * r_ema + (1.0 - ALPHA_EMA) * r_raw
    # contact retention: the radius may shrink by at most DELTA_M per step
    r_ret = np.maximum(r_ema, (1.0 - DELTA_M) * r_state)
    r_state = np.clip(r_ret, R_MIN, R_MAX)
    return r_state, r_ema, count


# --------------------------------------------------------------------------
# Physics kernel
# --------------------------------------------------------------------------
def simulate(policy: str, pos0: np.ndarray, noise: np.ndarray, wall_r: float = 0.0):
    """Run one episode. `noise` is (N_STEPS, N, 3), shared across policies.

    `wall_r` > 0 activates a soft radial containment field (bounded airspace /
    geofence): agents beyond WALL_MARGIN*wall_r feel an inward restoring
    acceleration. It is a boundary condition, not an inter-agent coupling, so it
    creates no graph links.
    """
    pos = pos0.copy()
    vel = np.zeros_like(pos)

    r_state = np.full(N_AGENTS, R_MAX)
    r_ema = np.full(N_AGENTS, R_MAX)
    last_count = np.zeros(N_AGENTS)

    ju_applied = ju_commanded = 0.0
    ju_jerk_acc = 0.0
    u_prev = None
    sat_steps = 0
    deg_acc = r_acc = frag_acc = 0.0
    min_pair = np.inf
    collide_events = 0
    collide_pairs = set()
    order_acc = disp_acc = 0.0
    tail_n = 0
    n_done = 0

    frag_final = largest_scc_final = scc_frac_final = 0.0
    ncomp_final = mindeg_final = 0
    lam2_final = 0.0
    finite_ok = True

    for t in range(N_STEPS):
        # ---- pairwise geometry ----
        diff = pos[None, :, :] - pos[:, None, :]          # diff[i,j] = p_j - p_i
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        # ---- interaction radius ----
        if policy == "alf":
            r_state, r_ema, last_count = alf_radius_update(
                dist, r_state, r_ema, last_count)
        else:
            r_state = np.full(N_AGENTS, R_MAX)

        mask = dist <= r_state[:, None]
        deg = mask.sum(axis=1)

        # ---- bounded pairwise control (identical per-link gains) ----
        inv_d = 1.0 / np.maximum(dist, D_FLOOR)
        w = np.where(mask, (D_WMIN / np.maximum(dist, D_WMIN)) ** BETA_W, 0.0)

        g = np.tanh((dist - D_EQ) / D_SCALE)
        rep = np.where(mask & (dist < D_SEP),
                       -K_REP * (inv_d - 1.0 / D_SEP), 0.0)
        rep = np.maximum(rep, -REP_CLAMP)
        g = np.where(mask, g + rep, 0.0)

        ehat = diff * inv_d[:, :, None]                   # unit vector i -> j
        u = C_COH * (w[:, :, None] * g[:, :, None] * ehat).sum(axis=1)

        dvel = vel[None, :, :] - vel[:, None, :]
        u += C_VEL * (w[:, :, None] * dvel).sum(axis=1)

        # ---- downwash: upper agent a perturbs lower agent b inside its cone ----
        dzz = pos[:, None, 2] - pos[None, :, 2]           # dzz[a,b] = z_a - z_b
        rho_h = np.hypot(pos[:, None, 0] - pos[None, 0, 0],
                         pos[:, None, 1] - pos[None, 0, 1])
        cone = (dzz > 0.0) & (dzz < H_DW) & (rho_h > 1e-6) & (rho_h < R_DW)
        safe_rho = np.maximum(rho_h, 1e-6)
        decay = np.where(cone, np.exp(-np.abs(dzz) / H_DW) * (1.0 - rho_h / R_DW),
                         0.0)

        exy_x = np.where(cone, (pos[None, :, 0] - pos[:, None, 0]) / safe_rho, 0.0)
        exy_y = np.where(cone, (pos[None, :, 1] - pos[:, None, 1]) / safe_rho, 0.0)

        down = decay.sum(axis=0)
        dw = np.stack([A_DW * K_SPREAD * (decay * exy_x).sum(axis=0),
                       A_DW * K_SPREAD * (decay * exy_y).sum(axis=0),
                       -A_DW * down], axis=1)

        # ---- actuation ----
        u_cmd = u
        u_app = np.clip(u_cmd, -A_MAX, A_MAX)
        sat_steps += int((np.abs(u_cmd) > A_MAX).any(axis=1).sum())

        ju_commanded += float((u_cmd ** 2).sum())
        ju_applied += float((u_app ** 2).sum())
        if u_prev is not None:
            # control chatter: mean squared step-to-step acceleration variation
            ju_jerk_acc += float(((u_app - u_prev) ** 2).sum())
        u_prev = u_app

        # ---- integrate: drag, downwash and containment are external ----
        drag = -C_DRAG * np.linalg.norm(vel, axis=1, keepdims=True) * vel
        turb = SIGMA_TURB * np.sqrt(DT) * noise[t]

        wall = np.zeros_like(pos)
        if wall_r > 0.0:
            rad = np.linalg.norm(pos, axis=1, keepdims=True)
            over = np.maximum(rad - WALL_MARGIN * wall_r, 0.0)
            safe = np.maximum(rad, 1e-9)
            wall = -C_WALL * over * (pos / safe)

        acc = u_app + drag + dw + wall + turb

        vel = vel + DT * acc
        pos = pos + DT * vel

        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            finite_ok = False
            break

        n_done += 1

        # ---- metrics ----
        d_sym = np.minimum(dist, dist.T)
        adj = d_sym <= np.maximum(r_state[:, None], r_state[None, :])
        np.fill_diagonal(adj, False)
        adj_sym = adj | adj.T

        ncomp, lsize = weak_components(adj)
        n_scc, scc_size = strong_components(adj)
        frag = 1.0 - lsize / N_AGENTS
        frag_acc += frag

        deg_acc += float(deg.mean())
        r_acc += float(r_state.mean())

        off_diag = d_sym[np.isfinite(d_sym)]
        if off_diag.size:
            min_pair = min(min_pair, float(off_diag.min()))
        close = np.argwhere(np.triu(d_sym < D_BODY, k=1))
        if close.size:
            collide_events += int(close.shape[0])
            for a, b in close:
                collide_pairs.add((int(a), int(b)))

        if t >= N_STEPS // 2:
            sp = np.linalg.norm(vel, axis=1)
            tot = float(sp.sum())
            order_acc += (float(np.linalg.norm(vel.sum(axis=0)) / tot)
                          if tot > 0 else 0.0)
            cen = pos.mean(axis=0)
            disp_acc += float(np.linalg.norm(pos - cen, axis=1).mean())
            tail_n += 1

        if t == N_STEPS - 1:
            frag_final = frag
            ncomp_final = ncomp
            largest_scc_final = scc_size / N_AGENTS
            scc_frac_final = n_scc
            lam2_final = fiedler_value(adj_sym)
            mindeg_final = int(deg.min())

    n = max(n_done, 1)
    tail_n = max(tail_n, 1)
    return {
        "frag_final": frag_final,
        "frag_mean": frag_acc / n,
        "largest_scc_final": largest_scc_final,
        "scc_final_frac": scc_frac_final,
        "n_components_final": ncomp_final,
        "lambda2_final": lam2_final,
        "min_degree_final": mindeg_final,
        "mean_degree": deg_acc / n,
        "mean_radius": r_acc / n,
        "min_radius": float(r_state.min()),
        "max_radius": float(r_state.max()),
        "ju_applied": ju_applied / (N_AGENTS * n),
        "ju_commanded": ju_commanded / (N_AGENTS * n),
        "ju_jerk": ju_jerk_acc / (N_AGENTS * max(n - 1, 1)),
        "sat_fraction": sat_steps / (N_AGENTS * n),
        "order_tail": order_acc / tail_n,
        "dispersion_tail": disp_acc / tail_n,
        "min_pair_dist": (min_pair if np.isfinite(min_pair) else float("nan")),
        "collision_events": collide_events,
        "collision_pairs": len(collide_pairs),
        "finite": int(finite_ok),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/output")
    ap.add_argument("--ratios", default="1.0,2.5,5.0",
                    help="density levels as multiples of rho_k^3D")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--tags", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    metrics_path = os.path.join(outdir, "wsl_3d_metrics.csv")
    suffix = f"_{args.tags}" if args.tags else ""
    stream_path = os.path.join(outdir, f"wsl_3d_metrics{suffix}.csv")

    def log(*a):
        if not args.quiet:
            print(*a, flush=True)

    log("=" * 74)
    log("ALF / ICRA 2027 -- 3D quadrotor-dynamics robustness validation")
    log("=" * 74)
    log(f"host        : {platform.node()} | python {platform.python_version()} "
        f"| numpy {np.__version__}")
    log(f"utc         : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log(f"N={N_AGENTS}  dt={DT}  T={T_END}  steps={N_STEPS}")
    log(f"r_b={R_PROBE}  r_max={R_MAX}  k={K_TARGET}  gamma={GAMMA}  "
        f"alpha={ALPHA_EMA}  dcount={DELTA_COUNT}  d_m={DELTA_M}")
    log(f"rho_k(3D)   = k/((4/3)pi r_max^3) = {RHO_K_3D:.6f}")
    log(f"physics     : a_max={A_MAX} c_coh={C_COH} c_vel={C_VEL} d_eq={D_EQ} "
        f"cd={C_DRAG} sig={SIGMA_TURB} dw={A_DW} c_wall={C_WALL}")
    log(f"policies    = {POLICIES}   seeds = {args.seeds} (base {BASE_SEED})")
    log(f"output      = {stream_path}")
    log("-" * 74)

    fh = open(stream_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    fh.flush()

    t_start = time.time()
    total = len(ratios) * args.seeds * len(POLICIES)
    done = 0

    for ratio in ratios:
        rho = ratio * RHO_K_3D
        # spherical swarm: uniform-in-ball initial condition at volumetric density rho
        ball_r = (3.0 * N_AGENTS / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        log(f"[density] ratio={ratio:.2f}  rho={rho:.6f}  ball_radius={ball_r:.3f}  "
            f"mean_deg@r_max={rho * (4/3) * np.pi * R_MAX**3:.2f}")

        for s in range(args.seeds):
            seed = BASE_SEED + s
            rng = np.random.default_rng(seed)
            dirs = rng.standard_normal((N_AGENTS, 3))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            rad = ball_r * rng.random(N_AGENTS) ** (1.0 / 3.0)
            pos0 = dirs * rad[:, None]
            noise = rng.standard_normal((N_STEPS, N_AGENTS, 3))

            for pol in POLICIES:
                t0 = time.time()
                m = simulate(pol, pos0, noise, wall_r=ball_r)
                row = {
                    "density_ratio": f"{ratio:.4f}",
                    "rho": f"{rho:.8f}",
                    "domain_side": f"{ball_r:.4f}",
                    "policy": pol,
                    "seed": seed,
                    **{k: (f"{v:.8g}" if isinstance(v, float) else v)
                       for k, v in m.items()},
                    "wall_time_s": f"{time.time() - t0:.3f}",
                }
                writer.writerow(row)
                fh.flush()
                os.fsync(fh.fileno())
                done += 1
            if (s + 1) % 5 == 0 or s == 0:
                log(f"  seed {seed}: done {done}/{total}  "
                    f"elapsed {time.time() - t_start:.1f}s")

    fh.close()
    log("-" * 74)
    log(f"wrote {stream_path}  ({done} rows, {time.time() - t_start:.1f}s)")

    if os.path.abspath(stream_path) != os.path.abspath(metrics_path):
        with open(stream_path, "rb") as src, open(metrics_path, "wb") as dst:
            dst.write(src.read())
        log(f"copied  -> {metrics_path}")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "N_AGENTS": N_AGENTS, "dt": DT, "T_END": T_END, "n_steps": N_STEPS,
        "r_probe": R_PROBE, "r_max": R_MAX, "k_target": K_TARGET, "gamma": GAMMA,
        "alpha_ema": ALPHA_EMA, "delta_count": DELTA_COUNT, "delta_m": DELTA_M,
        "rho_k_3d": RHO_K_3D,
        "a_max": A_MAX, "c_coh": C_COH, "c_vel": C_VEL, "d_eq": D_EQ,
        "d_sep": D_SEP, "k_rep": K_REP, "c_drag": C_DRAG, "sigma_turb": SIGMA_TURB,
        "downwash": {"A": A_DW, "H": H_DW, "R": R_DW, "k_spread": K_SPREAD},
        "d_body": D_BODY, "beta_w": BETA_W,
        "seeds": [BASE_SEED + i for i in range(args.seeds)],
        "density_ratios": ratios,
        "csv": os.path.basename(stream_path),
    }
    with open(os.path.join(outdir, "wsl_3d_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log("\n=== per-(density,policy) means ===")
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(stream_path) as f:
        for r in csv.DictReader(f):
            key = (r["density_ratio"], r["policy"])
            for col in ("frag_final", "frag_mean", "mean_degree", "mean_radius",
                        "min_radius", "ju_applied", "ju_commanded", "ju_jerk",
                        "sat_fraction", "order_tail", "dispersion_tail",
                        "min_pair_dist", "collision_events", "collision_pairs",
                        "finite"):
                try:
                    agg[key][col].append(float(r[col]))
                except (TypeError, ValueError):
                    pass
    for key in sorted(agg, key=lambda k: (float(k[0]), k[1])):
        a = agg[key]
        log(f"  ratio={key[0]:>6} {key[1]:>4} : frag={np.mean(a['frag_final']):.4f} "
            f"deg={np.mean(a['mean_degree']):5.2f} r={np.mean(a['mean_radius']):.2f} "
            f"rmin={np.mean(a['min_radius']):.2f} Ju={np.mean(a['ju_applied']):7.3f} "
            f"jerk={np.mean(a['ju_jerk']):7.4f} "
            f"sat={np.mean(a['sat_fraction']):.3f} ord={np.mean(a['order_tail']):.3f} "
            f"disp={np.mean(a['dispersion_tail']):6.2f} "
            f"mind={np.mean(a['min_pair_dist']):.2f} "
            f"coll={np.sum(a['collision_events']):.0f}/{np.sum(a['collision_pairs']):.0f} "
            f"fin={np.mean(a['finite']):.2f}")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
