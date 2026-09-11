# Anonymous artifact: Finite-Count Feedback in Adaptive Flocking

This is the artifact payload for a double-blind submission. Nothing in this
repository identifies the authors, their institution, or their public code host.

The study factorises a bounded metric interaction radius into a **measurement
aperture** and a **multiplicative command base**, and reports a regime-partitioned
operating result with an effective switching density
`rho* = 2.6 * rho_k`, `rho_k = k / (pi * r_max^2)`.

## Environment

Python 3.10 or newer.

```
pip install -r requirements.txt
```

`matplotlib` is needed only by `replot_density_paper_figure.py`. `numpy` and
`scipy` are needed only by the 3D quadrotor validation kernel; every 2D result in
the paper reproduces from the standard library alone. No simulator, ROS, GPU or
external service is required. The flocking dynamics in `eal_mfg_icra/core.py` are
the same shared analytical local velocity-alignment model used for every 2D
reported number.

## Layout

```
eal_mfg_icra/
  core.py                        simulator + DensityAdaptiveNeighborhood
  metrics.py                     fragmentation / SCC / lambda_2 / contact-break
  scripts/
    aperture_base_factorial.py   Table I, the 2x2 aperture-base factorial
    aperture_base_decomposition.py   A/B/I decomposition and per-cell separability
    connectivity_margin_audit.py     Fiedler margin, min-degree, disconnection
    parameter_sensitivity.py         dead-band and bias-robustness sweep
    published_range_headtohead.py    head-to-head against published range laws
    replot_density_paper_figure.py   Figure 1
    run_wsl_3d_quadrotor_validation.py   3D quadrotor kernel (drag / turbulence / downwash)
    wsl_3d_paired_stats.py               paired bootstrap on the 3D metrics
experiments/                     frozen per-episode artifacts (inputs)
paper/evidence/                  frozen tables and bootstrap outputs
paper/output/                    frozen 3D validation metrics and paired statistics
```

## Reproducing the reported numbers

Four scripts recompute their outputs **from the frozen artifacts in this
repository**, without touching the simulator:

```
python -m eal_mfg_icra.scripts.aperture_base_factorial --out experiments/aperture_base_factorial_20260909 --statistics-only
python -m eal_mfg_icra.scripts.aperture_base_decomposition --run experiments/aperture_base_factorial_20260909 --tag sparse --out paper/evidence
python -m eal_mfg_icra.scripts.aperture_base_decomposition --run experiments/aperture_base_factorial_unsaturated_20260910 --tag unsaturated --out paper/evidence
python -m eal_mfg_icra.scripts.connectivity_margin_audit --raw experiments/adaptive_feedback_matched/crdf_matched_raw.csv --out paper/evidence/connectivity_margin_paired.csv
python -m eal_mfg_icra.scripts.replot_density_paper_figure
```

Two further scripts re-run the deterministic simulator rather than reading a
frozen episode table, because their designs sweep parameters that were not
persisted as separate per-episode files. They take no frozen-artifact flag:

```
python -m eal_mfg_icra.scripts.parameter_sensitivity --out experiments/parameter_sensitivity
python -m eal_mfg_icra.scripts.published_range_headtohead --out experiments/published_range_headtohead
```

The 3D validation is shipped as a kernel plus its frozen output. The paired
statistics behind the reported `J_u` and Fiedler numbers regenerate with no
simulation, and every value is identical to the shipped JSON (the file's byte
layout may differ; the bootstrap RNG is fixed, so the numbers do not):

```
python eal_mfg_icra/scripts/wsl_3d_paired_stats.py --csv paper/output/wsl_3d_metrics.csv --json-out paper/output/wsl_3d_paired_stats.json
```

Re-running the 3D kernel itself costs CPU time (~3 minutes for the full 160
episodes). Note that the kernel's `--ratios` default is not the configuration
used in the paper; pass it explicitly, exactly as recorded in
`paper/output/wsl_3d_run_meta.json`:

```
python eal_mfg_icra/scripts/run_wsl_3d_quadrotor_validation.py --ratios 1.0,2.0,4.0,8.0 --seeds 20 --outdir paper/output
```

The frozen-metrics scripts are reproducible at zero cost; the simulator re-runs
are reproducible because the seed banks are fixed and disjoint between
development and evaluation. This distinction is stated here rather than glossed
over.

## Content addressing

Every headline artifact is content-addressed. The 12-hex SHA-256 prefixes quoted
in the manuscript resolve to the following files in this payload:

| Prefix | File | Role |
|---|---|---|
| `3d222a059ac8` | `experiments/aperture_base_factorial_20260909/factorial_statistics.csv` | factorial, Table I |
| `69b57748b19d` | `paper/evidence/aperture_base_decomposition_sparse.csv` | sparse decomposition |
| `0b5a3728ba1f` | `paper/evidence/aperture_base_decomposition_unsaturated.csv` | unsaturated decomposition |
| `3746cc41fec3` | `paper/evidence/crdf_matched_bootstrap_summary.csv` | density-sweep bootstrap |
| `d92c65a97672` | `paper/evidence/finite_sensing_robustness_cell_summary.csv` | sensing robustness |

**Filename collision, read this before hashing.** The density-sweep bootstrap
exists under the same filename in two places with *different* content:
`experiments/adaptive_feedback_matched/crdf_matched_bootstrap_summary.csv` is the
raw per-run output, while `paper/evidence/crdf_matched_bootstrap_summary.csv` is
the frozen table the manuscript cites. The cited prefix `3746cc41fec3` belongs to
the `paper/evidence/` copy; hashing the other path will not match.

Full 64-hex digests are recorded in the `manifest.json` files next to each
artifact, together with the digest of every source file that produced it.

## What is deliberately not here

* Raw per-agent-step traces (`cells/agent_steps.csv.gz`, several hundred MB) are
  omitted. The decomposition reads `episode_summary.csv` and
  `factorial_statistics.csv` only, so the traces add bulk without adding
  reproducibility.
* Study lines that the final 8-page paper does not cite are omitted
  (corridor/Gazebo transfer, formal out-of-distribution, continuous density
  intervention, real-trajectory replay). None of them is reported in the paper.
* Because the simulator core is shipped, those study lines can be regenerated
  from scratch if a reviewer requires them.
* The package `__init__` in this payload is a stub: the upstream development tree
  eagerly imports reinforcement-learning and mean-field research modules that
  this paper does not use, and they are intentionally not distributed here.

## Citation

Citations are omitted in this anonymous version. Please refer to the submission
itself.
