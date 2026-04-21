# Experiment 4: Calibration Ensemble (Section 3.3)

## Scientific Question
How does the choice of calibration algorithm affect model performance, and how
does calibration uncertainty interact with model structural uncertainty?

## Configs
- `_base_bow_calibration.yaml` — Shared settings (Bow at Banff, RDRS forcing)
- `algorithms/[model]/config_[algorithm].yaml` — Per-model/algorithm nested configs
- `[model]/config_bow_[model]_[algorithm].yaml` — Fully expanded configs (flat)
- `[model]/config_bow_[model]_[algorithm]_seed[N].yaml` — Seed variants

## Experimental Design

### Full Design: 17 Algorithms × 8 Models × 5 Seeds = 680

### What actually ran: 470 configs

Not all combinations are valid. Two constraints reduce the full factorial:

1. **Gradient-based algorithms (L-BFGS, Adam) require a differentiable model.**
   FUSE, SUMMA, and HYPE are Fortran-based and expose no gradient; these two
   algorithms are excluded for those models (17 → 15 algorithms).

2. **Fortran models run a single seed only.**
   FUSE, SUMMA, and HYPE use internal (SCE-based) calibration or are too
   expensive per iteration to justify the 5-seed sweep. They run seed 42 only.

| Model group | Models | Algorithms | Seeds | Configs |
|---|---|---|---|---|
| JAX-based | HBV, HEC-HMS, SAC-SMA, TOPMODEL, XAJ | 17 (all) | 5 (42, 1042, 2042, 3042, 4042) | 425 |
| Fortran-based | FUSE, SUMMA, HYPE | 15 (no L-BFGS, Adam) | 1 (42) | 45 |
| **Total** | **8** | | | **470** |

### Models (8)
HBV, HEC-HMS, TOPMODEL, XAJ, SAC-SMA, FUSE, SUMMA, HYPE

### Algorithms (17)
| Category | Algorithms |
|---|---|
| Gradient-free | DDS, SCE-UA, PSO, DE, CMA-ES, GA, SA, Nelder-Mead, Basin-Hopping |
| Bayesian | Bayesian Optimization, DREAM |
| Approximate | GLUE, ABC |
| Gradient-based (JAX only) | L-BFGS, Adam |
| Multi-objective | NSGA-II, MOEA/D |

## Seed Handling

Each stochastic algorithm is sensitive to its random starting point. To
quantify this sensitivity the experiment repeats each algorithm–model pair
with five deterministic seeds: **42, 1042, 2042, 3042, 4042**.

- Seeds are set via `system.random_seed` in each config file.
- The seed controls the optimizer's initial population / starting point and
  any internal random draws (e.g., DDS perturbation, PSO velocities).
- Results across seeds are reported as **median ± IQR** (interquartile range)
  to characterise algorithm robustness.
- The **best seed** (highest evaluation-period KGE) is used when comparing
  algorithms head-to-head in the paper figures.

Note: the decision ensemble experiment (06) now pins `random_seed: 42`
(PR [#28](https://github.com/symfluence-org/SYMFLUENCE/pull/28)) so that
FUSE decision-tree results are exactly reproducible.

## Baseline / Reference Performance

To interpret whether algorithm differences are practically meaningful, use
these reference points for the Bow at Banff domain:

| Reference | KGE | Notes |
|---|---|---|
| Uncalibrated (default params) | ~0.2–0.4 | Model-dependent; HBV ≈ 0.35 |
| Single-algorithm best (DDS, seed 42) | ~0.75–0.85 | Typical strong result |
| Ensemble best (any algorithm, any seed) | ~0.85–0.90 | Upper bound for this domain |
| Meaningful improvement threshold | >0.02 KGE | Below this, differences are within seed variability |

Within-algorithm seed spread (IQR) is typically 0.01–0.03 KGE for robust
algorithms (DDS, SCE-UA, PSO) and 0.03–0.08 for sensitive ones (Nelder-Mead,
Basin-Hopping). An algorithm that consistently beats another by less than the
seed IQR is not meaningfully better.

## Key Configuration Choices
- Bow at Banff lumped domain with RDRS forcing
- KGE objective, 1000 iterations per run
- Warm start disabled (`skip_warm_start: true`) for fair algorithm comparison
- Calibration period: 2004–2007; Evaluation period: 2008–2009; Spin-up: 2002–2003

## Reproducibility

All 470 experiments run end-to-end from configuration files alone:

```bash
symfluence workflow run --config hbv/config_bow_hbv_dds.yaml
```

Data acquisition (DEM, forcing, streamflow) is handled automatically:
- RDRS forcing is downloaded via cloud access
- WSC streamflow for station 05BB001 is fetched from the GeoMet API
  (PR [#36](https://github.com/symfluence-org/SYMFLUENCE/pull/36) —
  `streamflow_data_provider: WSC` now auto-enables download)
- DEM and geospatial attributes acquired from cloud sources
  (PR [#35](https://github.com/symfluence-org/SYMFLUENCE/pull/35) —
  canonical `data/` layout eliminates path mismatches)

No manual data placement or preprocessing is required.

## Paper Figures
- Figure 6: Calibration algorithm performance by model
- Figure 7: Convergence trajectories
