# Paper 3 — Raw Configuration Files (configs_dir/)

This directory contains the original SYMFLUENCE configuration files used to produce
the results in **"From Configuration to Prediction: A Unified Framework for
Hydrological Model Intercomparison and Ensemble Analysis"**.

Each subdirectory corresponds to one experiment in the paper.

## Directory Map

| Dir | Experiment | Paper Section | Configs | Description |
|-----|-----------|---------------|---------|-------------|
| `01_domain_definition/` | Domain definition | Section 2 | 4 | Point, lumped, semi-distributed, regional domains |
| `02_model_ensemble/` | Model ensemble | Section 3.1 | 28 | 27-model intercomparison (Table 1, Figure 2) |
| `03_forcing_ensemble/` | Forcing ensemble | Section 3.2 | 15 | 14 forcing products + ERA5 baseline (Figure 5) |
| `04_calibration_ensemble/` | Calibration ensemble | Section 3.3 | 470 | 17 algorithms x 8 models x 5 seeds (Figure 6) |
| `05_benchmarking/` | Benchmarking | Section 3.4 | 1 | Statistical benchmark comparison (Figure 8) |
| `06_decision_ensemble/` | Decision ensemble | Section 3.5 | 1 | 64 FUSE model structures (Figure 9) |
| `07_sensitivity_analysis/` | Sensitivity analysis | Section 3.6 | 24 | Sobol sensitivity for 22 models (Figure 10) |
| `08_large_sample/` | Large sample | Section 3.7 | 117 | 59 Icelandic catchments, FUSE + GR4J (Figure 11) |
| `09_large_domain/` | Large domain | Section 3.8 | 3 | Iceland distributed modeling (Figure 12) |
| `10_multivariate_evaluation/` | Multivariate eval | Section 3.9 | 8 | GRACE TWS + streamflow calibration (Figure 13) |
| `11_data_pipeline/` | Data pipeline | Section 3.10 | 4 | Automated data acquisition demos |
| `12_parallel_scaling/` | Parallel scaling | Section 3.10 | 93 | TauDEM, calibration, actors scaling (Figure 14) |

**Total: ~768 configuration files**

## Experiment Details

### 01 — Domain Definition (4 configs)
- `config_paradise_summa_optimization.yaml` — Point-scale (Paradise SNOTEL, WA)
- `config_Bow_lumped_era5.yaml` — Lumped catchment (Bow at Banff, AB)
- `config_Bow_lumped_elev_sd_routing_era5.yaml` — Semi-distributed with elevation bands
- `config_iceland_tutorial.yaml` — Regional distributed (Iceland)

### 02 — Model Ensemble (28 configs)
One config per hydrological model for the Bow at Banff domain. Models included:
SUMMA, FUSE, jFUSE, HBV, HYPE, RHESSys, CRHM, PRMS, CLM, VIC, MESH, mHM,
SWAT, GR4J, SAC-SMA, XAJ, XAJ+Snow17, HEC-HMS, TOPMODEL, SUMMA+MODFLOW,
GSFLOW, CLM+ParFlow, ParFlow, WRF-Hydro, WFLOW, WATFLOOD, LSTM, NGEN.

**Excluded:** MIKE SHE (proprietary), duplicate MESH elevation band variants.

### 03 — Forcing Ensemble (15 configs)
Paradise Creek with SUMMA under 15 forcing products:
ERA5, AORC, RDRS, HRRR, CONUS404, and 10 NEX-GDDP-CMIP6 GCMs.

### 04 — Calibration Ensemble (470 configs)
Organized by model subdirectory (`hbv/`, `hechms/`, `topmodel/`, `xinanjiang/`,
`sacsma/`, `fuse/`, `summa/`, `hype/`).

Each model has configs for up to 17 calibration algorithms (DDS, SCE-UA, PSO,
DE, CMA-ES, GA, SA, Nelder-Mead, Basin-Hopping, Bayesian Optimization, GLUE,
L-BFGS, Adam, ABC, NSGA-II, MOEA/D, DREAM) across 5 random seeds.

### 05 — Benchmarking (1 config)
Evaluates ensemble against persistence, climatology, and statistical benchmarks.

### 06 — Decision Ensemble (1 config)
FUSE model with enumerated structural decisions (64 combinations).

### 07 — Sensitivity Analysis (24 configs)
Sobol global sensitivity analysis for 22 models.
**Excluded:** LSTM (neural network), WATFLOOD (failed objective function).

### 08 — Large Sample (117 configs)
- `fuse_v3/` — 51 FUSE decision analysis configs for Icelandic catchments
- `gr4j_v2/` — 66 GR4J+CemaNeige configs for comparison

### 09 — Large Domain (3 configs)
Iceland-wide distributed modeling with FUSE and CARRA forcing.

### 10 — Multivariate Evaluation (8 configs)
- `bow_grace_tws/` — Streamflow + GRACE TWS joint calibration (6 configs)
- `iceland_scf_trend/` — Snow cover fraction trend analysis (1 config)
- `paradise_sca_sm/` — Snow cover area + soil moisture (1 config)

### 11 — Data Pipeline (4 configs)
Automated data acquisition for Paradise, Bow, and Iceland domains.

### 12 — Parallel Scaling (93 configs)
- `exp1_taudem/` — TauDEM watershed delineation scaling (46 configs)
- `exp2_calibration/` — DDS calibration parallelism (36 configs)
- `exp3_distributed/` — SUMMA actors distributed execution (7 configs)
- `base/` — Base configs for scaling experiments (4 configs)

## Notes

- These are the **original** configs with absolute paths from the development machine.
  For portable versions with relative paths, see `../configs_minimal/`.
- All configs follow the SYMFLUENCE YAML schema (6 sections: Global, Geospatial,
  Model Agnostic, Model Specific, Evaluation, Optimization).
- Settings with value `default` use SYMFLUENCE's built-in defaults.
