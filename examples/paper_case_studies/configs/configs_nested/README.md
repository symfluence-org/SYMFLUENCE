# Paper 3 — Minimal Portable Configurations (configs_minimal/)

Portable, documented SYMFLUENCE configuration files for reproducing the experiments
in **"From Configuration to Prediction"**. These configs use relative paths and
only include non-default settings.

## Prerequisites

```bash
pip install symfluence
```

SYMFLUENCE requires Python 3.9+ and installs all model backends automatically.
Some models (SUMMA, mizuRoute, WRF-Hydro) require pre-compiled executables — see
the [SYMFLUENCE documentation](https://symfluence.readthedocs.io) for installation guides.

## Data Requirements

Each domain requires specific input data. SYMFLUENCE can automatically download
most datasets when `DATA_ACCESS: cloud` is set.

| Domain | Location | Data Sources | Approx. Size |
|--------|----------|-------------|-------------|
| Paradise Creek | WA, USA (46.78°N, 121.75°W) | ERA5, SNOTEL, AORC, NEX-GDDP | ~2 GB |
| Bow at Banff | AB, Canada (51.17°N, 115.57°W) | ERA5, RDRS, WSC streamflow, GRACE | ~5 GB |
| Iceland (LamaH-Ice) | Iceland-wide | CARRA, IMO streamflow | ~20 GB |

### Manual data setup

If not using cloud access, create a `data/` directory alongside each config:

```
data/
├── domain_[DOMAIN_NAME]/
│   ├── shapefiles/           # Catchment, river, pour point shapefiles
│   ├── forcing/              # Meteorological forcing data
│   ├── observations/         # Streamflow, SWE, GRACE observations
│   └── parameters/           # Soil, land cover, DEM data
```

## How to Run

Each experiment can be run with:

```bash
symfluence run <config.yaml>
```

For configs using base config inheritance (prefixed with `_base_`), specify both:

```bash
symfluence run --base _base_bow_lumped.yaml models/config_summa.yaml
```

### Quick start — run a single model

```bash
cd configs_minimal/02_model_ensemble/
symfluence run --base _base_bow_lumped.yaml models/config_summa.yaml
```

### Run an entire ensemble

```bash
cd configs_minimal/02_model_ensemble/
for config in models/*.yaml; do
    symfluence run --base _base_bow_lumped.yaml "$config"
done
```

### Generate large-sample configs

```bash
cd configs_minimal/08_large_sample/
python generate_configs.py --template _template_fuse.yaml --output ./generated/fuse/
python generate_configs.py --template _template_gr4j.yaml --output ./generated/gr4j/
```

## Directory Structure

```
configs_minimal/
├── 01_domain_definition/          Section 2: 4 domain scale configs
├── 02_model_ensemble/             Section 3.1: 27-model ensemble
│   ├── _base_bow_lumped.yaml      Shared base config
│   └── models/                    Per-model overrides (27 files)
├── 03_forcing_ensemble/           Section 3.2: 14 forcing products
│   ├── _base_paradise_summa.yaml  Shared base config
│   └── forcings/                  Per-forcing overrides (15 files)
├── 04_calibration_ensemble/       Section 3.3: 17 algorithms × 8 models
│   ├── _base_bow_calibration.yaml Shared base config
│   └── algorithms/                Per-model/algorithm configs
├── 05_benchmarking/               Section 3.4: benchmark comparison
├── 06_decision_ensemble/          Section 3.5: 64 FUSE structures
├── 07_sensitivity_analysis/       Section 3.6: Sobol sensitivity
│   ├── _base_bow_sensitivity.yaml Shared base config
│   └── models/                    Per-model overrides (22 files)
├── 08_large_sample/               Section 3.7: 59 Icelandic catchments
│   ├── _template_fuse.yaml        FUSE template (DOMAIN_ID placeholder)
│   ├── _template_gr4j.yaml        GR4J template
│   └── generate_configs.py        Config generation script
├── 09_large_domain/               Section 3.8: Iceland distributed
├── 10_multivariate_evaluation/    Section 3.9: GRACE TWS calibration
├── 11_data_pipeline/              Section 3.10: Data processing
└── 12_parallel_scaling/           Section 3.10: Parallel execution
```

## Config Inheritance

Experiments 2, 3, 4, and 7 use a **base config + override** pattern:

- `_base_*.yaml` contains shared settings (domain, time period, forcing, etc.)
- Model/forcing-specific configs override only the settings that differ
- This reduces duplication and makes the experimental design transparent

## Expected Outputs

| Experiment | Output | Runtime (approx.) |
|-----------|--------|-------------------|
| 01 Domain | Delineated shapefiles, forcing, model setup | 10-30 min |
| 02 Models | 27 calibrated model runs + evaluation metrics | 2-8 hrs per model |
| 03 Forcing | 15 calibrated SUMMA runs under different forcings | 1-2 hrs per forcing |
| 04 Calibration | 680 calibration trajectories | 30 min - 2 hrs each |
| 05 Benchmark | Benchmark statistics table | 5 min |
| 06 Decisions | 64 FUSE structure calibrations | 4-8 hrs |
| 07 Sensitivity | 22 Sobol sensitivity index sets | 1-4 hrs per model |
| 08 Large sample | 118 catchment calibrations (FUSE + GR4J) | 30 min each |
| 09 Large domain | Distributed Iceland simulation | 2-6 hrs |
| 10 Multivariate | 3 calibration experiments + evaluation | 2-4 hrs each |
| 11 Pipeline | Data downloads + preprocessing | 30 min - 2 hrs |
| 12 Scaling | Timing benchmarks at various core counts | Varies |

## Differences from Raw Configs

Compared to `../configs_dir/` (raw configs), these minimal configs:

1. **Replace absolute paths** with `./data/` and `./`
2. **Remove redundant defaults** — only non-default settings included
3. **Add documentation** — inline comments explain each setting
4. **Use config inheritance** — base configs reduce duplication
5. **Include generation scripts** — for large-sample experiment automation

## Paper Reference

Eythorsson, D., Clark, M.P., et al. (2025). From Configuration to Prediction:
A Unified Framework for Hydrological Model Intercomparison and Ensemble Analysis.
