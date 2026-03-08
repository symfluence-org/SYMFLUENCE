# Experiment 10: Multivariate Evaluation (Section 3.9)

## Scientific Question
Does calibrating to multiple observation types (streamflow + GRACE TWS)
improve overall model realism compared to single-variable calibration?

## Configs
- `config_streamflow_only.yaml` — Exp 10a: streamflow KGE only (DDS, 1000 iterations)
- `config_tws_only.yaml` — Exp 10b: GRACE TWS correlation only (DDS, 1000 iterations)
- `config_joint.yaml` — Exp 10c: joint streamflow + TWS (NSGA-II, 2000 evaluations)
- `config_moead_joint.yaml` — Exp 10d: joint streamflow + TWS (MOEA/D, 2000 evaluations)

## Key Configuration Choices
- SUMMA model at Bow at Banff with extended period (2002-2017)
- 16-parameter calibration (snow, soil, groundwater) for fair comparison
- Single-objective: DDS with 1,000 iterations
- Multi-objective: NSGA-II and MOEA/D with 2,000 evaluations each
- `MULTIVAR_EVALUATION: true` for post-hoc multi-variable assessment
- GRACE JPL RL06 mascon product for total water storage

## Paper Figures
- Figure 26: Study domain and observations for multivariate calibration
- Figure 27: Single-objective and multi-objective calibration results
- Figure 28: Independent validation across observation products

## Data Requirements
- RDRS forcing data for Bow at Banff
- WSC streamflow (station 05BB001)
- GRACE JPL RL06 mascon TWS data
- CanSWE in-situ SWE observations
