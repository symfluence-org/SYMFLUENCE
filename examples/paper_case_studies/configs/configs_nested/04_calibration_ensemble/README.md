# Experiment 4: Calibration Ensemble (Section 3.3)

## Scientific Question
How does the choice of calibration algorithm affect model performance, and how
does calibration uncertainty interact with model structural uncertainty?

## Configs
- `_base_bow_calibration.yaml` — Shared settings (Bow at Banff, RDRS forcing)
- `algorithms/[model]/config_[algorithm].yaml` — Per-model/algorithm configs

## Design: 17 Algorithms x 8 Models x 5 Seeds = 680 Experiments

### Models (8)
HBV, HEC-HMS, TOPMODEL, XAJ, SAC-SMA, FUSE, SUMMA, HYPE

### Algorithms (17)
Gradient-free: DDS, SCE-UA, PSO, DE, CMA-ES, GA, SA, Nelder-Mead, Basin-Hopping
Bayesian: Bayesian Optimization, DREAM
Approximate: GLUE, ABC
Gradient-based: L-BFGS, Adam
Multi-objective: NSGA-II, MOEA/D

### Seeds
Each algorithm-model combination is run with seeds: 42, 1042, 2042, 3042, 4042

## Key Configuration Choices
- Bow at Banff lumped domain with RDRS forcing
- KGE objective, 1000 iterations per run
- Warm start disabled for fair algorithm comparison

## Paper Figures
- Figure 6: Calibration algorithm performance by model
- Figure 7: Convergence trajectories

## Full Config Set
The minimal configs include one representative (DDS) per model. The full 470
configs are in `../../configs_dir/04_calibration_ensemble/`.
