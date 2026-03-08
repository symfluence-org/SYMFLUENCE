# Experiment 7: Sensitivity Analysis (Section 3.6)

## Scientific Question
Which parameters control model behavior most strongly, and how does parameter
sensitivity vary across models?

## Configs
- `_base_bow_sensitivity.yaml` — Shared settings (Bow at Banff, Sobol analysis)
- `models/config_[model]_sensitivity.yaml` — Per-model overrides (22 files)

## Models (22)
All 27 ensemble models except:
- LSTM (neural network — not applicable to Sobol analysis)
- WATFLOOD (failed objective function computation)
- MIKE SHE (proprietary, excluded from ensemble)

## Key Configuration Choices
- Sobol global sensitivity analysis via Latin Hypercube Sampling
- 100 LHS samples per model, seed 22 for reproducibility
- KGE objective function
- Same domain/period as model ensemble (Experiment 2)

## Paper Figures
- Figure 10: Sobol sensitivity indices across models

## Usage
```bash
symfluence run --base _base_bow_sensitivity.yaml models/config_summa_sensitivity.yaml
```
