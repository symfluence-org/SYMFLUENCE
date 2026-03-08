# Experiment 6: Model Decision Ensemble (Section 3.5)

## Scientific Question
How does the choice of model process representations (decisions) within a
flexible modeling framework affect performance?

## Config
- `config_fuse_decisions.yaml` — FUSE with enumerated structural decisions

## Key Configuration Choices
- FUSE model with 64 structural combinations from decision tree
- Decisions enumerated: rainfall error (2) x surface runoff (2) = 4 combinations
  with fixed architecture, percolation, evaporation, interflow, routing, snow
- `ANALYSES: [decision_analysis]` triggers automatic enumeration
- Each structure independently calibrated with DDS (1000 iterations)

## Paper Figures
- Figure 9: Decision ensemble performance spread
