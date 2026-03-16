# Experiment 2: Model Ensemble (Section 3.1)

## Scientific Question
How do 27 different hydrological models compare when applied to the same domain
with identical forcing, calibration, and evaluation settings?

## Configs
- `_base_bow_lumped.yaml` — Shared settings (domain, period, forcing, calibration)
- `models/config_[model].yaml` — Per-model overrides (27 files)

## Models (27)
Process-based lumped: HBV, HYPE, GR4J, SAC-SMA, XAJ, XAJ+Snow17, HEC-HMS, TOPMODEL
Flexible framework: SUMMA, FUSE, jFUSE
Distributed/gridded: CLM, VIC, MESH, mHM, SWAT, PRMS, RHESSys, CRHM
Coupled: SUMMA+MODFLOW, GSFLOW, CLM+ParFlow, ParFlow+Snow17, WRF-Hydro
Community: WFLOW, WATFLOOD, NGEN
Machine learning: LSTM

## Key Configuration Choices
- All models use ERA5 forcing at Bow at Banff (2002-2009)
- DDS calibration with 1000 iterations, KGE objective
- Calibration: 2004-2007, Evaluation: 2008-2009, Spinup: 2002-2003

## Paper Figures/Tables
- Table 1: Model characteristics and parameter counts
- Figure 2: Model ensemble performance comparison
- Figure 3: Hydrograph comparison

## Usage
```bash
symfluence run --base _base_bow_lumped.yaml models/config_summa.yaml
```
