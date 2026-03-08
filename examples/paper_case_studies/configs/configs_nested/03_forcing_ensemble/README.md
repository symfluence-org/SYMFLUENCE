# Experiment 3: Forcing Ensemble (Section 3.2)

## Scientific Question
How sensitive is model performance to the choice of meteorological forcing data,
and how does forcing uncertainty compare to model structural uncertainty?

## Configs
- `_base_paradise_summa.yaml` — Shared settings (Paradise SNOTEL domain, SUMMA)
- `forcings/config_[forcing].yaml` — Per-forcing overrides (15 files)

## Forcing Products (15)
Reanalysis: ERA5, RDRS, HRRR, AORC, CONUS404
GCM (NEX-GDDP-CMIP6): ACCESS-CM2, CanESM5, CNRM-CM6-1, GFDL-ESM4, INM-CM5-0,
IPSL-CM6A-LR, MPI-ESM1-2-HR, MRI-ESM2-0, NorESM2-LM, UKESM1-0-LL

## Projection Configs (30)
- `projections/config_proj_[source]_params_[gcm].yaml` — 30 projection configs
- 10 GCMs × 3 parameter sources (AORC, ERA5, RDRS calibrated)
- SSP2-4.5 scenario from 2015-2100
- Used for Figure 6: SWE projections under climate change

## Key Configuration Choices
- Paradise Creek point-scale domain (SNOTEL station 679)
- SUMMA model calibrated to SWE (RMSE objective)
- Calibration: Oct 2015 - Sep 2018, Evaluation: Oct 2018 - Sep 2020

## Paper Figures
- Figure 4: Forcing ensemble SWE simulations
- Figure 5: Combined performance and parameter analysis across forcings
- Figure 6: SWE projections under SSP2-4.5 (projection configs)

## Data Requirements
- ERA5, RDRS, HRRR, AORC, CONUS404 gridded products
- 10 NEX-GDDP-CMIP6 GCM outputs (historical + SSP2-4.5)
- SNOTEL station 679 SWE observations
