# Tungnaá at Maríufossar: HBV and SAC-SMA/Snow-17 benchmarks

Two independent daily, lumped models use the existing Belgingur CARRA basin forcing. They run alongside SUMMA and HYPE without modifying their domains.

| Model | Calibrated parameters | Budget |
|---|---|---|
| HBV-96 (`jhbv`) | 14 physical parameters: tt, cfmax, **sfcf**, cfr, cwh, fc, lp, beta, k0, k1, k2, uzl, perc, maxbas | DDS, 2,000 iterations, seed 42, one worker |
| SAC-SMA + Snow-17 (`jsacsma`) | 16 SAC-SMA and 10 Snow-17 parameters, including **SCF**, phase threshold and seasonal melt factors | DDS, 2,000 iterations, seed 42, one worker |

The numerical HBV smoothing parameter is held fixed. Native package bounds are recorded in each domain's `parameter_bounds.json`; snowfall correction bounds are 0.5–1.5 for HBV and 0.7–1.4 for Snow-17. The objective is daily discharge KGE. Internal snow and storage outputs are requested; neither setup includes explicit glacier ice depletion. They are benchmarks, not glacierized model replacements.

Both use 2014 spinup, 2015–2016 calibration and 2017–2018 retrospective evaluation. This matches the SUMMA comparison period; it does not match the 20-year HYPE experiment. The evaluation years have already been inspected in previous work and are not fresh validation for subsequent model selection.

## Inputs and validation

`output/tungnaa_mariufossar_simple/launch_models.py` stages independent inputs under `/Users/darri/repos/SYMFLUENCE_data/domain_Tungnaa_Mariufossar_HBV_belgingur` and `domain_Tungnaa_Mariufossar_SACSMA_belgingur`. It reads the existing expanded SUMMA domain's unmodified Belgingur files, applies the five HRU area weights, and aggregates complete three-hourly intervals to daily mean temperature and precipitation rate. Precipitation is conserved. Existing elevation corrections are already in these inputs and are not reapplied.

The basin area is 1144.9559968 km², consistent with the forcing weights and SUMMA runoff conversion. The native model preprocessors perform unit conversion and calculate Hamon PET from daily basin temperature and area-weighted latitude. Both resulting input series contain 1,826 days with identical P, T and PET. This staged daily input also avoids unweighted HRU averaging and the HBV adapter's hourly-versus-daily source assumptions.

Each domain contains source-file SHA256 hashes and weights in `forcing_provenance.json`, validated daily totals in `forcing_checks.json`, its `config.yaml`, launch record, and workflow status/log. No new CARRA acquisition is needed.

## Run and monitor

The configs are `config_tungnaa_mariufossar_hbv_belgingur_2014_2018.yaml` and `config_tungnaa_mariufossar_sacsma_belgingur_2014_2018.yaml` alongside this README. They specify a native three-stage workflow over prepared input: `model_specific_preprocessing`, `run_model`, `calibrate_model`.

The launch helper invokes these native steps through `symfluence workflow step ...`, with a forcing-unit/date check between preprocessing and execution. Given staged inputs, `symfluence workflow run --config CONFIG` runs the configured stages directly. The YAML alone does not acquire and recreate the upstream SUMMA preprocessing: share the prepared daily inputs and observations as well, or rerun the documented staging helper with the source domain available.

Progress: `workflow_status.json` and `workflow.launch.log` in each domain. Final outputs and calibration results use native `simulations/` and `optimization/` directories. Initial model runs are uncalibrated and should not be reported as calibrated performance.

## Area compatibility fix required by these runs

The first attempt exposed a mismatch between canonical `domain.catchment_area` / `CATCHMENT_AREA` and plugins expecting `catchment_area_km2` / `CATCHMENT_AREA_KM2`. The calibration workers silently used 1,000 km²; initial runners omitted a valid m³/s conversion. Both attempts were stopped and preserved under `invalid_area_attempt/` with an invalid-results marker.

Local SYMFLUENCE changes add a compatibility accessor in `core/config/models/domain.py` and recognize the canonical area key in `core/calibration/workers/inmemory_worker.py`. Five regression tests pass in `tests/unit/optimization/test_inmemory_config_area.py`. These fixes are local and unpublished; include them when sharing the setup. The supervisor now verifies the worker's area before starting and the initial output's area, finite discharge and mm/day-to-m³/s conversion before calibration. This fix does not change model physics or existing SUMMA/HYPE inputs.
