# Tungnaá at Maríufossar: initial SUMMA / CARRA workflow

Run from the SYMFLUENCE repository with the existing Python environment:

```bash
.venv/bin/symfluence workflow run --config examples/02_watershed_modelling/config_tungnaa_mariufossar_summa_carra.yaml
```

The configuration writes to `/Users/darri/repos/SYMFLUENCE_data/domain_Tungnaa_Mariufossar`.
The experiment is `carra_smoke_201707`, covering 2017-07-01 03:00 through
2017-07-31 21:00 at a three-hour forcing interval. This is an uncalibrated,
cold-start integration test, not an evaluated runoff forecast.

## Inputs and spatial setup

- LamaH-Ice gauge **86**, station **V261**, named **Maríufoss**, river **Tungnaá**.
- Published total upstream basin (`A_basins_total_upstrm`), **1,141.063 km²**.
- Gauge coordinates: **64.09962986 N, 19.05375185 W**, transformed from the
  published ISN93 point, rather than treating ISN93 coordinates as degrees.
- One GRU and **5 HRUs** from 200 m elevation bands, with dominant MODIS land cover
  assigned within each band.
- National DEM, soil and land-cover rasters are cropped copies of the existing
  Iceland regional domain inputs. Categorical rasters use nearest-neighbour
  alignment to the DEM grid.
- CARRA1 west-domain meteorology from the native SYMFLUENCE CDS acquisition handler.
- Daily discharge from the original LamaH-Ice `ID_86.csv`; original quality flags
  remain in the raw CSV. The handler now preserves native daily means and missing days; the archived
  July smoke run used interpolated observations. Calibration uses daily values.

The basin boundary and attribute rasters are staged inputs. The configured
workflow performs project setup, pour-point creation, discretization,
observation processing, forcing acquisition/reuse, model-agnostic preprocessing,
model-ready-store creation, SUMMA preprocessing, execution and postprocessing.

## Outputs and interpretation

- `settings/SUMMA/`: file manager, decisions, parameters, attributes and initial state.
- `data/forcing/`: CARRA input and remapped forcing.
- `simulations/carra_smoke_201707/SUMMA/`: model NetCDF outputs and run logs.
- `results/carra_smoke_201707_results.csv`: postprocessed daily runoff results.
- `data/model_output/carra_smoke_201707_results.nc`: standardized output.
- `_workLog_Tungnaa_Mariufossar/`: configuration snapshots and workflow provenance.

HRU area is explicitly requested in SUMMA output so native runoff postprocessing
can use area weighting. No external river-routing model is configured. An initial
16-HRU elevation/land-cover trial produced invalid basin runoff because this SUMMA
build skips pure open-water HRUs. Its generated products are preserved under
`trial_archive/`; the active workflow uses elevation bands and does not represent
lakes explicitly.

Three stale units labels retained by the native CARRA acquisition handler were
corrected in the staged raw forcing: specific humidity to kg/kg and the converted
shortwave/longwave radiation rates to W/m². Numerical data were unchanged; the
domain's `forcing_metadata_corrections.json` records the adjustment.

LamaH-Ice reports **10.2% glacier cover**. Snow/ice land-cover HRUs are retained,
but glacier ice storage/dynamics have not been configured or validated. Before
calibration or forecasting, resolve glacier treatment, establish a multi-year
spin-up, assess routing and groundwater behaviour, and evaluate against genuine
daily discharge observations with an explicit quality-flag policy.

Source: LamaH-Ice, DOI `10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91`.

## Verified run

The final workflow completed on 2026-09-23 with zero failed steps. Checks confirm
five matching HRU IDs across attributes, forcing and model output; one forcing
file; 247 timestamps; and finite, nonnegative HRU and basin runoff throughout.
`output_checks.json` and `spatial_checks.json` in the domain record these checks.
The July cold-start runoff has a large positive bias; these checks establish
that the integration runs, not that the model is calibrated or forecast-ready.

## Five-year calibration workflow (launched 2026-09-23)

Configuration: `config_tungnaa_mariufossar_summa_carra_2014_2018.yaml`.
Experiment: `carra_2014_2018_dds`, same domain directory.

- Spin-up: 2014-01-01 through 2014-12-31.
- Calibration: 2015-01-01 through 2016-12-31.
- Held-out evaluation: 2017-01-01 through 2018-12-31.
- CARRA forcing: three-hourly, 2014-01-01 00:00 through 2018-12-31 21:00.
- DDS: 1,000 iterations, daily streamflow KGE, seed 42, perturbation factor 0.2.
- Local parameters: k_soil, aquiferBaseflowRate, albedoMax, snowfrz_scale.
- Basin parameters: routingGammaShape, routingGammaScale.
- Best-parameter final evaluation uses the same homegrown solver as calibration.

The native workflow acquires all 60 months, rebuilds inputs, runs the baseline,
then calibrates and performs final evaluation. Acquisition must finish before
calibration starts; CDS queue latency prevents a firm completion estimate.
The original daily observations are regenerated with the native observation step
because the workflow reused its old observation cache on initial launch.
Original quality flags remain in the source archive; no additional quality-code
filter is imposed. All discharge values in these five years are present and positive.
The two glacier/lake representation limitations described above still apply;
this experiment is a calibrated baseline, not validation of those processes.

Prior generated settings, observations and derived forcing are preserved under
`trial_archive/before_carra_2014_2018_dds/`; the July simulation outputs remain intact.
New CARRA downloads get corrected units directly from the acquisition handler.

Launch metadata: `carra_2014_2018_dds.launch.json` in the domain.
Live log: `_workLog_Tungnaa_Mariufossar/carra_2014_2018_dds.launch.log`.
To resume an interrupted download/workflow, run:

```bash
.venv/bin/symfluence workflow run --config examples/02_watershed_modelling/config_tungnaa_mariufossar_summa_carra_2014_2018.yaml
```

Completed temporary monthly CARRA chunks are reusable during interrupted acquisition.
Do not rerun `output/tungnaa_mariufossar/launch_five_year.py`: it is a one-time
archive-and-launch helper and deliberately refuses to overwrite its archive.
