# Tungnaá at Maríufossar: HYPE glacier workflow

Separate domain: `/Users/darri/repos/SYMFLUENCE_data/domain_Tungnaa_Mariufossar_HYPE`.
Same LamaH-Ice basin and gauge 86 / V261 as the SUMMA experiment. Existing SUMMA
acquisition remains running and its config has not been changed.

## Sequence

1. July 2009 cold-start pipeline test: `config_tungnaa_mariufossar_hype_carra_200907.yaml`.
2. After successful month verification, launch the five-year baseline:
   `config_tungnaa_mariufossar_hype_carra_2009_2013.yaml`.
3. Extend HYPE to 2009–2018 (10 years), then potentially 2004–2018 (15 years),
   after multi-year checks and calibration design. The SUMMA download will provide
   reusable CARRA coverage for 2014–2018. These extensions are not launched.

The five-year config reserves 2009 for spin-up, 2010–2011 for calibration and
2012–2013 for evaluation, but currently runs a baseline simulation, without a
parameter optimizer. Glacier-aware calibration and its bounds require further
setup after the pipeline test; no calibrated model is claimed.

Both configs use the native SYMFLUENCE workflow and native CARRA acquisition.
CARRA acquisition still depends on the CDS queue. The HYPE monthly request adds
one request stream; acquisition concurrency within either workflow is unchanged.
All 2009–2013 daily discharge observations are present and positive.

## Glacier representation

- One lumped basin for the initial pipeline test.
- Glacier fraction 0.102 from LamaH-Ice basin attributes (`g_frac`). This is rounded;
  the separately published `g_area` is 116.077 km². Land-cover proportions outside
  the glacier are rescaled while preserving their relative proportions.
- Correct HYPE special-class code **3**, not internal-lake code 2.
- Glacier type 1 (ice cap) in `GlacierData.txt`; finite initial ice volume comes
  from HYPE's default area-volume relationship, not observed ice thickness.
- Explicit provisional degree-day glacier melt coefficient 5 mm/°C/day and
  threshold 0°C, ice density 0.85. These are starting values, not calibrated values.
- Daily glacier volume (`GLCV`), area (`GLCA`), melt (`GMLT`), runoff (`COUT`),
  evaporation and snow outputs are requested.
- Glacier elevation, changing glacier inventory through time, lake storage, and
  detailed internal routing are not resolved by this lumped baseline.

Sources: LamaH-Ice DOI `10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91`;
[SMHI HYPE file reference](https://hypeweb.smhi.se/wp-content/uploads/sites/11/2023/03/hype_file_reference.pdf);
[SMHI model description](https://hypeweb.smhi.se/wp-content/uploads/sites/11/2023/03/hype_model_description.pdf).
The installed HYPE 5.36.0 source was checked for class codes, glacier parameters,
and output definitions.

## Execution and checks

The first native workflow was launched while final diagnostic-output and inclusive
end-date corrections were being tested. The progression controller waits for it
to finish, then re-enters native downstream preprocessing, execution and
postprocessing with the corrected code (no repeated smoke acquisition).

Before five-year launch it verifies a single glacier SLC, 10.2% glacier cover,
normalized class fractions, exactly 31 daily outputs, finite nonnegative runoff,
and positive glacier area and volume. It records `hype_smoke_checks.json`, archives
month-derived inputs/settings, then runs the five-year SYMFLUENCE config. A failed
check stops progression; it does not start the larger acquisition.

- `hype_glacier_smoke_200907.launch.log`: initial native workflow.
- `hype_smoke_verification_workflow.log`: corrected native downstream workflow.
- `hype_progression.json` and `hype_progression.log`: progression state/errors.
- `hype_glacier_2009_2013.launch.log`: five-year workflow, once checks pass.

Controller: `output/tungnaa_mariufossar_hype/advance_after_smoke.py` in the repo.
Do not run duplicate controllers or repeat the one-time staging helper.
A local synthetic-input executable check confirmed 31 days of output, nonzero
melt and declining finite ice storage; that check is not a watershed validation.

## Portable full-workflow configuration (verified setup 2026-09-28)

Use `config_tungnaa_mariufossar_hype_full_workflow.yaml` for a new installation.
It has no user-specific filesystem paths and enables native attribute acquisition
and native watershed delineation. From the SYMFLUENCE checkout, run:

```bash
symfluence workflow run --config examples/02_watershed_modelling/config_tungnaa_mariufossar_hype_full_workflow.yaml
```

Normal installation prerequisites apply: the Python environment, HYPE and TauDEM
binaries, and CDS credentials/dataset terms. `system.code_dir` and `system.data_dir`
use standard defaults; `SYMFLUENCE_DATA_DIR` can select a local output root.
LamaH-Ice discharge defaults to the managed dataset cache and auto-downloads when
missing. There is no need for the one-time staging helper or an input bundle.

A fresh domain `Tungnaa_Mariufossar_HYPE_full` passed setup_project,
create_pour_point, acquire_attributes, define_domain, discretize_domain and
process_observed_data. Attribute acquisition used native providers; observations
reused the shared LamaH-Ice cache (the missing-cache downloader has unit coverage).
Native Copernicus DEM/TauDEM delineation produced 1145.479 km² versus the
reference geometry's 1141.005 km² in EPSG:6933: +0.392%, with intersection-over-union
0.974618. This is a independently delineated boundary, not a byte-identical copy
of the published basin. The fixed 10.2% glacier fraction is a basin-average
approximation carried from the published catchment.

The source audit matched origin/develop at 98efeb80. The glacier special-class
correction, configurable glacier parameters and inclusive end-date correction
are still uncommitted local changes and must be incorporated into the version
Berglind runs. The path-free config is therefore not yet certified against an
unmodified public develop checkout. Full CARRA-to-output validation is pending
the July 2009 acquisition; no extra forcing request was started for this setup test.
