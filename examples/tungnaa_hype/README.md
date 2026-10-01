# Berglind's starting point: Tungnaá at Maríufossar, HYPE + Belgingur CARRA

Prepared 1 October 2026. This is a fresh, reproducible experiment specification—not a claim that the basin model is fully validated or that a new calibration will reproduce Darri's fitted score.

## Required code version

Use `develop` including the Belgingur/HYPE handoff changes. A release or checkout
predating this change does not contain the required adapter and glacier fixes.
For a new installation:

```bash
git clone --branch develop https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For an existing clean checkout, run `git switch develop` and `git pull --ff-only`,
then update its Python installation with `python -m pip install -e .`.
Record `git rev-parse HEAD` with the run for reproducibility.

Use an existing working SYMFLUENCE environment if available. Native prerequisites include a Fortran compiler and the platform's normal MPI/GDAL/geospatial dependencies; the repository installation guide covers platform setup. Set output storage explicitly **before installing binaries or running workflows**:

```bash
export SYMFLUENCE_DATA_DIR="$HOME/SYMFLUENCE_data"
symfluence binary install hype taudem
symfluence binary doctor
```

The current HYPE reference binary uses upstream revision `fa9def0d359685c9cde0763f7040e253c345bb65`, with no tracked native source modifications. The required fixes are in SYMFLUENCE's Python adapter; there is no private glacier executable to obtain from Darri. The installer follows its configured upstream version, so record the installed revision/version: a later build is not guaranteed bit-identical to the reference. Do not transfer the macOS binary to a different platform.

## First run: one-month pipeline check

```bash
symfluence workflow run --config examples/tungnaa_hype/smoke.yaml
python examples/tungnaa_hype/check_results.py --config examples/tungnaa_hype/smoke.yaml
```

This runs ordinary project setup, DEM/soil/land-cover acquisition, delineation, elevation discretization, observations, Belgingur acquisition, generic remapping, model-ready store creation, HYPE preprocessing, execution and postprocessing. It uses a separate `Tungnaa_Mariufossar_HYPE_shared_smoke` domain. Expected native output: July 1–31, 2009, 31 daily discharges and finite positive glacier area/volume. This cold-start month tests interfaces, not performance. Cryosphere calibration is disabled because a month has no complete annual balance interval.

Belgingur forcing uses public monthly NetCDF files and **does not require CDS credentials**. The adapter downloads one complete Iceland monthly file at a time, then saves a spatial subset and removes the temporary full file; leave space for that temporary download as well as retained inputs and model outputs. The LamaH-Ice observations are downloaded through SYMFLUENCE. Network access to the archive and ordinary geospatial attribute providers is required. No private local data paths or pre-staged files are referenced by these configs.

## Full period and calibration

After the smoke output check passes:

```bash
symfluence workflow run --config examples/tungnaa_hype/calibration.yaml
python examples/tungnaa_hype/check_results.py --config examples/tungnaa_hype/calibration.yaml
```

The full config starts from acquisition and preprocessing, not `calibrate_model` alone. It uses a separate `Tungnaa_Mariufossar_HYPE_shared` domain and does not overwrite Darri's previous fits. Re-running the same command uses normal workflow state; do not add `--force-rerun` merely to resume after a failure.

| Role | Inclusive dates |
|---|---|
| Complete simulation | 2009-01-01–2023-09-30 |
| Unscored spinup | 2009-01-01–2012-09-30 |
| Calibration, WY2013–16 | 2012-10-01–2016-09-30 |
| Evaluation, WY2017–21 | 2016-10-01–2021-09-30 |
| Dry transfer, WY2022–23 | 2021-10-01–2023-09-30 |

The simulation carries states through the volcanic years; their observations are excluded from scoring. Spinup is 1,369 days. DDS uses 3,000 trials, one process and seed 42. There is **no inherited optimized initial guess**. Objective: 0.6 discharge KGE + 0.2 snow score + 0.2 annual glacier-balance score. Error scales (.2 snow fraction, 1 m water equivalent) are objective normalization choices, not measured observational uncertainties. Annual constraints conservatively require both September endpoints inside calibration, so WY2014–16 contribute; WY2013 does not.

Results live below `$SYMFLUENCE_DATA_DIR/domain_Tungnaa_Mariufossar_HYPE_shared/`. The calibrated native outputs are in `optimization/HYPE/dds_hype_belgingur_wy2013_2023_fresh/final_evaluation/`. `check_results.py` checks native glacier classes/fractions, daily coverage and finite outputs, then prints separate discharge scores for calibration, evaluation and dry transfer. Its checks do not certify full water balance or glacier realism. Use `_workLog_*` and `optimization/HYPE` for workflow/calibration logs and progress.

## Physical assumptions and interpretation

- LamaH-Ice gauge/catchment 86, V261, Tungnaá at Maríufossar.
- Native delineation and 200 m elevation bands. The reference preprocessing produced five bands and about 1,145 km². Generated native geometry determines area; the stale 1,141 km² configuration override was removed. This is not a byte-identical supplied watershed boundary.
- Initial glacier fraction 0.11304, ice-cap type 1, allocated to upper elevation bands. Initial volume follows HYPE's area–volume relation, preserved across bands; it is not an observed thickness reconstruction.
- Native glacier special class is **3**, with explicit degree-day ice-melt parameters. Fractional snow cover and snow/glacier diagnostic outputs are enabled.
- This retains the checked degree-day baseline. Optional radiation-melt/routing/percolation experiments are not enabled.
- `ttpi` controls the rain/snow transition width; it is **not** a snowfall multiplier.
- Three-hour Belgingur forecast timestamps remain as published; precipitation/radiation rates are not deaccumulated. Forecast leads and interval alignment still need provider clarification, so no precise subdaily timing claim is made. HYPE runs daily.
- The reference run had evaluation KGE about .719 and dry KGE .602, but used inherited fitted seeds and previously inspected data. Those numbers are context, not acceptance thresholds for this new fresh run. Spring/autumn snow errors, very low simulated ET, schematic routing, and annual glacier-observation support remain under investigation.

## Validation status of this handoff

The changes are tested in an isolated checkout based on develop revision `98efeb8036b5f89950f3b8584ed8a7e1b70a0514`, with regression tests for Belgingur conversion/grid handling, daily LamaH observations, HYPE glaciers, fractional snow, and cryosphere scoring. The result checker is exercised on existing native output. These checks are distinct from running all downloads and the full 15-year workflow on Berglind's machine. Start with the smoke config; the complete fresh acquisition-to-calibration handoff has not been rerun here.

The shared configs intentionally retain restartable workflow behavior and portable default paths. Changing only `system.data_dir` (or the environment variable above) selects a different local storage root. Keep the domain/experiment names separate from any existing trials.

Do not use `workflow run --dry-run` as a no-side-effect preview with this base revision: the temporary handoff check reached actual attribute acquisition and was stopped. Configuration validation and the regression suite passed; that attempted preview is not an end-to-end validation.
