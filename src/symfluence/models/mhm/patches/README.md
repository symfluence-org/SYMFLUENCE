# mHM non-hourly time step patch

A **local SYMFLUENCE patch, not an upstream UFZ release**, applied by

```sh
symfluence binary install mhm --patched
```

The patched build checks out upstream [mHM](https://git.ufz.de/mhm/mhm) commit
`fd474cc3` (after 5.13.4, Jul 2026), verifies the patch with `git apply --check`,
applies it, builds as usual and copies the patch to
`bin/applied_timestep_meteo_index.patch`. The mHM preprocessor refuses
`MHM_TIMESTEP` other than 1 unless it finds that file next to the executable.
Plain `symfluence binary install mhm` is unchanged (configured branch, no patch).

## The defect

`meteo_handler%update_timestep` (`src/meteo/mo_meteo_handler.f90`) picks the forcing
record for model step `tt` as

```fortran
iMeteoTS = ceiling(tt / nint(24 / nTstepForcingDay))
```

i.e. it divides by the number of *hours* per forcing record, which is only the
number of *model steps* per forcing record when the model step is 1 h. With daily
forcing and `timestep = 24` each forcing day is used for 24 model days; with
`timestep = 6`, for 6 model steps of 6 h. The run completes without error but
consumes only the first 1/24 (or 1/6) of the forcing. The patch divides by
`nTStepDay / nTstepForcingDay` (model steps per forcing record) in both the
single-read and the chunked-read branch. The result is unchanged for `timestep = 1`.
The bug is still present on upstream `main` as of 14 Oct 2026.

## Verification (Tungnaá at Maríufossar, 14 Oct 2026)

Distributed 0.03° Tungnaá setup (235 cells, daily CARRA forcing 2009-01-01 to
2023-09-30, uncalibrated default parameters), daily discharge 2012-10-01 to 2023-09-30:

| Build, step | Correlation with patched 1 h | Mean flow ratio | Max basin snowpack |
|---|---|---|---|
| Upstream, 1 h | bit-identical | 1.000 | 1,340 mm |
| Upstream, 24 h | −0.09 | 0.98 | ≈ 12,700 mm (winter forcing repeated for years) |
| Patched, 6 h | 0.935 | 1.007 | 1,321 mm |
| Patched, 24 h | 0.739 | 1.007 | 1,596 mm |

The remaining 24 h difference is model physics (no diurnal temperature cycle for
snowmelt, daily routing step), not forcing indexing; recalibrate at the step you use.
