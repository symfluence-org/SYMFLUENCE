# GSFLOW 2.4.0 glacier conservation patch

A **local SYMFLUENCE patch, not an upstream USGS release**, applied by

```sh
symfluence binary install gsflow --patched
```

The patched build checks out upstream [gsflow_v2](https://github.com/rniswon/gsflow_v2)
commit `cd71e820` (GSFLOW 2.4.0, 24 Jul 2026), verifies the patch with `git apply --check`,
applies it, builds as usual and copies the patch to `bin/applied_glacier_conservation.patch`.
Plain `symfluence binary install gsflow` is unchanged (configured branch, no patch).

It ports the PRMS 5.2.1.1 glacier conservation fix
(`symfluence/models/prms/patches/`) to the PRMS code embedded in GSFLOW 2.4.0, which
still contains the same defects, and fixes two GSFLOW-specific accounting errors:

| File | Change |
|---|---|
| `prms/srunoff.f90` | Basal ice melt no longer acts as the snowmelt flag (that skipped rain on the snow-free part of glacier HRUs); it infiltrates below the ice directly. |
| `prms/srunoff.f90` | Surface runoff from the ice-free part of active glacier HRUs is added to runoff (previously computed, then dropped). |
| `gsflow/gsflow_module.f90`, `prms/sm_read_control_file.f90`, `prms/glacr_melt.f90` | New control `glacier_dynamics` (default 1 = upstream dynamic geometry). 0 skips the annual geometry/ELA update: fixed glacier geometry. |
| `prms/snowcomp.f90` | With `glacier_dynamics: 0`, the 1 October snow on ice becomes firn, as in PRMS. Upstream GSFLOW books it as snow evaporation, which also subtracts it from glacier storage (removed twice). |
| `prms/srunoff.f90` | Coupled GSFLOW only: the glacier terminus outflow `glacr_flow` (melt and rain released by the firn/snow/ice reservoirs) infiltrates at the terminus HRU, with any excess becoming surface runoff. Upstream adds it only to `hru_outflow`, which feeds PRMS channel routing; SFR never sees it, so the water was lost. PRMS-only runs still route it and are unchanged. The build adds the `srunoff.o` → `prms_glacr.mod` Makefile dependency this needs. |

`glacier_dynamics: 0` is a **fixed-geometry baseline**, not a validated retreat or
finite-ice model. Use separate pure land and glacier HRUs (`glacier_frac_init` 0/1).

## Verification (Tungnaá at Maríufossar, 13 Oct 2026)

GSFLOW in PRMS-only mode on the PRMS glacier V2 inputs (6 HRUs, 3 pure glacier HRUs,
climate_hru CARRA forcing, 2009-01-01 to 2023-09-30), scored with
`symfluence.models.prms.balance.check_water_balance` at full output precision
(`statvarOut_format: 5`; the default prints 4 significant digits, ±0.25 mm noise):

| Build | Max daily residual | Cumulative residual | Discharge vs patched PRMS 5.2.1.1 |
|---|---|---|---|
| Upstream `cd71e820` | 68.3 mm | +284.6 mm | r 0.99983, mean ratio 1.0018 |
| Patched, `glacier_dynamics: 0` | 0.0006 mm | 0.018 mm | r 1.000000, mean ratio 1.00000 |
| Patched, `glacier_dynamics: 1` | 68.2 mm | +264.8 mm | upstream annual geometry behaviour retained |

A non-glacier coupled GSFLOW run (the audited Tungnaá member, MODFLOW-NWT, 5,386 days)
is bit-identical with the unpatched build.

## Coupled glacier outflow (`glacr_flow`, 6 Nov 2026)

Found on the Sogið GSFLOW setup and rechecked on the promoted Tungnaá V3b member
(coupled GSFLOW, 58 HRUs, 2009-01-01 to 2023-09-30). Same inputs, old patched build
versus this one; the old build reproduces V3b's `statvar.dat` byte for byte:

| Build | Mean StreamOut WY2013–22 | Summary budget error | Error with the glacier storage loss (4.09e9 m³) counted |
|---|---|---|---|
| Before (`glacr_flow` dropped) | 77.5 m³/s | +12.4% | +22% (8.2e9 m³ lost) |
| With the fix | 94.7 m³/s | −7.0% | +2.1% |

GSFLOW's summary budget has no glacier storage term, so on its own the reported error
is not a closure check. Glacier mass balance and glacier storage are unchanged. The streamflow
increase (+17 m³/s, about 19% of precipitation) is water the old build lost, which the
calibration had compensated for with rain and snow multipliers.
