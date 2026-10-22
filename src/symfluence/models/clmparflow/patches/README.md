# ParFlow-CLM v3.15.0 patches

**Local SYMFLUENCE patches, not upstream ParFlow releases**, applied by

```sh
symfluence binary install clmparflow --patched
```

The patched build checks out ParFlow `v3.15.0`, verifies each patch with `git apply --check`, applies it,
builds hypre `v2.31.0` (needed for the PFMG preconditioner) and ParFlow with `-DPARFLOW_HAVE_CLM=ON` (`mpi1`
AMPS layer when `mpicc`/`mpif90` are available, `seq` otherwise), and copies the patches to
`bin/applied_*.patch`. Plain `symfluence binary install clmparflow` is unchanged. Basin mode
(`CLMPARFLOW_BASIN_MODE`) needs the patched build.

| Patch | Files | Change |
|---|---|---|
| `parflow3150_clm_glacier_water.patch` | new `clm/clm_hydro_ice.F90`, `clm/clm_main.F90`, `clm/clm.F90`, `clm/CMakeLists.txt` | Land-ice tiles pass water to ParFlow. Upstream, `clm_hydro_soil` runs on soil tiles only, so `qflx_infl` (the only flux handed to ParFlow) is never set on land ice: rain, snowpack outflow and ice melt on glaciers vanish, and `pfreadout` overwrites `h2osoi_liq` from ParFlow saturation every step, so ice melted by `clm_meltfreeze` vanishes too. The new routine sets `qflx_infl = qflx_top_soil - qflx_evap_grnd + ice melt/dt` on land-ice tiles, books any sublimation beyond supply as an ice deficit and restores the ice column (unlimited ice store, fixed glacier area). Each rank logs istep, melt, deficit and infiltration per tile and step to `glacier_ice.txt.<rank>`. |
| `parflow3150_clm_restart_read.patch` | `clm/drv_restart.F90` | The restart reader skips `snowage_vis`/`snowage_nir`, which the writer stores, so every read after them is misaligned ("I/O past end of record" on `clm_ic = 1`). |
| `parflow3150_clm_ice_albedo.patch` | `clm/clm_soilalb.F90` | Land-ice albedo (CLM 3.0: 0.80 visible / 0.55 near-infrared, compiled in) can be set per run in `clm_albice.txt` (two numbers). Without the file the compiled values are used. |

## Verification (Tungnaá at Maríufossar, 17–22 Oct 2026)

- Two-year 1 km hillslope transect (124 cells, 30 glacier): water-balance residual 24.3 % of precipitation
  upstream (equal to the glacier water) and 0.13 % patched.
- Full basin (1,140 cells, 139 glacier, 2 ranks), hand-tuned on WY2013–16: KGE 0.867 / 0.880 / 0.680
  (calibration / evaluation WY2017–21 / dry WY2022–23), glacier net balance r 0.84 and RMSE 0.49 m w.e. against
  LamaH-Ice. The basin-mode code here reproduces that run's inputs byte for byte and a two-day segment's
  pressure, saturation, outflow and glacier logs bit for bit.

The ice store is unlimited and the glacier area fixed: a fixed-geometry baseline, not a retreat model.
