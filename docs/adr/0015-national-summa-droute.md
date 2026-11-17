# 0015 National SUMMA + dRoute (Iceland, landslíkan, paradigm 1)

**Date:** 2026-11-16 · **Status: PREPARED** (inputs built, block runs and routing tested; full run pending) ·
Branch `feature/hype-mf6-national`

## Context

The landslíkan report compares three paradigms on one fabric and one forcing: SUMMA + routing (process-based,
semi-distributed, glacier), HYPE + MODFLOW 6 (ADR 0012–0014) and ParFlow-CLM (ADR 0016). The best Tungnaá SUMMA
member is the glacier build (`summa_v2_19gru_glacier`, ashleymedin `glacier` branch b851c3df plus
`patches/glacier_diagnostics_routing.patch`), 19 GRUs, 200 m bands, CARRA 3-hourly, mizuRoute.

Decisions (user, 2026-11-16): glacier SUMMA for the ice units with sparse geometry; period 1991–2025 as HYPE; the
server may run the full model.

## Design

**Inputs from the fabric, no shapefile pipeline** (`models/summa/national.py`, as national HYPE):

- GRU = fabric unit (10,042; gruId = unit id, so routing needs no remapping).
- HRU = elevation band from `hru_classes`: 200 m in the 769 units with ice, 400 m elsewhere; a band under 5 % of
  its unit is merged into its smaller neighbour. 16,450 HRUs (ice run 2,948, land run 13,502). hruId = unit × 100 + k.
- Two runs with the same decisions and parameters: **ice** (769 GRUs; glacier attributes and a two-domain cold
  state) and **land** (9,273 GRUs, ordinary files). The glacier build stores its grids densely over every GRU
  (ygrid × xgrid × grid × gru); keeping ice-free units out gives 149 × 150 × 769 arrays (~150 MB) instead of 10,042.
- Ice geometry per GRU from the fabric rasters, not per-GRU warping: block size a multiple of 25 m, ≥ 100 m and
  ≤ 150 cells a side; `cell2hruId` is the block's majority HRU from the 25 m bands, `glacierMask` ≥ 50 % glacier,
  `bed_elev` = block-mean raw DEM − block-mean ice thickness. 10,477 km² and 3,409 Gt of ice (class table 10,470 km²).
  The surface is the raw IslandsDEM the class bands were cut from, not `dem_filled.tif` (the conditioned routing
  surface, hydraulic potential under ice): with that, 773 km² of ice landed in the wrong bands.
- Vegetation: dominant non-water land class (IGBP as national HYPE, except cultivated land → grassland 10: Iceland's
  hayfields; as cropland 12, three low-lying land HRUs failed in January 1991 in the native solver, "residual and
  iteration increment are of the same sign", and run as 10); soil: fabric soil group → ROSETTA (coarse →
  LOAMY SAND, andosol → SANDY LOAM, organic → SILT LOAM; priors).
- Parameters: the Tungnaá glacier calibration, uniform, pinned to the finished `elevgrad` calibration's
  best-parameter JSON (`BEST_PARAMS`, score 0.876), not read from the template's `trialParams.nc`: a running Tungnaá
  calibration rewrites that file (2026-11-16 21:27, adding `vGn_alpha`/`vGn_n`/`theta_res`), and a rebuild from it
  made ice HRU 654302 fail at its first step ("coupled step below the minimum step length"). Each parameter on the
  template's dimension (only
  `routingGamma*` are GRU parameters; a local parameter put on `gru` is read at HRU indices and fails with a bare
  `read_param/netcdf-file_open/` once a block's HRUs pass nGRU).
- Decisions, tables and parameter-info files copied from the Tungnaá glacier member. Daily output.

**Forcing** (`national.write_forcing`): the 3-h CARRA land-cell archive (`landslikan.carra 3h`, new: 8 variables
as scaled int16, 46 MB a month, gap-free UTC period ends) and the fabric's `forcing_carra` weights; temperature
lapsed (−6.5 K/km) from CARRA orography to the HRU, pressure hypsometric, specific humidity capped at saturation,
longwave × (T/T₀)⁴. One NetCDF per water year and run, ~0.8 GB a year: written just before each chunk and deleted
after it.

**Runs** (`national_run.py`): chunks of whole water years (default 5). Each chunk runs every GRU block of both
runs (`-g start count -r e`), blocks balanced by HRU count (ice HRUs × 2), `HDF5_USE_FILE_LOCKING=FALSE`; the block
restarts are merged along hru/gru into the next chunk's initial conditions.

**Routing** (`national_run.route`): GRU `averageRoutedRunoff` (m/s) × unit area → dRoute Muskingum–Cunge on the
fabric's `droute/` (topology.nc, 19 transfers by commissioning year, 413 level-pool lake nodes), hourly steps,
daily means. dRoute from `/Users/darri/repos/dRoute` main (#17), not the SYMFLUENCE wheel (no `load_hydrofabric`).

## Measured (2026-11-16, Mac, v1.1 fabric)

- Build: 90 s. Forcing: 8 s and 65 MB per month for both runs.
- Physics: 0.5–0.8 ms per HRU and 3-h step on land (100 GRUs); ice blocks 2 ms in the first month (cold glacier
  start). Full period ≈ 350 CPU-h → about 16–20 h on the server's 22 free cores.
- **Setup ≈ 0.5 s per HRU**, all in `t2l_lookup_soil` (the enthalpy lookup tables of `nrgConserv enthalpyForm`),
  ≈ 2.2 CPU-h per start of the whole model: hence multi-year chunks. Compressed attribute files added 0.3 s per
  GRU more (each GRU read decompresses the whole variable); they are written uncompressed.
- dRoute: 1.1 s to load, 8.5 ms per hourly step for 10,042 reaches (≈ 45 min for 1991–2025). Mass-balance residual
  0.9 % after 10 days of constant inflow from a cold start: to check before the full run.
- Restarts: land blocks write and merge. **Glacier blocks cannot write restarts**: the glacier branch calls
  `writeRestartGrid` inside the GRU loop, which redefines the grid dimensions and fails at the second glacier GRU
  (`iCreate[grid] String match to name in use`). Fix: `patches/glacier_restart_grid.patch` (call once after the
  loop); `installs/summa-glacier` rebuilt with it 2026-11-16: two 8-GRU ice blocks write restarts, and a 16-GRU
  run starts from their merge. Glacier restarts size `ygrid`/`xgrid` to each block's largest GRU (rest = netCDF
  default fill), so `merge_restarts` pads every non-concatenated dimension to a common size (grids to the cold
  state's 149 × 150) before joining.

## Open

1. ~~Rebuild summa-glacier with the restart patch~~ (local and server `~/landslikan_models`, 2026-11-16); next the
   whole-country smoke month (staged on the server) and 1991–2025.
2. Spin-up: the Tungnaá cold state is uniform; run WY1992–1996 twice and start the scored run from the second.
3. Calibration: the Tungnaá values are priors only; a national calibration like ADR 0013 (KGE at the 72 gauges,
   VÍ glacier mass balance) after the first run.
4. dRoute mass-balance residual; lakes' level-pool parameters are defaults.
5. Lakes and open water are not SUMMA HRU classes (dominant non-water class): lake evaporation is missing.
