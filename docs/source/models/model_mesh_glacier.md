# Staging native MESH glacier surfaces

The shared geometry builder and MESH input writer are available as Python APIs.
MESH preprocessing also supports an opt-in `native_ice_surface` workflow mode.
It consumes a prepared geometry manifest after all ordinary input fixers, stages
the transformed CLASS/hydrology/DDB files privately, then installs them in the
model's forcing directory for the normal native runner. Existing configurations
default to `none`. No baseline configuration enables glacier mode.

```yaml
system:
  workflow_steps: [model_specific_preprocessing, run_model]
model:
  hydrological_model: MESH
  mesh:
    glacier_mode: native_ice_surface
    glacier_manifest: /absolute/path/geometry.json
    glacier_unit_ids: [lower, upper]
    glacier_albedo_visible: 0.4
    glacier_albedo_infrared: 0.25
    glacier_temperature_c: -3
```

Add these keys to a normal prepared-domain config, then run
`symfluence workflow run --config config.yaml --force-rerun`. Geometry preparation
from outline/thickness sources remains an explicit preceding operation. The
schema requires the manifest, ordered unit IDs, albedos and temperature together;
finite glacier storage is not a supported mode. Glacier calibration still needs
separate parameter-propagation and physical validation.

```python
from symfluence.geospatial.glacier_geometry import build_glacier_manifest
from symfluence.models.mesh.preprocessing.glacier_writer import write_glacier_bundle

geometry = build_glacier_manifest(
    "spatial_units.gpkg", "glacier_outlines.gpkg",
    ["thickness_projected.tif"], id_column="unit_id",
)
report = write_glacier_bundle(
    "prepared_mesh_inputs", "new_mesh_ice_inputs", geometry,
    unit_ids=["lower", "upper"],  # Exact native CLASS/DDB column order.
    albedo_visible=0.4, albedo_infrared=0.25, temperature_c=-3,
)
```

The example albedos and temperature are explicit pilot assumptions, not calibrated
values or universal glacier defaults. The writer retains the staged snow state
and native snow-to-ice conversion defaults. It sets glacier IWF to 1. Supported
CLASS input has three read layers and the standard field markers.

## Geometry contract

- Supply non-overlapping spatial-unit polygons with stable IDs. Generic land-cover
  classes are not spatial-unit polygons. Supply outlines separately from thickness.
- Inputs must share a projected metre CRS; reproject rasters explicitly before
  calling, documenting the resampling and source products. The builder integrates
  exact raster-cell/polygon overlap areas, including fractional boundary cells.
- Ice area comes from outlines. Missing, zero or invalid thickness inside ice
  raises an error; it cannot silently turn ice into land. Overlapping positive
  thickness tiles are rejected rather than selecting their maximum.
- Each unit records land/ice areas, ice fraction, thickness coverage, ice volume
  and water-equivalent volume at the stated density. Raster and vector sources
  are recorded. In-memory geometry should have its provenance recorded by the caller.
- Outline area must agree with native MESH GRU area within the writer's tolerance.
  Do not relax this to hide a mismatch in basin boundary or spatial support.

## Native MESH contract

The writer accepts a finalized clean input directory containing CLASS, hydrology
and drainage database files. Every input GRU must occur in exactly one subbasin;
distributed land-cover classes need an explicit spatial adapter first. The
supported DDB convention has one zero dummy GRU column after all active columns.
`GridArea` must declare square metres. Additional GRU-indexed DDB variables are
rejected until an explicit remapping is implemented.

Each source GRU is split into its positive land and ice fractions. The DDB
fractions retain the original total area and subbasin structure. Routing and
forcing files are copied unchanged. Counted GRU hydrology arrays are expanded
with the same parent mapping, and CLASS NM is updated. Uncounted trailing
parameters stay uncounted. Soil and canopy parameters are replaced on ice blocks
with explicit CLASS ice codes, ice albedos and cold-start ice states.

The destination must be new and outside the source. Files are staged privately
and published as a complete directory. A zero-ice manifest preserves native input
files byte-for-byte. Do not use an input directory containing simulation outputs
or unrelated large caches; the writer copies the supplied directory.

Terrestrial CLASS initial-condition fixes, vegetation fixes and field overrides
preserve recognized ice blocks (top-layer SAND=-4). Elevation regeneration from
an ice block is rejected: build elevation units before splitting land and ice.
Full meshflow regeneration or GRU trimming must not be applied to the staged
bundle. Calibration targeting preserves ice blocks as described below; multi-year
calibration readiness still requires physical checks.

## Physical scope and validation

The output manifest explicitly records `mode: native_ice_surface` and
`finite_ice_storage: false`. Thickness-derived volume is provenance, not a claim
that CLASS depletes a finite glacier store. Finite-storage physics, restart
conservation and native energy/water budgets still need separate implementation
or verification.

Tests cover partial cells, missing/overlapping thickness, CRS and unit overlap,
zero ice, identity/order matching, native fraction conservation, hydrology
expansion, unchanged forcing/source files, and two successive terrestrial fixer
passes preserving the glacier fields. Run:

```sh
.venv/bin/python -m pytest tests/unit/models/mesh/test_glacier_writer.py \
  tests/unit/models/mesh/test_parameter_fixer.py \
  tests/unit/models/mesh/test_preprocessor.py -q
```

## Experimental finite store and process accounting

`MESH_GLACIER_MODE: finite_ice_guard` (typed: `model.mesh.glacier_mode`)
adds `MESH_glacier_storage.ini` to the finalized workflow inputs. Initial mass
is public geometry water-equivalent volume divided by glacier area, in kg/m²
of glacier. It is independent of CLASS's fixed thermal-layer thickness.

This mode requires the private native patch in
`src/symfluence/models/mesh/patches/`, a serial executable, and an adjacent
`<executable>.glacier.json` capability manifest with `protocol:
SYMFLUENCE_GLACIER_V1` and the executable's SHA-256 under `sha256`. Apply the patch
to a fresh isolated source checkout with `apply_glacier_ledger.py CHECKOUT`,
then build MESH normally. The shared installed executable is not replaced.
An old executable or mismatching hash is rejected. This is an experimental
source patch, not an upstream MESH capability or installer release.

The native ledger records melt, signed ice vapor loss, snow-to-ice conversion,
pond refreezing into snow, and melt energy. Pond refreezing is **not** counted
as direct bulk-ice gain. The persistent inventory is updated by conversion
minus melt minus vapor loss. If a step would exhaust the store, execution
fails before committing that step. Partial output cannot be scored. This is
an exhaustion guard, **not** a dynamic exposed-bed or glacier-retreat model;
thermal geometry and glacier area remain fixed.

Only `RESUMEFLAG` and `SAVERESUMEFLAG` values `0` and `4` are supported by the
workflow guard. Full sequential restart includes a versioned `.glacier-v1`
sidecar with inventory, cumulative fluxes, tile identities, step count and the
next simulation start date. Set the resumed run's start date to that checkpoint
date so the forcing reader advances correctly. Missing states, changed tile
layout, and mismatched dates fail explicitly. Preserve the complete native
restart file set together, not only the glacier sidecar.

Calibration targeting now protects ice blocks from terrestrial soil/canopy
changes. `ICE_ALBEDO_VISIBLE`, `ICE_ALBEDO_INFRARED`, and `ICE_LNZ0` target only
ice CLASS fields. `FREZTH` (temperature threshold), `SWELIM`, and `SNDENLIM`
target ice entries in the **counted** GRU hydrology section; `FRZTH` is a
different soil control. Unread trailing soil parameters are rejected rather
than silently calibrated. Glacier parameters require explicit
`MESH_PARAM_BOUNDS`; no search ranges or thickness calibration are assumed.

A completed pilot and restart checks do not certify a multi-year calibration.
Check energy, inventories, snow observations, annual glacier balances and
parameter sensitivity over the intended forcing period first. In particular,
native summary `STGW`/`DSTGWACC` are based on **time-averaged** storage, not the
final instantaneous inventory. Use actual checkpoint endpoints, including groundwater storage. Soil drainage
into that store is an internal transfer and must not also be counted as an
external loss.

## Explicit land-cover/elevation response units

`model.mesh.spatial_units` and `model.mesh.spatial_forcing` are an opt-in pair.
The first is a projected-metre polygon file with unique `unit_id`, `template_mid`,
`elevation` (metres), and `area_m2` columns. `template_mid` identifies a land-cover
CLASS template produced by meshflow; it is not inferred from polygon order.
The second is NetCDF with `time, subbasin` variables `TA` (K), `PRES` (Pa),
`QA` (kg/kg), `UV` (m/s), `PRE` (kg/m²/s), `FSIN` and `FLIN` (W/m²), and a
`unit_id(subbasin)` coordinate matching the polygon order exactly.

The standard preprocessing step retains every template, builds unique native
tiles, installs their forcing columns, and then applies the glacier writer.
It checks area conservation, overlap, unit identity and finite forcing. The
adapter writes an explicit Gregorian reference timestamp, including midnight:
this native MESH reader does not reliably parse date-only CF reference units.

This mode requires `run_mode: noroute`. Bookkeeping links accumulate every unit
at the last active rank and then a separate zero-area outlet, so native basin
output includes the whole watershed. These links are **not** a river network.
The runner compares daily native basin precipitation with the area-weighted
input forcing, rejecting last-unit-only summaries even when dates are complete.

The native `noroute` mode skips river-class parameters. For `wf_lzs`, the runner
therefore writes explicit grid `flz` and `pwr` in `MESH_parameters.nc`, refreshed
from the current hydrology parameters before each execution. This also preserves
calibration targeting. Only uniform positive FLZ/PWR are currently supported;
an independently supplied parameter NetCDF is never overwritten.

The Tungnaá diagnostic configuration under `output/tungnaa_mesh_spatial` uses
200 m elevation bands intersected with MODIS land cover and exact RGI outlines,
plus the public thickness inventory. Existing five-band CARRA forcing is
remapped by area overlap, with a residual within-band temperature lapse and
pressure/humidity adjustment. This is not a new CARRA download or an independent
verification of CARRA's native orography. Precipitation receives no additional
multiplier. Open-water polygons retain explicit wetland-proxy labels; lake
physics, physical channel routing, evolving glacier area and exposed-bed
transition remain unsupported in this pilot.

## Spatial diagnostics and optional travel-time routing

Set `model.mesh.spatial_diagnostics: true` with explicit spatial units to enable
native daily grid output for precipitation, runoff, evaporation, snow fraction,
solid/liquid snow storage and groundwater storage. The runner checks the daily
clock and unit dimensions, finite values, snow-fraction bounds and area-weighted
precipitation/runoff/evaporation against native basin output. Partial outputs
cannot pass by merely having a valid basin summary. Inactive bundled output
settings are backed up before replacement; different active custom settings
are rejected.

An optional `model.mesh.spatial_routing_kernel` CSV has columns `unit_id`,
`lag_days`, `weight`. Each unit must appear, integer daily lags must be 0–365,
and nonnegative weights must sum to one separately for each unit. Native
per-unit runoff is translated conservatively; delayed water beyond the run end
is retained in `MESH_routing_tail.csv` and reported as end routing storage.
`MESH_spatial_streamflow.csv` is the discharge used by the calibration worker
when a kernel is configured. Missing routed output fails rather than falling
back to unrouted runoff. The worker respects nested `domain.calibration_period`
and averages subdaily observations for these daily outputs.

This is fixed linear travel-time routing, not native channel hydraulics, lake
routing or feedback to terrestrial groundwater. Kernels require independently
documented drainage geometry and travel-time assumptions. Spinup must cover
initially empty routing storage. Report both terrestrial water balance and
routing storage, rather than equating routed outflow with same-day runoff.

The follow-up Tungnaá configuration is
`output/tungnaa_mesh_precalibration/workflow.yaml`. Its CARRA temperature reference
uses the downloaded native CARRA orography, whose grid centers match Belgingur
to within 0.43 m. Raw archive comparisons show that the five source-band
model-ready temperatures are *unlapsed*, despite the upstream configuration
flag. The corrected forcing therefore uses actual native source elevations,
with pressure/humidity adjustments and unchanged precipitation. See the
[official CARRA dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-carra-single-levels?tab=overview)
and the local `orography_audit.json` for source and verification.

The diagnostic reference kernel uses the corrected Wflow D8 terrain network,
with fixed channel/hillslope velocities of 1/0.1 m/s; these are assumptions,
not velocities fitted to evaluation performance. Native MESH remains in
`noroute` mode internally. Read the precalibration report before any optimizer
launch: excess spring snow, unavailable MODIS pixel QA and the terrestrial
open-water surrogate remain qualifications. A better discharge score does not
resolve them, and the configuration's workflow steps do not launch calibration.

## Snow and glacier calibration constraints

Opt in with `model.mesh.cryosphere_constraints: true` and
`cryosphere_observations: /path/to/cryosphere`. This requires explicit spatial
units/diagnostics and `finite_ice_guard`. The directory contains the LamaH-ICE
semicolon tables `modis_fractional_snow_cover_and_glacier_albedo.csv` and
`glacier_timeseries.csv`. No downloads or silent observation substitution occur.

The runner additionally emits native hourly `SNO_H.csv` and `LQWSSNO_H.csv`.
Only a 60-minute native timestep is accepted: these are then individual-step
endpoints, not daily or subhourly averages. Annual glacier balance combines
bulk-ice ledger change with solid and liquid snow change at the exact
September-end boundaries. The run must include the preceding boundary.

Use complete October–September scoring periods. Every selected year requires
at least 20 paired spring dates (April–June), 20 late-summer dates
(August–September), and a finite annual observed glacier balance/positive area.
Winter snow is diagnostic only because the available catchment CSV lacks
pixel QA. The snow score equally weights the two seasonal mean squared errors:
`1 - sqrt((MSE_spring + MSE_late_summer)/2) / snow_cover_error_scale`.
The glacier score is `1 - max(RMSE_fixed_area, RMSE_reported_area) /
glacier_mb_error_scale`. Taking the larger area-support error avoids selecting
whichever support flatters the model; neither representation implies evolving
modeled glacier area. Defaults are 0.2 snow fraction and 1 m w.e.; these are
explicit objective scales, not measured observation uncertainties.

Set `optimization.metric: COMPOSITE` and `composite_metric` with strictly
positive weights for exactly `kge`, `snow_score`, `glacier_score` (the pilot uses
0.6/0.2/0.2). Raw KGE remains available. When constraints are enabled, missing
components, invalid weights or a discharge-only objective are rejected. Missing
annual observations or incomplete output fail scoring rather than dropping a
constraint. `cryosphere_metrics.json` records periods, counts, annual values,
area support and component scores.

`ZSNL` is the CLASS limiting snow depth for full cover (mapped to `SNOLIM` in
`CLASSA`), not a ponding-depth parameter. In explicit spatial configurations,
land hydrology updates protect glacier entries and MODIS-17 water proxies.
Glacier albedo parameters target ice only and require explicit bounds. Run
process sensitivity before expanding parameter ranges; larger `ZSNL` did not
resolve Tungnaá's spring snow-cover bias. The scoped constrained workflow is
`output/tungnaa_mesh_snow_constraints/workflow.yaml`; its steps replay the model
without starting optimization.

## Frozen-soil infiltration inputs

`enable_frozen_soil: true` requires explicit, counted native inputs. Configure
`model.mesh.frozen_soil_parameters`, or supply a complete native hydrology file.
There are no universal basin defaults for these empirical settings. For example,
this is a **process-pilot assumption set**, not a calibrated recommendation:

```yaml
model:
  mesh:
    enable_frozen_soil: true
    frozen_soil_parameters:
      soil_porosity: 0.46002
      soil_depth: 0.35
      surface_saturation: 0.75
      ice_lens_temperature: -2.0
      infiltration_coefficient: 2.0
      opportunity_time: 0.0
```

The writer adds `SOIL_POR_MAX`, `SOIL_DEPTH`, `S0`, `T_ICE_LENS` and one
`t0_ACC` per native calendar year to the GRU-independent section. It adds a
counted `FRZC` array to the GRU-dependent section, after spatial/glacier splitting.
Ice and mapped open-water proxy tiles get `FRZC=0` (native bypass); soil tiles get
the configured positive coefficient. A zero opportunity time selects the native
empirical estimate. The native module uses shared porosity/depth/saturation
assumptions; these are not spatially resolved soil measurements.

The runner validates these values before every execution, including calibration.
Old enabled configurations with missing fields now fail instead of silently
bypassing the process. `FRZTH` and `RCHARG` are not substitutes: the supported
native reader does not recognize them. Inactive legacy tails are removed when
writing explicit frozen-soil inputs. `FLZ` and `PWR` remain the supported lower-zone
baseflow parameters and are refreshed into spatial grid overrides separately.

Native `FRZC=0` skips frozen-soil restriction; it does **not** force zero
infiltration. For diagnostics, use incoming snowpack drainage minus
`MELTRUNOFF` to measure admission to the CLASS soil/pond path. The raw native
`SNOWINFIL` variable can retain its incoming value on rejection branches.
Snowpack drainage includes rain as well as melt; admission is not equivalent to
actual soil infiltration or groundwater recharge. Report `DRAINSOL`, overland
runoff, and storage changes separately.

Native compatibility caveat: the inspected MESH `FROZEN.F90` and
`RUNCLASS36_module.f90` hard-code daily/annual resets at step 48. With
`TIMESTEPFLAG 60`, these resets never occur. Input validation does not certify
this native implementation: empirical frozen-soil runs at non-30-minute
steps require a tested native clock correction. The Tungnaá fixed-parameter
experiment records the minimal patch and a separate replay in
`output/tungnaa_mesh_frozen_fix/frozen_clock.patch`; it is not yet an installer
patch or a production-calibration approval. Standard CLASS infiltration
(`FROZENSOILINFILFLAG 0`) does not use this optional module.
