---
name: add-data-handler
description: >-
  Add or modify a SYMFLUENCE data handler — forcing, geospatial attributes,
  remote-sensing products, or streamflow/observation datasets — across the
  acquisition → preprocessing → model-ready pipeline. Covers the registry +
  import-list mechanism, base-class contracts, mixins, config-key dispatch, and
  output-path conventions.
when_to_use:
  - Adding a new dataset (e.g. NLDAS, MSWEP, a soil/snow product, a gauge network)
  - Debugging why a handler "isn't found" / isn't being picked up
  - Understanding how data flows from a remote source into the model-ready store
---

# Adding & Understanding SYMFLUENCE Data Handlers

SYMFLUENCE ingests 70+ datasets through a uniform plugin pipeline. This skill is
the complete map of that subsystem plus the recipe for extending it. All paths
are relative to `src/symfluence/data/` unless noted.

## 1. The three-layer pipeline

Data flows through three pluggable layers, each with its own base class and
registry. Most new datasets need a handler in **only one** layer (usually
acquisition). Know which layer you're touching before you start.

```
REMOTE SOURCE
   │
   ▼  acquisition/handlers/*.py        @R.acquisition_handlers.add('NAME')
ACQUISITION          BaseAcquisitionHandler.download(output_dir) -> Path
   │                 raw files → data/forcing/raw_data/{DATASET}/  or  data/attributes/{cat}/
   ▼  preprocessing/                   (forcing standardisation, attribute zonal stats)
PREPROCESSING        dataset_handlers/  → BaseDatasetHandler  (@R.dataset_handlers.add)
   │                 attribute_processors/ → BaseAttributeProcessor (zonal stats → CSV)
   │                 cfif/  = CF-Intermediate Format (model-neutral variable names/units)
   │                 resampling/  = EASYMORE remap grid → HRU (basin_averaged_data/)
   ▼  model_ready/                     ModelReadyStoreBuilder.build_all()
MODEL-READY          forcings_builder / observations_builder / attributes_builder
                     → data/model_ready/{forcings,observations,attributes}/  (CF-1.8 NetCDF)

OBSERVATIONS take a parallel track:
   observation/handlers/*.py  @R.observation_handlers.add('name')
   BaseObservationHandler.acquire() -> Path ; .process(raw) -> Path
   → data/observations/{type}/{raw,preprocessed}/
```

Layer choice:
- **New forcing or attribute dataset to download** → acquisition handler (§4).
- **New gauge / in-situ observation network** → observation handler (§7).
- **New forcing dataset that needs its own variable renaming / unit conversion
  before remapping** → also add a preprocessing dataset handler (§8).
- **New derived attribute** (a new zonal-stat over a raster) → attribute
  processor (§8), but usually you only add the acquisition handler and reuse the
  generic processors.

## 2. The mechanism: explicit import list + registries (read this first)

A handler registers itself (via its `@register(...)` decorator) the moment its
module is imported. Modules are imported through an **explicit hardcoded list**,
*not* an auto-scan — so adding a file is not enough; you must also list it.

1. `acquisition/handlers/__init__.py` defines `_handler_modules = [...]`
   (a literal list of module names, ~lines 19-91) and imports each in a
   try/except loop (lines 93-100). Import failures are swallowed at **debug**
   level — a handler whose module raises ImportError (e.g. a missing heavy dep)
   silently won't register (see §9 troubleshooting). `nldas` and `mswep` are in
   this list (lines 85, 45).
2. That `__init__` runs because `acquisition/__init__.py:32` does
   `from . import handlers` on first import of the acquisition package — which
   happens when SYMFLUENCE imports the data layer. (`core/_bootstrap.py` does
   NOT touch data handlers; it only seeds delineation/BMI/metric registries and
   discovers external plugins.)
3. The observation layer works the same way via **explicit imports** in
   `observation/handlers/__init__.py` (`from .ana import ...`, etc.). The
   preprocessing `dataset_handlers/__init__.py` likewise imports its modules.

**Consequence: to add a handler you (a) drop a correctly-decorated `.py` file in
the right `handlers/` directory AND (b) add its module name to that directory's
`__init__.py` import list (the `_handler_modules` list for acquisition, or an
explicit `from .yourmod import YourHandler` for observation). Forgetting step (b)
is the #1 reason a new handler is "not found".** You do NOT edit `_bootstrap.py`.

Registration goes through the unified `R` facade in `core/registries.py`;
the lookup facades (`AcquisitionRegistry` etc., deriving from
`BaseRegistry` in `base_registry.py`) are read-only:

| Lookup facade | `R.*` attribute (register here) | Key normalisation |
|---------------|--------------------------------|-------------------|
| `AcquisitionRegistry` (`acquisition/registry.py`) | `@R.acquisition_handlers.add('NAME')` | lowercased |
| `ObservationRegistry` (`observation/registry.py`) | `@R.observation_handlers.add('name')` | lowercased |
| `DatasetRegistry` (`preprocessing/dataset_handlers/dataset_registry.py`) | `@R.dataset_handlers.add('name')` | lowercased |

Keys are case-insensitive (`'ERA5'` == `'era5'`). You may stack multiple
`add(...)` decorators for aliases — NLDAS does this
(`nldas.py`: `'NLDAS'`, `'NLDAS2'`, `'NLDAS-2'`).

Lookup:
```python
handler = AcquisitionRegistry.get_handler('ERA5', config, logger)  # raises DataAcquisitionError if missing
AcquisitionRegistry.list_datasets()                                # enumerate registered names
```

## 3. Where data lands (path conventions)

Paths come from `path_manager.py` under `SYMFLUENCE_DATA_DIR/domain_{NAME}/`:

```
data/forcing/raw_data/{DATASET}/      raw downloaded forcing       (raw_forcing_dir)
data/forcing/merged_data/             standardised, CFIF names     (merged_forcing_dir)
data/forcing/basin_averaged_data/     remapped to HRUs
data/attributes/{elevation,soilclass,landclass,climate,...}/   static attributes
data/observations/{type}/{raw,preprocessed}/                   observations
data/model_ready/{forcings,observations,attributes}/           CF-1.8 model-ready store
cache/raw_forcing/{DATASET}_{hash16}.nc                        content-addressed forcing cache
```

The acquisition base resolves these for you — don't hardcode. Use the
`output_dir` passed to `download()`, and `self._attribute_dir(subdir)` for
attribute outputs (`base.py:91-93`).

## 4. Acquisition handler contract

`BaseAcquisitionHandler(ABC, ConfigurableMixin, CoordinateUtilsMixin)` —
`acquisition/base.py:24`.

**Constructor** (provided; don't override unless you must call `super().__init__`):
```python
def __init__(self, config, logger, reporting_manager=None)
```
After construction you have:
- `self.config` — typed `SymfluenceConfig` (config is coerced from dict if needed)
- `self.logger`
- `self.bbox` — `{'lat_min','lat_max','lon_min','lon_max'}`
- `self.start_date`, `self.end_date` — `pd.Timestamp` (from `domain.time_start/end`)

**The one abstract method you must implement** (`base.py:95`):
```python
@abstractmethod
def download(self, output_dir: Path) -> Path:
    """Acquire data into output_dir; return the path written."""
```

**Inherited helpers** (use these, don't reinvent):
- `self._get_config_value(lambda: self.config.x.y, default=..., dict_key='KEY')`
  — the canonical dual Pydantic-path / flat-dict accessor.
- `self._skip_if_exists(path, force=None)` — idempotency; respects `FORCE_DOWNLOAD`.
- `self._get_earthdata_credentials()` / `self._get_earthdata_token()` — NASA auth
  (config, then `EARTHDATA_USERNAME`/`PASSWORD`/`TOKEN` env, then `.netrc`, then an
  in-process earthaccess login). `self._earthdata_auth()` also gives redirect-aware
  sessions and earthaccess login (`data/acquisition/earthdata_auth.py`); don't parse
  `.netrc` yourself (use `utils.read_netrc_credentials`).
- `self._attribute_dir(subdir)` — get/create `attributes/{subdir}`.
- `self.plot_diagnostics(file_path)` — auto spatial/distribution plots if a
  `reporting_manager` was injected.

Module-level helpers in `acquisition/utils.py`:
- `create_robust_session(max_retries=5, ...)` — `requests.Session` with backoff.
- `download_file_streaming(url, target, ...)` — atomic chunked download via `.part`.
- `atomic_write(target)` — context manager for atomic file writes.
- `resolve_credentials(host, env_prefix, config)` / `get_cds_credentials(config)`.

## 5. Mixins — opt in by multiple inheritance

Order: `class XAcquirer(BaseAcquisitionHandler, RetryMixin, ChunkedDownloadMixin, SpatialSubsetMixin)`.

**RetryMixin** (`mixins/retry.py`) — transient-failure resilience:
```python
self.execute_with_retry(fn, max_retries=3, base_delay=60, backoff_factor=2.0,
                        retryable_exceptions=(IOError, ConnectionError))
self.is_retryable_http_error(err)   # 429/503/timeout → True; 401/404 → False
self.is_retryable_cds_error(err)
```
Use for any network download from a flaky source (HydroShare, GES DISC, CDS).

**ChunkedDownloadMixin** (`mixins/chunked.py`) — large time ranges:
```python
self.generate_temporal_chunks(start, end, freq='MS')   # [(s,e), ...]; 'MS'|'YS'|'D'|'W'
self.generate_year_month_list(start, end)              # [(yyyy, mm), ...]
self.download_chunks_parallel(chunks, fn, max_workers=2)
self.merge_netcdf_chunks(files, out, time_slice=..., cleanup=True)
self.get_netcdf_encoding(ds, compression=True, complevel=1)  # for ds.to_netcdf(encoding=...)
```
Use for monthly/yearly reanalysis (ERA5, NLDAS, ESA CCI).

**SpatialSubsetMixin** (`mixins/spatial.py`) — clip to domain bbox:
```python
self.get_coord_names(ds)                       # auto-detect lat/lon dim names
self.subset_xarray_bbox(ds, handle_lon_wrap=True, time_slice=...)  # 0-360↔-180/180, dateline
self.subset_numpy_mask(ds, ...)                # rotated/irregular grids
self.subset_rasterio_window(src_path, out_path)  # GeoTIFFs
self.bbox_to_cds_area()    # [N,W,S,E] for CDS
self.bbox_to_geojson()     # for AppEEARS / CMR
self.bbox_to_wcs_params()  # OGC WCS SUBSET tuples
```

## 6. Categories & templates

(Run `symfluence list forcings` / `symfluence list observations` for the datasets
registered right now.) Pick the closest existing handler and copy its shape.
Representative examples:

| Category | Pattern | Reference handler | Mixins |
|----------|---------|-------------------|--------|
| Cloud reanalysis (Zarr) | open zarr → spatial subset → monthly loop → schema → NetCDF | `era5.py` (`era5_processing.py` for SUMMA schema) | — |
| CDS/OPeNDAP reanalysis | auth → monthly chunks → server-side crop → merge | `era5_cds.py`, `nldas.py` | Retry, Chunked, Spatial |
| Rclone/Drive forcing | rclone pull → subset → NetCDF | `mswep.py` | — |
| Static attribute (dual-source) | try primary (HydroShare zip) → fallback (WCS) → rasterio crop | `soilgrids.py`, `soilgrids_properties.py` | Retry |
| Tiled DEM | tile math → batch download → mosaic → crop | `dem.py` (`_TileDownloadMixin`) | Retry |
| Remote sensing (NASA CMR) | CMR search → earthaccess download HDF → QC → domain mean | `modis_lai.py` (extends `BaseEarthaccessAcquirer`) | — |
| Remote sensing (AppEEARS) | submit task → poll → download results | `modis_et.py` (extends `BaseAppEEARSAcquirer`) | — |
| Satellite SM/ET/TWS | OPeNDAP/cloud → spatial+temporal subset → NetCDF | `smap.py`, `grace.py`, `gleam_et.py` | varies |

**Specialized base classes** (subclass instead of `BaseAcquisitionHandler` when
they fit — they add domain helpers):
- `BaseEarthaccessAcquirer` (`handlers/earthaccess_base.py`): `_search_granules_cmr`,
  `_get_download_urls`, `_download_with_earthaccess`, `_count_available_granules`.
- `BaseAppEEARSAcquirer` (`handlers/appeears_base.py`): `_appeears_login/logout`,
  `_submit_appeears_task`, `_wait_for_task`, `_download_appeears_results`.
- `era5_processing.py` (functions, not a base): `era5_to_summa_schema`,
  `calculate_wind_speed`, `calculate_specific_humidity`, `validate_forcing_data`.

**Minimal template:**
```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Acquire MYDATA <one line>."""
from pathlib import Path

from symfluence.core.registries import R
from symfluence.data.acquisition.base import BaseAcquisitionHandler
from symfluence.data.acquisition.mixins.retry import RetryMixin


@R.acquisition_handlers.add('MYDATA')
class MyDataAcquirer(BaseAcquisitionHandler, RetryMixin):
    """Download MYDATA and write a domain-subset NetCDF/GeoTIFF."""

    def download(self, output_dir: Path) -> Path:
        out = output_dir / f"domain_{self.config.domain.name}_mydata.nc"
        if self._skip_if_exists(out):
            return out
        # ... self.bbox, self.start_date/end_date, self._get_config_value(...) ...
        # network I/O wrapped in self.execute_with_retry(...)
        return out
```

## 7. Observation handlers (different contract)

`BaseObservationHandler` (`observation/base.py:96`) — for gauges, snow courses,
flux towers used in calibration/evaluation. Two abstract methods:
```python
def acquire(self) -> Path:            # download/locate raw data
def process(self, input_path) -> Path:  # standardise → preprocessed/
```
Set class attributes `obs_type` (e.g. `"streamflow"`) and `source_name`.
Register with `@R.observation_handlers.add('name')` (lowercase).

Many datasets have **both** an acquisition handler (raw download) and an
observation handler (standardise + filter to experiment period) — see the two
`grdc.py` files: `acquisition/handlers/grdc.py` (`GRDCAcquirer.download`) and
`observation/handlers/grdc.py` (`GRDCHandler` calls the acquirer in `acquire()`,
then standardises in `process()`).

## 8. Preprocessing layer (only if needed)

**Dataset handlers** (`preprocessing/dataset_handlers/`, base
`base_dataset.py:210`, `@R.dataset_handlers.add('era5')`): convert a forcing
dataset's native variables to the **CFIF** standard before remapping. Implement
`get_variable_mapping`, `process_dataset`, `get_coordinate_names`,
`create_shapefile`, `merge_forcings`, `needs_merging`. Example: `era5_utils.py`.

**CFIF** (`preprocessing/cfif/`) = CF-Intermediate Format: model-neutral variable
names + units (`variables.py` `CFIF_VARIABLES`; `units.py` converters). Map your
dataset's variables onto these standard names so any model adapter can consume
them. `SUMMA_TO_CFIF_MAPPING` / `CFIF_TO_SUMMA_MAPPING` bridge legacy names.

**Attribute processors** (`preprocessing/attribute_processors/`, base
`base.py:28`): zonal stats over rasters → per-HRU CSV. Six exist
(elevation/climate/soil/landcover/geology/hydrology); you rarely add one. Output
CSVs are consumed by `model_ready/attributes_builder.py` into a grouped NetCDF.

**Resampling** (`preprocessing/resampling/`): EASYMORE weight generation/application
remapping the forcing grid to HRU polygons → `basin_averaged_data/`.

## 9. Wiring into dispatch — how your handler gets invoked

Registering is necessary but not sufficient; the pipeline must select your name
from config. Key config keys (read in `data_manager.py` / `acquisition_service.py`):

- **Forcing**: `forcing.dataset` (`FORCING_DATASET`) selects the handler;
  `domain.data_access` (`DATA_ACCESS`) chooses MAF (default) vs CLOUD path.
- **Attributes**: `domain.dem_source`, `domain.land_class_source`,
  `domain.attribute_profile` (`core` / `camels_spat` / `full` — see
  `acquisition/attribute_profiles.py`, `ProfileDataset.handler_name` is the
  registry key).
- **Observations**: `data.additional_observations` (comma list) and
  `data.streamflow_data_provider` (USGS/WSC/SMHI/…); plus
  `evaluation.{grace,modis_snow,snotel,fluxnet,usgs_gw}.download` flags that
  auto-append types looked up in `R.observation_handlers`.
- **Cache**: `data.force_download`, `data.forcing_cache_size_gb` (3.0),
  `data.forcing_cache_ttl_days` (30) — content-addressed cache in
  `cache/forcing_cache.py` (`RawForcingCache`, keyed by dataset+bbox+time+vars).

If a new dataset is a member of an attribute profile, add a `ProfileDataset`
entry in `attribute_profiles.py`. If it's a forcing dataset selected by
`forcing.dataset`, the registry name *is* the selector — no extra wiring.

**Troubleshooting "my handler isn't found":**
1. **Did you add the module name to `handlers/__init__.py`'s `_handler_modules`
   list?** This is the most common miss — the file existing is not enough.
2. Does the module import cleanly? The import loop swallows ImportErrors at debug
   level — run `python -c "import symfluence.data.acquisition.handlers.mydata"`
   to surface the real error (often a missing heavy dep). Guard heavy imports
   inside methods (lazy import), per project convention.
3. Is the file named without a leading `_` and in the correct `handlers/` dir?
4. Is the config selector spelled to match a registered key (case-insensitive)?
5. Confirm directly:
   `python -c "import symfluence.data.acquisition; from symfluence.data.acquisition.registry import AcquisitionRegistry as R; print(sorted(R.list_datasets()))"`

## 10. Conventions

Follow the repo conventions in `CLAUDE.md` (SPDX header, 120-col lines, Python
3.11+, lazy heavy imports, `# noqa: BLE001` for resilient I/O). Handler-specific:
file `{product}.py` (lowercase), class `{Product}Acquirer` / `{Product}Handler`,
config access via `self._get_config_value(...)`, and never let a `BLE001` catch
hide a registration-blocking import.

## 11. Step-by-step: add an acquisition handler

1. Identify category (§6) and copy the closest reference handler.
2. Create `acquisition/handlers/{name}.py` with SPDX header,
   `@R.acquisition_handlers.add('NAME')` (+ aliases), correct base + mixins.
3. **Add `'{name}'` to the `_handler_modules` list in
   `acquisition/handlers/__init__.py`** (§2 — without this it never registers).
4. Implement `download(self, output_dir) -> Path`: skip-if-exists → fetch (chunk
   if large, retry if flaky) → spatial subset to `self.bbox` → write NetCDF/GeoTIFF
   → return the path.
5. If forcing needing standardisation: add a `dataset_handlers/` handler mapping
   to CFIF (§8).
6. Wire dispatch (§9): document/confirm the `forcing.dataset` value or add a
   `ProfileDataset` entry.
7. Verify registration:
   `python -c "import symfluence.data.acquisition; from symfluence.data.acquisition.registry import AcquisitionRegistry as R; print('NAME' in [n.upper() for n in R.list_datasets()])"`
8. Smoke-test `download()` on a tiny bbox + short date range. Confirm output
   lands in the expected path (§3) and re-running skips (idempotency).
9. `ruff check src/symfluence/` and `mypy src/symfluence/` on the new file.

## 12. Key file reference

| Concern | File |
|---------|------|
| Acquisition base | `acquisition/base.py` |
| Acquisition registry | `acquisition/registry.py`; unified `core/registries.py` |
| Import list (MUST edit) | `acquisition/handlers/__init__.py` (`_handler_modules`); triggered by `acquisition/__init__.py:32` |
| Mixins | `acquisition/mixins/{retry,chunked,spatial}.py` |
| HTTP/cred utils | `acquisition/utils.py` |
| Earthaccess / AppEEARS bases | `acquisition/handlers/earthaccess_base.py`, `appeears_base.py` |
| ERA5 schema helpers | `acquisition/handlers/era5_processing.py` |
| Attribute profiles | `acquisition/attribute_profiles.py` |
| Orchestration / dispatch | `data_manager.py`, `acquisition/acquisition_service.py` |
| Paths | `path_manager.py` |
| Forcing cache | `cache/forcing_cache.py` |
| Preprocessing dataset handlers | `preprocessing/dataset_handlers/base_dataset.py`, `dataset_registry.py` |
| CFIF standard | `preprocessing/cfif/{variables,units}.py` |
| Attribute processors | `preprocessing/attribute_processors/base.py` |
| Resampling | `preprocessing/resampling/` |
| Model-ready builders | `model_ready/store_builder.py` (+ `forcings/observations/attributes_builder.py`) |
| Observation base/registry | `observation/base.py`, `observation/registry.py` |
| Worked example (both layers) | `acquisition/handlers/grdc.py` + `observation/handlers/grdc.py` |
| Recent additions to copy | `acquisition/handlers/nldas.py`, `mswep.py` |
