# SYMFLUENCE CARRA2 forcing plugin (experimental)

Install alongside the current SYMFLUENCE checkout:

```sh
uv pip install --python .venv/bin/python -e plugins/symfluence-carra2
```

Set `domain.data_access: cloud`, `forcing.dataset: CARRA2`,
`forcing.time_step_size: 10800`, an explicit N/W/S/E bounding box, and
three-hour UTC start/end times. Run the normal `acquire_forcings` workflow step.
This checkout adds CARRA2 to the core forcing configuration whitelist.
CDS credentials and access to `reanalysis-pan-carra` are required.

The `symfluence.plugins` entry point registers acquisition and preprocessing
handlers. It writes standard forcing variables on the native polar grid;
the CARRA2 point-cell polygon/remapping implementation uses the projection and
cell spacing decoded from GRIB, rather than a default CARRA1 projection.

## Temporal convention

Both state variables and fluxes use forecasts initialized at 00/12 UTC.
Fluxes are differences between accumulations at leads 6,9,12,15,18 hours.
Outputs are three-hour rates labelled at the **interval end**. State variables
are instantaneous forecasts at that end, not three-hour means or analyses.
`time_bounds` records the accumulation interval. One preceding calendar day
is fetched for boundary coverage. No differencing across forecast cycles,
interpolation, or gap filling is permitted. Duplicate/missing intervals,
missing values inside the requested area and unfamiliar parameters fail validation.
Negative increments beyond 0.01 mm precipitation or 0.1 W/m² radiation
per three-hour interval fail validation. Smaller negative increments are
redistributed within their forecast cycle, preserving its net accumulation
when nonnegative. Net-negative trace cycles become zero, with the residual
recorded explicitly in numerical_correction_audit. These are explicit numerical
acceptance budgets, not proof of the archive’s upstream packing precision.

Requests are sequential, grouped by week and split at calendar-month boundaries.
JSON request manifests and native GRIB files are retained in `carra2_requests/`.
Final NetCDF publication is atomic. Retained raw files consume extra disk space.
Version 0.1.2 retains requested native cells as (time, point), with native row/column
indices and actual projected cell polygons. Outside-area bitmap cells are
excluded; missing cells inside the requested area are never filled.
Set data.force_download: true when replacing older processed caches; raw GRIB
requests are still reused. The plugin supports the native polar stereographic grid;
unfamiliar layouts fail explicitly rather than guessing coordinates.

## Validation status

Offline tests include native synthetic GRIB decoding, registered acquisition
output, forecast-cycle transitions, interval volume conservation, and missing
coverage rejection. The real CDS sample passed conversion through the normal acquisition workflow
on 2026-10-07: eight three-hour outputs, 1,633 native cells, all seven fields finite.
Spatial remapping to 21 Tungnaá units passed independent weight reconstruction
and complete polygon coverage, without gap filling. Evidence is in
output/tungnaa_carra2_recovery/validation.json. Native orography/lapse adjustments,
seasonal coverage and long-period suitability remain unverified; this is an
experimental pilot, not production validation.

```sh
.venv/bin/python -m pytest plugins/symfluence-carra2/tests -q
```

## References

- [CARRA2 CDS dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-pan-carra)
- [CARRA2 user guide](https://confluence.ecmwf.int/spaces/CKB/pages/583771203/Copernicus+pan-Arctic+Regional+Reanalysis+CARRA2+Data+User+Guide)
- [MET Norway request examples](https://github.com/metno/carra2_cds)

The live CDS form currently names downward longwave
`thermal_surface_radiation_downwards`, unlike one example script. The request
uses the live catalogue name. Belgingur's published forecast leads/interval
alignment remain unconfirmed; equality with CARRA2 is not expected.
