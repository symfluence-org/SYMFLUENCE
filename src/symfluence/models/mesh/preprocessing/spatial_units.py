"""Explicit land-cover/elevation units with one forcing column per native tile.

This is a non-routing response-unit discretization. The final rank aggregates
all units for basin diagnostics; links do not represent physical river channels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ..errors import MESHConfigurationError, MESHInputError, MESHOutputError


def prepare_spatial_units(directory, units_path, forcing_path):
    directory = Path(directory)
    units = gpd.read_file(units_path)
    needed = {"unit_id", "template_mid", "elevation", "area_m2"}
    if not needed <= set(units) or units.unit_id.duplicated().any():
        raise MESHConfigurationError("Spatial units require unique IDs, template MIDs, elevation and area")
    if (
        units.crs is None
        or not units.crs.is_projected
        or any(axis.unit_conversion_factor != 1 for axis in units.crs.axis_info)
    ):
        raise MESHConfigurationError("Spatial unit geometry must use a metre projected CRS")
    if not np.allclose(units.area, units.area_m2, rtol=1e-6, atol=0.1):
        raise MESHInputError("Spatial unit geometry and declared areas disagree")
    if abs(units.geometry.union_all().area - units.area.sum()) > 1:
        raise MESHInputError("Spatial units overlap")
    n = len(units)
    if n < 2 or (units.area_m2 <= 0).any():
        raise MESHConfigurationError("At least two positive-area units required")
    path = directory / "MESH_parameters_CLASS.ini"
    lines = path.read_text().splitlines()
    starts = [i for i, l in enumerate(lines) if "05 5xFCAN" in l]
    ends = [i + 1 for i, l in enumerate(lines) if "19 RCAN/SCAN" in l]
    blocks = [lines[a:b] for a, b in zip(starts, ends)]
    mids = [int(float(next(l for l in block if "13 XSLP" in l).split()[4])) for block in blocks]
    if len(set(mids)) != len(mids) or not set(units.template_mid) <= set(mids):
        raise MESHInputError("Missing or ambiguous land-cover templates")
    parents = [mids.index(mid) for mid in units.template_mid]
    new = []
    for i, parent in enumerate(parents):
        for line in blocks[parent]:
            if "13 XSLP" in line:
                values = line.split()[:4] + [str(1001 + i)]
                line = "  ".join(values) + "  13 XSLP/XDRAINH/MANN/KSAT/MID"
            new.append(line)
        new.append("")
    header = lines[: starts[0]]
    idx = next(i for i, l in enumerate(header) if "04 DEGLAT" in l)
    vals = header[idx].split()[:9]
    vals[7:9] = [str(n + 1), str(n)]
    header[idx] = " ".join(vals) + "  " + header[idx][header[idx].index("04 DEGLAT") :]
    hydro = (directory / "MESH_parameters_hydrology.ini").read_text().splitlines()
    section = next(i for i, l in enumerate(hydro) if "GRU class dependent hydrologic parameters" in l)
    ci = next(i for i in range(section + 1, len(hydro)) if re.match(r"^\s*\d+\s", hydro[i]))
    count = int(hydro[ci].split()[0])
    seen = 0
    for i in range(ci + 1, len(hydro)):
        if seen == count:
            break
        body, sep, comment = hydro[i].partition("#")
        v = body.split()
        if not v or not re.fullmatch(r"[A-Za-z_]\w*", v[0]):
            continue
        values = v[1:]
        if len(values) == 1:
            values *= len(mids)
        if len(values) != len(mids):
            raise MESHInputError("Hydrology template count mismatch")
        hydro[i] = v[0] + " " + " ".join(values[j] for j in parents) + (" #" + comment if sep else "")
        seen += 1
    if seen != count:
        raise MESHInputError("Incomplete hydrology section")
    with xr.open_dataset(directory / "MESH_drainage_database.nc") as source:
        if source.sizes.get("subbasin") != 1:
            raise MESHInputError("Spatial adapter expects a fresh lumped template")
        if not np.isclose(float(source.GridArea.sum()), units.area_m2.sum(), rtol=1e-5):
            raise MESHInputError("Native and spatial basin areas disagree")
        ddb = source.drop_dims("NGRU").isel(subbasin=np.zeros(n + 1, dtype=int)).load()
    ddb = ddb.assign_coords(subbasin=np.arange(1, n + 2))
    ddb["GRU"] = (("subbasin", "NGRU"), np.pad(np.eye(n), ((0, 1), (0, 1))))
    for name in ["GridArea", "Area"]:
        if name in ddb:
            ddb[name] = ("subbasin", np.r_[units.area_m2.values, 0.0], {"units": "m2"})
    ddb["Rank"] = ("subbasin", np.arange(1, n + 2, dtype="int32"))
    # Native basin output clamps the outlet rank to NAA (the last active
    # grid). Accumulate all response units there before the dummy outlet.
    # RUNMODE noroute ensures these bookkeeping links add no channel physics.
    ddb["Next"] = ("subbasin", np.array([n] * (n - 1) + [n + 1, 0], dtype="int32"))
    ddb["DA"] = (
        "subbasin",
        np.r_[units.area_m2.values[:-1], units.area_m2.sum(), units.area_m2.sum()],
        {"units": "m2"},
    )
    ddb["AL"] = ("subbasin", np.r_[np.sqrt(units.area_m2.values), 1.0])
    centroids = gpd.GeoSeries(units.geometry.centroid, crs=units.crs).to_crs(4326)
    ddb["lat"] = ("subbasin", np.r_[centroids.y.values, centroids.y.mean()], {"units": "degrees_north"})
    ddb["lon"] = ("subbasin", np.r_[centroids.x.values, centroids.x.mean()], {"units": "degrees_east"})
    with xr.open_dataset(forcing_path) as source:
        if list(source.unit_id.values.astype(str)) != list(units.unit_id.astype(str)):
            raise MESHInputError("Spatial forcing identity/order mismatch")
        forcing = source.drop_vars("unit_id").load()
    variables = ["TA", "PRES", "QA", "UV", "PRE", "FSIN", "FLIN"]
    if any(forcing[v].dims != ("time", "subbasin") or not np.isfinite(forcing[v]).all() for v in variables):
        raise MESHInputError("Invalid spatial forcing dimensions or values")
    if (forcing.PRE < 0).any():
        raise MESHInputError("Negative spatial precipitation")
    # Native MESH reserves the last grid as a zero-area outlet. It must not
    # consume a real response unit or exclude the highest glacier band.
    forcing = forcing.isel(subbasin=np.r_[np.arange(n), 0]).assign_coords(subbasin=np.arange(1, n + 2))
    forcing.time.attrs.update(standard_name="time", long_name="time", axis="T")
    options = (directory / "MESH_input_run_options.ini").read_text()
    if not re.search(r"RUNMODE\s+noroute\b", options):
        raise MESHConfigurationError(
            "Spatial response units require noroute; an explicit river network is needed for channel routing"
        )
    path.write_text("\n".join(header + new + lines[ends[-1] :]) + "\n")
    (directory / "MESH_parameters_hydrology.ini").write_text("\n".join(hydro) + "\n")
    ddb.to_netcdf(directory / "MESH_drainage_database.nc", mode="w")
    encoding = {v: {"dtype": "float32", "zlib": True} for v in variables}
    encoding["time"] = {
        "dtype": "float64",
        "calendar": "gregorian",
        "units": "hours since 1900-01-01 00:00:00",
        "_FillValue": None,
    }
    forcing.to_netcdf(
        directory / "MESH_forcing_safe.nc",
        mode="w",
        encoding=encoding,
    )
    # xarray normalizes midnight out of CF units. This MESH reader's native
    # parse_datetime requires an explicit time-of-day; date-only references
    # otherwise leave uninitialized clock components despite a zero exit code.
    from netCDF4 import Dataset

    with Dataset(directory / "MESH_forcing_safe.nc", "r+") as native:
        native.variables["time"].units = "hours since 1900-01-01 00:00:00"
    (directory / "spatial_units.json").write_text(
        json.dumps(
            {
                "units": units.drop(columns="geometry").to_dict("records"),
                "topology": "aggregation only; no channel routing",
            },
            indent=2,
        )
    )


def preserve_template_grus(directory):
    """Append MESH's dummy column before generic fixers can trim real land cover."""
    directory = Path(directory)
    count = (directory / "MESH_parameters_CLASS.ini").read_text().count("05 5xFCAN")
    path = directory / "MESH_drainage_database.nc"
    with xr.open_dataset(path) as source:
        ddb = source.load()
    values = ddb.GRU.values
    if values.shape[1] == count + 1 and np.all(values[:, -1] == 0):
        return
    if values.shape[1] != count or not np.allclose(values.sum(axis=1), 1):
        raise MESHInputError("Ambiguous spatial-template GRU padding")
    if any("NGRU" in v.dims for k, v in ddb.data_vars.items() if k != "GRU"):
        raise MESHConfigurationError("Unsupported GRU-indexed template field")
    attrs = dict(ddb.GRU.attrs)
    dims = ddb.GRU.dims
    ddb = ddb.drop_dims("NGRU")
    ddb["GRU"] = (dims, np.column_stack([values, np.zeros(len(values))]), attrs)
    ddb.to_netcdf(path, mode="w")


def check_spatial_output(directory):
    """Reject last-unit-only native basin output even when dates are complete."""
    directory = Path(directory)
    if not (directory / "spatial_units.json").exists():
        return
    path = directory / "results" / "Basin_average_water_balance.csv"
    observed = pd.read_csv(path, skipinitialspace=True)
    dates = pd.to_datetime(observed.YEAR.astype(str), format="%Y") + pd.to_timedelta(observed.JDAY - 1, unit="D")
    with (
        xr.open_dataset(directory / "MESH_drainage_database.nc") as ddb,
        xr.open_dataset(directory / "MESH_forcing_safe.nc") as forcing,
    ):
        seconds = np.diff(forcing.time.values).astype("timedelta64[s]").astype(float)
        if not len(seconds) or not np.all(seconds == seconds[0]) or seconds[0] <= 0:
            raise MESHConfigurationError("Spatial forcing requires a regular positive timestep")
        precipitation = (forcing.PRE * ddb.GridArea / ddb.GridArea.sum()).sum("subbasin")
        expected = (precipitation * seconds[0]).to_series().resample("D").sum().reindex(dates)
    if not np.allclose(observed.PREC, expected, rtol=1e-4, atol=1e-3):
        raise MESHInputError("Spatial MESH basin precipitation does not match area-weighted forcing")
    check_spatial_diagnostics(directory, observed, dates)


def prepare_spatial_baseflow(directory):
    """Keep calibrated baseflow active without enabling artificial channels.

    Native noroute skips river classes (NRVR=0), so FLZ/PWR in the channel
    section are otherwise silently ignored. Refresh grid overrides from the
    current hydrology file before each execution, including calibration.
    """
    directory = Path(directory)
    if not (directory / "spatial_units.json").exists():
        return
    options_path = directory / "MESH_input_run_options.ini"
    options = options_path.read_text()
    if not re.search(r"BASEFLOWFLAG\s+wf_lzs\b", options):
        raise MESHConfigurationError("Spatial adapter currently requires wf_lzs baseflow")
    hydro = (directory / "MESH_parameters_hydrology.ini").read_text()
    with xr.open_dataset(directory / "MESH_drainage_database.nc") as source:
        parameters = source[["lat", "lon", "crs"]].load()
        n = source.sizes["subbasin"]
    for key in ("FLZ", "PWR"):
        matches = re.findall(rf"^\s*{key}\s+([^#\n]+)", hydro, re.MULTILINE)
        if len(matches) != 1:
            raise MESHInputError(f"Ambiguous spatial baseflow parameter {key}")
        values = np.array([float(v) for v in matches[0].split()])
        if not len(values) or not np.isfinite(values).all() or not np.all(values == values[0]) or values[0] <= 0:
            raise MESHConfigurationError(f"Spatial baseflow requires a positive uniform {key}")
        parameters[key.lower()] = ("subbasin", np.full(n, values[0]), {"grid_mapping": "crs"})
    match = re.search(r"^INPUTPARAMSFORM(?:FLAG)?\s+([^\n#]+)", options, re.MULTILINE)
    if not match:
        raise MESHConfigurationError("Spatial baseflow requires explicit INPUTPARAMSFORM")
    destination = directory / "MESH_parameters.nc"
    if destination.exists():
        with xr.open_dataset(destination) as existing:
            if existing.attrs.get("symfluence_spatial_baseflow") != 1:
                raise MESHConfigurationError("Cannot overwrite an independently supplied MESH_parameters.nc")
    parameters.attrs["symfluence_spatial_baseflow"] = 1
    parameters.to_netcdf(destination, mode="w")
    if not {"nc", "netcdf"} & set(match[1].split()):
        options = options[: match.end()] + " nc " + options[match.end() :]
        options_path.write_text(options)


SPATIAL_DIAGNOSTIC_FIELDS = ('PREC', 'RFF', 'ET', 'FSNO', 'SNO', 'LQWSSNO', 'STGGW')


def prepare_spatial_diagnostics(directory, cryosphere_endpoints=False):
    """Enable compact native daily fields without the five-point CLASS limit."""
    directory = Path(directory)
    if not (directory / 'spatial_units.json').exists():
        raise MESHConfigurationError('Spatial diagnostic output requires an explicit unit manifest')
    options_path = directory / 'MESH_input_run_options.ini'
    options = options_path.read_text()
    previous = re.search(r'^OUT(?:FIELDS|FILES)FLAG\s+(\S+)', options, re.MULTILINE)
    options, count = re.subn(r'^OUT(?:FIELDS|FILES)FLAG[^\n]*', 'OUTFIELDSFLAG on', options, flags=re.MULTILINE)
    if count != 1:
        raise MESHInputError('Expected exactly one native output-fields control flag')
    path = directory / 'outputs_balance.txt'
    contents = ''.join(f'{v} d csv shedorder print_date\n' for v in SPATIAL_DIAGNOSTIC_FIELDS)
    daily_contents = contents
    if cryosphere_endpoints:
        step = re.search(r'^TIMESTEPFLAG\s+(\d+)', options, re.MULTILINE)
        if not step or int(step[1]) != 60:
            raise MESHConfigurationError('Cryosphere endpoints currently require a 60-minute native timestep')
        contents += 'SNO h csv shedorder print_date\nLQWSSNO h csv shedorder print_date\n'
    managed = (directory / 'spatial_diagnostics.json').exists() and path.exists() and path.read_text() in (
        daily_contents, daily_contents + 'SNO h csv shedorder print_date\nLQWSSNO h csv shedorder print_date\n')
    if path.exists() and path.read_text() != contents and not managed:
        if previous and previous[1].lower() in ('1', 'on'):
            raise MESHConfigurationError('Refusing to replace independently configured native output fields')
        # Base settings ship an inactive template. Preserve it before enabling
        # the explicit spatial contract; never discard an active custom setup.
        backup = directory / 'outputs_balance.pre_spatial.txt'
        if not backup.exists():
            backup.write_text(path.read_text())
    path.write_text(contents)
    options_path.write_text(options)
    (directory / 'spatial_diagnostics.json').write_text(json.dumps({'fields': SPATIAL_DIAGNOSTIC_FIELDS, 'hourly_endpoints': cryosphere_endpoints}))


def check_spatial_diagnostics(directory, basin, dates):
    """Require complete per-unit output and independently reconcile aggregation."""
    directory = Path(directory)
    if not (directory / 'spatial_diagnostics.json').exists():
        return
    with xr.open_dataset(directory / 'MESH_drainage_database.nc') as ddb:
        weights = ddb.GridArea.values / ddb.GridArea.sum().item()
    for variable in SPATIAL_DIAGNOSTIC_FIELDS:
        frame = pd.read_csv(directory / 'results' / f'{variable}_D.csv', header=None, index_col=0)
        frame = frame.dropna(axis=1, how='all')
        clock = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if not clock.equals(pd.DatetimeIndex(dates)) or frame.shape[1] != len(weights):
            raise MESHOutputError(f'Incomplete or misaligned spatial diagnostic: {variable}')
        values = frame.to_numpy()
        if not np.isfinite(values).all():
            raise MESHOutputError(f'Nonfinite spatial diagnostic: {variable}')
        if variable == 'FSNO' and (values.min() < -1e-5 or values.max() > 1.00001):
            raise MESHInputError('Invalid native fractional snow cover')
        if variable in ('PREC', 'RFF', 'ET') and not np.allclose(
            values @ weights, basin[variable], rtol=1e-4, atol=1e-3
        ):
            raise MESHInputError(f'Spatial diagnostic aggregation mismatch: {variable}')
