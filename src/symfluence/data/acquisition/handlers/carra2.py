# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CARRA2 (pan-Arctic regional reanalysis) forcing acquisition from the CDS.

Sequential, resumable CDS requests with explicit forecast-cycle accounting.
Requires ``eccodes`` to decode GRIB; it is imported only when reading data.

Configuration: ``FORCING_DATASET: CARRA2``, ``DATA_ACCESS: cloud``,
``FORCING_TIME_STEP_SIZE: 10800``, an explicit bounding box and three-hour UTC
start/end times. CDS credentials with access to ``reanalysis-pan-carra``.

Temporal convention: states and fluxes come from forecasts initialized at
00/12 UTC. Fluxes are differences of accumulations at leads 6, 9, 12, 15 and
18 h, written as three-hour rates labelled at the interval end; states are the
instantaneous forecast at that end. ``time_bounds`` records each interval. No
differencing across forecast cycles, interpolation or gap filling is done:
duplicate or missing intervals, missing values inside the requested area and
unexpected parameters fail. Negative increments beyond 0.01 mm precipitation
or 0.1 W/m2 radiation per interval fail; smaller ones are redistributed within
their cycle, preserving its net accumulation, and the residual is recorded in
``numerical_correction_audit``.

Requests are sequential, weekly, split at month boundaries; request manifests
and native GRIB are kept in ``carra2_requests/`` and the final NetCDF is
published atomically. Requested native cells are kept as (time, point) with
native row/column indices; cell polygons use the native DX/DY. Only the native
polar-stereographic grid is supported. Status: experimental (validated on a
2026-10-07 Tungnaá sample, 1,633 native cells; long-period suitability and
orographic adjustment are unverified).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R
from symfluence.data.acquisition.base import BaseAcquisitionHandler

DATASET = 'reanalysis-pan-carra'
VERSION = 'carra2-core-1'
VARIABLES = ['2m_temperature', '2m_specific_humidity', '10m_wind_speed',
             'surface_pressure', 'total_precipitation',
             'surface_solar_radiation_downwards', 'thermal_surface_radiation_downwards']
# GRIB paramIds (verified against the CARRA2 CDS documentation).
FIELDS = {
    167: ('air_temperature', 'K', False),
    174096: ('specific_humidity', 'kg kg-1', False),
    207: ('wind_speed', 'm s-1', False),
    134: ('surface_air_pressure', 'Pa', False),
    228228: ('precipitation_flux', 'kg m-2 s-1', True),
    169: ('surface_downwelling_shortwave_flux', 'W m-2', True),
    175: ('surface_downwelling_longwave_flux', 'W m-2', True),
}

def request_for_days(days, area):
    dates = pd.DatetimeIndex(days)
    if len(set(zip(dates.year, dates.month))) != 1:
        raise ValueError('Requests must stay within one calendar month')
    return dict(level_type='single_levels', variable=VARIABLES,
                product_type='forecast', time=['00:00', '12:00'],
                leadtime_hour=['6', '9', '12', '15', '18'],
                year=[str(dates[0].year)], month=[f'{dates[0].month:02d}'],
                day=[f'{d:02d}' for d in dates.day], data_format='grib', area=area)

def interval_rate(previous, current, hours=3):
    delta = np.asarray(current) - np.asarray(previous)
    if not np.isfinite(delta).all() or np.min(delta) < -1e-5:
        raise ValueError('Nonfinite or decreasing forecast accumulation within a cycle')
    return np.maximum(delta, 0) / (hours * 3600)

def cycle_rates(accumulations, tolerance, hours=3):
    """Bound numerical negatives, conserving each cell's full-cycle increment.

    Tolerances are explicit acceptance budgets in accumulated native units,
    not inferred solely from the final GRIB packing (upstream packing is unknown).
    Tiny net-negative cycle totals cannot be represented by nonnegative fluxes;
    their residual is reported rather than hidden.
    """
    values = np.asarray(accumulations, dtype=float)
    delta = np.diff(values, axis=0)
    if not np.isfinite(delta).all() or np.min(delta) < -tolerance:
        raise ValueError(f'Decreasing accumulation exceeds numerical budget {tolerance}: {np.nanmin(delta)}')
    positive = np.maximum(delta, 0.)
    positive_total = positive.sum(axis=0)
    net = values[-1] - values[0]
    target = np.maximum(net, 0.)
    factor = np.divide(target, positive_total, out=np.zeros_like(target), where=positive_total > 0)
    corrected = positive * factor
    residual = corrected.sum(axis=0) - net
    audit = dict(negative_increments=int((delta < 0).sum()),
                 minimum_raw_increment=float(delta.min()), tolerance=float(tolerance),
                 max_absolute_increment_correction=float(np.max(np.abs(corrected-delta))),
                 max_cycle_volume_residual=float(np.max(np.abs(residual))),
                 mean_cycle_volume_residual=float(np.mean(residual)))
    return corrected / (hours*3600), audit


def read_grib(path, area=None):
    import eccodes as ec
    from pyproj import CRS
    records = {}
    grid = None
    grid_cache = {}
    with open(path, 'rb') as f:
        while (g := ec.codes_grib_new_from_file(f)) is not None:
            try:
                param = ec.codes_get(g, 'paramId')
                if param not in FIELDS:
                    raise ValueError(f'Unexpected CARRA2 paramId {param}: {ec.codes_get(g, "shortName")}')
                if ec.codes_get(g, 'jPointsAreConsecutive') or ec.codes_get(g, 'alternativeRowScanning'):
                    raise ValueError('Unsupported GRIB scan order')
                nx, ny = int(ec.codes_get(g, 'Nx')), int(ec.codes_get(g, 'Ny'))
                # CDS area subsets can retain the entire native grid, with a
                # bitmap outside the geographic request. Never treat that bitmap
                # as permission to drop holes inside the requested support.
                signature = ec.codes_get(g, 'md5GridSection')
                if signature not in grid_cache:
                    full_lat = ec.codes_get_array(g, 'latitudes').reshape(ny, nx)
                    full_lon = (ec.codes_get_array(g, 'longitudes').reshape(ny, nx) + 180) % 360 - 180
                    if area is None:
                        indices = np.arange(nx * ny)
                        shape = (ny, nx)
                    else:
                        north, west, south, east = map(float, area)
                        if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
                            raise ValueError('Invalid or dateline-crossing CARRA2 area')
                        selected = ((full_lat >= south) & (full_lat <= north)
                                    & (full_lon >= west) & (full_lon <= east))
                        indices = np.flatnonzero(selected)
                        if not len(indices):
                            raise ValueError('No native CARRA2 cells inside requested area')
                        # Packed native cells: no invented values or artificial
                        # rectangular corner cells. Cell polygons use native DX/DY.
                        shape = (1, len(indices))
                    lat = full_lat.ravel()[indices].reshape(shape)
                    lon = full_lon.ravel()[indices].reshape(shape)
                    grid_cache[signature] = (indices, lat, lon)
                indices, lat, lon = grid_cache[signature]
                if grid is None:
                    if ec.codes_get(g, 'gridType') != 'polar_stereographic':
                        raise ValueError('Expected native CARRA2 polar stereographic grid')
                    lad = ec.codes_get(g, 'LaDInDegrees')
                    lov = ec.codes_get(g, 'orientationOfTheGridInDegrees')
                    radius = ec.codes_get(g, 'radius')
                    proj = CRS.from_proj4(f'+proj=stere +lat_0=90 +lat_ts={lad} +lon_0={lov} +R={radius} +units=m').to_proj4()
                    attrs = dict(MAP_PROJ4_STR=proj, DX=float(ec.codes_get(g,'DxInMetres')), DY=float(ec.codes_get(g,'DyInMetres')))
                    if area is not None:
                        attrs.update(spatial_layout='packed_native_cells', native_grid_nx=nx,
                                     native_grid_ny=ny, requested_area=json.dumps(list(area)),
                                     _native_indices=indices)
                    grid = (lat, lon, attrs)
                else:
                    if lat.shape != grid[0].shape:
                        raise ValueError('Inconsistent grids within request')
                    # GRIB1 precipitation and GRIB2 state coordinates have
                    # different rounding precision. Permit <=5m, never a cell shift.
                    distance = np.hypot((lat-grid[0])*111200,
                                        (lon-grid[1])*111200*np.cos(np.deg2rad(grid[0])))
                    if np.max(distance) > 5 or (area is not None and
                            not np.array_equal(indices, grid[2]['_native_indices'])):
                        raise ValueError('Inconsistent grids within request')
                if ec.codes_get(g, 'bitmapPresent'):
                    bitmap = ec.codes_get_array(g, 'bitmap').ravel()[indices]
                    if not bitmap.all():
                        raise ValueError('Missing CARRA2 grid values inside requested support')
                cycle = pd.to_datetime(str(ec.codes_get(g, 'dataDate'))) + pd.Timedelta(minutes=int(ec.codes_get(g,'dataTime'))//100*60 + int(ec.codes_get(g,'dataTime'))%100)
                step = int(ec.codes_get(g, 'endStep'))
                if ec.codes_get(g,'stepUnits') != 1:
                    raise ValueError('Expected GRIB step units of hours')
                if FIELDS[param][2] and (ec.codes_get(g,'stepType') != 'accum' or ec.codes_get(g,'startStep') != 0):
                    raise ValueError('Expected accumulation from forecast initialization')
                key = (param, cycle, step)
                if key in records:
                    raise ValueError(f'Duplicate GRIB field {key}')
                values = ec.codes_get_values(g).ravel()[indices].reshape(lat.shape)
                if not np.isfinite(values).all() or (area is None and ec.codes_get(g, 'numberOfMissing')):
                    raise ValueError('Missing CARRA2 grid values')
                records[key] = values
            finally:
                ec.codes_release(g)
    if grid is None:
        raise ValueError('Empty GRIB file')
    return records, grid

def convert(path, start, end, area=None):
    records, (lat, lon, attrs) = read_grib(path) if area is None else read_grib(path, area=area)
    attrs = dict(attrs)
    native_indices = attrs.pop("_native_indices", None)
    times = pd.date_range(start, end, freq='3h')
    if len(times) == 0 or any(t.hour % 3 or t.minute or t.second for t in times):
        raise ValueError('CARRA2 requires three-hour UTC timestamps')
    # Flux timestamps are interval ends. Retain forecast state at that end.
    data = {}
    numerical_audit = {}
    for param, (name, units, accumulated) in FIELDS.items():
        values = []
        cycles = {}
        # Daily CDS requests include forecasts outside the output window.
        # Validate coverage first, then correct only cycles actually consumed.
        selected = []
        for t in times:
            candidates = [(c, s) for (p,c,s) in records if p == param and s in (9,12,15,18) and c + pd.Timedelta(hours=s) == t]
            if len(candidates) != 1:
                raise ValueError(f'Missing/ambiguous coverage for {name} at {t}')
            selected.append(candidates[0])
        if accumulated:
            # 0.01 kg/m2 precipitation (0.01 mm); 0.1 W/m2 over 3 hours.
            tolerance = .01 if param == 228228 else .1 * 10800
            for cycle in sorted({c for c,s in selected}):
                totals = [records[param,cycle,s] for s in [6,9,12,15,18]]
                try:
                    rates, audit = cycle_rates(totals, tolerance)
                except ValueError as exc:
                    raise ValueError(f'{name}, forecast cycle {cycle}: {exc}') from exc
                cycles[cycle] = rates
                numerical_audit[f'{name}:{cycle.isoformat()}'] = audit
        for c,s in selected:
            value = records[param,c,s]
            if accumulated:
                value = cycles[c][(s-9)//3]
            values.append(value)
        data[name] = (('time','y','x'), np.asarray(values,dtype='float32'), dict(units=units, cell_methods='time: mean' if accumulated else 'time: point'))
    if native_indices is None:
        ds = xr.Dataset(data, coords=dict(time=times, latitude=(('y','x'),lat), longitude=(('y','x'),lon)), attrs=attrs)
    else:
        point_data = {name: (('time', 'point'), item[1][:, 0, :], item[2]) for name, item in data.items()}
        ds = xr.Dataset(point_data, coords=dict(time=times,
                        latitude=('point',lat.ravel()), longitude=('point',lon.ravel()),
                        native_row=('point',native_indices // attrs['native_grid_nx']),
                        native_col=('point',native_indices % attrs['native_grid_nx'])), attrs=attrs)
    ds['time_bounds'] = (('time','bounds'), np.column_stack([times.values - np.timedelta64(3,'h'),times.values]))
    ds.time.attrs['bounds'] = 'time_bounds'
    for name in ('time', 'time_bounds'):
        ds[name].encoding.update(units='hours since 1970-01-01 00:00:00', calendar='proleptic_gregorian')
    ds.attrs.update(source=DATASET, temporal_convention='Interval end UTC; fluxes from differences of 6,9,12,15,18 h accumulations; states are forecast values at interval end', forecast_cycles='00/12 UTC', converter_version=VERSION, numerical_correction_audit=json.dumps(numerical_audit), accumulation_correction='Bounded negative increments; redistribute correction over positive increments in same forecast cycle to retain net accumulation where nonnegative')
    return ds

@R.acquisition_handlers.add('CARRA2')
class CARRA2Acquirer(BaseAcquisitionHandler):
    """Download CARRA2 single-level forecasts and write SYMFLUENCE forcing NetCDF."""

    def download(self, output_dir: Path) -> Path:
        import cdsapi
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.bbox:
            raise ValueError('CARRA2 requires a bounding box')
        area = [self.bbox[k] for k in ("lat_max","lon_min","lat_min","lon_max")]
        expected = pd.date_range(self.start_date,self.end_date,freq='3h')
        if expected.empty or expected[-1] != self.end_date:
            raise ValueError('End time must lie on three-hour grid')
        tag = hashlib.sha256(json.dumps([str(self.start_date),str(self.end_date),area,VARIABLES,VERSION]).encode()).hexdigest()[:16]
        final = output_dir/f'CARRA2_{tag}.nc'
        if final.exists():
            with xr.open_dataset(final) as ds:
                if pd.DatetimeIndex(ds.time.values).equals(expected):
                    return final
        cache = output_dir/'carra2_requests'
        cache.mkdir(exist_ok=True)
        client = None
        chunks = []
        # Weekly output blocks bound memory. Requests are split at month boundaries.
        for i in range(0,len(expected),56):
            ts = expected[i:i+56]
            days = pd.date_range(ts[0].normalize()-pd.Timedelta(days=1),ts[-1].normalize())
            paths = []
            for _,group in pd.Series(days,index=days).groupby(days.to_period('M')):
                request = request_for_days(group.values,area)
                key = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:16]
                raw = cache/f'{key}.grib'
                (cache/f'{key}.json').write_text(json.dumps(request,indent=2))
                if not raw.exists():
                    temp = raw.with_suffix('.part')
                    self.logger.info('Downloading CARRA2 %s', key)
                    if client is None:
                        client = cdsapi.Client()
                    client.retrieve(DATASET,request,str(temp))
                    temp.replace(raw)
                paths.append(raw)
            combined = cache/'block.grib'
            with combined.open('wb') as target:
                for p in paths:
                    with p.open('rb') as source:
                        import shutil
                        shutil.copyfileobj(source,target)
            ds = convert(combined,ts[0],ts[-1],area=area)
            chunk = cache/f'{tag}_{i}.nc'
            ds.to_netcdf(chunk)
            chunks.append(chunk)
            combined.unlink()
        with xr.open_mfdataset(chunks,combine='by_coords') as ds:
            if not pd.DatetimeIndex(ds.time.values).equals(expected):
                raise ValueError('Incomplete output time coverage')
            temporary = final.with_suffix('.part.nc')
            ds.to_netcdf(temporary)
        temporary.replace(final)
        return final
