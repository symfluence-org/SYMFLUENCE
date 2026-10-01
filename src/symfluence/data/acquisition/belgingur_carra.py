# SPDX-License-Identifier: GPL-3.0-or-later
"""Belgingur's CARRA forecast archive, normalized for native preprocessing.

No forecast-lead or interval shift is inferred: timestamps are retained as
published. Precipitation and radiation are already rates, not accumulations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .base import BaseAcquisitionHandler

BASE_URL = 'http://ftp.betravedur.is/LV/icebox/carra/'
# source name, canonical name, accepted unit, scale, offset, canonical unit
VARIABLES = [
    ('air_temperature_at_2m_agl', 'air_temperature', 'C', 1., 273.15, 'K'),
    ('air_pressure_at_surface', 'surface_air_pressure', 'hPa', 100., 0., 'Pa'),
    ('lwe_precipitation_rate', 'precipitation_flux', 'mm hr-1', 1/3600., 0., 'kg m-2 s-1'),
    ('relative_humidity_at_2m_agl', 'relative_humidity', '%', 1., 0., '%'),
    ('wind_speed_at_10m_agl', 'wind_speed', 'm s-1', 1., 0., 'm s-1'),
    ('downward_shortwave_flux', 'surface_downwelling_shortwave_flux', 'W m-2', 1., 0., 'W m-2'),
    ('downward_longwave_flux', 'surface_downwelling_longwave_flux', 'W m-2', 1., 0., 'W m-2'),
]


def normalize_belgingur(ds, bbox, start, end, source_url):
    """Subset the native Lambert grid and convert units without resampling."""
    times = pd.to_datetime(ds.Times.values.astype(str), format='%Y-%m-%d_%H:%M:%S')
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Belgingur timestamps must be unique and increasing')
    if len(times) > 1 and not (np.diff(times.values) == np.timedelta64(3, 'h')).all():
        raise ValueError('Belgingur archive must have a continuous three-hour clock')
    lat, lon = ds.XLAT.values, ds.XLONG.values
    if 'MAP_PROJ4_STR' not in ds.attrs:
        raise ValueError('Missing native-grid projection metadata')
    mask = ((lat >= bbox['lat_min']-.1) & (lat <= bbox['lat_max']+.1)
            & (lon >= bbox['lon_min']-.1) & (lon <= bbox['lon_max']+.1))
    rows, cols = np.where(mask)
    if not len(rows):
        raise ValueError('Belgingur Iceland grid does not intersect the requested domain')
    d = ds.isel(south_north=slice(rows.min(), rows.max()+1),
                west_east=slice(cols.min(), cols.max()+1))
    use_time = np.flatnonzero((times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end)))
    if not len(use_time):
        raise ValueError('No Belgingur timestamps within requested period')
    d = d.isel(Time=use_time)
    out = xr.Dataset(coords={
        'time': times[use_time],
        'latitude': (('y', 'x'), d.XLAT.values),
        'longitude': (('y', 'x'), d.XLONG.values),
    })
    out.latitude.attrs['units'] = 'degrees_north'
    out.longitude.attrs['units'] = 'degrees_east'
    for src, dest, expected, scale, offset, units in VARIABLES:
        actual = d[src].attrs.get('units')
        if actual != expected:
            raise ValueError(f'{src}: expected units {expected!r}, got {actual!r}')
        data = np.asarray(d[src].transpose('Time', 'south_north', 'west_east').values)*scale+offset
        if not np.isfinite(data).all():
            raise ValueError(f'{src}: nonfinite data in requested spatial subset')
        out[dest] = (('time', 'y', 'x'), data.astype('float32'))
        out[dest].attrs = {'units': units, 'source_variable': src}
    # The archive omits q units. Derive q from documented T/RH/P units instead,
    # using the same Magnus convention as the CDS adapter. Do not silently guess.
    tc = out.air_temperature - 273.15
    vapor = out.relative_humidity/100 * 611.2*np.exp(17.67*tc/(tc+243.5))
    out['specific_humidity'] = .622*vapor/(out.surface_air_pressure-.378*vapor)
    out.specific_humidity.attrs = {'units': 'kg kg-1', 'derivation': 'Magnus from temperature, RH and surface pressure; archive q has no units'}
    out.attrs = {k: v for k, v in ds.attrs.items() if k in ('MAP_PROJ4_STR', 'DX', 'DY', 'TITLE', 'HISTORY')}
    out.attrs.update(forcing_source='belgingur', source_url=source_url,
                     temporal_convention='Published timestamps unchanged; rate interval alignment and forecast leads unconfirmed',
                     precipitation_conversion='Published mm/hour divided by 3600; no deaccumulation')
    return out


class BelgingurCARRAAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Do not combine a CDS series with a Belgingur series in one raw directory.
        for path in output_dir.glob('*.nc'):
            with xr.open_dataset(path) as existing:
                if existing.attrs.get('forcing_source') != 'belgingur':
                    raise ValueError(f'Refusing to mix Belgingur forcing with another source: {path}')
        for month in pd.period_range(self.start_date, self.end_date, freq='M'):
            stamp = month.strftime('%Y%m')
            final = output_dir / f'{self.domain_name}_CARRA_BELGINGUR_{stamp}.nc'
            first = max(self.start_date, month.start_time)
            last = min(self.end_date, month.end_time.floor('3h'))
            expected = pd.date_range(first, last, freq='3h')
            if final.exists():
                with xr.open_dataset(final) as cached:
                    if pd.DatetimeIndex(cached.time.values).equals(expected):
                        self.logger.info('Reusing Belgingur month %s', stamp)
                        continue
            url = BASE_URL + f'{stamp}-carra-sfc_wod.nc'
            archive = output_dir / f'.belgingur_{stamp}.download'
            temporary = final.with_suffix('.nc.partial')
            self.logger.info('Downloading Belgingur CARRA %s', url)
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with archive.open('wb') as stream:
                        for block in response.iter_content(1024*1024):
                            stream.write(block)
                with xr.open_dataset(archive) as source:
                    result = normalize_belgingur(source, self.bbox, first, last, url)
                if not pd.DatetimeIndex(result.time.values).equals(expected):
                    raise ValueError(f'Incomplete requested time coverage for {stamp}')
                result.to_netcdf(temporary, encoding={v: {'zlib': True, 'complevel': 4} for v in result.data_vars})
                result.close()
                temporary.replace(final)
            finally:
                archive.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
        return output_dir
