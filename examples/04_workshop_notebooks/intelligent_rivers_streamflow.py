"""Small data checks shared by Intelligent Rivers notebook 04e and its tests."""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.metrics import kge, nse, pbias
from symfluence.core.mixins.project import resolve_data_subdir

# Keep only meteorological variables; no model states or discharge as predictors.
FORCING_UNITS = {
    'airtemp': {'K', 'kelvin'},
    'pptrate': {'kg m-2 s-1', 'kg m^-2 s^-1'},
    'SWRadAtm': {'W m-2', 'W m^-2'},
    'LWRadAtm': {'W m-2', 'W m^-2'},
    'airpres': {'Pa'},
    'spechum': {'kg kg-1', 'kg kg^-1', 'g g-1'},
    'windspd': {'m s-1', 'm s^-1'},
}


def load_bundle(folder: Path):
    """Read the checksum-verified Ayilam subset; retain zeros and reject gaps."""
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'CAMELS subset checksum mismatch: {name}')
    basin = gpd.read_file(folder / 'catchment.geojson').to_crs(4326)
    obs = pd.read_csv(folder / 'streamflow_observed.csv', parse_dates=['datetime'])
    obs = obs.set_index('datetime').discharge_cms
    days = pd.date_range('2000-01-01', '2015-12-31', freq='D')
    if not obs.index.equals(days) or not np.isfinite(obs).all() or (obs < 0).any():
        raise ValueError('Ayilam observations must be complete, finite, nonnegative daily flow.')
    if len(basin) != 1 or basin.gauge_id.astype(str).iloc[0] != '15007':
        raise ValueError('Expected the single CAMELS-IND catchment 15007.')
    return basin, obs, provenance


def daily_summa_forcing(project: Path, config):
    """Read the SUMMA manifest and average complete days for the native LSTM.

    Precipitation stays a mean rate (kg m-2 s-1). Multiply by 86400 for mm/day.
    Reject incomplete weather instead of compressing the LSTM time axis.
    """
    manifest = project / 'settings' / 'SUMMA' / 'forcingFileList.txt'
    folder = resolve_data_subdir(project, 'forcing') / 'SUMMA_input'
    files = []
    for line in manifest.read_text().splitlines():
        names = shlex.split(line.split('!')[0], comments=True)
        if names:
            files.append(folder / names[0])
    if not files:
        raise FileNotFoundError('SUMMA forcing manifest is empty. Complete preprocessing.')
    parts = []
    hashes = {}
    for path in files:
        with xr.open_dataset(path) as ds:
            for name, units in FORCING_UNITS.items():
                if name not in ds or ds[name].attrs.get('units', '').strip() not in units:
                    raise ValueError(f'{path.name}: missing {name} or unsupported units.')
                if set(ds[name].dims) != {'time', 'hru'} or ds.sizes['hru'] != 1:
                    raise ValueError('04e expects one HRU with time × hru weather variables.')
            parts.append(ds[list(FORCING_UNITS)].isel(hru=0).load().to_dataframe())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hourly = pd.concat(parts).sort_index()[list(FORCING_UNITS)]
    if hourly.index.has_duplicates:
        if (hourly.groupby(level=0).nunique(dropna=False) > 1).any().any():
            raise ValueError('Conflicting weather at duplicate timestamps.')
        hourly = hourly.loc[~hourly.index.duplicated()]
    start, end = pd.Timestamp(config.domain.time_start), pd.Timestamp(config.domain.time_end)
    hourly = hourly.loc[start:end]
    timestep = int(config.forcing.time_step_size)
    if timestep <= 0 or 86400 % timestep:
        raise ValueError('Forcing timestep must divide one day.')
    expected = pd.date_range(start, end, freq=pd.Timedelta(seconds=timestep))
    if not hourly.index.equals(expected) or not np.isfinite(hourly).all().all():
        raise ValueError('Weather must cover the experiment with no missing timesteps or values.')
    if (hourly.pptrate < 0).any():
        raise ValueError('Negative precipitation needs upstream inspection.')
    counts = hourly.resample('D').count()
    if (counts != 86400 // timestep).any().any():
        raise ValueError('Each daily LSTM input must contain a full day of weather.')
    return hourly.resample('D').mean(), hashes


def write_daily_forcing(daily: pd.DataFrame, path: Path):
    """Write only the seven daily weather variables and the single HRU ID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {name: (('time', 'hru'), daily[name].to_numpy()[:, None]) for name in FORCING_UNITS},
        coords={'time': daily.index, 'hru': [0], 'hruId': ('hru', [1])},
    )
    for name, units in FORCING_UNITS.items():
        ds[name].attrs['units'] = sorted(units)[0]
    ds.attrs['aggregation'] = 'Complete daily means of SUMMA model-input meteorology; UTC dates.'
    ds.to_netcdf(path)
    ds.close()


def comparison_scores(frame: pd.DataFrame, period: str):
    """Score all predictors on the same finite observed days."""
    lo, hi = [pd.Timestamp(value.strip()) for value in period.split(',')]
    window = frame.loc[lo:hi]
    paired = window.replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 2:
        raise ValueError('At least two common observed days are needed for comparison.')
    obs = paired['Observed']
    rows = []
    for name in paired.columns.drop('Observed'):
        sim = paired[name]
        rows.append({'predictor': name, 'NSE': nse(obs, sim), 'KGE': kge(obs, sim),
                     'PBIAS (%)': pbias(obs, sim), 'MAE (m3/s)': (sim - obs).abs().mean(),
                     'negative_days': int((sim < 0).sum()), 'paired_days': len(paired),
                     'calendar_days': len(pd.date_range(lo, hi, freq='D'))})
    return pd.DataFrame(rows).set_index('predictor'), paired
