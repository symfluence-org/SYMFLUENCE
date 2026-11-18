"""Small data checks shared by Intelligent Rivers notebook 04e and its tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.metrics import kge, nse, pbias

# LSTM inputs: meteorology only, no model states or discharge. Units as SUMMA writes them.
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


def daily_summa_forcing(project: Path, config) -> pd.DataFrame:
    """Daily means of SUMMA's hourly input weather (one HRU), for the native LSTM.

    Precipitation stays a mean rate (kg m-2 s-1); multiply by 86400 for mm/day.
    A missing hour raises an error instead of quietly shortening a day.
    """
    files = sorted((project / 'data' / 'forcing' / 'SUMMA_input').glob('*.nc'))
    if not files:
        raise FileNotFoundError('No SUMMA forcing yet. Run the preprocessing stages first.')
    with xr.open_mfdataset(files) as ds:
        hourly = ds[list(FORCING_UNITS)].isel(hru=0).to_dataframe()[list(FORCING_UNITS)]
    start, end = pd.Timestamp(config.domain.time_start), pd.Timestamp(config.domain.time_end)
    hourly = hourly.loc[start:end]
    expected = pd.date_range(start, end, freq=pd.Timedelta(seconds=int(config.forcing.time_step_size)))
    if not hourly.index.equals(expected) or hourly.isna().any().any():
        raise ValueError('Weather must cover the experiment with no missing timesteps or values.')
    return hourly.resample('D').mean()


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
