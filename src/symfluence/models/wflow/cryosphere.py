"""Calibration-only spatial snow and glacier constraints for Wflow."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigurationError, EvaluationError, ModelExecutionError


def calculate_constraints(output_dir, settings_dir, project_dir, period,
                          snow_scale=.2, mb_scale=1., snow_threshold=5.):
    start, end = [pd.Timestamp(x.strip()) for x in period.split(',')]
    if start > end or min(snow_scale, mb_scale, snow_threshold) <= 0:
        raise ConfigurationError('Explicit ordered calibration period and positive scales required')
    with xr.open_dataset(Path(settings_dir) / 'wflow_staticmaps.nc') as source:
        maps = source.load().sortby('y').sortby('x')
    with xr.open_dataset(Path(output_dir) / 'cryosphere.nc') as source:
        ds = source.sel(time=slice(start, end + pd.Timedelta(days=1))).load().sortby('y').sortby('x')
    dates = pd.DatetimeIndex(ds.time.values) - pd.Timedelta(days=1)
    # Include the boundary state immediately before the first calibration day.
    if not dates.equals(pd.date_range(start - pd.Timedelta(days=1), end)):
        raise ModelExecutionError('Incomplete cryosphere calibration output coverage')
    if not (np.array_equal(ds.x, maps.x) and np.array_equal(ds.y, maps.y)):
        raise ModelExecutionError('Cryosphere output grid differs from static maps')
    mask = np.isfinite(maps.wflow_subcatch.values)
    area = maps.wflow_cellarea.values[mask] / 1e6
    fraction = maps.wflow_glacierfrac.values[mask]
    if not ((area > 0).all() and np.isfinite(area).all() and
            np.isfinite(fraction).all() and ((fraction >= 0) & (fraction <= 1)).all() and
            0 < np.dot(area, fraction) < area.sum()):
        raise ConfigurationError('Invalid glacier areas')
    def values(name):
        v = ds[name].transpose('time', 'y', 'x').values[:, mask]
        if not np.isfinite(v).all() or (v < -.001).any():
            raise ModelExecutionError(f'Invalid cryosphere state {name}')
        return v
    snow = values('snow') + values('snowwater')
    ice = values('ice')
    snow_series = pd.Series((snow > snow_threshold) @ (area*(1-fraction)) /
                           np.dot(area, 1-fraction), index=dates).loc[start:end]
    obsdir = Path(project_dir) / 'data/observations/cryosphere'
    observations = pd.read_csv(obsdir / 'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';')
    obs = pd.Series(observations.fsca_outside_glaciers.to_numpy()/100,
                    index=pd.to_datetime(dict(year=observations.YYYY, month=observations.MM, day=observations.DD)))
    if obs.index.has_duplicates:
        raise EvaluationError('Duplicate snow observation dates')
    pair = pd.concat({'sim': snow_series, 'obs': obs.loc[start:end]}, axis=1).dropna()
    pair = pair[pair.obs.between(0, 1)]
    if len(pair) < 30:
        raise EvaluationError('Fewer than 30 paired calibration snow observations')
    glacier = pd.Series((ice+snow) @ (area*fraction), index=dates)  # mm km2
    annual = pd.read_csv(obsdir / 'glacier_timeseries.csv', sep=';', index_col=0)
    years = []
    for label, row in annual.iterrows():
        year = int(str(label))
        left, right = pd.Timestamp(year-1, 9, 30), pd.Timestamp(year, 9, 30)
        if left < start - pd.Timedelta(days=1) or right > end:
            continue
        if not np.isfinite(row.annual_net_MB) or not np.isfinite(row.g_area_dyn) or row.g_area_dyn <= 0:
            continue
        if left not in glacier.index or right not in glacier.index:
            raise EvaluationError(f'Missing glacier balance endpoint: {year}')
        sim = (glacier.loc[right] - glacier.loc[left]) / row.g_area_dyn / 1000
        years.append(dict(year=year, simulated_mwe=float(sim), observed_mwe=float(row.annual_net_MB)))
    if not years:
        raise EvaluationError('No complete calibration glacier-balance years')
    snow_rmse = float(np.sqrt(((pair.sim-pair.obs)**2).mean()))
    mb_rmse = float(np.sqrt(np.mean([(r['simulated_mwe']-r['observed_mwe'])**2 for r in years])))
    metrics = dict(snow_score=1-snow_rmse/snow_scale, glacier_score=1-mb_rmse/mb_scale,
                   snow_cover_rmse=snow_rmse, glacier_mb_rmse_mwe=mb_rmse,
                   snow_observation_count=len(pair), glacier_balance_year_count=len(years))
    (Path(output_dir) / 'cryosphere_metrics.json').write_text(json.dumps(dict(
        period=period, metrics=metrics, annual_glacier_balance=years,
        snow_presence_threshold_mm=snow_threshold,
        caveat='Snow presence proxy; fixed glacier fractions; shared snow state; objective scales are not observational uncertainty.'), indent=2))
    return metrics
