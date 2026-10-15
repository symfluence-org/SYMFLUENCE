"""Snow and glacier constraints for explicit finite-ice MESH response units.

Snow uses native fractional cover outside ice, including mapped water proxies.
Only April–June and August–September are scored because pixel QA is unavailable.
Annual glacier change includes bulk ice AND exact boundary snow/liquid storage.
Hourly storage is an endpoint only for the supported 60-minute native timestep.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import MESHConfigurationError, MESHInputError, MESHOutputError
from .glacier_runtime import _run_dates, check_finite_output


def _field(root, name, dates, count):
    path = root / name
    if not path.exists():
        path = root / 'results' / name
    frame = pd.read_csv(path, header=None, index_col=0).dropna(axis=1, how='all')
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if not frame.index.equals(dates) or frame.shape[1] != count + 1:
        raise MESHOutputError(f'Incomplete or misaligned cryosphere output: {name}')
    if not np.isfinite(frame.to_numpy()).all():
        raise MESHOutputError(f'Nonfinite cryosphere output: {name}')
    return frame.iloc[:, :count]


def calculate_constraints(output_dir, observation_dir, period, snow_scale=.2, mb_scale=1.):
    """Return soft constraint scores; missing/invalid evidence raises, never skips.

    Scoring requires complete October–September years and observations for every
    selected year. Use calibration years only for parameter selection. Reported
    annual glacier area is an alternative support, not modeled evolving geometry;
    the larger RMSE across fixed and reported area prevents favorable selection.
    """
    root = Path(output_dir)
    (root / 'cryosphere_metrics.json').unlink(missing_ok=True)
    obsdir = Path(observation_dir)
    if not np.isfinite([snow_scale, mb_scale]).all() or min(snow_scale, mb_scale) <= 0:
        raise MESHConfigurationError('Cryosphere error scales must be finite and positive')
    start, end = [pd.Timestamp(x.strip()) for x in str(period).split(',')]
    if (start.month, start.day, end.month, end.day) != (10, 1, 9, 30) or end <= start:
        raise MESHConfigurationError('Cryosphere scoring requires complete October–September years')
    check_finite_output(root)
    options = (root / 'MESH_input_run_options.ini').read_text()
    step = re.search(r'^TIMESTEPFLAG\s+(\d+)', options, re.MULTILINE)
    if not step or int(step[1]) != 60:
        raise MESHConfigurationError('Cryosphere endpoints require a 60-minute native timestep')
    run_start, run_end = [pd.Timestamp(datetime(y, 1, 1) + timedelta(days=d-1, hours=h, minutes=m))
                          for y, d, h, m in _run_dates(root)]
    if run_start > start - pd.Timedelta(hours=1) or run_end < end + pd.Timedelta(days=1):
        raise MESHConfigurationError('Simulation must include preceding glacier boundary and full scoring period')
    hours = pd.date_range(run_start, run_end, freq='h', inclusive='left')
    days = pd.date_range(run_start.normalize(), run_end-pd.Timedelta(days=1), freq='D')
    units = pd.DataFrame(json.loads((root / 'spatial_units.json').read_text())['units'])
    area = units.area_m2.to_numpy(float)
    ice = units.surface.eq('ice').to_numpy()
    if not np.isfinite(area).all() or (area <= 0).any() or not ice.any() or ice.all():
        raise MESHConfigurationError('Constraints require positive, finite land and ice areas')
    n = len(units)
    fsno = _field(root, 'FSNO_D.csv', days, n)
    if fsno.to_numpy().min() < -1e-5 or fsno.to_numpy().max() > 1.00001:
        raise MESHInputError('Invalid snow fractions')
    solid = _field(root, 'SNO_H.csv', hours, n)
    liquid = _field(root, 'LQWSSNO_H.csv', hours, n)
    if min(solid.to_numpy().min(), liquid.to_numpy().min()) < -1e-5:
        raise MESHOutputError('Negative snow storage')
    snow = solid + liquid
    model_sca = pd.Series(fsno.to_numpy()[:, ~ice] @ area[~ice] / area[~ice].sum(), index=days)
    observed = pd.read_csv(obsdir / 'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';')
    dates = pd.to_datetime(dict(year=observed.YYYY, month=observed.MM, day=observed.DD))
    sca_obs = pd.Series(observed.fsca_outside_glaciers.to_numpy(float)/100, index=dates)
    if not sca_obs.index.is_unique:
        raise MESHInputError('Duplicate snow observation dates')
    pairs = pd.concat({'sim': model_sca, 'obs': sca_obs}, axis=1).loc[start:end].dropna()
    pairs = pairs[pairs.obs.between(0, 1)]
    years = list(range(start.year+1, end.year+1))
    snow_metrics = {}
    season_errors = []
    for label, months in [('spring', [4, 5, 6]), ('late_summer', [8, 9])]:
        selected = pairs[pairs.index.month.isin(months)]
        for year in years:
            if (selected.index.year == year).sum() < 20:
                raise MESHInputError(f'Insufficient {label} snow observations in WY{year}')
        mse = float(((selected.sim-selected.obs)**2).mean())
        season_errors.append(mse)
        snow_metrics[label] = {'n': len(selected), 'rmse_fraction': float(np.sqrt(mse)),
                               'bias_fraction': float((selected.sim-selected.obs).mean())}
    snow_rmse = float(np.sqrt(np.mean(season_errors)))
    ledger = pd.read_csv(root / 'MESH_glacier_ledger.csv')
    ranks = np.flatnonzero(ice) + 1
    if set(ledger.tile.unique()) != set(ranks):
        raise MESHOutputError('Ice ledger tiles do not match spatial ice units')
    bulk = {}
    for rank in ranks:
        rows = ledger[ledger.tile == rank]
        if not np.array_equal(rows.step.to_numpy(), np.arange(1, len(hours)+1)):
            raise MESHOutputError('Incomplete or unordered glacier steps')
        bulk[rank] = pd.Series(rows.end_kgm2.to_numpy(), index=hours)
    mb = pd.read_csv(obsdir / 'glacier_timeseries.csv', sep=';', index_col=0)
    if not mb.index.is_unique:
        raise MESHInputError('Duplicate glacier observation years')
    annual = []
    for year in years:
        before = pd.Timestamp(year-1, 10, 1)-pd.Timedelta(hours=1)
        last = pd.Timestamp(year, 10, 1)-pd.Timedelta(hours=1)
        delta = np.array([bulk[rank].loc[last]-bulk[rank].loc[before]
                          + snow.loc[last].iloc[rank-1]-snow.loc[before].iloc[rank-1] for rank in ranks])
        observed = mb.reindex([year])[['annual_net_MB', 'g_area_dyn']].to_numpy(float)[0]
        if not np.isfinite(observed).all() or observed[1] <= 0:
            raise MESHInputError(f'Missing valid annual glacier observations for WY{year}')
        volume = float(delta @ area[ice] / 1000)
        annual.append({'year': year, 'fixed_area_mwe': volume / area[ice].sum(),
                       'reported_area_mwe': volume / (mb.loc[year, 'g_area_dyn']*1e6),
                       'observed_mwe': float(mb.loc[year, 'annual_net_MB'])})
    annual_frame = pd.DataFrame(annual)
    fixed_rmse = float(np.sqrt(((annual_frame.fixed_area_mwe-annual_frame.observed_mwe)**2).mean()))
    reported_rmse = float(np.sqrt(((annual_frame.reported_area_mwe-annual_frame.observed_mwe)**2).mean()))
    mb_rmse = max(fixed_rmse, reported_rmse)
    metrics = {'snow_score': 1-snow_rmse/snow_scale, 'glacier_score': 1-mb_rmse/mb_scale,
               'snow_rmse': snow_rmse, 'glacier_mb_rmse': mb_rmse,
               'glacier_fixed_area_rmse': fixed_rmse, 'glacier_reported_area_rmse': reported_rmse}
    report = dict(metrics, period=str(period), snow_seasons=snow_metrics, glacier_years=annual,
                  snow_season_weights={'spring': .5, 'late_summer': .5},
                  snow_error_scale=snow_scale, glacier_error_scale=mb_scale,
                  glacier_support='larger RMSE of fixed and reported annual areas',
                  pixel_QA_available=False)
    (root / 'cryosphere_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return metrics
