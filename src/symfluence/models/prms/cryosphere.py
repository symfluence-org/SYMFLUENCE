"""Calibration-only snow/annual glacier constraints for fixed pure-ice HRUs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_constraints(statvar, parameters, project_dir, period, snow_scale=.2, mb_scale=1.):
    if not period or snow_scale <= 0 or mb_scale <= 0:
        raise ValueError('Explicit period and positive cryosphere scales required')
    start, end = [pd.Timestamp(x.strip()) for x in period.split(',')]
    if start > end:
        raise ValueError('Reversed cryosphere period')
    params = {}
    for block in Path(parameters).read_text().split('** Parameters **')[1].split('####'):
        r = block.strip().splitlines()
        if r:
            params[r[0]] = np.asarray(r[4 + int(r[1]):], float)
    area = params['hru_area'] / 247.105381467
    ice = params['glacier_frac_init']
    if not np.isfinite(area).all() or (area <= 0).any() or not np.isin(ice, [0, 1]).all() or not 0 < np.dot(area, ice) < area.sum():
        raise ValueError('Finite areas and separate pure ice/land HRUs required')
    lines = Path(statvar).read_text().splitlines()
    n = int(lines[0]); v = np.loadtxt(statvar, skiprows=n + 1, ndmin=2)
    dates = pd.to_datetime(dict(year=v[:, 1].astype(int), month=v[:, 2].astype(int), day=v[:, 3].astype(int)))
    f = pd.DataFrame(v[:, 7:], columns=['_'.join(x.split()) for x in lines[1:n + 1]], index=dates)
    if f.index.has_duplicates:
        raise ValueError('Duplicate PRMS dates')
    f = f.loc[start:end]
    snow = f[[f'snowcov_area_{i+1}' for i in range(len(area))]]
    fractions = f[[f'glacier_frac_{i+1}' for i in range(len(area))]]
    if not np.isfinite(snow.to_numpy()).all() or not snow.ge(0).all().all() or not snow.le(1.001).all().all() or not np.allclose(fractions, ice):
        raise ValueError('Invalid snow cover or changing glacier fractions')
    modeled = pd.Series(snow.to_numpy() @ (area * (1-ice)) / np.dot(area, 1-ice), index=f.index)
    observations = Path(project_dir) / 'data/observations/cryosphere'
    o = pd.read_csv(observations / 'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';')
    observed = pd.Series(o.fsca_outside_glaciers.to_numpy()/100, index=pd.to_datetime(dict(year=o.YYYY, month=o.MM, day=o.DD)))
    pair = pd.concat([modeled.rename('s'), observed.loc[start:end].rename('o')], axis=1).dropna()
    pair = pair[pair.o.between(0, 1)]
    if len(pair) < 30:
        raise ValueError('Fewer than 30 paired snow observations')
    annual = pd.read_csv(observations / 'glacier_timeseries.csv', sep=';', index_col=0)
    years = []
    for y, row in annual.iterrows():
        left, right = pd.Timestamp(int(y)-1, 9, 30), pd.Timestamp(int(y), 9, 30)
        if left < start or right > end:
            continue
        if not np.isfinite(row.annual_net_MB) or not np.isfinite(row.g_area_dyn) or row.g_area_dyn <= 0:
            continue
        if left not in f.index or right not in f.index:
            raise ValueError(f'Missing PRMS balance endpoint {y}')
        sim = (f.loc[right, 'basin_gl_storage_1']-f.loc[left, 'basin_gl_storage_1'])*.0254*area.sum()/row.g_area_dyn
        if not np.isfinite(sim):
            raise ValueError('Nonfinite glacier balance')
        years.append(dict(year=int(y), simulated_mwe=float(sim), observed_mwe=float(row.annual_net_MB)))
    if not years:
        raise ValueError('No complete calibration glacier years')
    snow_rmse = float(np.sqrt(((pair.s-pair.o)**2).mean()))
    mb_rmse = float(np.sqrt(np.mean([(y['simulated_mwe']-y['observed_mwe'])**2 for y in years])))
    metrics = dict(snow_score=1-snow_rmse/snow_scale, glacier_score=1-mb_rmse/mb_scale,
                   snow_cover_rmse=snow_rmse, glacier_mb_rmse_mwe=mb_rmse,
                   snow_observation_count=len(pair), glacier_balance_year_count=len(years))
    Path(statvar).with_name('cryosphere_metrics.json').write_text(json.dumps(dict(period=period, metrics=metrics, annual_glacier_balance=years,
        caveat='Fixed September endpoints; glacier-reservoir change on observed area; objective scales are not observation uncertainties.'), indent=2))
    return metrics
