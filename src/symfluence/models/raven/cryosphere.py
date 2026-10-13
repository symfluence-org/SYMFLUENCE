"""Calibration-only annual glacier-storage constraint for native Raven."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def glacier_metrics(output, geometry, observations, period, scale=1.0):
    """Compare glacier ice + snow + liquid storage changes with observed balance.

    Uses September-end daily-average states and observed annual glacier area.
    This approximate observation operator is not an exact field-date balance.
    Both endpoints must lie inside calibration; no fractional snow proxy is used.
    """
    start, end = [pd.Timestamp(x.strip()) for x in period.split(',')]
    if start > end or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid calibration interval or glacier error scale')
    hrus = json.loads(Path(geometry).read_text())['hrus']
    weights = np.array([h['area_km2'] if h['ice'] else 0.0 for h in hrus])
    if not np.isfinite(weights).all() or weights.sum() <= 0 or (weights < 0).any():
        raise ValueError('Finite positive glacier area required')
    storage = None
    for variable in ['GLACIER_ICE', 'SNOW', 'SNOW_LIQ', 'GLACIER']:
        p = Path(output)/f'{variable}_Daily_Average_ByHRU.csv'
        header = pd.read_csv(p, header=None, nrows=1).iloc[0, 2:2+len(hrus)].astype(int)
        if header.tolist() != [h['id'] for h in hrus]:
            raise ValueError('Raven HRU order differs from glacier geometry')
        f = pd.read_csv(p, skiprows=2, header=None)
        s = pd.Series(f.iloc[:, 2:2+len(hrus)].to_numpy(float) @ weights,
                      index=pd.to_datetime(f.iloc[:, 1]))
        if s.index.has_duplicates or not np.isfinite(s).all():
            raise ValueError('Invalid glacier state output')
        if storage is not None and not storage.index.equals(s.index):
            raise ValueError('Glacier component timestamps differ')
        storage = s if storage is None else storage+s
    observed = pd.read_csv(observations, sep=';', index_col=0)
    rows = []
    for label, row in observed.iterrows():
        year = int(str(label))
        left, right = pd.Timestamp(year-1, 9, 30), pd.Timestamp(year, 9, 30)
        if left < start or right > end:
            continue
        if not np.isfinite(row.annual_net_MB) or not np.isfinite(row.g_area_dyn) or row.g_area_dyn <= 0:
            continue
        if left not in storage.index or right not in storage.index:
            raise ValueError(f'Missing glacier-balance endpoint for {year}')
        value = (storage.loc[right]-storage.loc[left])/row.g_area_dyn/1000
        rows.append(dict(year=year, simulated_mwe=float(value), observed_mwe=float(row.annual_net_MB)))
    if not rows:
        raise ValueError('No complete glacier-balance years inside calibration')
    rmse = float(np.sqrt(np.mean([(r['simulated_mwe']-r['observed_mwe'])**2 for r in rows])))
    result = dict(glacier_score=1-rmse/scale, glacier_mb_rmse_mwe=rmse, glacier_balance_year_count=len(rows))
    (Path(output)/'cryosphere_metrics.json').write_text(json.dumps(dict(period=period, metrics=result, annual=rows), indent=2))
    return result
