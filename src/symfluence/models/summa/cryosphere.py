# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration-only annual glacier mass-balance constraint for SUMMA glacier mode."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigurationError, EvaluationError, ModelExecutionError


def glacier_metrics(output_dir, observations, period, scale=1.0):
    """Compare simulated annual (Oct-Sep) glacier storage change with observed net balance.

    ``basin__GlacierStorage`` (Gt; ice, firn and snow on the glacier domains) is read
    from the SUMMA output in ``output_dir``. The change between consecutive
    30 September end-of-day states is divided by the observed dynamic glacier area
    (``g_area_dyn``, km2) and compared with ``annual_net_MB`` (m w.e.). Both endpoints
    must lie inside ``period``. This is an approximate observation operator (model
    September-end states, observed area), not an exact field-date balance.

    Returns ``glacier_score = 1 - RMSE/scale``, the RMSE and the year count, and
    writes ``cryosphere_metrics.json`` next to the output.
    """
    start, end = [pd.Timestamp(x.strip()) for x in str(period).split(',')]
    if start > end or not np.isfinite(scale) or scale <= 0:
        raise ConfigurationError('Invalid calibration interval or glacier error scale')
    storage = None
    for path in sorted(Path(output_dir).glob('*.nc')):
        with xr.open_dataset(path) as ds:
            if 'basin__GlacierStorage' in ds:
                storage = ds['basin__GlacierStorage'].sum('gru').to_series()
                break
    if storage is None:
        raise ModelExecutionError(f'basin__GlacierStorage not found in SUMMA output under {output_dir}')
    storage = storage[~storage.index.duplicated()].dropna()
    daily = storage.groupby(storage.index.normalize()).last()
    if not np.isfinite(daily).all():
        raise ModelExecutionError('Invalid glacier storage output')
    observed = pd.read_csv(observations, sep=';', index_col=0)
    rows = []
    for year, row in observed.iterrows():
        left, right = pd.Timestamp(int(year) - 1, 9, 30), pd.Timestamp(int(year), 9, 30)
        if left < start - pd.Timedelta(days=1) or right > end:
            continue
        if not np.isfinite(row.annual_net_MB) or not np.isfinite(row.g_area_dyn) or row.g_area_dyn <= 0:
            continue
        if left not in daily.index or right not in daily.index:
            raise EvaluationError(f'Missing glacier-balance endpoint for {year}')
        value = (daily.loc[right] - daily.loc[left]) * 1e3 / row.g_area_dyn  # Gt / km2 -> m w.e.
        rows.append(dict(year=int(year), simulated_mwe=float(value), observed_mwe=float(row.annual_net_MB)))
    if not rows:
        raise EvaluationError('No complete glacier-balance years inside calibration')
    rmse = float(np.sqrt(np.mean([(r['simulated_mwe'] - r['observed_mwe']) ** 2 for r in rows])))
    result = dict(glacier_score=1 - rmse / scale, glacier_mb_rmse_mwe=rmse, glacier_balance_year_count=len(rows))
    (Path(output_dir) / 'cryosphere_metrics.json').write_text(
        json.dumps(dict(period=str(period), metrics=result, annual=rows), indent=2))
    return result
