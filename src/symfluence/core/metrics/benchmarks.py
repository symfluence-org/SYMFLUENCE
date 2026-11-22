# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Reference (benchmark) predictors for split-sample skill scores.

A model score means little on its own: a KGE of 0.5 is good for some
catchments and poor for others. These benchmarks are the usual low bars
(Knoben et al., 2019; Schaefli & Gupta, 2007), built from observations only:

* ``mean_flow``    -- the calibration-period observed mean, everywhere. Its KGE
  on the calibration period is 1 - sqrt(2) = -0.41.
* ``climatology``  -- the calibration-period observed mean for each calendar
  day (day-of-year climatology).
* ``persistence``  -- the previous observation (one native timestep earlier).

Benchmarks are *fitted on the calibration period only* and then scored on each
configured period, exactly as the model is. Skill relative to a benchmark is

    skill = (score_model - score_benchmark) / (1 - score_benchmark)

for a score whose perfect value is 1 (KGE, NSE): 1 is perfect, 0 is no better
than the benchmark and negative is worse.

Pure functions: pandas/numpy only, no config access or logging.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ['BENCHMARK_NAMES', 'benchmark_series', 'constant_benchmark_kge', 'skill_score']

BENCHMARK_NAMES: Tuple[str, ...] = ('mean_flow', 'climatology', 'persistence')


def _infer_step(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if len(index) < 2:
        return None
    diffs = pd.Series(index[1:] - index[:-1])
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return pd.Timedelta(diffs.mode().iloc[0])


def _in_period(index: pd.DatetimeIndex, period: Tuple[Any, Any]) -> np.ndarray:
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if end == end.normalize():
        return np.asarray((index >= start) & (index < end + pd.Timedelta(days=1)))
    return np.asarray((index >= start) & (index <= end))


def benchmark_series(
    obs: pd.Series,
    calibration_period: Tuple[Any, Any],
) -> Dict[str, pd.Series]:
    """Benchmark predictions on ``obs``'s index, fitted on ``calibration_period``.

    Returns ``{name: series}`` for :data:`BENCHMARK_NAMES`. A benchmark that
    cannot be built (no calibration observations) is omitted. Persistence
    uses the observation one native step earlier; across a gap it is NaN.
    """
    obs = pd.to_numeric(obs, errors='coerce').dropna().sort_index()
    if obs.empty:
        return {}
    index = pd.DatetimeIndex(obs.index)
    calib = obs[_in_period(index, calibration_period)]
    out: Dict[str, pd.Series] = {}
    if not calib.empty:
        mean = float(calib.mean())
        out['mean_flow'] = pd.Series(mean, index=obs.index, name='mean_flow')

        calib_idx = pd.DatetimeIndex(calib.index)
        clim = calib.groupby([calib_idx.month, calib_idx.day]).mean()
        keys = list(zip(index.month, index.day))
        values = np.array([clim.get(k, np.nan) for k in keys], dtype=float)
        values = np.where(np.isfinite(values), values, mean)
        out['climatology'] = pd.Series(values, index=obs.index, name='climatology')

    step = _infer_step(index)
    if step is not None:
        prev = obs.copy()
        prev.index = index + step
        out['persistence'] = prev.reindex(index).rename('persistence')
    return out


def skill_score(model_score: Any, benchmark_score: Any) -> float:
    """Skill of ``model_score`` relative to ``benchmark_score`` (perfect = 1).

    NaN when either score is missing or the benchmark is already perfect.
    """
    try:
        m, b = float(model_score), float(benchmark_score)
    except (TypeError, ValueError):
        return float('nan')
    if not (np.isfinite(m) and np.isfinite(b)) or b >= 1.0:
        return float('nan')
    return (m - b) / (1.0 - b)


def constant_benchmark_kge(beta: Any) -> float:
    """KGE of a constant predictor with bias ratio ``beta`` (mean_sim / mean_obs).

    The correlation of a constant series is undefined; by convention it is 0,
    and the variability ratio is 0, so KGE = 1 - sqrt(2 + (beta - 1)^2). With
    beta = 1 (the mean of the scored observations) this is 1 - sqrt(2) = -0.41.
    """
    try:
        b = float(beta)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(b):
        return float('nan')
    return float(1.0 - np.sqrt(2.0 + (b - 1.0) ** 2))
