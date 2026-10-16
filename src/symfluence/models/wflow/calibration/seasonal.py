"""Volume-bias score for complete, paired water-year seasonal windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError


def seasonal_bias_score(sim: pd.Series, obs: pd.Series) -> float:
    """One minus mean absolute fractional bias for Oct–Mar, Apr–Jun, Jul–Sep.

    Each complete water-year/season receives equal weight. Missing days and
    nonpositive observed seasonal volumes invalidate the score.
    """
    if not sim.index.equals(obs.index) or sim.index.has_duplicates or len(sim) == 0:
        raise EvaluationError('Seasonal bias requires matching unique daily indices')
    if not sim.index.equals(pd.date_range(sim.index[0], sim.index[-1])):
        raise EvaluationError('Seasonal bias requires complete daily coverage')
    if not np.isfinite(sim).all() or not np.isfinite(obs).all():
        raise EvaluationError('Seasonal bias requires finite paired values')
    index = sim.index
    wy = index.year + (index.month >= 10)
    season = np.where((index.month >= 10) | (index.month <= 3), 0,
                      np.where(index.month <= 6, 1, 2))
    errors = []
    for year, group in sorted(set(zip(wy, season))):
        lo, hi = [(f'{year-1}-10-01', f'{year}-03-31'),
                  (f'{year}-04-01', f'{year}-06-30'),
                  (f'{year}-07-01', f'{year}-09-30')][group]
        dates = pd.date_range(lo, hi)
        mask = (wy == year) & (season == group)
        if not index[mask].equals(dates):
            raise EvaluationError('Seasonal bias requires whole seasons')
        denominator = float(obs.loc[dates].sum())
        if denominator <= 0:
            raise EvaluationError('Seasonal observed volume must be positive')
        errors.append(abs(float(sim.loc[dates].sum()) / denominator - 1))
    return float(1 - np.mean(errors))
