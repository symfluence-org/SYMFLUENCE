from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import EvaluationError
from symfluence.models.wflow.calibration.seasonal import seasonal_bias_score


def test_seasonal_errors_do_not_cancel():
    dates = pd.date_range('2012-10-01', '2014-09-30')
    obs = pd.Series(10., index=dates)
    sim = obs.copy()
    sim.loc[sim.index.month.isin([4, 5, 6])] *= 1.3
    sim.loc[sim.index.month.isin([7, 8, 9])] *= .7
    assert seasonal_bias_score(sim, obs) == pytest.approx(.8)
    assert seasonal_bias_score(obs, obs) == 1


def test_missing_and_partial_seasons_rejected():
    obs = pd.Series(1., index=pd.date_range('2012-10-01', '2013-09-30'))
    with pytest.raises(EvaluationError):
        seasonal_bias_score(obs.drop(obs.index[12]), obs.drop(obs.index[12]))
    with pytest.raises(EvaluationError):
        seasonal_bias_score(obs.iloc[1:], obs.iloc[1:])
    missing = obs.copy(); missing.iloc[4] = np.nan
    with pytest.raises(EvaluationError):
        seasonal_bias_score(missing, obs)
