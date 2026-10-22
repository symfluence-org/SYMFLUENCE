"""Scaling must not learn from spinup, validation or evaluation targets."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

torch = pytest.importorskip('torch')
from symfluence.models.lstm.preprocessor import LSTMPreProcessor


def test_explicit_training_dates_exclude_outside_extremes():
    p = LSTMPreProcessor.__new__(LSTMPreProcessor)
    p.logger = logging.getLogger('test.lstm.scaling')
    p.spatial_mode = 'lumped'
    p.device = torch.device('cpu')
    p.lookback = 2
    p.feature_scaler, p.target_scaler = StandardScaler(), StandardScaler()
    dates = pd.date_range('2012-09-28', periods=10, name='time')
    fit_dates = dates[3:6]
    values = np.array([10000., 10000., 10000., 1., 2., 3., 99999., 99999., 99999., 99999.])
    forcing = pd.DataFrame({'air_temperature': values}, index=pd.MultiIndex.from_product([dates, [1]], names=['time','hruId']))
    obs = pd.DataFrame({'streamflow': values * 2}, index=dates)
    X, y, common, _, _ = p.process_data(forcing, obs, scaler_fit_dates=fit_dates)
    assert p.feature_scaler.mean_[0] == pytest.approx(2.)
    assert p.target_scaler.mean_[0] == pytest.approx(4.)
    assert len(X) == len(y) == 8  # retain later dates for independent inference
    assert common[-1] == dates[-1]
