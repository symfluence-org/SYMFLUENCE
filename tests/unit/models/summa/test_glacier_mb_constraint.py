from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import EvaluationError, ModelExecutionError
from symfluence.models.summa.cryosphere import glacier_metrics


def _output(tmp_path, loss_gt_per_year):
    # Daily end-of-day storage (Gt), linear loss, two GRUs summed by the operator.
    time = pd.date_range('2012-01-01 21:00', '2016-09-30 21:00', freq='D')
    years = (time - time[0]).days / 365.25
    storage = 40. - loss_gt_per_year * years
    data = np.stack([storage * .25, storage * .75], axis=1)
    xr.Dataset({'basin__GlacierStorage': (('time', 'gru'), data)}, coords={'time': time}).to_netcdf(
        tmp_path / 'run_day.nc')
    obs = tmp_path / 'glacier.csv'
    obs.write_text(';annual_net_MB;summer_MB;winter_MB;g_frac_dyn;g_area_dyn\n'
                   + ''.join(f'{y};-2.0;;;;125.0\n' for y in range(2012, 2018)))
    return obs


def test_matching_balance_scores_one(tmp_path):
    obs = _output(tmp_path, loss_gt_per_year=0.25)  # 0.25 Gt / 125 km2 = 2.0 m w.e.
    m = glacier_metrics(tmp_path, obs, '2012-10-01, 2016-09-30')
    assert m['glacier_balance_year_count'] == 4  # WY2013-16 only
    assert m['glacier_mb_rmse_mwe'] == pytest.approx(0., abs=0.02)
    assert m['glacier_score'] == pytest.approx(1., abs=0.02)
    years = [r['year'] for r in json.loads((tmp_path / 'cryosphere_metrics.json').read_text())['annual']]
    assert years == [2013, 2014, 2015, 2016]


def test_double_melt_is_penalised(tmp_path):
    obs = _output(tmp_path, loss_gt_per_year=0.5)  # -4 m w.e. vs observed -2
    m = glacier_metrics(tmp_path, obs, '2012-10-01, 2016-09-30', scale=1.0)
    assert m['glacier_mb_rmse_mwe'] == pytest.approx(2., abs=0.03)
    assert m['glacier_score'] == pytest.approx(-1., abs=0.03)


def test_missing_variable_raises(tmp_path):
    xr.Dataset({'scalarSWE': (('time',), [1.])}, coords={'time': [pd.Timestamp('2013-01-01')]}).to_netcdf(
        tmp_path / 'x.nc')
    with pytest.raises(ModelExecutionError):
        glacier_metrics(tmp_path, tmp_path / 'none.csv', '2012-10-01, 2016-09-30')


def test_composite_score_weights_components():
    from symfluence.models.summa.calibration.worker_impl.worker_orchestration import _composite_score

    metrics = {'Calib_KGE': 0.7, 'KGE': 0.6, 'GLACIER_SCORE': 0.5}
    assert _composite_score(metrics, {'KGE': 0.8, 'GLACIER_SCORE': 0.2}) == pytest.approx(0.8 * 0.6 + 0.2 * 0.5)
    with pytest.raises(EvaluationError):
        _composite_score({'KGE': 0.6}, {'KGE': 0.8, 'GLACIER_SCORE': 0.2})
