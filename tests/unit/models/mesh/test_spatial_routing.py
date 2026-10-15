"""Fixed response-unit routing must preserve mass and retain the run-end tail."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.mesh.spatial_routing import route_spatial_runoff


def test_impulse_routing_retains_delayed_volume_and_rejects_loss(tmp_path):
    (tmp_path/'spatial_units.json').write_text(json.dumps({'units': [{'unit_id': 'a'}, {'unit_id': 'b'}]}))
    (tmp_path/'results').mkdir()
    xr.Dataset({'GridArea': ('subbasin', [86400000., 86400000., 0.])}).to_netcdf(tmp_path/'MESH_drainage_database.nc')
    pd.DataFrame([[1., 2., 0.], [0., 0., 0.]], index=pd.date_range('2013-01-01', periods=2)).to_csv(
        tmp_path/'results/RFF_D.csv', header=False)
    kernel = tmp_path/'kernel.csv'
    kernel.write_text('unit_id,lag_days,weight\na,0,0.25\na,1,0.75\nb,2,1\n')
    report = route_spatial_runoff(tmp_path, kernel)
    flow = pd.read_csv(tmp_path/'MESH_spatial_streamflow.csv')
    tail = pd.read_csv(tmp_path/'MESH_routing_tail.csv')
    np.testing.assert_allclose(flow.QOSIM1, [.25, .75])
    np.testing.assert_allclose(tail.discharge_cms, [2., 0.])
    assert report['volume_residual_m3'] == 0
    assert report['end_routing_storage_m3'] == 172800.
    kernel.write_text('unit_id,lag_days,weight\na,0,.9\nb,2,1\n')
    with pytest.raises(ValueError, match='preserve each unit volume'):
        route_spatial_runoff(tmp_path, kernel)


def test_worker_scores_only_nested_calibration_period_and_daily_observations(tmp_path):
    from unittest.mock import Mock, patch

    from symfluence.models.mesh.calibration.worker import MESHWorker

    dates = pd.date_range('2013-01-01', periods=60)
    q = np.arange(1., 61.)
    # Good calibration fit, deliberately wrong holdout: it must not enter score.
    simulated = q.copy()
    simulated[30:] *= 5
    pd.DataFrame({'YEAR': dates.year, 'JDAY': dates.dayofyear, 'QOSIM1': simulated}).to_csv(
        tmp_path/'MESH_spatial_streamflow.csv', index=False)
    obs_dates = pd.date_range(dates[0], periods=120, freq='12h')
    obs = np.repeat(q, 2) + np.tile([-0.5, 0.5], 60)
    config = {'domain': {'name': 'test', 'calibration_period': '2013-01-01, 2013-01-30'},
              'model': {'mesh': {'spatial_routing_kernel': 'kernel.csv', 'spinup_days': 0}},
              'paths': {'observations_path': '/example/observed.csv'}}
    worker = MESHWorker(logger=Mock())
    with patch('symfluence.models.mesh.calibration.worker.StreamflowMetrics.load_observations',
               return_value=(obs, obs_dates)) as loader:
        result = worker.calculate_metrics(tmp_path, config)
        assert result['kge'] == pytest.approx(1.)
        assert loader.call_args.args[0]['OBSERVATIONS_PATH'] == '/example/observed.csv'
        # The optimizer's final replay sends a typed config, unlike trial tasks.
        typed = Mock(spec=['to_dict'])
        typed.to_dict.return_value = config
        result = worker.calculate_metrics(tmp_path, typed)
        assert result['kge'] == pytest.approx(1.)
        typed.to_dict.assert_called_once_with()
        assert loader.call_args.args[0]['OBSERVATIONS_PATH'] == '/example/observed.csv'
        result = worker.calculate_metrics(tmp_path, config, period='2013-01-31, 2013-03-01')
        assert result['kge'] < 0
        result = worker.calculate_metrics(tmp_path, config, period='bad period')
        assert result['kge'] == worker.penalty_score
    (tmp_path/'MESH_spatial_streamflow.csv').unlink()
    (tmp_path/'Basin_average_water_balance.csv').write_text('YEAR,JDAY,RFF\n2013,1,1\n')
    result = worker.calculate_metrics(tmp_path, config)
    assert result['error'] == 'MESH output not found'
