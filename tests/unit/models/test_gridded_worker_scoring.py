"""Wflow, LISFLOOD, PCR-GLOBWB and CWatM score like every other streamflow worker.

They used to carry their own scoring: a 30-point minimum, a hard-coded -999
on failure, raw-resolution observations, and (Wflow, LISFLOOD) KGE/NSE only
whatever objective was configured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.cwatm.calibration.worker import CWatMWorker
from symfluence.models.lisflood.calibration.worker import LisfloodWorker
from symfluence.models.pcrglobwb.calibration.worker import PCRGLOBWBWorker
from symfluence.models.wflow.calibration.worker import WflowWorker

DAYS = pd.date_range('2001-01-01', '2001-12-31', freq='D')
OBS = 20 + 10 * np.sin(np.arange(len(DAYS)) / 58.0)


def _netcdf(name, variable):
    def write(directory, index, values):
        xr.Dataset({variable: (('time',), values)}, coords={'time': index}).to_netcdf(directory / name)
    return write


def _wflow_csv(directory, index, values):
    pd.DataFrame({'time': index, 'Q': values}).to_csv(directory / 'output.csv', index=False)


WORKERS = [
    pytest.param(WflowWorker, _wflow_csv, id='wflow'),
    pytest.param(LisfloodWorker, _netcdf('dis.nc', 'dis'), id='lisflood'),
    pytest.param(PCRGLOBWBWorker, _netcdf('discharge_dailyTot_output.nc', 'discharge'), id='pcrglobwb'),
    pytest.param(CWatMWorker, _netcdf('discharge_daily.nc', 'discharge'), id='cwatm'),
]


@pytest.fixture
def config(tmp_path):
    obs_dir = tmp_path / 'domain_x' / 'observations' / 'streamflow' / 'preprocessed'
    obs_dir.mkdir(parents=True)
    pd.DataFrame({'datetime': DAYS, 'discharge_cms': OBS}).to_csv(
        obs_dir / 'x_streamflow_processed.csv', index=False)
    return {'DOMAIN_NAME': 'x', 'SYMFLUENCE_DATA_DIR': str(tmp_path)}


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / 'run'
    directory.mkdir()
    return directory


@pytest.mark.parametrize('worker_cls, write', WORKERS)
def test_configured_objective_is_scored(worker_cls, write, config, run_dir):
    write(run_dir, DAYS, OBS * 1.1)
    cfg = dict(config, OPTIMIZATION_METRIC='RMSE')
    metrics = worker_cls(cfg).calculate_metrics(run_dir, cfg)
    assert metrics['rmse'] == pytest.approx(float(np.sqrt(np.mean((OBS * 0.1) ** 2))))
    assert metrics['kge'] > 0.8


@pytest.mark.parametrize('worker_cls, write', WORKERS)
def test_short_window_is_scored(worker_cls, write, config, run_dir):
    write(run_dir, DAYS, OBS)
    cfg = dict(config, CALIBRATION_PERIOD='2001-03-01, 2001-03-20')
    assert worker_cls(cfg).calculate_metrics(run_dir, cfg)['kge'] == pytest.approx(1.0)


@pytest.mark.parametrize('worker_cls, write', WORKERS)
def test_failure_is_the_configured_penalty(worker_cls, write, config, run_dir):
    cfg = dict(config, PENALTY_SCORE=-5.0)
    metrics = worker_cls(cfg).calculate_metrics(run_dir, cfg)
    assert metrics['kge'] == -5.0 and 'error' in metrics


@pytest.mark.parametrize('worker_cls, write', WORKERS[:3])
def test_subdaily_output_is_scored_as_daily_means(worker_cls, write, config, run_dir):
    hours = pd.date_range(DAYS[0], periods=len(DAYS) * 24, freq='h')
    # Daily mean equals the observation; the midnight value alone is far off.
    within_day = np.tile(np.r_[-10.0, np.full(23, 10.0 / 23)], len(DAYS))
    write(run_dir, hours, np.repeat(OBS, 24) + within_day)
    assert worker_cls(config).calculate_metrics(run_dir, config)['kge'] == pytest.approx(1.0)
