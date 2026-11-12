"""mHM discharge: one reader for the worker, the postprocessor and the extractor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.mhm.calibration.worker import MHMWorker
from symfluence.models.mhm.extractor import MHMResultExtractor, read_mhm_discharge

DAYS = pd.date_range('2001-01-01', periods=5, freq='D')
Q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def _write(path, data_vars):
    xr.Dataset(data_vars, coords={'time': DAYS}).to_netcdf(path)
    return path


def test_reads_named_variable(tmp_path):
    series = read_mhm_discharge(_write(tmp_path / 'discharge.nc', {'Qsim': (('time',), Q)}))
    assert list(series.index) == list(DAYS)
    np.testing.assert_allclose(series.values, Q)


def test_takes_first_gauge(tmp_path):
    path = _write(tmp_path / 'discharge.nc', {'Qsim': (('time', 'gauge'), np.c_[Q, Q * 10])})
    np.testing.assert_allclose(read_mhm_discharge(path).values, Q)


def test_reads_gauge_indexed_variable_of_mhm_5_13(tmp_path):
    path = _write(tmp_path / 'discharge.nc', {
        'Qobs_0000000001': (('time',), Q * 0), 'Qsim_0000000001': (('time',), Q)})
    np.testing.assert_allclose(read_mhm_discharge(path).values, Q)
    # ... and so does the extractor, which used to know only the bare names.
    np.testing.assert_allclose(MHMResultExtractor('MHM').extract_streamflow(tmp_path).values, Q)


def test_no_discharge_variable_is_an_error(tmp_path):
    with pytest.raises(ModelExecutionError, match='No discharge variable'):
        read_mhm_discharge(_write(tmp_path / 'discharge.nc', {'aET': (('time',), Q)}))


def test_worker_reads_settings_output_when_run_dir_is_empty(tmp_path):
    settings = tmp_path / 'settings'
    (settings / 'output').mkdir(parents=True)
    _write(settings / 'output' / 'discharge_run.nc', {'Qsim': (('time',), Q)})
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    config = {'DOMAIN_NAME': 'x', 'SYMFLUENCE_DATA_DIR': str(tmp_path), 'MHM_WORKER_WARMUP_DAYS': 2}
    series = MHMWorker(config=config).read_simulated_streamflow(
        run_dir, config, sim_dir=str(run_dir), settings_dir=str(settings))
    np.testing.assert_allclose(series.values, Q[2:])
