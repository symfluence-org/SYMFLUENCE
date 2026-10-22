"""Aligned timestamps still need a Fortran-compatible CF reference epoch."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from symfluence.models.mizuroute.runner import MizuRouteRunner


@pytest.mark.parametrize('epoch', ['2009-01-01T00:00:00+00:00', '2009-01-01T03:00:00+03:00'])
def test_aligned_iso_epoch_is_normalized_without_changing_times(tmp_path, monkeypatch, epoch):
    path = tmp_path / 'runoff.nc'
    ds = xr.Dataset({'averageRoutedRunoff': (('time', 'gru'), [[1e-6], [2e-6]]), 'gruId': ('gru', [1])}, coords={'time': [0., 10800.]})
    ds.time.attrs['units'] = f'seconds since {epoch}'
    ds.to_netcdf(path)
    with xr.open_dataset(path) as f:
        before = f.time.values.copy()
    monkeypatch.setattr('symfluence.models.mizuroute.runner.resolve_runoff_file', lambda **kwargs: path)
    runner = MizuRouteRunner.__new__(MizuRouteRunner)
    runner.config = SimpleNamespace(model=SimpleNamespace(hydrological_model='SUMMA', mizuroute=SimpleNamespace(from_model='SUMMA', time_rounding_freq='none')))
    runner.logger = logging.getLogger(__name__)
    runner.project_dir = tmp_path
    runner.domain_name = 'test'
    runner.experiment_id = 'test'
    assert runner.fix_time_precision() == path
    with xr.open_dataset(path) as f:
        np.testing.assert_array_equal(f.time.values, before)
        np.testing.assert_allclose(f.averageRoutedRunoff.values[:, 0], [1e-6, 2e-6])
    with xr.open_dataset(path, decode_times=False) as f:
        assert f.time.attrs['units'] == 'seconds since 2009-01-01 00:00:00'


def test_zero_exit_with_fatal_log_rejected_even_with_stale_output(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError

    log = tmp_path / 'routing.log'
    log.write_text('FATAL ERROR: init_time failed')
    (tmp_path / 'old.nc').touch()
    with pytest.raises(ModelExecutionError, match='fatal error'):
        MizuRouteRunner._validate_routing_output(log, tmp_path)


def test_missing_routing_output_rejected(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError

    log = tmp_path / 'routing.log'
    log.write_text('STOP')
    with pytest.raises(ModelExecutionError, match='no NetCDF output'):
        MizuRouteRunner._validate_routing_output(log, tmp_path)
