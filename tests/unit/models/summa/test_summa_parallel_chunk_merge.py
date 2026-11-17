"""SummaRunner._merge_files: per-GRU-chunk outputs are concatenated, not merged."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.summa.runner import SummaRunner

TIMES = pd.date_range('2000-01-01', periods=4, freq='h')


def _write_chunk(path, start, n):
    gru = np.arange(start, start + n)
    runoff = np.tile(gru.astype(float), (len(TIMES), 1)) * 10.0
    swe = np.tile(gru.astype(float), (len(TIMES), 1)) * 100.0
    ds = xr.Dataset(
        {
            'averageRoutedRunoff': (('time', 'gru'), runoff),
            'gruId': ('gru', gru),
            'hruId': ('hru', gru),
            'scalarSWE': (('time', 'hru'), swe),
            'basinMeta': ('time', np.arange(len(TIMES), dtype=float)),
        },
        coords={'time': TIMES},
    )
    ds.to_netcdf(path)


def _runner(tmp_path):
    runner = object.__new__(SummaRunner)
    runner.output_dir = tmp_path
    runner.logger = logging.getLogger('test_summa_merge')
    return runner


@pytest.mark.parametrize('chunks', [
    [(1, 1), (2, 1), (3, 1)],   # three single-GRU files
    [(1, 3), (4, 2)],           # uneven chunks 3 + 2
])
def test_chunks_concatenate_along_hru_and_gru(tmp_path, chunks):
    # Write in reverse so glob/file order differs from GRU order.
    for start, n in reversed(chunks):
        _write_chunk(tmp_path / f'exp_G{start:06d}-{start + n - 1:06d}_timestep.nc', start, n)

    out = tmp_path / 'exp_timestep.nc'
    _runner(tmp_path)._merge_files('exp_*_timestep.nc', out)

    total = sum(n for _, n in chunks)
    with xr.open_dataset(out, decode_times=False) as m:
        assert m.sizes['gru'] == total
        assert m.sizes['hru'] == total
        assert m.sizes['time'] == len(TIMES)
        expected = np.arange(1, total + 1)
        np.testing.assert_array_equal(m['gruId'].values, expected)
        np.testing.assert_array_equal(m['hruId'].values, expected)
        np.testing.assert_allclose(m['averageRoutedRunoff'].values[0], expected * 10.0)
        np.testing.assert_allclose(m['scalarSWE'].values[-1], expected * 100.0)
        # time-only variables are kept once, not duplicated per chunk
        assert m['basinMeta'].dims == ('time',)
        np.testing.assert_allclose(m['basinMeta'].values, np.arange(len(TIMES)))
        assert 'seconds since 1990-1-1' in m['time'].attrs['units']


def test_mismatched_chunk_raises(tmp_path):
    _write_chunk(tmp_path / 'exp_G000001-000001_timestep.nc', 1, 1)
    bad = xr.Dataset(
        {'averageRoutedRunoff': (('time', 'gru'), np.ones((2, 1))), 'gruId': ('gru', [2]),
         'hruId': ('hru', [2]), 'scalarSWE': (('time', 'hru'), np.ones((2, 1))),
         'basinMeta': ('time', np.zeros(2))},
        coords={'time': TIMES[:2]},
    )
    bad.to_netcdf(tmp_path / 'exp_G000002-000002_timestep.nc')

    out = tmp_path / 'exp_timestep.nc'
    with pytest.raises(ModelExecutionError):
        _runner(tmp_path)._merge_files('exp_*_timestep.nc', out)
    assert not out.exists()
