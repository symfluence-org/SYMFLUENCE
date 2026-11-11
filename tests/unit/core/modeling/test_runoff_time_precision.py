"""Shared runoff time-precision fix (runoff_loader.fix_time_precision)."""
from __future__ import annotations

import netCDF4 as nc4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.utilities.runoff_loader import (
    fix_time_precision,
    parse_cf_time_units,
)

REF = pd.Timestamp('1990-01-01')


def _write(path, units, values):
    with nc4.Dataset(path, 'w') as ds:
        ds.createDimension('time', len(values))
        ds.createDimension('gru', 1)
        t = ds.createVariable('time', 'f8', ('time',))
        t.units = units
        t.calendar = 'standard'
        t[:] = values
        q = ds.createVariable('averageRoutedRunoff', 'f8', ('time', 'gru'))
        q[:] = np.arange(len(values))[:, None] * 1e-8
        g = ds.createVariable('gruId', 'i4', ('gru',))
        g[:] = [1]


def _stamps(path):
    with xr.open_dataset(path) as ds:
        return pd.DatetimeIndex(ds.time.values)


@pytest.mark.parametrize('unit,per_hour', [('hours', 1.0), ('minutes', 60.0), ('days', 1 / 24)])
def test_non_second_units_since_1990_are_decoded_in_their_own_unit(tmp_path, unit, per_hour):
    """'hours since 1990-01-01' used to be read as seconds and collapse to one stamp."""
    path = tmp_path / 'a.nc'
    h0 = (pd.Timestamp('2010-10-01') - REF) / pd.Timedelta(hours=1)
    _write(path, f'{unit} since 1990-01-01 00:00:00', (h0 + np.arange(48)) * per_hour)
    expected = pd.date_range('2010-10-01', periods=48, freq='h')

    fix_time_precision(path)

    np.testing.assert_array_equal(_stamps(path), expected)
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.time.attrs['units'].startswith(unit)


def test_sub_hourly_axis_is_not_rounded_into_duplicates(tmp_path):
    path = tmp_path / 'b.nc'
    s0 = (pd.Timestamp('2010-10-01 00:30') - REF).total_seconds()
    _write(path, 'seconds since 1990-01-01 00:00:00', s0 + 1800 * np.arange(8) + 0.3)

    fix_time_precision(path, rounding_freq='h')

    np.testing.assert_array_equal(
        _stamps(path), pd.date_range('2010-10-01 00:30', periods=8, freq='30min')
    )


def test_every_stamp_is_rounded_not_only_when_the_first_is_off(tmp_path):
    path = tmp_path / 'c.nc'
    s0 = (pd.Timestamp('2010-10-01') - REF).total_seconds()
    values = s0 + 3600.0 * np.arange(6)
    values[1:] += 0.4
    _write(path, 'seconds since 1990-01-01 00:00:00', values)

    fix_time_precision(path)

    with nc4.Dataset(path) as ds:
        assert (np.asarray(ds['time'][:]) % 3600 == 0).all()


def test_clean_file_is_not_rewritten(tmp_path):
    path = tmp_path / 'd.nc'
    s0 = (pd.Timestamp('2010-10-01') - REF).total_seconds()
    _write(path, 'seconds since 1990-01-01 00:00:00', s0 + 3600.0 * np.arange(6))
    before = path.read_bytes()

    assert fix_time_precision(path) == path

    assert path.read_bytes() == before
    assert not path.with_suffix('.tmp.nc').exists()


def test_summa_microsecond_jitter_after_an_exact_first_step_is_removed(tmp_path):
    """~13 us jitter from step 2 on (first stamp exact) must still be fixed."""
    path = tmp_path / 'j.nc'
    s0 = (pd.Timestamp('2004-01-01') - REF).total_seconds()
    values = s0 + 3600.0 * np.arange(72)
    values[1:] += 13e-6
    _write(path, 'seconds since 1990-01-01 00:00:00', values)

    fix_time_precision(path)

    with nc4.Dataset(path) as ds:
        steps = np.diff(np.asarray(ds['time'][:]))
    assert (steps == 3600.0).all()


def test_irregular_sub_hourly_stamps_fall_back_to_whole_seconds(tmp_path):
    """Hourly rounding would merge 02:00 and 02:10; seconds rounding keeps both."""
    path = tmp_path / 'e.nc'
    s0 = (pd.Timestamp('2010-10-01') - REF).total_seconds()
    offsets = np.array([0.0, 3600.0, 7200.0, 7800.0, 10800.0, 14400.0])
    _write(path, 'seconds since 1990-01-01 00:00:00', s0 + offsets + 2e-5)

    fix_time_precision(path, rounding_freq='h')

    with nc4.Dataset(path) as ds:
        np.testing.assert_array_equal(np.asarray(ds['time'][:]), s0 + offsets)


def test_rounding_that_would_merge_stamps_is_refused(tmp_path):
    """Sub-second steps collide even at whole seconds: refuse, do not rewrite."""
    path = tmp_path / 'g.nc'
    s0 = (pd.Timestamp('2010-10-01') - REF).total_seconds()
    _write(path, 'seconds since 1990-01-01 00:00:00', s0 + 0.2 * np.arange(6))
    before = path.read_bytes()

    with pytest.raises(ModelExecutionError, match='duplicates'):
        fix_time_precision(path, rounding_freq='h')

    assert path.read_bytes() == before


def test_unknown_units_and_calendars_are_left_alone(tmp_path):
    path = tmp_path / 'f.nc'
    _write(path, 'months since 1990-01-01', np.arange(4, dtype=float))
    before = path.read_bytes()
    assert fix_time_precision(path) == path
    assert path.read_bytes() == before
    assert parse_cf_time_units('months since 1990-01-01') is None
    assert parse_cf_time_units('seconds') is None


def test_unreadable_or_missing_file_raises(tmp_path):
    with pytest.raises(ModelExecutionError, match='not found'):
        fix_time_precision(tmp_path / 'missing.nc')
    garbage = tmp_path / 'garbage.nc'
    garbage.write_bytes(b'not a netcdf file' * 100)
    with pytest.raises(ModelExecutionError, match='Failed to open'):
        fix_time_precision(garbage)


def test_mizuroute_runner_uses_the_shared_implementation():
    """One implementation: the mizuRoute runner must not carry its own copy."""
    import inspect

    from symfluence.models.mizuroute.runner import MizuRouteRunner

    source = inspect.getsource(MizuRouteRunner.fix_time_precision)
    assert 'shared_fix_time_precision(' in source
    assert 'since 1990-01-01' not in source
