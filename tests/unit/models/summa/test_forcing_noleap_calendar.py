"""noleap forcing: standard-calendar time axis and leap days at the data step.

The time fixer could not convert cftime (noleap) dates at all, labelled the
standard-calendar seconds axis with the source calendar, and the leap-day
insertion assumed hourly data (24 steps, Feb 28 23:00 boundary).
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("cftime")

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.summa.forcing_processor import SummaForcingProcessor

pytestmark = [pytest.mark.unit, pytest.mark.quick]

REFERENCE = pd.Timestamp('1990-01-01')


def _processor(tmp_path: Path, step: int) -> SummaForcingProcessor:
    return SummaForcingProcessor(
        config=SimpleNamespace(forcing=SimpleNamespace(time_step_size=step)),
        logger=logging.getLogger("test.summa.noleap"),
        forcing_basin_path=tmp_path,
        forcing_summa_path=tmp_path,
        intersect_path=tmp_path,
        catchment_path=tmp_path,
        project_dir=tmp_path,
        setup_dir=tmp_path,
        domain_name="toy",
        forcing_dataset="gcm",
        data_step=step,
        gruId="HRU_ID",
        hruId="HRU_ID",
        catchment_name="catchment.shp",
    )


def _as_dates(seconds) -> pd.DatetimeIndex:
    return REFERENCE + pd.to_timedelta(np.asarray(seconds), unit='s')


def _noleap_dataset(step_hours: int) -> xr.Dataset:
    time = xr.date_range('2004-02-27', '2004-03-02 21:00', freq=f'{step_hours}h',
                         calendar='noleap', use_cftime=True)
    return xr.Dataset({'air_temperature': (('time', 'hru'), np.full((len(time), 1), 270.0))},
                      coords={'time': time, 'hru': [0]})


# ---------------------------------------------------------------------------
# time axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('step_hours', [1, 3])
def test_noleap_dates_keep_their_labels_on_a_standard_axis(tmp_path, step_hours):
    path = tmp_path / 'noleap.nc'
    _noleap_dataset(step_hours).to_netcdf(path)
    processor = _processor(tmp_path, step_hours * 3600)

    with xr.open_dataset(path) as source:
        labels = [t.isoformat() for t in source.time.values]
        fixed = processor._fix_time_coordinate_comprehensive(source.copy(), path.name)

    assert processor._source_calendar == 'noleap'
    assert fixed.time.attrs['calendar'] == 'standard'
    dates = _as_dates(fixed.time.values)
    assert [d.isoformat() for d in dates] == labels
    assert not ((dates.month == 2) & (dates.day == 29)).any()

    # Whoever decodes the written axis gets the same dates back.
    out = tmp_path / 'fixed.nc'
    fixed.to_netcdf(out)
    with xr.open_dataset(out) as back:
        assert [pd.Timestamp(t).isoformat() for t in back.time.values] == labels


def test_undecoded_noleap_offsets_are_decoded_in_their_calendar(tmp_path):
    """2005-01-01 is day 5 * 365 of a noleap axis starting 2000-01-01."""
    dataset = xr.Dataset(
        {'air_temperature': (('time', 'hru'), np.full((2, 1), 270.0))},
        coords={'time': ('time', np.array([5 * 365.0, 5 * 365.0 + 1.0]),
                         {'units': 'days since 2000-01-01 00:00:00', 'calendar': 'noleap'}),
                'hru': [0]})
    processor = _processor(tmp_path, 86400)

    fixed = processor._fix_time_coordinate_comprehensive(dataset, 'raw.nc')

    assert list(_as_dates(fixed.time.values)) == [pd.Timestamp('2005-01-01'), pd.Timestamp('2005-01-02')]
    assert fixed.time.attrs['calendar'] == 'standard'


def test_standard_axis_is_unchanged(tmp_path):
    time = pd.date_range('2004-02-27', '2004-03-02 23:00', freq='h')
    dataset = xr.Dataset({'air_temperature': (('time', 'hru'), np.full((len(time), 1), 270.0))},
                         coords={'time': time, 'hru': [0]})
    processor = _processor(tmp_path, 3600)

    fixed = processor._fix_time_coordinate_comprehensive(dataset, 'std.nc')

    assert processor._source_calendar == 'standard'
    assert fixed.time.attrs['calendar'] == 'standard'
    assert _as_dates(fixed.time.values).equals(time)


def test_calendar_that_cannot_be_mapped_is_rejected(tmp_path):
    time = xr.date_range('2004-02-27', periods=8, freq='D', calendar='360_day', use_cftime=True)
    dataset = xr.Dataset({'air_temperature': (('time', 'hru'), np.full((len(time), 1), 270.0))},
                         coords={'time': time, 'hru': [0]})

    with pytest.raises(ModelExecutionError, match="360_day"):
        _processor(tmp_path, 86400)._fix_time_coordinate_comprehensive(dataset, '360.nc')


# ---------------------------------------------------------------------------
# leap-day insertion
# ---------------------------------------------------------------------------

def _write_processed(tmp_path: Path, times: pd.DatetimeIndex, name: str = 'toy_gcm_2004-02-01-00-00-00.nc') -> None:
    seconds = (times - REFERENCE).total_seconds().values
    hours = seconds / 3600.0
    xr.Dataset(
        {'air_temperature': (('time', 'hru'), np.column_stack([hours, 2 * hours])),
         'hruId': ('hru', np.array([7, 8], dtype='int32'))},
        coords={'time': ('time', seconds, {'units': 'seconds since 1990-01-01 00:00:00', 'calendar': 'standard'}),
                'hru': [0, 1]},
    ).to_netcdf(tmp_path / name)


def _noleap_times(freq: str, offset: str = '0h') -> pd.DatetimeIndex:
    times = pd.date_range('2004-02-27', '2004-03-03', freq=freq, inclusive='left') + pd.Timedelta(offset)
    return times[~((times.month == 2) & (times.day == 29))]


@pytest.mark.parametrize('freq, step, offset, n_steps', [
    ('h', 3600, '0h', 24),
    ('3h', 10800, '0h', 8),
    ('6h', 21600, '0h', 4),
    ('h', 3600, '30min', 24),     # steps not on the hour
])
def test_leap_day_is_inserted_at_the_data_step(tmp_path, freq, step, offset, n_steps):
    times = _noleap_times(freq, offset)
    _write_processed(tmp_path, times)
    processor = _processor(tmp_path, step)
    processor._source_calendar = 'noleap'

    processor._insert_noleap_leap_days()

    with xr.open_dataset(tmp_path / 'toy_gcm_leapday_2004-02-29-00-00-00.nc', decode_times=False) as leap:
        assert leap.time.attrs['calendar'] == 'standard'
        expected = pd.date_range('2004-02-29', periods=n_steps, freq=freq) + pd.Timedelta(offset)
        assert _as_dates(leap.time.values).equals(expected)
        # the test field is linear in time, so interpolation must reproduce it
        hours = leap.time.values / 3600.0
        np.testing.assert_allclose(leap.air_temperature.values, np.column_stack([hours, 2 * hours]))
        np.testing.assert_array_equal(leap.hruId.values, [7, 8])

    # together with the source steps the axis is now gap-free
    full = np.sort(np.concatenate([(times - REFERENCE).total_seconds().values, leap.time.values]))
    np.testing.assert_allclose(np.diff(full), step)


def test_leap_day_from_an_earlier_run_is_rebuilt_and_claimed(tmp_path):
    _write_processed(tmp_path, _noleap_times('3h'))
    processor = _processor(tmp_path, 10800)
    processor._source_calendar = 'noleap'
    processor._insert_noleap_leap_days()

    processor._written_forcing_names = set()
    processor._insert_noleap_leap_days()

    assert 'toy_gcm_leapday_2004-02-29-00-00-00.nc' in processor._written_forcing_names


def test_no_leap_day_is_inserted_when_the_forcing_has_one(tmp_path):
    _write_processed(tmp_path, pd.date_range('2004-02-27', '2004-03-03', freq='3h', inclusive='left'))
    processor = _processor(tmp_path, 10800)
    processor._source_calendar = 'noleap'

    processor._insert_noleap_leap_days()

    assert not list(tmp_path.glob('*leapday*'))


def test_other_gaps_around_the_leap_day_are_an_error(tmp_path):
    times = _noleap_times('3h')
    _write_processed(tmp_path, times[times != pd.Timestamp('2004-02-28 21:00')])
    processor = _processor(tmp_path, 10800)
    processor._source_calendar = 'noleap'

    with pytest.raises(ModelExecutionError, match='other gaps'):
        processor._insert_noleap_leap_days()


def test_step_that_does_not_divide_a_day_is_an_error(tmp_path):
    _write_processed(tmp_path, _noleap_times('3h'))
    processor = _processor(tmp_path, 25200)   # 7 h
    processor._source_calendar = 'noleap'

    with pytest.raises(ModelExecutionError, match='does not divide a day'):
        processor._insert_noleap_leap_days()
