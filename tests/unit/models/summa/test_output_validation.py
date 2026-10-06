from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.models.summa.output_validation import validate_output_coverage


@pytest.mark.parametrize('case', ['complete', 'truncated', 'gap', 'duplicate', 'wrong_start'])
def test_output_coverage(tmp_path: Path, case):
    fm = tmp_path / 'fileManager.txt'
    fm.write_text("simStartTime '2009-01-01 00:00'\nsimEndTime '2009-01-02 21:00'\noutFilePrefix 'test'\n")
    times = pd.date_range('2009-01-01', '2009-01-02 21:00', freq='3h')
    if case == 'truncated':
        times = times[:-1]
    elif case == 'gap':
        times = times.delete(5)
    elif case == 'duplicate':
        times = times.insert(5, times[5])
    elif case == 'wrong_start':
        times = times[1:]
    xr.Dataset(coords={'time': times}).to_netcdf(tmp_path / 'test_timestep.nc')
    if case == 'complete':
        assert validate_output_coverage(fm, tmp_path, 10800)
    else:
        with pytest.raises(ValidationError):
            validate_output_coverage(fm, tmp_path, 10800)


def test_missing_output(tmp_path):
    fm = tmp_path / 'fileManager.txt'
    fm.write_text("simStartTime '2009-01-01'\nsimEndTime '2009-01-02'\noutFilePrefix 'test'\n")
    with pytest.raises(ValidationError, match='No SUMMA timestep'):
        validate_output_coverage(fm, tmp_path)


@pytest.mark.parametrize('truncate', [False, True])
def test_routed_coverage(tmp_path, truncate):
    from symfluence.models.summa.output_validation import validate_routing_coverage
    times = pd.date_range('2009-01-01', periods=16, freq='3h')
    summa = tmp_path / 'test_timestep.nc'
    xr.Dataset(coords={'time': times}).to_netcdf(summa)
    xr.Dataset(coords={'time': times[:-1] if truncate else times}).to_netcdf(tmp_path / 'test.h.2009.nc')
    if truncate:
        with pytest.raises(ValidationError, match='mizuRoute output coverage'):
            validate_routing_coverage(summa, tmp_path, 'test')
    else:
        assert validate_routing_coverage(summa, tmp_path, 'test')


@pytest.mark.parametrize('explicit_step', [True, False])
@pytest.mark.parametrize('case', ['roundoff', 'shifted', 'interior_shift', 'gap'])
def test_timestamp_tolerance(tmp_path, explicit_step, case):
    fm = tmp_path / 'fileManager.txt'
    fm.write_text("simStartTime '2022-09-01'\nsimEndTime '2023-08-31 23:00'\noutFilePrefix 'test'\n")
    times = pd.date_range('2022-09-01', '2023-08-31 23:00', freq='h')
    # Reproduce fractional-day decoding noise across the whole year.
    import numpy as np
    times = times + pd.to_timedelta(np.arange(len(times)) % 3 * 13312, unit='ns')
    if case == 'shifted':
        times = times + pd.Timedelta(seconds=1)
    elif case == 'interior_shift':
        values = times.values.copy()
        values[100] += np.timedelta64(1, 's')
        times = pd.DatetimeIndex(values)
    elif case == 'gap':
        times = times.delete(100)
    xr.Dataset(coords={'time': times}).to_netcdf(tmp_path / 'test_timestep.nc')
    step = 3600 if explicit_step else None
    if case == 'roundoff':
        assert validate_output_coverage(fm, tmp_path, step)
    else:
        with pytest.raises(ValidationError, match='incomplete output coverage'):
            validate_output_coverage(fm, tmp_path, step)
