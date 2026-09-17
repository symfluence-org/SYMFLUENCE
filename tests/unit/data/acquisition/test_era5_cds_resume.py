# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
from __future__ import annotations

"""
Regression tests for CDS month-chunk resume.

A CDS request spends most of its wall clock queued server-side — an hour per
month is not unusual — so a long record takes many hours to acquire. The ARCO
pathway skips chunks already on disk; the CDS pathway did not, which made any
interruption restart from the first month and discard the whole download.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.era5_cds import ERA5CDSAcquirer


@pytest.fixture
def acquirer():
    probe = ERA5CDSAcquirer.__new__(ERA5CDSAcquirer)
    probe.logger = logging.getLogger('test-era5-cds-resume')
    probe.logger.addHandler(logging.NullHandler())
    probe.domain_name = 'TestDomain'
    return probe


def _write_chunk(directory, year, month, n_times=24):
    path = directory / f"TestDomain_era5_cds_processed_{year}{month:02d}_temp.nc"
    ds = xr.Dataset(
        {'t2m': (('time',), np.arange(n_times, dtype='float64'))},
        coords={'time': pd.date_range(f'{year}-{month:02d}-01', periods=n_times, freq='h')},
    )
    ds.to_netcdf(path)
    ds.close()
    return path


def test_existing_month_is_reused(acquirer, tmp_path):
    written = _write_chunk(tmp_path, 2005, 6)

    assert acquirer._cached_month_chunk(2005, 6, tmp_path) == written


def test_missing_month_is_downloaded(acquirer, tmp_path):
    assert acquirer._cached_month_chunk(2005, 6, tmp_path) is None


def test_empty_month_is_redownloaded(acquirer, tmp_path):
    """A file with no timesteps is a truncated download, not a usable chunk."""
    _write_chunk(tmp_path, 2005, 6, n_times=0)

    assert acquirer._cached_month_chunk(2005, 6, tmp_path) is None


def test_corrupt_month_is_redownloaded(acquirer, tmp_path):
    """A half-written NetCDF must not be mistaken for a completed month."""
    path = tmp_path / "TestDomain_era5_cds_processed_200506_temp.nc"
    path.write_bytes(b'not a netcdf file')

    assert acquirer._cached_month_chunk(2005, 6, tmp_path) is None


def test_only_the_matching_month_is_reused(acquirer, tmp_path):
    _write_chunk(tmp_path, 2005, 6)

    assert acquirer._cached_month_chunk(2005, 6, tmp_path) is not None
    assert acquirer._cached_month_chunk(2005, 7, tmp_path) is None
    assert acquirer._cached_month_chunk(2006, 6, tmp_path) is None


def test_partial_record_resumes_at_the_gap(acquirer, tmp_path):
    """The real case: 69 of 96 months present -> only the remainder is fetched."""
    for month in range(1, 13):
        if month <= 8:
            _write_chunk(tmp_path, 2005, month)

    cached = [m for m in range(1, 13)
              if acquirer._cached_month_chunk(2005, m, tmp_path) is not None]
    missing = [m for m in range(1, 13)
               if acquirer._cached_month_chunk(2005, m, tmp_path) is None]

    assert cached == list(range(1, 9))
    assert missing == list(range(9, 13))


@pytest.mark.parametrize('defect', [None, 'missing_hour', 'wrong_grid', 'missing_variable'])
def test_extended_record_reuses_only_complete_matching_month(acquirer, tmp_path, defect):
    acquirer.start_date = pd.Timestamp('2023-01-01')
    acquirer.end_date = pd.Timestamp('2024-12-31 23:00')
    acquirer.bbox_to_cds_area = lambda **kw: [32.5, 77., 32.25, 77.5]
    times = pd.date_range('2023-02-01', '2023-02-28 23:00', freq='h')
    if defect == 'missing_hour':
        times = times.delete(10)
    variables = ['air_temperature', 'surface_air_pressure', 'wind_speed',
                 'specific_humidity', 'precipitation_flux',
                 'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux']
    if defect == 'missing_variable':
        variables.pop()
    lat = [32.5, 32.25] if defect != 'wrong_grid' else [31.5, 31.25]
    ds = xr.Dataset({v: (('time', 'latitude', 'longitude'), np.ones((len(times), 2, 3))) for v in variables},
                    coords={'time': times, 'latitude': lat, 'longitude': [77., 77.25, 77.5]})
    source = tmp_path / 'domain_TestDomain_ERA5_CDS_2023_2023.nc'
    ds.to_netcdf(source)
    before = source.read_bytes()
    result = acquirer._month_from_merged_record(2023, 2, tmp_path)
    assert source.read_bytes() == before
    if defect:
        assert result is None
    else:
        with xr.open_dataset(result) as recovered:
            xr.testing.assert_equal(recovered, ds)
