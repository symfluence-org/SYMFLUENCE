"""Extending a CDS record must not select the older, shorter cache copy."""
from __future__ import annotations

import logging

import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import FileOperationError
from symfluence.models.summa.forcing_processor import SummaForcingProcessor


def record(tmp_path, name, start, end):
    path = tmp_path / name
    xr.Dataset(coords={'time': pd.date_range(start, end, freq='h')}).to_netcdf(path)
    return path


def selector():
    obj = SummaForcingProcessor.__new__(SummaForcingProcessor)
    obj.logger = logging.getLogger(__name__)
    return obj


def test_longer_record_wins_over_earlier_cache(tmp_path):
    old = record(tmp_path, 'domain_ERA5_remapped_grus_oldlonghash12345.nc', '2023-01-01', '2023-01-31 23:00')
    new = record(tmp_path, 'domain_ERA5_remapped_grus_CDS_2023_2024.nc', '2023-01-01', '2023-02-28 23:00')
    assert selector()._select_forcing_by_time([old, new]) == [new]
    assert selector()._select_forcing_by_time([new, old]) == [new]


def test_monthly_files_sort_by_data_not_name(tmp_path):
    jan = record(tmp_path, 'z.nc', '2023-01-01', '2023-01-31 23:00')
    feb = record(tmp_path, 'a.nc', '2023-02-01', '2023-02-28 23:00')
    assert selector()._select_forcing_by_time([feb, jan]) == [jan, feb]


def test_contained_month_is_not_added_twice(tmp_path):
    entire = record(tmp_path, 'entire.nc', '2023-01-01', '2023-03-31 23:00')
    middle = record(tmp_path, 'month.nc', '2023-02-01', '2023-02-28 23:00')
    assert selector()._select_forcing_by_time([middle, entire]) == [entire]


def test_partial_overlap_requires_resolution(tmp_path):
    a = record(tmp_path, 'a.nc', '2023-01-01', '2023-02-15 23:00')
    b = record(tmp_path, 'b.nc', '2023-02-01', '2023-03-01 23:00')
    with pytest.raises(FileOperationError, match='Partially overlapping'):
        selector()._select_forcing_by_time([a, b])


def test_contained_bounds_do_not_hide_missing_timestamps(tmp_path):
    wide = tmp_path / 'with_gap.nc'
    xr.Dataset(coords={'time': pd.to_datetime(['2023-01-01', '2023-01-03'])}).to_netcdf(wide)
    gap = record(tmp_path, 'gap.nc', '2023-01-02', '2023-01-02 23:00')
    with pytest.raises(FileOperationError, match='overlapping'):
        selector()._select_forcing_by_time([wide, gap])
