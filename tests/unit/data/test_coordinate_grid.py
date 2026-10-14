"""Spatial identity regressions for positional EASYMORE cache reuse."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.preprocessing.resampling.coordinate_grid import (
    forcing_grid_signature,
    require_same_forcing_grid,
    weights_match_forcing_coordinates,
)
from symfluence.data.preprocessing.resampling.weight_applier import RemappingWeightApplier
from symfluence.data.preprocessing.resampling.weight_generator import RemappingWeightGenerator


def grid(path, lat_shift=0.0, lon_shift=0.0, reverse=False, extra_column=False):
    lat = np.array([[64.0, 64.01, 64.02], [64.025, 64.035, 64.045]]) + lat_shift
    lon = np.array([[-18.9, -18.85, -18.8], [-18.9, -18.85, -18.8]]) + lon_shift
    if reverse:
        lat, lon = lat[::-1], lon[::-1]
    if extra_column:
        lat = np.c_[lat, lat[:, -1] + 0.01]
        lon = np.c_[lon, lon[:, -1] + 0.05]
    xr.Dataset(
        coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
        attrs={"MAP_PROJ4_STR": "+proj=lcc", "DX": 2500, "DY": 2500},
    ).to_netcdf(path)
    return path


def weights(path, lat=64.01, lon=-18.85):
    # A valid spatial subset; the maximum indices do not describe grid extent.
    pd.DataFrame({"rows": [0], "cols": [1], "lat_s": [lat], "lon_s": [lon], "weight": [1.0]}).to_csv(path, index=False)
    return path


def test_same_grid_different_files_and_longitude_convention(tmp_path):
    first = grid(tmp_path / "first.nc")
    second = grid(tmp_path / "second.nc", lon_shift=360.0)
    assert forcing_grid_signature(first) == forcing_grid_signature(second)
    require_same_forcing_grid([first, second])


@pytest.mark.parametrize(
    "kwargs", [{"lat_shift": 0.01}, {"lon_shift": -0.02}, {"reverse": True}, {"extra_column": True}]
)
def test_shift_reorder_or_shape_change_rejected(tmp_path, kwargs):
    first = grid(tmp_path / "first.nc")
    changed = grid(tmp_path / "changed.nc", **kwargs)
    with pytest.raises(ValueError, match="separate weights"):
        require_same_forcing_grid([first, changed])


def test_valid_subset_cache_accepted_without_count_heuristic(tmp_path):
    source = grid(tmp_path / "source.nc")
    mapping = weights(tmp_path / "mapping.csv")
    assert weights_match_forcing_coordinates(mapping, source)
    generator = RemappingWeightGenerator.__new__(RemappingWeightGenerator)
    generator.logger = Mock()
    assert generator._weights_match_forcing_grid(mapping, source)


def test_same_shape_shifted_cache_rejected(tmp_path):
    source = grid(tmp_path / "source.nc", lat_shift=0.01)
    assert not weights_match_forcing_coordinates(weights(tmp_path / "mapping.csv"), source)


@pytest.mark.parametrize("row,col", [(-1, 1), (0.5, 1), (2, 1), (0, 3)])
def test_invalid_indices_rejected(tmp_path, row, col):
    source = grid(tmp_path / "source.nc")
    mapping = weights(tmp_path / "mapping.csv")
    frame = pd.read_csv(mapping)
    frame["rows"] = row
    frame["cols"] = col
    frame.to_csv(mapping, index=False)
    assert not weights_match_forcing_coordinates(mapping, source)


def test_regular_lat_lon_grid(tmp_path):
    source = tmp_path / "source.nc"
    xr.Dataset(coords={"latitude": [64.0, 64.5], "longitude": [-19.0, -18.85, -18.0]}).to_netcdf(source)
    assert weights_match_forcing_coordinates(weights(tmp_path / "mapping.csv", lat=64.0), source)


def test_unverifiable_cache_not_accepted(tmp_path):
    source = grid(tmp_path / "source.nc")
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"ID_s": [1], "weight": [1.0]}).to_csv(mapping, index=False)
    assert not weights_match_forcing_coordinates(mapping, source)


def test_invalid_coordinates_rejected(tmp_path):
    source = tmp_path / "bad.nc"
    xr.Dataset(coords={"latitude": [np.nan], "longitude": [-19.0]}).to_netcdf(source)
    with pytest.raises(ValueError, match="Invalid forcing coordinates"):
        forcing_grid_signature(source)


def test_bad_mapping_cannot_validate_existing_output(tmp_path):
    source = grid(tmp_path / "source.nc", lon_shift=0.02)
    mapping = weights(tmp_path / "mapping.csv")
    output = tmp_path / "existing.nc"
    output.write_bytes(b"x" * 2000)
    applier = RemappingWeightApplier.__new__(RemappingWeightApplier)
    applier.logger = Mock()
    assert not applier.apply_weights(source, mapping, output)
    assert output.read_bytes() == b"x" * 2000


def test_grid_preflight_precedes_weight_generation_and_processed_skip(tmp_path):
    from symfluence.data.preprocessing.forcing_resampler import ForcingResampler

    first = grid(tmp_path / "first.nc")
    changed = grid(tmp_path / "changed.nc", lon_shift=0.02)
    obj = ForcingResampler.__new__(ForcingResampler)
    obj.forcing_basin_path = tmp_path / "out"
    obj.project_dir = tmp_path
    obj.merged_forcing_path = tmp_path
    obj.logger = Mock()
    obj.dataset_handler = Mock()
    obj.dataset_handler.get_coordinate_names.return_value = ("latitude", "longitude")
    obj._file_processor = Mock()
    obj.file_processor.get_forcing_files.return_value = [first, changed]
    obj._filter_forcing_files_by_period = lambda files: files
    obj._weight_generator = Mock()
    with pytest.raises(ValueError, match="Forcing grid changes"):
        obj._create_parallelized_weighted_forcing()
    obj.weight_generator.create_weights.assert_not_called()
    obj.file_processor.filter_processed_files.assert_not_called()
