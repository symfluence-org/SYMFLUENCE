# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the model-ready attributes reader and its builder hru_id support.

Covers ``open_canonical_attributes`` / ``AttributesReader`` plus the builder
change that stamps an ``hru_id`` coordinate into the per-HRU groups so they can
be joined safely (the reason the store was previously write-only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip('netCDF4')
gpd = pytest.importorskip('geopandas')
pytest.importorskip('xarray')

from shapely.geometry import box

from symfluence.data.model_ready import open_canonical_attributes
from symfluence.data.model_ready.attributes_builder import AttributesNetCDFBuilder

DOMAIN = 'test'
HRU_IDS = ['hru_0', 'hru_1', 'hru_2']
ELEV = np.array([870.0, 1320.5, 1980.0])


def _catchment_shp(tmp_path: Path) -> None:
    path = tmp_path / 'shapefiles' / 'catchment' / f'{DOMAIN}_HRUs_lumped.shp'
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({
        'HRU_ID': HRU_IDS,
        'HRU_area': np.full(3, 2.0e6),
        'geometry': [box(i, 50, i + 1, 51) for i in range(3)],
    }, crs='EPSG:4326').to_file(path)


def _dem_intersection_shp(tmp_path: Path, *, reversed_order: bool = False) -> None:
    """DEM intersection carrying HRU_ID + elev_mean. When reversed_order, the
    rows are in the opposite order to the catchment to prove the join is
    order-independent rather than positional."""
    path = (
        tmp_path / 'shapefiles' / 'catchment_intersection'
        / 'with_dem' / 'catchment_with_dem.shp'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    order = slice(None, None, -1) if reversed_order else slice(None)
    gpd.GeoDataFrame({
        'HRU_ID': list(HRU_IDS)[order],
        'elev_mean': ELEV[order],
        'geometry': [box(i, 50, i + 1, 51) for i in range(3)],
    }, crs='EPSG:4326').to_file(path)


def _soilclass_csv(tmp_path: Path) -> None:
    path = tmp_path / 'attributes' / 'soilclass' / f'{DOMAIN}_attributes.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'sand_0-5cm_mean': [40.0, 50.0, 60.0],
        'clay_0-5cm_mean': [15.0, 20.0, 25.0],
    }).to_csv(path, index=False)


def _build(tmp_path: Path) -> Path:
    result = AttributesNetCDFBuilder(project_dir=tmp_path, domain_name=DOMAIN).build()
    assert result is not None and result.exists()
    return result


def test_open_returns_none_when_absent(tmp_path):
    assert open_canonical_attributes(tmp_path, DOMAIN) is None


def test_builder_stamps_hru_id_into_terrain(tmp_path):
    _catchment_shp(tmp_path)
    _dem_intersection_shp(tmp_path)
    result = _build(tmp_path)

    ds = netCDF4.Dataset(str(result), 'r')
    try:
        terrain = ds.groups['terrain']
        assert 'hru_id' in terrain.variables
        assert [str(x) for x in terrain.variables['hru_id'][:]] == HRU_IDS
    finally:
        ds.close()


def test_reader_groups_and_has_group(tmp_path):
    _catchment_shp(tmp_path)
    _dem_intersection_shp(tmp_path)
    _build(tmp_path)

    reader = open_canonical_attributes(tmp_path, DOMAIN)
    assert reader is not None
    assert 'terrain' in reader.groups
    assert reader.has_group('terrain')
    assert not reader.has_group('nonexistent')
    assert reader.has_variable('terrain', 'elev_mean')


def test_per_hru_values_is_order_independent(tmp_path):
    """The join keys on hru_id, so reversed intersection rows still map right."""
    _catchment_shp(tmp_path)
    _dem_intersection_shp(tmp_path, reversed_order=True)
    _build(tmp_path)

    reader = open_canonical_attributes(tmp_path, DOMAIN)
    mapping = reader.per_hru_values('terrain', 'elev_mean')

    assert mapping is not None
    assert mapping['hru_0'] == pytest.approx(870.0)
    assert mapping['hru_1'] == pytest.approx(1320.5)
    assert mapping['hru_2'] == pytest.approx(1980.0)


def test_variable_reduce_mean(tmp_path):
    _catchment_shp(tmp_path)
    _dem_intersection_shp(tmp_path)
    _build(tmp_path)

    reader = open_canonical_attributes(tmp_path, DOMAIN)
    mean_elev = reader.variable('terrain', 'elev_mean', reduce='mean')
    assert mean_elev == pytest.approx(float(ELEV.mean()))


def test_find_variable_for_soil_sand_clay(tmp_path):
    """CLM's path: locate sand/clay in soil_extended by keyword, mean-reduced."""
    _catchment_shp(tmp_path)
    _soilclass_csv(tmp_path)
    _build(tmp_path)

    reader = open_canonical_attributes(tmp_path, DOMAIN)
    assert reader.has_group('soil_extended')
    sand = reader.find_variable('soil_extended', ['sand'], reduce='mean')
    clay = reader.find_variable('soil_extended', ['clay'], reduce='mean')
    assert sand == pytest.approx(50.0)
    assert clay == pytest.approx(20.0)


def test_missing_group_and_variable_return_none(tmp_path):
    _catchment_shp(tmp_path)
    _dem_intersection_shp(tmp_path)
    _build(tmp_path)

    reader = open_canonical_attributes(tmp_path, DOMAIN)
    assert reader.variable('nonexistent', 'x') is None
    assert reader.variable('terrain', 'nonexistent') is None
    assert reader.per_hru_values('nonexistent', 'x') is None
