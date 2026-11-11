# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Geology area weights outside North America; explicit shapefile roles."""
from __future__ import annotations

import inspect
import logging

import pytest

gpd = pytest.importorskip('geopandas')

from shapely.geometry import box

from symfluence.data.preprocessing.attribute_processors import geology
from symfluence.data.preprocessing.resampling.shapefile_processor import ShapefileProcessor


def test_geology_no_longer_uses_north_america_albers():
    assert 'ESRI:102008' not in inspect.getsource(geology)


def test_ensure_wgs84_role_is_explicit_not_guessed_from_parent_dirs(tmp_path):
    # data root contains "catchment"; the forcing grid shapefile must still be a SOURCE
    root = tmp_path / 'my_catchment_data' / 'shapefiles' / 'forcing'
    root.mkdir(parents=True)
    shp = root / 'forcing_ERA5.shp'
    gpd.GeoDataFrame({'ID': [1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs='EPSG:4326').to_file(shp)

    proc = object.__new__(ShapefileProcessor)
    proc.logger = logging.getLogger('test_roles')
    proc._get_config_value = lambda typed, default=None, dict_key=None: 'HRU_ID'

    assert proc.ensure_wgs84(shp, is_target=False) == shp          # a path, not a (path, field) tuple
    assert proc.ensure_wgs84(shp) == shp                           # heuristic looks at the file name only


def test_ensure_wgs84_target_role_returns_hru_field(tmp_path):
    shp = tmp_path / 'dom_HRUs_GRUs.shp'          # no "catchment" anywhere in the name
    gpd.GeoDataFrame({'HRU_ID': [1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs='EPSG:4326').to_file(shp)
    proc = object.__new__(ShapefileProcessor)
    proc.logger = logging.getLogger('test_roles')
    proc._get_config_value = lambda typed, default=None, dict_key=None: 'HRU_ID'
    result = proc.ensure_wgs84(shp, is_target=True)
    assert isinstance(result, tuple) and result[1] == 'HRU_ID'
