# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Geofabric subsetting writes compact integer ids with a consistent topology."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

gpd = pytest.importorskip('geopandas')

from shapely.geometry import LineString, box

from symfluence.core.exceptions import GeospatialError
from symfluence.geospatial.geofabric.delineators.subsetter import GeofabricSubsetter

INT32_MAX = np.iinfo('int32').max


def _subsetter():
    sub = object.__new__(GeofabricSubsetter)
    sub.logger = MagicMock()
    return sub


def _cells(n):
    return [box(-19.0 + 0.1 * i, 64.0, -18.9 + 0.1 * i, 64.1) for i in range(n)]


def _lines(n):
    return [LineString([(-19.0 + 0.1 * i, 64.05), (-18.9 + 0.1 * i, 64.05)]) for i in range(n)]


def test_hydrobasins_ids_are_compacted_with_consistent_topology():
    hybas = [7120034520, 7120034530, 7120034540]            # > 2**31
    next_down = [7120034530, 7120034540, 7120099990]        # last drains out of the subset
    basins = gpd.GeoDataFrame({'HYBAS_ID': hybas, 'NEXT_DOWN': next_down}, geometry=_cells(3), crs='EPSG:4326')
    rivers = basins.copy()

    _subsetter()._add_symfluence_columns(basins, rivers, 'HYDROSHEDS')

    assert basins['GRU_ID'].tolist() == [1, 2, 3]
    assert basins['gru_to_seg'].tolist() == [1, 2, 3]
    assert rivers['LINKNO'].tolist() == [1, 2, 3]
    assert rivers['DSLINKNO'].tolist() == [2, 3, 0]
    assert basins['SRC_ID'].tolist() == [str(v) for v in hybas]
    assert rivers['SRC_DSLNK'].tolist() == [str(v) for v in next_down]
    for frame, col in [(basins, 'GRU_ID'), (rivers, 'LINKNO'), (rivers, 'DSLINKNO')]:
        assert frame[col].max() <= INT32_MAX


def test_merit_ids_that_fit_are_left_alone():
    basins = gpd.GeoDataFrame({'COMID': [71000012, 71000013]}, geometry=_cells(2), crs='EPSG:4326')
    rivers = gpd.GeoDataFrame({'COMID': [71000012, 71000013], 'NextDownID': [71000013, 0],
                               'slope': [0.01, 0.01]}, geometry=_lines(2), crs='EPSG:4326')
    _subsetter()._add_symfluence_columns(basins, rivers, 'MERIT')
    assert basins['GRU_ID'].tolist() == [71000012, 71000013]
    assert 'SRC_ID' not in basins.columns


def test_nws_string_ids_become_one_integer_namespace():
    basins = gpd.GeoDataFrame({'divide_id': ['cat-10', 'cat-11', 'cat-12'],
                               'toid': ['nex-11', 'nex-12', 'tnx-99']}, geometry=_cells(3), crs='EPSG:5070')
    rivers = gpd.GeoDataFrame({'id': ['wb-10', 'wb-11', 'wb-12'],
                               'toid': ['nex-11', 'nex-12', 'tnx-99']}, geometry=_lines(3), crs='EPSG:5070')
    sub = _subsetter()
    sub._add_symfluence_columns(basins, rivers, 'NWS')

    assert basins['GRU_ID'].tolist() == [10, 11, 12]
    assert basins['gru_to_seg'].tolist() == rivers['LINKNO'].tolist() == [10, 11, 12]
    assert rivers['DSLINKNO'].tolist() == [11, 12, 0]       # every link points at a LINKNO or 0
    assert set(rivers['DSLINKNO']) - {0} <= set(rivers['LINKNO'])
    assert basins['SRC_ID'].tolist() == ['cat-10', 'cat-11', 'cat-12']
    sub.logger.warning.assert_called_once()


def test_nws_unexpected_id_format_raises():
    basins = gpd.GeoDataFrame({'divide_id': ['basin_a'], 'toid': ['nex-1']}, geometry=_cells(1), crs='EPSG:5070')
    rivers = gpd.GeoDataFrame({'id': ['wb-1'], 'toid': ['nex-1']}, geometry=_lines(1), crs='EPSG:5070')
    with pytest.raises(GeospatialError, match="cat-"):
        _subsetter()._add_symfluence_columns(basins, rivers, 'NWS')


def test_merit_fabric_config_includes_up4(tmp_path):
    sub = _subsetter()
    sub.hydrofabric_types = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GeofabricSubsetter.__mro__[1], '__init__', lambda self, config, logger: None)
        GeofabricSubsetter.__init__(sub, {}, MagicMock())
    assert sub.hydrofabric_types['MERIT']['upstream_cols'] == ['up1', 'up2', 'up3', 'up4']


def test_grus_discretization_rejects_ids_beyond_int32(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin

    from symfluence.core.exceptions import DiscretizationError
    from symfluence.geospatial.discretization.attributes import grus

    dem = tmp_path / 'dem.tif'
    with rasterio.open(dem, 'w', driver='GTiff', height=10, width=30, count=1, dtype='float32',
                       crs='EPSG:4326', transform=from_origin(-19.0, 64.1, 0.01, 0.01)) as dst:
        dst.write(np.full((10, 30), 500.0, dtype='float32'), 1)
    shp = tmp_path / 'basins.shp'
    gpd.GeoDataFrame({'GRU_ID': [7120034520, 7120034530], 'GRU_area': [1.0e6, 1.0e6]},
                     geometry=_cells(2), crs='EPSG:4326').to_file(shp)

    disc = MagicMock()
    disc.dem_path = dem
    disc.delineation_suffix = 'subset'
    disc._get_config_value.return_value = False
    disc._get_file_path.side_effect = [shp, tmp_path / 'hrus.shp']
    disc._read_shapefile.side_effect = lambda p: gpd.read_file(p)
    with pytest.raises(DiscretizationError, match='32-bit'):
        grus.discretize(disc)
