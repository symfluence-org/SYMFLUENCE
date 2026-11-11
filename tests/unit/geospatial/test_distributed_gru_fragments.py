# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Watershed fragments sharing a GRU_ID are dissolved, not dropped."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

gpd = pytest.importorskip('geopandas')

from shapely.geometry import LineString, box

from symfluence.geospatial.geofabric.delineators.distributed_delineator import GeofabricDelineator


def _delineator():
    d = object.__new__(GeofabricDelineator)
    d.logger = MagicMock()
    d.graph = MagicMock()
    d.graph.drop_degenerate_reaches.side_effect = lambda rivers, logger=None: rivers
    d._get_fabric_config = lambda rivers: {'river_id_col': 'LINKNO'}
    return d


def test_diagonal_fragments_are_dissolved_and_area_conserved():
    # Watershed 1 = two squares touching only at a corner (4-connected
    # polygonization returns them as two polygons with the same DN).
    basins = gpd.GeoDataFrame(
        {'DN': [1, 1, 2]},
        geometry=[box(500000, 7000000, 501000, 7001000),
                  box(501000, 7001000, 501500, 7001500),
                  box(501000, 7000000, 502000, 7001000)],
        crs='EPSG:32627',
    )
    rivers = gpd.GeoDataFrame({'LINKNO': [1, 2]},
                              geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
                              crs='EPSG:32627')
    total = basins.geometry.area.sum()

    out, _ = _delineator()._process_geofabric(basins, rivers)

    assert sorted(out['GRU_ID']) == [1, 2]
    assert out['GRU_area'].sum() == pytest.approx(total, rel=1e-6)
    gru1 = out[out['GRU_ID'] == 1].iloc[0]
    assert gru1['GRU_area'] == pytest.approx(1000.0 * 1000.0 + 500.0 * 500.0, rel=1e-6)
    assert gru1.geometry.geom_type == 'MultiPolygon'
    assert np.array_equal(out['gru_to_seg'].values, out['GRU_ID'].values)


def test_polygonize_writes_raster_crs(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    pytest.importorskip('osgeo')
    from rasterio.transform import from_origin

    from symfluence.geospatial.geofabric.processors.gdal_processor import GDALProcessor

    data = np.ones((4, 4), dtype='int32')
    data[:, 2:] = 2
    with rasterio.open(tmp_path / 'elv-watersheds.tif', 'w', driver='GTiff', height=4, width=4, count=1,
                       dtype='int32', crs='EPSG:32627', transform=from_origin(500000, 7000000, 100, 100),
                       nodata=-1) as dst:
        dst.write(data, 1)

    GDALProcessor(MagicMock()).run_gdal_processing(tmp_path)

    out = gpd.read_file(tmp_path / 'basin-watersheds.shp')
    assert out.crs is not None and out.crs.to_epsg() == 32627
