from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.exceptions import GeospatialError
from symfluence.models.wflow.glacier_geometry import mapped_glaciers


def test_fraction_and_volume_conservation(tmp_path):
    transform = from_origin(0, 200, 100, 100)
    outlines = tmp_path / 'ice.gpkg'
    gpd.GeoDataFrame(geometry=[box(0, 0, 100, 200)], crs=3057).to_file(outlines)
    with rasterio.open(tmp_path / 'test_thickness.tif', 'w', driver='GTiff',
                       height=2, width=2, count=1, dtype='float64', crs=3057,
                       transform=transform, nodata=-9999) as dst:
        dst.write(np.array([[100., 200.], [300., 400.]]), 1)
    fraction, store, meta = mapped_glaciers(np.ones((1, 1), bool),
        from_origin(0, 200, 200, 200), outlines, tmp_path, box(0, 0, 100, 200))
    assert fraction[0, 0] == .5
    assert store[0, 0] == pytest.approx(200*917)
    assert meta['initial_ice_water_km3'] == pytest.approx(200*917*.5*200**2/1e12)


def test_rejects_lost_glacier_boundary(tmp_path):
    outlines = tmp_path / 'ice.gpkg'
    gpd.GeoDataFrame(geometry=[box(0, 0, 200, 200)], crs=3057).to_file(outlines)
    with rasterio.open(tmp_path / 'test_thickness.tif', 'w', driver='GTiff',
                       height=2, width=2, count=1, dtype='float64', crs=3057,
                       transform=from_origin(0, 200, 100, 100)) as dst:
        dst.write(np.ones((2, 2))*100, 1)
    with pytest.raises(GeospatialError, match='boundary'):
        mapped_glaciers(np.array([[True, False], [True, False]]),
                        from_origin(0, 200, 100, 100), outlines, tmp_path, box(0, 0, 200, 200))
