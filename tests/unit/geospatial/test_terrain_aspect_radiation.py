# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Aspect, slope and annual radiation rasters derived from a DEM."""
from __future__ import annotations

import logging

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('pvlib')

from rasterio.transform import from_origin

from symfluence.geospatial.raster_utils import (
    calculate_annual_radiation,
    calculate_aspect,
    quantile_class_thresholds,
    radiation_raster_is_current,
    terrain_slope_aspect,
)

LOG = logging.getLogger('test_terrain')
N = 20
PIXEL = 30.0  # metres


def _plane(rising_towards: str, rise_per_pixel: float = PIXEL) -> np.ndarray:
    """A plane rising towards the given compass direction (row 0 = north)."""
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    return {
        'north': (N - 1 - rows), 'south': rows, 'east': cols, 'west': (N - 1 - cols),
    }[rising_towards] * rise_per_pixel + 100.0


def _write(path, data, crs='EPSG:32627', transform=None, nodata=-9999.0):
    transform = transform or from_origin(500000, 7100000, PIXEL, PIXEL)
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data.astype('float32'), 1)
    return path


@pytest.mark.parametrize("rising,facing_deg,facing_class", [
    ('north', 180.0, 3), ('south', 0.0, 1), ('east', 270.0, 4), ('west', 90.0, 2),
])
def test_aspect_is_downslope_compass_bearing(tmp_path, rising, facing_deg, facing_class):
    dem = _plane(rising)
    transform = from_origin(500000, 7100000, PIXEL, PIXEL)
    slope, aspect, valid = terrain_slope_aspect(dem, transform, rasterio.crs.CRS.from_epsg(32627), -9999.0)
    assert valid.all()
    np.testing.assert_allclose(aspect % 360.0, facing_deg, atol=1e-6)
    # 30 m rise per 30 m pixel = 45 degrees, NOT arctan(30)
    np.testing.assert_allclose(np.degrees(slope), 45.0, atol=1e-6)

    out = calculate_aspect(_write(tmp_path / 'dem.tif', dem), tmp_path / 'aspect.tif', 4, LOG)
    assert out is not None
    with rasterio.open(out) as src:
        assert set(np.unique(src.read(1))) == {facing_class}   # 1=N, 2=E, 3=S, 4=W


def test_geographic_dem_uses_metric_spacing():
    # 0.001 deg pixels at 64N: ~111 m north-south but only ~49 m east-west
    transform = from_origin(-19.0, 64.0 + N * 0.0005, 0.001, 0.001)
    crs = rasterio.crs.CRS.from_epsg(4326)
    dz = 50.0
    slope_ns, _, _ = terrain_slope_aspect(_plane('north', dz), transform, crs, None)
    slope_ew, aspect_ew, _ = terrain_slope_aspect(_plane('east', dz), transform, crs, None)
    lat = 64.0
    np.testing.assert_allclose(np.tan(slope_ns), dz / 111194.93e-3, rtol=1e-3)
    np.testing.assert_allclose(
        np.tan(slope_ew), dz / (111194.93e-3 * np.cos(np.radians(lat))), rtol=2e-3)
    np.testing.assert_allclose(aspect_ew, 270.0, atol=1e-6)


def test_nodata_is_not_differentiated_as_elevation(tmp_path):
    dem = _plane('north')
    dem[8:12, 8:12] = -9999.0
    transform = from_origin(500000, 7100000, PIXEL, PIXEL)
    slope, aspect, valid = terrain_slope_aspect(dem, transform, None, -9999.0)
    assert not valid[8:12, 8:12].any() and np.isnan(slope[9, 9])
    # neighbours of the hole keep the true 45 degree slope
    np.testing.assert_allclose(np.degrees(slope[valid]), 45.0, atol=1e-6)
    np.testing.assert_allclose(aspect[valid], 180.0, atol=1e-6)

    out = calculate_aspect(_write(tmp_path / 'dem.tif', dem), tmp_path / 'aspect.tif', 8, LOG)
    with rasterio.open(out) as src:
        classes = src.read(1)
    assert (classes[8:12, 8:12] == -9999).all()
    assert set(np.unique(classes[valid])) == {5}   # south of 8 classes


def test_annual_radiation_iceland_south_beats_north(tmp_path):
    """At 64N a south-facing slope receives more than a north-facing one; the
    old raster sampled 00:00 UTC only and was zero everywhere in Iceland."""
    # roof-shaped ridge running east-west: north half faces north, south half faces south
    rows = np.arange(N, dtype=float)
    ridge = (N / 2 - np.abs(rows - (N - 1) / 2.0)) * (PIXEL * 0.5) + 500.0
    dem = np.repeat(ridge[:, None], N, axis=1)
    dem[0, 0] = -9999.0
    out = calculate_annual_radiation(_write(tmp_path / 'dem.tif', dem), tmp_path / 'rad.tif', LOG)
    assert out is not None and radiation_raster_is_current(out)
    with rasterio.open(out) as src:
        rad = src.read(1)
        assert src.nodata == -9999.0
    assert rad[0, 0] == -9999.0
    north_face = rad[2:8, 2:-2]
    south_face = rad[12:18, 2:-2]
    assert north_face.min() > 100.0            # kWh m-2 yr-1, not zero
    assert south_face.max() < 3000.0
    assert south_face.mean() > 1.3 * north_face.mean()


def test_radiation_classes_are_a_count_not_a_band_width(tmp_path):
    values = np.linspace(100000.0, 140000.0, N * N).reshape(N, N)
    raster = _write(tmp_path / 'rad.tif', values)
    thresholds = quantile_class_thresholds(raster, 5)
    assert len(thresholds) == 6               # 5 classes, not 8000 bands
    assert thresholds[0] == pytest.approx(100000.0) and thresholds[-1] == pytest.approx(140000.0)
    counts, _ = np.histogram(values, bins=thresholds)
    assert counts.min() >= 0.9 * values.size / 5
