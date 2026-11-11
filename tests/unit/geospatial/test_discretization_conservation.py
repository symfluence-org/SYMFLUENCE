# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Discretization: co-registration, coverage, MIN_HRU_SIZE and area conservation."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

gpd = pytest.importorskip('geopandas')
rasterio = pytest.importorskip('rasterio')

from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.exceptions import DiscretizationError
from symfluence.geospatial.discretization.attributes import combined
from symfluence.geospatial.discretization.core import DomainDiscretizer
from symfluence.geospatial.discretization.hru_utils import (
    check_area_conservation,
    fill_nodata_from_nearest,
    merge_small_hrus,
)
from symfluence.geospatial.zonal import fill_empty_zones, zonal_stats_covered

LOG = logging.getLogger('test_discretization_conservation')
X0, Y0 = 500000.0, 7100000.0
CRS = 'EPSG:32627'


def _raster(path, data, pixel, *, x0=X0, y0=Y0, crs=CRS, nodata=None, dtype='float32'):
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                       count=1, dtype=dtype, crs=crs, transform=from_origin(x0, y0, pixel, pixel),
                       nodata=nodata) as dst:
        dst.write(data.astype(dtype), 1)
    return path


def _discretizer(dem_path=None, min_hru_size=0.0):
    disc = object.__new__(DomainDiscretizer)
    disc.logger = MagicMock()
    disc.dem_path = dem_path
    disc._get_config_value = lambda typed, default=None, dict_key=None: (
        min_hru_size if dict_key == 'MIN_HRU_SIZE' else default)
    return disc


def _gru(xmin=X0 + 100, ymin=Y0 - 1900, xmax=X0 + 1900, ymax=Y0 - 100, crs=CRS, gru_id=1):
    return gpd.GeoDataFrame({'GRU_ID': [gru_id]}, geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)


def _elev_land(tmp_path, land_pixel=100.0, land_x0=X0, land_crs=CRS):
    """Elevation rising eastward (two 200 m bands) and land classes split north/south."""
    cols = np.arange(20)
    elev = np.repeat((100.0 + 20.0 * cols)[None, :], 20, axis=0)        # 100..480 m
    dem = _raster(tmp_path / 'dem.tif', elev, 100.0, nodata=-9999.0)
    n = int(round(2000.0 / land_pixel))
    land = np.ones((n, n))
    land[n // 2:, :] = 2
    land_path = _raster(tmp_path / 'land.tif', land, land_pixel, x0=land_x0, crs=land_crs,
                        nodata=255, dtype='uint8')
    info = {
        'elevation': {'path': dem, 'type': 'continuous', 'band_size': 200.0, 'class_name': 'elevClass'},
        'landclass': {'path': land_path, 'type': 'discrete', 'class_name': 'landClass'},
    }
    return info


def _combo_areas(hrus):
    proj = hrus.to_crs(CRS)
    return {(int(r.elevClass), int(r.landClass)): r.geometry.area for r in proj.itertuples()}


class TestCombinedCoRegistration:
    def test_same_grid_reference(self, tmp_path):
        info = _elev_land(tmp_path)
        hrus = combined._create_combined_attribute_hrus(_discretizer(), _gru(), info, ['elevation', 'landclass'])
        areas = _combo_areas(hrus)
        assert set(areas) == {(1, 1), (1, 2), (2, 1), (2, 2)}
        assert sum(areas.values()) == pytest.approx(1800.0 * 1800.0, rel=1e-9)

    def test_different_resolution_no_broadcast_error(self, tmp_path):
        # land raster at 50 m against a 100 m DEM used to raise a broadcast ValueError
        info = _elev_land(tmp_path, land_pixel=50.0)
        hrus = combined._create_combined_attribute_hrus(_discretizer(), _gru(), info, ['elevation', 'landclass'])
        areas = _combo_areas(hrus)
        assert set(areas) == {(1, 1), (1, 2), (2, 1), (2, 2)}
        assert sum(areas.values()) == pytest.approx(1800.0 * 1800.0, rel=1e-9)
        # north/south split at the raster's middle row is preserved
        assert areas[(1, 1)] == pytest.approx(areas[(1, 2)], rel=1e-9)

    def test_offset_grid_is_registered_by_coordinates(self, tmp_path):
        # Land raster shifted 1 km west and split east/west at x = X0 + 1000 in
        # map coordinates. Reading it by array position (old behaviour) would
        # put the boundary 1 km off.
        info = _elev_land(tmp_path)
        land = np.ones((20, 30))
        land[:, 20:] = 2                      # x >= X0 - 1000 + 2000 = X0 + 1000
        info['landclass']['path'] = _raster(tmp_path / 'land_offset.tif', land, 100.0,
                                            x0=X0 - 1000.0, nodata=255, dtype='uint8')
        hrus = combined._create_combined_attribute_hrus(_discretizer(), _gru(), info, ['elevation', 'landclass'])
        proj = hrus.to_crs(CRS)
        for row in proj.itertuples():
            minx, _, maxx, _ = row.geometry.bounds
            if int(row.landClass) == 1:
                assert maxx <= X0 + 1000.0 + 1e-6
            else:
                assert minx >= X0 + 1000.0 - 1e-6

    def test_grus_in_other_crs_are_reprojected(self, tmp_path):
        info = _elev_land(tmp_path)
        grus = _gru().to_crs('EPSG:4326')
        hrus = combined._create_combined_attribute_hrus(_discretizer(), grus, info, ['elevation', 'landclass'])
        assert len(hrus) == 4
        assert hrus.crs == grus.crs
        assert hrus.to_crs(CRS).geometry.area.sum() == pytest.approx(1800.0 * 1800.0, rel=1e-4)

    def test_aspect_class_is_written(self, tmp_path):
        info = _elev_land(tmp_path)
        info['aspect'] = dict(info.pop('landclass'), class_name='aspectClass')
        hrus = combined._create_combined_attribute_hrus(_discretizer(), _gru(), info, ['elevation', 'aspect'])
        assert 'aspectClass' in hrus.columns
        assert sorted(set(hrus['aspectClass'])) == [1, 2]

    def test_raster_not_covering_gru_raises_clear_error(self, tmp_path):
        info = _elev_land(tmp_path)
        far = _gru(X0 + 50000, Y0 - 1900, X0 + 51000, Y0 - 100)
        with pytest.raises(DiscretizationError, match='GRU 1'):
            combined._create_combined_attribute_hrus(_discretizer(), far, info, ['elevation', 'landclass'])

    def test_nodata_holes_are_filled_and_area_conserved(self, tmp_path):
        info = _elev_land(tmp_path)
        with rasterio.open(info['landclass']['path'], 'r+') as dst:
            land = dst.read(1)
            land[4:8, 4:8] = 255
            dst.write(land, 1)
        disc = _discretizer()
        grus = _gru()
        hrus = combined._create_combined_attribute_hrus(disc, grus, info, ['elevation', 'landclass'])
        assert hrus.geometry.area.sum() == pytest.approx(1800.0 * 1800.0, rel=1e-9)
        assert set(hrus['landClass']) == {1, 2}
        assert any('no landclass data' in str(c) for c in disc.logger.warning.call_args_list)
        assert check_area_conservation(hrus, grus, LOG) < 1e-9


class TestSingleAttributeNodataHoles:
    def test_holes_filled_from_nearest_class(self, tmp_path):
        land = np.ones((20, 20))
        land[:, 10:] = 2
        land[5:9, 2:6] = 255          # hole inside class 1
        raster = _raster(tmp_path / 'land.tif', land, 100.0, nodata=255, dtype='uint8')
        disc = _discretizer()
        grus = _gru()
        hrus = disc._create_multipolygon_hrus(grus, raster, None, 'landClass')
        assert len(hrus) == 2
        assert hrus.geometry.area.sum() == pytest.approx(1800.0 * 1800.0, rel=1e-9)
        assert disc.logger.warning.called

    def test_fill_helper(self):
        data = np.array([[1, 1, 9], [9, 2, 2]])
        valid = data != 9
        inside = np.ones_like(valid)
        filled, n = fill_nodata_from_nearest(data, valid, inside)
        assert n == 2 and filled[0, 2] in (1, 2) and filled[1, 0] in (1, 2)
        assert not (filled == 9).any()


class TestAreaConservationWarning:
    def test_warns_when_hrus_do_not_cover_gru(self):
        grus = _gru()
        hrus = gpd.GeoDataFrame({'GRU_ID': [1], 'HRU_ID': [1]},
                                geometry=[box(X0 + 100, Y0 - 1900, X0 + 1900, Y0 - 400)], crs=CRS)
        logger = MagicMock()
        rel = check_area_conservation(hrus, grus, logger)
        assert rel == pytest.approx(300.0 / 1800.0, rel=1e-9)
        logger.warning.assert_called_once()

    def test_silent_when_conserved(self):
        grus = _gru()
        logger = MagicMock()
        assert check_area_conservation(grus.assign(HRU_ID=1), grus, logger) == pytest.approx(0.0, abs=1e-12)
        logger.warning.assert_not_called()


class TestMinHruSize:
    def _hrus(self):
        # GRU 1: big west block (2 km2), tiny strip (0.02 km2) touching both big
        # blocks but sharing a longer edge with the east block (2 km2);
        # GRU 2: one tiny HRU on its own
        return gpd.GeoDataFrame(
            {
                'GRU_ID': [1, 1, 1, 2],
                'HRU_ID': [1, 2, 3, 4],
                'landClass': [10, 20, 30, 40],
            },
            geometry=[
                box(X0, Y0, X0 + 1000, Y0 + 2000),
                box(X0 + 1000, Y0, X0 + 1100, Y0 + 200),          # sliver: 100 m x 200 m
                box(X0 + 1100, Y0, X0 + 2100, Y0 + 2000).union(box(X0 + 1000, Y0 + 200, X0 + 1100, Y0 + 2000)),
                box(X0 + 5000, Y0, X0 + 5100, Y0 + 100),
            ],
            crs=CRS,
        )

    def test_small_hru_joins_longest_shared_boundary_neighbour(self):
        hrus = self._hrus()
        merged = merge_small_hrus(hrus, 1.0, LOG)
        assert sorted(merged['landClass']) == [10, 30, 40]         # sliver gone, lone tiny HRU kept
        assert merged.geometry.area.sum() == pytest.approx(hrus.geometry.area.sum(), rel=1e-12)
        east = merged[merged['landClass'] == 30].geometry.iloc[0]
        # the sliver shares 200 m with the west block and 100 + 200 m with the east block
        assert east.area == pytest.approx(2000.0 * 1100.0, rel=1e-12)

    def test_zero_threshold_is_a_no_op(self):
        hrus = self._hrus()
        assert merge_small_hrus(hrus, 0.0, LOG) is hrus

    def test_clean_and_prepare_applies_min_hru_size_km2(self, tmp_path):
        dem = _raster(tmp_path / 'dem.tif', np.full((60, 80), 500.0), 100.0, y0=Y0 + 4000, nodata=-9999.0)
        hrus = self._hrus()
        out = _discretizer(dem, min_hru_size=1.0)._clean_and_prepare_hru_gdf(hrus)
        assert len(out) == 3
        assert out['HRU_ID'].tolist() == [1, 2, 3]
        assert out['HRU_area'].sum() == pytest.approx(hrus.geometry.area.sum(), rel=1e-6)
        out0 = _discretizer(dem, min_hru_size=0.0)._clean_and_prepare_hru_gdf(self._hrus())
        assert len(out0) == 4


class TestZonalCoverage:
    def _setup(self, tmp_path):
        # 100 m DEM; one HRU is a 20 m square that contains no pixel centre
        elev = np.repeat((100.0 + 10.0 * np.arange(20))[None, :], 20, axis=0)
        dem = _raster(tmp_path / 'dem.tif', elev, 100.0, nodata=-9999.0)
        hrus = gpd.GeoDataFrame(
            {'GRU_ID': [1, 1, 2], 'HRU_ID': [1, 2, 3], 'landClass': [1, 2, 3]},
            geometry=[
                box(X0, Y0 - 2000, X0 + 1000, Y0),
                box(X0 + 1205, Y0 - 1020, X0 + 1225, Y0 - 1000),     # tiny, inside pixel col 12
                box(X0 + 90000, Y0 - 100, X0 + 90100, Y0),           # entirely off the raster
            ],
            crs=CRS,
        )
        return dem, hrus, elev

    def test_tiny_zone_gets_touched_pixel_value(self, tmp_path):
        _, hrus, elev = self._setup(tmp_path)
        results, empty = zonal_stats_covered(
            hrus.geometry, elev, affine=from_origin(X0, Y0, 100.0, 100.0), nodata=-9999.0)
        assert results[1]['mean'] == pytest.approx(220.0)
        assert empty == [2]
        logger = MagicMock()
        unfilled = fill_empty_zones(results, empty, hrus.geometry, group_ids=[1, 1, 2],
                                    zone_ids=[1, 2, 3], logger=logger)
        assert unfilled == []
        assert results[2]['mean'] == pytest.approx(220.0)        # nearest zone with data
        assert '3<-2' in str(logger.warning.call_args)

    def test_donor_prefers_same_group(self, tmp_path):
        _, hrus, elev = self._setup(tmp_path)
        results, empty = zonal_stats_covered(
            hrus.geometry, elev, affine=from_origin(X0, Y0, 100.0, 100.0), nodata=-9999.0)
        fill_empty_zones(results, empty, hrus.geometry, group_ids=[1, 2, 1], logger=MagicMock())
        assert results[2]['mean'] == pytest.approx(results[0]['mean'])

    def test_hru_elev_mean_never_minus_9999(self, tmp_path):
        dem, hrus, _ = self._setup(tmp_path)
        out = _discretizer(dem)._clean_and_prepare_hru_gdf(hrus)
        assert (out['elev_mean'] > 0).all()
        assert out.set_index('landClass').loc[2, 'elev_mean'] == pytest.approx(220.0)
