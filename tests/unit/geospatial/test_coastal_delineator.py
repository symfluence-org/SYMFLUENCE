# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the coastal-strip division fallbacks in CoastalWatershedDelineator.

These pin the behavior of the two fallback methods used when the primary
Voronoi division fails (RTI architecture review item 20):

- ``_divide_coastal_strip_by_buffer_method``: progressive metre-band claims,
  then nearest-basin assignment of leftover fragments via ``sjoin_nearest``.
- ``_divide_coastal_strip_by_extending_boundaries``: per-basin 1 km buffer
  claims.

Before the rewrite the buffer method used degree-sized bands (millimetres in
the projected CRS the caller supplies), which collapsed the whole contiguous
strip onto the first basin in row order; these tests guard the fixed
nearest-based partition.
"""
from __future__ import annotations

import logging
import types

import pytest

pytest.importorskip("geopandas")

import geopandas as gpd  # noqa: E402
from shapely.geometry import box  # noqa: E402

from symfluence.geospatial.geofabric.delineators.coastal_delineator import (  # noqa: E402
    CoastalWatershedDelineator,
)

pytestmark = [pytest.mark.unit]

UTM_CRS = "EPSG:32627"


@pytest.fixture
def river_basins():
    """Three 10x10 km basins side by side along the x axis (projected CRS)."""
    return gpd.GeoDataFrame(
        {'GRU_ID': [1, 2, 3]},
        geometry=[
            box(0, 0, 10_000, 10_000),
            box(10_000, 0, 20_000, 10_000),
            box(20_000, 0, 30_000, 10_000),
        ],
        crs=UTM_CRS,
    )


@pytest.fixture
def coastal_strip():
    """A 2 km coastal band along the full southern edge of the basins."""
    return gpd.GeoDataFrame(geometry=[box(0, -2_000, 30_000, 0)], crs=UTM_CRS)


@pytest.fixture
def delineator():
    """The methods under test only use ``self.logger``."""
    return types.SimpleNamespace(logger=logging.getLogger("test_coastal"))


class TestBufferMethod:
    def test_partitions_strip_among_touching_basins(self, delineator, river_basins, coastal_strip):
        result = CoastalWatershedDelineator._divide_coastal_strip_by_buffer_method(
            delineator, coastal_strip, river_basins
        )

        assert result is not None and not result.empty
        # Every basin claims a share; the strip must not collapse onto basin 1
        # (the pre-fix degenerate behavior assigned it 59.9998 of 60 km2).
        assert set(result['parent_basin']) == {1, 2, 3}
        areas = result.set_index('parent_basin').geometry.area
        total = coastal_strip.geometry.area.sum()
        # No coastal area is lost.
        assert areas.sum() == pytest.approx(total, rel=1e-6)
        # The claim bands partition the first kilometre evenly: each basin gets
        # at least (almost) its own 10 km2 inner band. The contiguous outer
        # band is a single fragment equidistant to all three basins and goes
        # whole to the tie-winning first match — documented fallback behavior.
        for gru_id in (1, 2, 3):
            assert areas[gru_id] > 9.5e6  # ~its 10 km2 inner band, in m2

    def test_detached_fragment_goes_to_nearest_basin(self, delineator, river_basins):
        # One fragment clearly nearest basin 3, far beyond the 1 km claim bands.
        strip = gpd.GeoDataFrame(
            geometry=[box(24_000, -7_000, 26_000, -5_000)], crs=UTM_CRS
        )
        result = CoastalWatershedDelineator._divide_coastal_strip_by_buffer_method(
            delineator, strip, river_basins
        )

        assert result is not None
        assert list(result['parent_basin']) == [3]
        assert result.geometry.area.sum() == pytest.approx(strip.geometry.area.sum(), rel=1e-9)

    def test_reprojects_basins_when_crs_differs(self, delineator, river_basins):
        # The caller reprojects the strip to UTM but historically passed basins
        # in their original CRS; the method must align them before intersecting.
        strip = gpd.GeoDataFrame(
            geometry=[box(24_000, -7_000, 26_000, -5_000)], crs=UTM_CRS
        )
        basins_geographic = river_basins.to_crs("EPSG:4326")
        result = CoastalWatershedDelineator._divide_coastal_strip_by_buffer_method(
            delineator, strip, basins_geographic
        )

        assert result is not None
        assert list(result['parent_basin']) == [3]

    def test_output_crs_matches_strip(self, delineator, river_basins, coastal_strip):
        result = CoastalWatershedDelineator._divide_coastal_strip_by_buffer_method(
            delineator, coastal_strip, river_basins
        )
        assert result.crs == coastal_strip.crs

    def test_empty_strip_returns_none(self, delineator, river_basins):
        empty = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1).buffer(0).difference(box(0, 0, 1, 1))], crs=UTM_CRS)
        result = CoastalWatershedDelineator._divide_coastal_strip_by_buffer_method(
            delineator, empty, river_basins
        )
        assert result is None


class TestExtendingBoundariesMethod:
    def test_claims_one_km_band_per_basin(self, delineator, river_basins, coastal_strip):
        result = CoastalWatershedDelineator._divide_coastal_strip_by_extending_boundaries(
            delineator, coastal_strip, river_basins
        )

        assert result is not None and not result.empty
        assert set(result['parent_basin']) == {1, 2, 3}
        # This method claims a single 1 km buffer band; with a 2 km strip the
        # claimed area is the inner band only.
        assert result.geometry.area.sum() < coastal_strip.geometry.area.sum()


class TestVoronoiDivision:
    """The primary division must cover the whole strip (outer regions included)."""

    def _tessellate(self, delineator, river_basins, strip):
        centroids = river_basins.copy()
        centroids.geometry = river_basins.geometry.centroid.buffer(1.0)
        return CoastalWatershedDelineator._create_voronoi_tessellation(delineator, centroids, cover=strip)

    def test_outer_regions_are_kept_and_strip_area_conserved(self, delineator):
        # 3 x 2 block of basins with a coastal ring around it: every basin is on
        # the hull, so every plain-Voronoi region is unbounded (previously dropped)
        basins = gpd.GeoDataFrame(
            {'GRU_ID': range(1, 7)},
            geometry=[box(10_000 * i, 10_000 * j, 10_000 * (i + 1), 10_000 * (j + 1))
                      for j in range(2) for i in range(3)],
            crs=UTM_CRS,
        )
        land = box(-3_000, -3_000, 33_000, 23_000)
        strip = gpd.GeoDataFrame(geometry=[land.difference(basins.union_all())], crs=UTM_CRS)

        voronoi = self._tessellate(delineator, basins, strip)
        assert voronoi is not None and len(voronoi) == 6

        pieces = gpd.overlay(strip, voronoi, how='intersection')
        assert pieces.geometry.area.sum() == pytest.approx(strip.geometry.area.sum(), rel=1e-6)
        assert CoastalWatershedDelineator._check_strip_area_conserved(delineator, strip, pieces)

    def test_works_with_fewer_than_four_basins(self, delineator, river_basins, coastal_strip):
        voronoi = self._tessellate(delineator, river_basins, coastal_strip)
        assert voronoi is not None and len(voronoi) == 3
        pieces = gpd.overlay(coastal_strip, voronoi, how='intersection')
        assert pieces.geometry.area.sum() == pytest.approx(coastal_strip.geometry.area.sum(), rel=1e-6)
        # each third of the strip goes to the basin above it
        assert sorted(round(a) for a in pieces.geometry.area) == [20_000_000] * 3

    def test_lost_area_is_reported(self, river_basins, coastal_strip):
        from unittest.mock import MagicMock
        d = types.SimpleNamespace(logger=MagicMock())
        half = gpd.GeoDataFrame(geometry=[box(0, -2_000, 15_000, 0)], crs=UTM_CRS)
        assert not CoastalWatershedDelineator._check_strip_area_conserved(d, coastal_strip, half)
        d.logger.error.assert_called_once()
