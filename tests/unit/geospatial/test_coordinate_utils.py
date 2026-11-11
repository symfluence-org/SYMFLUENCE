"""
Tests for coordinate_utils module.

Tests coordinate transformation utilities including:
- Bounding box parsing
- Coordinate format conversion
- CRS transformations
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestBoundingBoxParsing:
    """Tests for bounding box coordinate parsing."""

    def test_parse_bbox_standard_format(self):
        """Test parsing standard lat_max/lon_min/lat_min/lon_max format."""
        bbox_str = "47.0/8.0/46.0/9.0"
        parts = bbox_str.split("/")

        lat_max, lon_min, lat_min, lon_max = map(float, parts)

        assert lat_max == 47.0
        assert lon_min == 8.0
        assert lat_min == 46.0
        assert lon_max == 9.0

    def test_parse_bbox_negative_coordinates(self):
        """Test parsing bounding box with negative coordinates."""
        bbox_str = "45.0/-120.0/44.0/-119.0"
        parts = bbox_str.split("/")

        lat_max, lon_min, lat_min, lon_max = map(float, parts)

        assert lat_max == 45.0
        assert lon_min == -120.0
        assert lat_min == 44.0
        assert lon_max == -119.0

    def test_parse_bbox_decimal_precision(self):
        """Test parsing bounding box with high decimal precision."""
        bbox_str = "46.12345/8.67890/46.00001/8.99999"
        parts = bbox_str.split("/")

        lat_max, lon_min, lat_min, lon_max = map(float, parts)

        assert abs(lat_max - 46.12345) < 1e-10
        assert abs(lon_min - 8.67890) < 1e-10
        assert abs(lat_min - 46.00001) < 1e-10
        assert abs(lon_max - 8.99999) < 1e-10

    def test_parse_bbox_invalid_format(self):
        """Test parsing invalid bounding box format raises error."""
        bbox_str = "invalid/format"

        with pytest.raises(ValueError):
            parts = bbox_str.split("/")
            lat_max, lon_min, lat_min, lon_max = map(float, parts)

    def test_parse_bbox_missing_values(self):
        """Test parsing bounding box with missing values raises error."""
        bbox_str = "47.0/8.0/46.0"  # Missing lon_max

        with pytest.raises(ValueError):
            parts = bbox_str.split("/")
            lat_max, lon_min, lat_min, lon_max = map(float, parts)


class TestBoundingBoxValidation:
    """Tests for bounding box validation."""

    def test_validate_bbox_lat_range(self):
        """Test validation of latitude range."""
        # Valid: lat_max > lat_min
        bbox_valid = (47.0, 8.0, 46.0, 9.0)  # lat_max, lon_min, lat_min, lon_max
        assert bbox_valid[0] > bbox_valid[2]  # lat_max > lat_min

        # Invalid: lat_max <= lat_min
        bbox_invalid = (46.0, 8.0, 47.0, 9.0)
        assert bbox_invalid[0] < bbox_invalid[2]  # lat_max < lat_min is invalid

    def test_validate_bbox_lon_range(self):
        """Test validation of longitude range."""
        # Valid: lon_max > lon_min
        bbox_valid = (47.0, 8.0, 46.0, 9.0)
        assert bbox_valid[3] > bbox_valid[1]  # lon_max > lon_min

        # Note: For bboxes crossing the antimeridian, lon_max < lon_min may be valid

    def test_validate_bbox_latitude_bounds(self):
        """Test latitude values are within valid range (-90 to 90)."""
        valid_lats = [0.0, 45.0, -45.0, 89.9, -89.9]
        invalid_lats = [90.1, -90.1, 180.0, -180.0]

        for lat in valid_lats:
            assert -90 <= lat <= 90

        for lat in invalid_lats:
            assert not (-90 <= lat <= 90)

    def test_validate_bbox_longitude_bounds(self):
        """Test longitude values are within valid range (-180 to 180)."""
        valid_lons = [0.0, 90.0, -90.0, 179.9, -179.9]
        invalid_lons = [180.1, -180.1, 360.0, -360.0]

        for lon in valid_lons:
            assert -180 <= lon <= 180

        for lon in invalid_lons:
            assert not (-180 <= lon <= 180)


class TestCoordinateConversion:
    """Tests for coordinate format conversion."""

    def test_lon_0_360_to_180(self):
        """Test conversion from 0-360 to -180-180 longitude range."""
        lon_360 = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        lon_180 = np.where(lon_360 > 180, lon_360 - 360, lon_360)

        expected = np.array([0.0, 90.0, 180.0, -90.0, 0.0])
        np.testing.assert_array_almost_equal(lon_180, expected)

    def test_lon_180_to_0_360(self):
        """Test conversion from -180-180 to 0-360 longitude range."""
        lon_180 = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])
        lon_360 = np.where(lon_180 < 0, lon_180 + 360, lon_180)

        expected = np.array([180.0, 270.0, 0.0, 90.0, 180.0])
        np.testing.assert_array_almost_equal(lon_360, expected)


class TestUTMZoneDetection:
    """Tests for UTM zone detection from coordinates."""

    def test_utm_zone_from_longitude(self):
        """Test UTM zone calculation from longitude."""
        # UTM zone = floor((lon + 180) / 6) + 1
        test_cases = [
            (-180.0, 1),   # Zone 1: -180 to -174
            (-120.0, 11),  # Zone 11: -126 to -120
            (0.0, 31),     # Zone 31: -6 to 0
            (8.5, 32),     # Zone 32: 6 to 12 (Switzerland)
            (120.0, 51),   # Zone 51: 114 to 120
            (174.0, 60),   # Zone 60: 174 to 180
        ]

        for lon, expected_zone in test_cases:
            zone = int((lon + 180) / 6) + 1
            assert zone == expected_zone, f"lon={lon} expected zone {expected_zone}, got {zone}"

    def test_utm_hemisphere_from_latitude(self):
        """Test UTM hemisphere detection from latitude."""
        test_cases = [
            (46.5, 'N'),   # Northern hemisphere
            (-33.9, 'S'),  # Southern hemisphere
            (0.0, 'N'),    # Equator is typically N
        ]

        for lat, expected_hemisphere in test_cases:
            hemisphere = 'N' if lat >= 0 else 'S'
            assert hemisphere == expected_hemisphere


class TestAreaCalculation:
    """Tests for geographic area calculation utilities."""

    def test_area_degrees_to_meters(self):
        """Test approximate area conversion from degrees to meters."""
        # At latitude 45: 1 degree ~ 78.8 km (lon), 111 km (lat)
        lat = 45.0
        deg_lat = 0.01  # 0.01 degree latitude ~ 1.11 km
        deg_lon = 0.01  # 0.01 degree longitude ~ 0.788 km at lat 45

        # Approximate area in km^2
        area_km2 = deg_lat * 111.0 * deg_lon * 111.0 * np.cos(np.radians(lat))

        # Should be roughly 0.87 km^2
        assert 0.5 < area_km2 < 1.5

    def test_area_increases_at_equator(self):
        """Test that longitude degree spans are largest at equator."""
        # At equator, 1 degree longitude ~ 111 km
        # At lat 60, 1 degree longitude ~ 55.5 km

        lat_equator = 0.0
        lat_60 = 60.0

        lon_span_equator = 111.0 * np.cos(np.radians(lat_equator))  # ~ 111 km
        lon_span_60 = 111.0 * np.cos(np.radians(lat_60))  # ~ 55.5 km

        assert lon_span_equator > lon_span_60


class TestGridCoordinates:
    """Tests for grid coordinate utilities."""

    def test_grid_cell_centers(self):
        """Test calculation of grid cell center coordinates."""
        # Grid from lat 46-47, lon 8-9 with 0.25 degree resolution
        lat_min, lat_max = 46.0, 47.0
        lon_min, lon_max = 8.0, 9.0
        resolution = 0.25

        # Create grid edges
        lat_edges = np.arange(lat_min, lat_max + resolution, resolution)
        lon_edges = np.arange(lon_min, lon_max + resolution, resolution)

        # Cell centers are midpoints between edges
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

        # First cell center should be at (46.125, 8.125)
        assert abs(lat_centers[0] - 46.125) < 0.001
        assert abs(lon_centers[0] - 8.125) < 0.001

    def test_grid_cell_bounds(self):
        """Test calculation of grid cell bounding boxes."""
        center_lat = 46.5
        center_lon = 8.5
        half_size = 0.125  # Half of 0.25 degree cell

        cell_bounds = (
            center_lon - half_size,  # min_lon
            center_lat - half_size,  # min_lat
            center_lon + half_size,  # max_lon
            center_lat + half_size,  # max_lat
        )

        assert cell_bounds == (8.375, 46.375, 8.625, 46.625)


class TestGlobalBoundingBox:
    """A box spanning 360 degrees is the globe, not an empty box."""

    @pytest.mark.parametrize("lon_min,lon_max", [(-180.0, 180.0), (0.0, 360.0)])
    def test_full_circle_normalises_to_global(self, lon_min, lon_max):
        from symfluence.geospatial.coordinate_utils import BoundingBox

        bbox = BoundingBox(-60.0, 60.0, lon_min, lon_max)
        as_360 = bbox.normalize_longitude('0-360')
        assert (as_360.lon_min, as_360.lon_max) == (0.0, 360.0)
        as_180 = bbox.normalize_longitude('-180-180')
        assert (as_180.lon_min, as_180.lon_max) == (-180.0, 180.0)

    def test_partial_box_still_normalised(self):
        from symfluence.geospatial.coordinate_utils import BoundingBox

        out = BoundingBox(60.0, 66.0, -25.0, -13.0).normalize_longitude('0-360')
        assert (out.lon_min, out.lon_max) == (335.0, 347.0)
