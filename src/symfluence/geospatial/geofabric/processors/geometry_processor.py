# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geometry processing operations for geofabric data.

Provides geometry cleaning, simplification, and winding order correction.
Used primarily by coastal delineation methods.

Refactored from geofabric_utils.py (2026-01-01)
"""

from typing import Any, Optional

from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid


class GeometryProcessor:
    """
    Geometry operations for geofabric processing.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def clean_geometries(geometry) -> Optional[Any]:
        """
        Clean and validate geometry.

        Uses make_valid() to fix invalid geometries (self-intersections, etc.).

        Args:
            geometry: Shapely geometry object

        Returns:
            Cleaned geometry, or None if geometry is None or invalid
        """
        if geometry is None or not geometry.is_valid:
            return None
        try:
            return make_valid(geometry)
        except (ValueError, AttributeError, GEOSException):
            # Shapely geometry operations can raise various errors
            return None

    @staticmethod
    def simplify_geometry(geometry, tolerance: float = 1) -> Any:
        """
        Simplify geometry while preserving topology.

        Args:
            geometry: Shapely geometry object
            tolerance: Simplification tolerance (default: 1)

        Returns:
            Simplified geometry, or original geometry if simplification fails
        """
        try:
            return geometry.simplify(tolerance, preserve_topology=True)
        except (ValueError, AttributeError, GEOSException):
            # Return original geometry if simplification fails
            return geometry

    @staticmethod
    def fix_polygon_winding(geometry) -> Optional[Any]:
        """
        Ensure correct winding order for polygon geometries.

        OGC standard requires exterior ring to be counter-clockwise (CCW)
        and holes to be clockwise (CW).

        Handles both Shapely 2.0+ (orient method) and older versions.

        Args:
            geometry: Shapely Polygon or MultiPolygon

        Returns:
            Geometry with corrected winding order, or None if geometry is None
        """
        if geometry is None:
            return None

        try:
            # Try Shapely 2.0+ method first
            if geometry.geom_type == 'Polygon':
                return geometry.orient(1.0)
            elif geometry.geom_type == 'MultiPolygon':
                return geometry.__class__([geom.orient(1.0) for geom in geometry.geoms])
        except AttributeError:
            # Fallback for older Shapely versions
            if geometry.geom_type == 'Polygon':
                # Make exterior ring counter-clockwise
                if not geometry.exterior.is_ccw:
                    geometry = Polygon(
                        list(geometry.exterior.coords)[::-1],
                        [list(interior.coords)[::-1] for interior in geometry.interiors]
                    )
            elif geometry.geom_type == 'MultiPolygon':
                # Fix each polygon in the multipolygon
                polygons = []
                for poly in geometry.geoms:
                    if not poly.exterior.is_ccw:
                        poly = Polygon(
                            list(poly.exterior.coords)[::-1],
                            [list(interior.coords)[::-1] for interior in poly.interiors]
                        )
                    polygons.append(poly)
                geometry = MultiPolygon(polygons)

        return geometry
