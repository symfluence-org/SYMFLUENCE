# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geometry Validator

Validates and repairs geometries in GeoDataFrames.
"""

import logging

from shapely.validation import make_valid


class GeometryValidator:
    """
    Validates and repairs geometries in GeoDataFrames.

    Dissolve operations can create invalid geometries (self-intersections,
    invalid rings, etc.) that cause bus errors or crashes when processed
    by EASYMORE or other GIS operations.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize geometry validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def validate_and_repair(self, gdf):
        """
        Validate and repair geometries in a GeoDataFrame.

        Args:
            gdf: GeoDataFrame with potentially invalid geometries

        Returns:
            GeoDataFrame with validated and repaired geometries
        """
        invalid_count = 0
        repaired_count = 0

        def repair_geometry(geom):
            nonlocal invalid_count, repaired_count

            if geom is None or geom.is_empty:
                return geom

            if not geom.is_valid:
                invalid_count += 1
                try:
                    repaired = make_valid(geom)
                    if repaired.is_valid and not repaired.is_empty:
                        repaired_count += 1
                        return repaired

                    self.logger.warning(f"Could not repair geometry: {geom.geom_type}")
                    return geom

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Error repairing geometry: {e}")
                    return geom

            return geom

        gdf = gdf.copy()
        gdf['geometry'] = gdf['geometry'].apply(repair_geometry)

        if invalid_count > 0:
            self.logger.info(
                f"Found {invalid_count} invalid geometries, repaired {repaired_count}"
            )
        else:
            self.logger.debug("All geometries are valid")

        return gdf
