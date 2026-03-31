# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geometry utility functions for hydrological modeling.

Provides common geometry operations including:
- Geometry cleaning and validation
- Catchment centroid calculation with proper CRS handling
- Area calculations with automatic CRS detection
- Spatial aggregation utilities
"""

import logging
from typing import Optional, Tuple, Union

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid

from symfluence.core.mixins import LoggingMixin

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def clean_geometry(
    geometry: Union[Polygon, MultiPolygon, GeometryCollection],
    logger: Optional[logging.Logger] = None
) -> Optional[Union[Polygon, MultiPolygon]]:
    """
    Clean and validate geometries, ensuring only Polygon or MultiPolygon.

    Args:
        geometry: Shapely geometry object
        logger: Optional logger for debug messages

    Returns:
        Cleaned Polygon or MultiPolygon, or None if invalid/empty
    """
    if geometry is None or geometry.is_empty:
        return None

    try:
        # Handle GeometryCollection - extract only Polygons
        if isinstance(geometry, GeometryCollection):
            polygons = []
            for geom in geometry.geoms:
                if (
                    isinstance(geom, Polygon)
                    and geom.is_valid
                    and not geom.is_empty
                ):
                    polygons.append(geom)
                elif isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        if (
                            isinstance(poly, Polygon)
                            and poly.is_valid
                            and not poly.is_empty
                        ):
                            polygons.append(poly)

            if not polygons:
                return None
            elif len(polygons) == 1:
                geometry = polygons[0]
            else:
                geometry = MultiPolygon(polygons)

        # Ensure we have a valid Polygon or MultiPolygon
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            return None

        # Fix invalid geometries
        if not geometry.is_valid:
            geometry = make_valid(geometry)

            # Check again after repair
            if not isinstance(geometry, (Polygon, MultiPolygon)):
                return None

        return geometry if geometry.is_valid and not geometry.is_empty else None

    except Exception as e:  # noqa: BLE001 — geospatial resilience
        if logger:
            logger.debug(f"Error cleaning geometry: {str(e)}")
        return None


def calculate_catchment_centroid(
    catchment_gdf: 'gpd.GeoDataFrame',
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float]:
    """
    Calculate catchment centroid with proper CRS handling.

    Ensures accurate centroid calculation by:
    1. Detecting or assuming geographic CRS (EPSG:4326)
    2. Calculating appropriate UTM zone from bounds
    3. Projecting to UTM for accurate centroid
    4. Converting back to geographic coordinates (lon, lat)

    Args:
        catchment_gdf: GeoDataFrame of catchment polygon(s)
        logger: Optional logger for debug/info messages

    Returns:
        Tuple of (longitude, latitude) in decimal degrees

    Raises:
        ImportError: If geopandas is not available
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for centroid calculation")

    # Ensure CRS is defined
    if catchment_gdf.crs is None:
        if logger:
            logger.warning("Catchment CRS not defined, assuming EPSG:4326")
        catchment_gdf = catchment_gdf.set_crs(epsg=4326)

    # Convert to geographic coordinates if not already
    catchment_geo = catchment_gdf.to_crs(epsg=4326)

    # Get approximate center from bounds (for UTM zone calculation)
    bounds = catchment_geo.total_bounds  # (minx, miny, maxx, maxy)
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # Calculate appropriate UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1

    # Northern hemisphere: 326xx, Southern: 327xx
    epsg_code = f"326{utm_zone:02d}" if center_lat >= 0 else f"327{utm_zone:02d}"

    # Project to UTM for accurate centroid calculation
    catchment_utm = catchment_geo.to_crs(f"EPSG:{epsg_code}")

    # Calculate centroid in projected coordinates
    centroid_utm = catchment_utm.geometry.centroid.iloc[0]

    # Create GeoDataFrame for reprojection
    centroid_gdf = gpd.GeoDataFrame(
        geometry=[centroid_utm],
        crs=f"EPSG:{epsg_code}"
    )

    # Convert back to geographic coordinates
    centroid_geo = centroid_gdf.to_crs(epsg=4326)

    # Extract coordinates
    lon = centroid_geo.geometry.x[0]
    lat = centroid_geo.geometry.y[0]

    if logger:
        logger.info(
            f"Calculated catchment centroid: {lon:.6f}°E, {lat:.6f}°N "
            f"(UTM Zone {utm_zone})"
        )

    return lon, lat


def calculate_catchment_area_km2(
    catchment_gdf: 'gpd.GeoDataFrame',
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Calculate total catchment area in km².

    Automatically detects appropriate UTM projection for accurate area calculation.

    Args:
        catchment_gdf: GeoDataFrame of catchment polygon(s)
        logger: Optional logger for debug/info messages

    Returns:
        Total area in square kilometers

    Raises:
        ImportError: If geopandas is not available
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for area calculation")

    # Ensure CRS is defined
    if catchment_gdf.crs is None:
        if logger:
            logger.warning("Catchment CRS not defined, assuming EPSG:4326")
        catchment_gdf = catchment_gdf.set_crs(epsg=4326)

    # Use geopandas estimate_utm_crs() for automatic UTM selection
    try:
        utm_crs = catchment_gdf.estimate_utm_crs()
        catchment_proj = catchment_gdf.to_crs(utm_crs)
    except AttributeError:
        # Fallback for older geopandas versions
        catchment_geo = catchment_gdf.to_crs(epsg=4326)
        bounds = catchment_geo.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        epsg_code = f"326{utm_zone:02d}" if center_lat >= 0 else f"327{utm_zone:02d}"
        catchment_proj = catchment_geo.to_crs(f"EPSG:{epsg_code}")

    # Calculate area in m² and convert to km²
    area_km2 = catchment_proj.geometry.area.sum() / 1e6

    if logger:
        logger.info(f"Calculated catchment area: {area_km2:.2f} km²")

    return area_km2


def calculate_feature_centroids(
    gdf: 'gpd.GeoDataFrame',
    temp_epsg: int = 3857,
    logger: Optional[logging.Logger] = None
) -> 'gpd.GeoSeries':
    """
    Calculate centroids for each feature in a GeoDataFrame with CRS handling.

    If the CRS is geographic (e.g. EPSG:4326), it projects to a temporary
    projected CRS (default EPSG:3857) to calculate centroids, then projects back.

    Args:
        gdf: Input GeoDataFrame
        temp_epsg: Temporary projected EPSG code (default: 3857)
        logger: Optional logger for debug messages

    Returns:
        GeoSeries of centroids in the original CRS
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for centroid calculation")

    if gdf.empty:
        return gdf.geometry

    # Check if CRS is geographic
    if gdf.crs and gdf.crs.is_geographic:
        if logger:
            logger.debug(f"Projecting to EPSG:{temp_epsg} for centroid calculation")
        # Project to temporary CRS, calculate centroid, project back
        try:
            return gdf.to_crs(epsg=temp_epsg).geometry.centroid.to_crs(gdf.crs)
        except Exception as e:  # noqa: BLE001 — geospatial resilience
            if logger:
                logger.warning(f"Projection failed, using original CRS: {e}")
            return gdf.geometry.centroid
    else:
        return gdf.geometry.centroid


def validate_and_fix_crs(
    gdf: 'gpd.GeoDataFrame',
    assumed_epsg: int = 4326,
    logger: Optional[logging.Logger] = None
) -> 'gpd.GeoDataFrame':
    """
    Validate GeoDataFrame CRS and assign default if missing.

    Args:
        gdf: GeoDataFrame to validate
        assumed_epsg: EPSG code to assume if CRS is missing
        logger: Optional logger for warnings

    Returns:
        GeoDataFrame with valid CRS
    """
    if gdf.crs is None:
        if logger:
            logger.warning(
                f"CRS not defined, assuming EPSG:{assumed_epsg}"
            )
        return gdf.set_crs(epsg=assumed_epsg)
    return gdf


class GeospatialUtilsMixin(LoggingMixin):
    """
    Mixin providing geospatial utility methods.

    Provides:
        - Centroid calculation with proper CRS handling
        - Area calculations with automatic UTM projection
        - CRS validation and conversion utilities
    """

    def calculate_catchment_centroid(
        self,
        catchment_gdf: 'gpd.GeoDataFrame'
    ) -> Tuple[float, float]:
        """Delegate to calculate_catchment_centroid function."""
        return calculate_catchment_centroid(
            catchment_gdf=catchment_gdf,
            logger=self.logger
        )

    def calculate_catchment_area_km2(
        self,
        catchment_gdf: 'gpd.GeoDataFrame'
    ) -> float:
        """Delegate to calculate_catchment_area_km2 function."""
        return calculate_catchment_area_km2(
            catchment_gdf=catchment_gdf,
            logger=self.logger
        )

    def calculate_feature_centroids(
        self,
        gdf: 'gpd.GeoDataFrame',
        temp_epsg: int = 3857
    ) -> 'gpd.GeoSeries':
        """Delegate to calculate_feature_centroids function."""
        return calculate_feature_centroids(
            gdf=gdf,
            temp_epsg=temp_epsg,
            logger=self.logger
        )

    def validate_and_fix_crs(
        self,
        gdf: 'gpd.GeoDataFrame',
        assumed_epsg: int = 4326
    ) -> 'gpd.GeoDataFrame':
        """Delegate to validate_and_fix_crs function."""
        return validate_and_fix_crs(
            gdf=gdf,
            assumed_epsg=assumed_epsg,
            logger=self.logger
        )
