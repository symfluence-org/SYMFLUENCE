# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GRU-based domain discretization using whole watershed units.

Uses Grouped Response Units directly as Hydrologic Response Units without
further subdivision, suitable for lumped or semi-distributed modeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS

from symfluence.core.exceptions import DiscretizationError
from symfluence.geospatial.zonal import fill_empty_zones, zonal_stats_covered

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Use Grouped Response Units (GRUs) as Hydrologic Response Units (HRUs) without further discretization.

    Returns:
        Path: Path to the output HRU shapefile.
    """
    # Determine default name based on delineation suffix
    # The delineation_suffix reflects how the domain was actually created
    # (e.g., "delineate" for TauDEM, "lumped", "subset_{geofabric}", etc.)
    delineation_suffix = discretizer.delineation_suffix
    default_name = f"{discretizer.domain_name}_riverBasins_{delineation_suffix}.shp"
    delineate_coastal = discretizer._get_config_value(
        lambda: discretizer.config.domain.delineation.delineate_coastal_watersheds,
        dict_key='DELINEATE_COASTAL_WATERSHEDS',
    )
    if delineate_coastal:
        default_name = f"{discretizer.domain_name}_riverBasins_with_coastal.shp"
    elif delineation_suffix == "point":
        default_name = f"{discretizer.domain_name}_riverBasins_point.shp"

    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=default_name,
    )

    # Use backward-compatible catchment subpath
    default_name = f"{discretizer.domain_name}_HRUs_GRUs.shp"
    hru_output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath=discretizer._get_catchment_subpath(default_name),
        default_name=default_name,
    )

    gru_gdf = discretizer._read_shapefile(gru_shapefile)
    gru_gdf["HRU_ID"] = range(1, len(gru_gdf) + 1)
    gru_gdf["hru_type"] = "GRU"

    discretizer.logger.debug("Calculating elevation and centroid statistics for HRUs")

    # Get CRS information
    with rasterio.open(discretizer.dem_path) as src:
        dem_crs = src.crs
        discretizer.logger.debug(f"DEM CRS: {dem_crs}")

    shapefile_crs = gru_gdf.crs
    discretizer.logger.debug(f"Shapefile CRS: {shapefile_crs}")

    # Check if CRS match
    if dem_crs != shapefile_crs:
        discretizer.logger.debug(
            f"CRS mismatch detected. Reprojecting shapefile from {shapefile_crs} to {dem_crs}"
        )
        gru_gdf_projected = gru_gdf.to_crs(dem_crs)
    else:
        discretizer.logger.debug("CRS match - no reprojection needed")
        gru_gdf_projected = gru_gdf.copy()

    # Use rasterstats with the raster array and transform
    try:
        with rasterio.open(discretizer.dem_path) as src:
            dem_array = src.read(1)
            dem_transform = src.transform
            dem_nodata = src.nodata

        # GRUs smaller than a DEM pixel are retried with every touched pixel
        # and, failing that, filled from the nearest GRU (never silently -9999).
        zs, empty = zonal_stats_covered(
            gru_gdf_projected.geometry,
            dem_array,
            affine=dem_transform,
            stats=["mean"],
            nodata=dem_nodata if dem_nodata is not None else -9999,
            logger=discretizer.logger,
            label="GRU mean elevation",
        )
        fill_empty_zones(
            zs,
            empty,
            gru_gdf_projected.geometry,
            zone_ids=gru_gdf["GRU_ID"].tolist() if "GRU_ID" in gru_gdf.columns else None,
            logger=discretizer.logger,
            label="GRU mean elevation",
        )
        gru_gdf["elev_mean"] = [
            item["mean"] if item.get("mean") is not None else -9999 for item in zs
        ]
        discretizer.logger.debug(
            f"Successfully calculated elevation statistics for {len(gru_gdf)} HRUs"
        )

    except Exception as e:  # noqa: BLE001 — geospatial resilience
        discretizer.logger.error(f"Error calculating zonal statistics: {str(e)}")
        # Fallback: set all elevation means to -9999
        gru_gdf["elev_mean"] = -9999
        discretizer.logger.warning(
            "Setting all elevation means to -9999 due to calculation error"
        )

    # Calculate centroids in projected CRS for accuracy
    # Project to UTM for accurate centroid calculation if not already in UTM
    try:
        if gru_gdf.crs.is_geographic:
            utm_crs = gru_gdf.estimate_utm_crs()
            gru_gdf_utm = gru_gdf.to_crs(utm_crs)
        else:
            # Already in projected coordinate system
            gru_gdf_utm = gru_gdf.copy()
            utm_crs = gru_gdf.crs

        centroids_utm = gru_gdf_utm.geometry.centroid
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))

        gru_gdf["center_lon"] = centroids_wgs84.x
        gru_gdf["center_lat"] = centroids_wgs84.y

        discretizer.logger.debug(
            "Calculated centroids in WGS84: "
            f"lat range {centroids_wgs84.y.min():.6f} to {centroids_wgs84.y.max():.6f}, "
            f"lon range {centroids_wgs84.x.min():.6f} to {centroids_wgs84.x.max():.6f}"
        )

    except Exception as e:  # noqa: BLE001 — geospatial resilience
        discretizer.logger.error(f"Error calculating centroids: {str(e)}")
        # Fallback: try to use existing center_lat/center_lon if they exist and look reasonable
        if "center_lat" in gru_gdf.columns and "center_lon" in gru_gdf.columns:
            # Check if existing values look like actual lat/lon (rough check)
            if (
                gru_gdf["center_lat"].between(-90, 90).all()
                and gru_gdf["center_lon"].between(-180, 180).all()
            ):
                discretizer.logger.debug("Using existing center_lat/center_lon coordinates")
            else:
                discretizer.logger.warning(
                    "Existing center_lat/center_lon appear to be in projected coordinates, setting to default values"
                )
                gru_gdf["center_lat"] = 0.0
                gru_gdf["center_lon"] = 0.0
        else:
            gru_gdf["center_lat"] = 0.0
            gru_gdf["center_lon"] = 0.0

    if "COMID" in gru_gdf.columns:
        gru_gdf["GRU_ID"] = gru_gdf["COMID"]
    elif "fid" in gru_gdf.columns:
        gru_gdf["GRU_ID"] = gru_gdf["fid"]

    # GRU/HRU ids are written to 32-bit integer variables downstream (SUMMA
    # attributes and forcing, mizuRoute topology). Refuse ids that would wrap.
    numeric_ids = pd.to_numeric(gru_gdf["GRU_ID"], errors="coerce")
    too_big = numeric_ids.abs() > np.iinfo("int32").max
    if too_big.any():
        example = gru_gdf["GRU_ID"][too_big].iloc[0]
        raise DiscretizationError(
            f"GRU_ID values such as {example!r} do not fit 32-bit integers and would overflow "
            "in the model input files. Re-run the domain definition (geofabric "
            "subsetting now assigns compact ids and keeps the source id in SRC_ID), or "
            "renumber GRU_ID in the river basins shapefile."
        )

    gru_gdf["HRU_area"] = gru_gdf["GRU_area"]
    gru_gdf["HRU_ID"] = gru_gdf["GRU_ID"]

    gru_gdf.to_file(hru_output_shapefile)
    discretizer.logger.debug(f"GRUs saved as HRUs to {hru_output_shapefile}")

    return hru_output_shapefile
