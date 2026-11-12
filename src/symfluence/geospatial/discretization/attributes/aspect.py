# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Aspect-based domain discretization for slope direction classification.

Creates HRUs by classifying terrain aspect (slope direction) into
directional classes (N, NE, E, SE, S, SW, W, NW or custom divisions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from symfluence.geospatial.raster_utils import calculate_aspect

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on aspect (slope direction) using MultiPolygon HRUs.

    Returns:
        Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
    """
    # Determine default name based on method
    default_name = f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp"
    delineate_coastal = discretizer._get_config_value(
        lambda: discretizer.config.domain.delineation.delineate_coastal_watersheds,
        dict_key='DELINEATE_COASTAL_WATERSHEDS',
    )
    if delineate_coastal:
        default_name = f"{discretizer.domain_name}_riverBasins_with_coastal.shp"

    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=default_name,
    )

    dem_raster = discretizer._get_file_path(
        path_key="DEM_PATH",
        name_key="DEM_NAME",
        default_subpath="attributes/elevation/dem",
        default_name=f"domain_{discretizer.domain_name}_elv.tif"
    )

    aspect_raster = discretizer._get_file_path(
        path_key="ASPECT_PATH",
        name_key="ASPECT_NAME",
        default_subpath="attributes/aspect",
        default_name="aspect.tif"
    )

    # Use backward-compatible catchment subpath
    default_name = f"{discretizer.domain_name}_HRUs_aspect.shp"
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath=discretizer._get_catchment_subpath(default_name),
        default_name=default_name,
    )

    aspect_class_number = int(discretizer._get_config_value(
        lambda: discretizer.config.domain.aspect_class_number,
        dict_key='ASPECT_CLASS_NUMBER',
    ))

    if not aspect_raster.exists():
        discretizer.logger.info("Aspect raster not found. Calculating aspect...")
        aspect_raster = calculate_aspect(
            dem_raster, aspect_raster, aspect_class_number, discretizer.logger
        )
        if aspect_raster is None:
            raise ValueError("Failed to calculate aspect")

    gru_gdf, aspect_classes = discretizer._read_and_prepare_data(
        gru_shapefile, aspect_raster
    )
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, aspect_raster, aspect_classes, "aspectClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf, gru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Aspect-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None
