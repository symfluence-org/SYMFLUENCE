# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Elevation-based domain discretization for altitude band classification.

Creates HRUs by dividing catchments into elevation bands with configurable
band sizes, supporting temperature lapse rate and snowmelt modeling.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on elevation within each GRU.

    Returns:
        Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
    """
    # Determine default name based on method
    default_name = f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp"
    delineate_coastal = discretizer._get_config_value(
        lambda: discretizer.config.domain.delineation.delineate_coastal_watersheds,
        default=False
    )
    if delineate_coastal:
        default_name = f"{discretizer.domain_name}_riverBasins_with_coastal.shp"

    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=default_name,
    )

    # Note: DEM path is already resolved in discretizer.dem_path, but re-resolving here as per original code pattern
    dem_raster = discretizer._get_file_path(
        path_key="DEM_PATH",
        name_key="DEM_NAME",
        default_subpath="attributes/elevation/dem",
        default_name=f"domain_{discretizer.domain_name}_elv.tif"
    )

    # Use backward-compatible catchment subpath
    default_name = f"{discretizer.domain_name}_HRUs_elevation.shp"
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath=discretizer._get_catchment_subpath(default_name),
        default_name=default_name,
    )

    elevation_band_size = float(discretizer._get_config_value(
        lambda: discretizer.config.domain.elevation_band_size,
        default=200.0
    ))
    gru_gdf, elevation_thresholds = discretizer._read_and_prepare_data(
        gru_shapefile, dem_raster, elevation_band_size
    )
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, dem_raster, elevation_thresholds, "elevClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf, gru_gdf)
        # Suppress shapefile column name truncation warning (known limitation)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Column names longer than 10.*')
            hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Elevation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None
