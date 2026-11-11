# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Soil class-based domain discretization for pedological classification.

Creates HRUs based on soil type classifications, enabling soil-specific
parameterization of infiltration, water holding capacity, and conductivity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on soil classifications using MultiPolygon HRUs.

    Returns:
        Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
    """
    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp",
    )

    soil_raster = discretizer._get_file_path(
        path_key="SOIL_CLASS_PATH",
        name_key="SOIL_CLASS_NAME",
        default_subpath="attributes/soilclass",
        default_name=f"domain_{discretizer.domain_name}_soil_classes.tif",
    )

    # Use backward-compatible catchment subpath
    default_name = f"{discretizer.domain_name}_HRUs_soilclass.shp"
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath=discretizer._get_catchment_subpath(default_name),
        default_name=default_name,
    )

    gru_gdf, soil_classes = discretizer._read_and_prepare_data(
        gru_shapefile, soil_raster
    )
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, soil_raster, soil_classes, "soilClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf, gru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Soil-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None
