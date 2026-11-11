# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Radiation-based domain discretization for solar energy classification.

Creates HRUs based on annual solar radiation patterns derived from terrain
geometry, enabling energy-aware hydrological response unit delineation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from symfluence.geospatial.raster_utils import (
    calculate_annual_radiation,
    quantile_class_thresholds,
    radiation_raster_is_current,
)

if TYPE_CHECKING:
    from ..core import DomainDiscretizer


def discretize(discretizer: "DomainDiscretizer") -> Optional[object]:
    """
    Discretize the domain based on radiation properties using MultiPolygon HRUs.

    Returns:
        Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
    """
    gru_shapefile = discretizer._get_file_path(
        path_key="RIVER_BASINS_PATH",
        name_key="RIVER_BASINS_NAME",
        default_subpath="shapefiles/river_basins",
        default_name=f"{discretizer.domain_name}_riverBasins_{discretizer.delineation_suffix}.shp",
    )

    dem_raster = discretizer._get_file_path(
        path_key="DEM_PATH",
        name_key="DEM_NAME",
        default_subpath="attributes/elevation/dem",
        default_name=f"domain_{discretizer.domain_name}_elv.tif"
    )

    radiation_raster = discretizer._get_file_path(
        path_key="RADIATION_PATH",
        name_key="RADIATION_NAME",
        default_subpath="attributes/radiation",
        default_name="annual_radiation.tif",
    )

    # Use backward-compatible catchment subpath
    default_name = f"{discretizer.domain_name}_HRUs_radiation.shp"
    output_shapefile = discretizer._get_file_path(
        path_key="CATCHMENT_PATH",
        name_key="CATCHMENT_SHP_NAME",
        default_subpath=discretizer._get_catchment_subpath(default_name),
        default_name=default_name,
    )

    radiation_class_number = int(discretizer._get_config_value(
        lambda: discretizer.config.domain.radiation_class_number,
        default=1
    ))

    if not radiation_raster.exists():
        discretizer.logger.info(
            "Annual radiation raster not found. Calculating radiation..."
        )
        radiation_raster = calculate_annual_radiation(
            dem_raster, radiation_raster, discretizer.logger
        )
        if radiation_raster is None:
            raise ValueError("Failed to calculate annual radiation")
    else:
        warn_if_stale_radiation_raster(discretizer, radiation_raster)

    # RADIATION_CLASS_NUMBER is a number of classes (quantiles of the whole
    # raster), the same definition the combined discretization uses.
    gru_gdf = discretizer._read_shapefile(gru_shapefile)
    radiation_thresholds = quantile_class_thresholds(
        radiation_raster, radiation_class_number, discretizer.logger
    )
    hru_gdf = discretizer._create_multipolygon_hrus(
        gru_gdf, radiation_raster, radiation_thresholds, "radiationClass"
    )

    if hru_gdf is not None and not hru_gdf.empty:
        hru_gdf = discretizer._clean_and_prepare_hru_gdf(hru_gdf, gru_gdf)
        hru_gdf.to_file(output_shapefile)
        discretizer.logger.info(
            f"Radiation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}"
        )

        return output_shapefile
    else:
        discretizer.logger.error(
            "No valid HRUs were created. Check your input data and parameters."
        )
        return None


def warn_if_stale_radiation_raster(discretizer: "DomainDiscretizer", radiation_raster) -> None:
    """Warn when a reused radiation raster was not written by the current method."""
    if not radiation_raster_is_current(radiation_raster):
        discretizer.logger.warning(
            f"Reusing existing radiation raster {radiation_raster}, which was not "
            "produced by the current SYMFLUENCE radiation calculation. Rasters "
            "written by earlier versions sampled the sun once a day at 00:00 UTC "
            "and are not usable; delete the file to recompute it (ignore this "
            "message if you supplied the raster yourself)."
        )
