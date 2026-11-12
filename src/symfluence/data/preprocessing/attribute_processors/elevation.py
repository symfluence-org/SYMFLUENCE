# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Elevation attribute processor.

Handles DEM processing, slope/aspect generation, and elevation statistics calculation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

try:
    from osgeo import gdal
except ImportError:
    gdal = None
from scipy.stats import circmean

from symfluence.core.mixins.project import resolve_data_subdir

from .base import BaseAttributeProcessor


class ElevationProcessor(BaseAttributeProcessor):
    """Processor for elevation, slope, and aspect attributes."""

    def __init__(self, config: Dict[str, Any], logger):
        """Initialize elevation processor."""
        super().__init__(config, logger)

        # Create or access directories
        attributes_dir = resolve_data_subdir(self.project_dir, 'attributes')
        self.dem_dir = attributes_dir / 'elevation' / 'dem'
        self.slope_dir = attributes_dir / 'elevation' / 'slope'
        self.aspect_dir = attributes_dir / 'elevation' / 'aspect'

        # Create directories if they don't exist
        self.slope_dir.mkdir(parents=True, exist_ok=True)
        self.aspect_dir.mkdir(parents=True, exist_ok=True)

    def find_dem_file(self) -> Path:
        """
        Find the DEM file in the elevation/dem directory.

        Returns:
            Path to DEM file

        Raises:
            FileNotFoundError: If no DEM files found
        """
        dem_files = list(self.dem_dir.glob("*.tif"))

        if not dem_files:
            self.logger.error(f"No DEM files found in {self.dem_dir}")
            raise FileNotFoundError(f"No DEM files found in {self.dem_dir}")

        # Use the first found DEM file
        return dem_files[0]

    def generate_slope_and_aspect(self, dem_file: Path) -> Dict[str, Path]:
        """
        Generate slope and aspect rasters from the DEM using GDAL.

        Args:
            dem_file: Path to input DEM file

        Returns:
            Dictionary with paths to dem, slope, and aspect rasters

        Raises:
            Exception: If GDAL processing fails
        """
        if gdal is None:
            raise ImportError(
                "GDAL Python bindings required for slope/aspect generation. "
                "Install via: brew install gdal (macOS), apt install gdal-bin libgdal-dev (Linux), "
                "or pip install gdal==$(gdal-config --version)"
            )
        self.logger.info(f"Generating slope and aspect from DEM: {dem_file}")

        # Create output file paths
        slope_file = self.slope_dir / f"{dem_file.stem}_slope.tif"
        aspect_file = self.aspect_dir / f"{dem_file.stem}_aspect.tif"

        try:
            # Prepare the slope options
            slope_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                slopeFormat='degree',
                alg='Horn'
            )

            # Generate slope
            self.logger.info("Calculating slope...")
            gdal.DEMProcessing(
                str(slope_file),
                str(dem_file),
                'slope',
                options=slope_options
            )

            # Prepare the aspect options
            aspect_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                alg='Horn',
                zeroForFlat=True
            )

            # Generate aspect
            self.logger.info("Calculating aspect...")
            gdal.DEMProcessing(
                str(aspect_file),
                str(dem_file),
                'aspect',
                options=aspect_options
            )

            self.logger.info(f"Slope saved to: {slope_file}")
            self.logger.info(f"Aspect saved to: {aspect_file}")

            return {
                'dem': dem_file,
                'slope': slope_file,
                'aspect': aspect_file
            }

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error generating slope and aspect: {str(e)}")
            raise

    def calculate_statistics(self, raster_file: Path, attribute_name: str) -> Dict[str, float]:
        """
        Calculate zonal statistics for a raster.

        Args:
            raster_file: Path to raster file
            attribute_name: Name of attribute (elevation, slope, aspect)

        Returns:
            Dictionary of statistics with proper prefixes
        """
        self.logger.info(f"Calculating statistics for {attribute_name} from {raster_file}")

        # Define statistics to calculate
        stats = ['min', 'mean', 'max', 'std']

        # Special handling for aspect (circular statistics)
        if attribute_name == 'aspect':
            def calc_circmean(x):
                """Calculate circular mean of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular mean, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circmean(rad))
                return result

            def calc_circstd(x):
                """Calculate circular standard deviation of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular std, convert back to degrees
                rad = np.radians(x)
                try:
                    from scipy.stats import circstd
                    result = np.degrees(circstd(rad))
                    return result
                except ImportError:
                    return np.nan

            # Add custom circular statistics
            stats = ['min', 'max']
            custom_stats = {
                'circmean': calc_circmean
            }

            from importlib.util import find_spec
            if find_spec("scipy"):
                custom_stats['circstd'] = calc_circstd
            else:
                self.logger.warning("scipy.stats.circstd not available, skipping circular std")

            zonal_out = zonal_stats(
                str(self.catchment_path),
                str(raster_file),
                stats=stats,
                add_stats=custom_stats,
                all_touched=True
            )
        else:
            # Regular statistics for elevation and slope
            zonal_out = zonal_stats(
                str(self.catchment_path),
                str(raster_file),
                stats=stats,
                all_touched=True
            )

        # Format the results
        results = {}
        if zonal_out:
            for i, zonal_result in enumerate(zonal_out):
                # Use HRU ID as key if available, otherwise use index
                is_lumped = self._get_config_value(lambda: self.config.domain.definition_method, dict_key='DOMAIN_DEFINITION_METHOD') == 'lumped'
                if not is_lumped:
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self._get_config_value(
                        lambda: self.config.paths.catchment_hruid,
                        dict_key='CATCHMENT_SHP_HRUID',
                    )
                    hru_id = catchment.iloc[i][hru_id_field]
                    key_prefix = f"HRU_{hru_id}_"
                else:
                    key_prefix = ""

                # Add each statistic to results
                for stat, value in zonal_result.items():
                    if value is not None:
                        results[f"{key_prefix}{attribute_name}_{stat}"] = value

        return results

    def process(self) -> Dict[str, float]:
        """
        Process elevation, slope, and aspect attributes.

        Returns:
            Dictionary of elevation-related attributes
        """
        results = {}

        try:
            # Find the DEM file
            dem_file = self.find_dem_file()

            # Generate slope and aspect
            raster_files = self.generate_slope_and_aspect(dem_file)

            # Calculate statistics for each raster
            for attribute_name, raster_file in raster_files.items():
                stats = self.calculate_statistics(raster_file, attribute_name)

                # Add statistics with proper prefixes
                for stat_name, value in stats.items():
                    if "." in stat_name:  # If stat_name already has hierarchical structure
                        results[stat_name] = value
                    else:
                        # Extract any HRU prefix if present
                        if stat_name.startswith("HRU_"):
                            hru_part = stat_name.split("_", 2)[0] + "_" + stat_name.split("_", 2)[1] + "_"
                            clean_stat = stat_name.replace(hru_part, "")
                            prefix = hru_part
                        else:
                            clean_stat = stat_name
                            prefix = ""

                        # Remove attribute prefix if present in the stat name
                        clean_stat = clean_stat.replace(f"{attribute_name}_", "")
                        results[f"{prefix}{attribute_name}.{clean_stat}"] = value

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error processing elevation attributes: {str(e)}", exc_info=True)

        return results
