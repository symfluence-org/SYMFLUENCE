# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Soil attribute processor.

Handles soil properties including:
- Soil texture (sand, silt, clay) from SOILGRIDS
- Hydraulic properties (porosity, field capacity, wilting point, Ksat)
- USDA texture classification
- Soil depth from Pelletier dataset
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np

try:
    from osgeo import gdal
except ImportError:
    gdal = None
from rasterstats import zonal_stats

from .base import BaseAttributeProcessor


class SoilProcessor(BaseAttributeProcessor):
    """Processor for soil attributes."""

    def process(self) -> Dict[str, Any]:
        """
        Process soil attributes.

        Returns:
            Dictionary of soil attributes
        """
        results = {}

        # Process soil texture from SOILGRIDS
        texture_results = self._process_soilgrids_texture()
        results.update(texture_results)

        # Process soil hydraulic properties derived from texture classes
        hydraulic_results = self._derive_hydraulic_properties(texture_results)
        results.update(hydraulic_results)

        # Process soil depth attributes from Pelletier dataset
        depth_results = self._process_pelletier_soil_depth()
        results.update(depth_results)

        return results

    def _process_soilgrids_texture(self) -> Dict[str, Any]:
        """
        Process soil texture data from SOILGRIDS.

        Returns:
            Dict[str, Any]: Dictionary of soil texture attributes
        """
        results = {}

        # Define paths to SOILGRIDS data (configurable via SOILGRIDS_DIR)
        soilgrids_dir = Path(self._get_config_value(
            lambda: None,
            default=str(self.project_dir / 'data' / 'attributes' / 'soilclass' / 'soilgrids'),
            dict_key='SOILGRIDS_DIR',
        ))

        # Define soil components and depths to process
        components = ['clay', 'sand', 'silt']
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        stats = ['mean', 'min', 'max', 'std']

        # Process each soil component at each depth
        for component in components:
            for depth in depths:
                # Try acquisition naming first, then legacy nested layout
                tif_path = self._find_soilgrids_file(
                    soilgrids_dir, component, depth
                )

                if tif_path is None:
                    continue

                self.logger.info(f"Processing soil {component} at depth {depth}")

                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(tif_path),
                    stats=stats,
                    all_touched=True
                )

                # Get scale and offset for proper unit conversion
                scale, offset = self._read_scale_and_offset(tif_path)

                # Update results
                is_lumped = self._is_lumped()

                if is_lumped:
                    for stat in stats:
                        if zonal_out and len(zonal_out) > 0 and stat in zonal_out[0]:
                            value = zonal_out[0][stat]
                            if value is not None:
                                # Apply scale and offset if they exist
                                if scale is not None:
                                    value = value * scale
                                if offset is not None:
                                    value = value + offset
                                results[f"soil.{component}_{depth}_{stat}"] = value
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self._get_config_value(
                        lambda: self.config.paths.catchment_hruid,
                        dict_key='CATCHMENT_SHP_HRUID',
                    )

                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"

                            for stat in stats:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat]
                                    # Apply scale and offset if they exist
                                    if scale is not None:
                                        value = value * scale
                                    if offset is not None:
                                        value = value + offset
                                    results[f"{prefix}soil.{component}_{depth}_{stat}"] = value

        # Calculate USDA soil texture class percentages
        self._calculate_usda_texture_classes(results)

        return results

    def _find_pelletier_file(
        self, pelletier_dir: Path, attr_key: str
    ) -> Optional[Path]:
        """Find a Pelletier TIF file, checking acquisition and legacy naming."""
        # Acquisition naming: domain_{name}_pelletier_{attr_key}.tif
        matches = list(pelletier_dir.glob(f"*pelletier_{attr_key}.tif"))
        if matches:
            return matches[0]

        # Legacy naming maps
        _LEGACY_NAMES = {
            'soil_thickness': 'upland_hill-slope_soil_thickness.tif',
            'regolith_thickness': 'upland_hill-slope_regolith_thickness.tif',
            'sedimentary_thickness': 'upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif',
            'average_thickness': 'average_soil_and_sedimentary-deposit_thickness.tif',
            'hillslope_valley_average': 'hill-slope_and_valley-bottom_average_soil_and_sedimentary-deposit_thickness.tif',
        }
        legacy_name = _LEGACY_NAMES.get(attr_key)
        if legacy_name:
            legacy = pelletier_dir / legacy_name
            if legacy.exists():
                return legacy

        self.logger.debug(f"Pelletier file not found: {attr_key}")
        return None

    def _find_soilgrids_file(
        self, soilgrids_dir: Path, component: str, depth: str
    ) -> Optional[Path]:
        """Find a SoilGrids TIF file, checking acquisition and legacy naming."""
        # Acquisition naming: domain_{name}_soilgrids_{component}_{depth}_mean.tif
        matches = list(soilgrids_dir.glob(f"*soilgrids_{component}_{depth}_mean.tif"))
        if matches:
            return matches[0]

        # Legacy nested: {component}/{component}_{depth}_mean.tif
        legacy = soilgrids_dir / component / f"{component}_{depth}_mean.tif"
        if legacy.exists():
            return legacy

        # Legacy flat
        flat = soilgrids_dir / f"{component}_{depth}_mean.tif"
        if flat.exists():
            return flat

        self.logger.debug(f"Soil component file not found: {component} {depth}")
        return None

    def _derive_hydraulic_properties(self, texture_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive hydraulic properties from soil texture using pedotransfer functions.

        Args:
            texture_results: Dictionary containing soil texture information

        Returns:
            Dict[str, Any]: Dictionary of derived hydraulic properties
        """
        results = {}

        # Define pedotransfer functions for common hydraulic properties
        # These are based on the relationships in Saxton and Rawls (2006)

        is_lumped = self._is_lumped()

        if is_lumped:
            # For lumped catchment
            # Calculate weighted average properties across all depths
            depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
            depth_weights = [5, 10, 15, 30, 40, 100]  # Thickness of each layer in cm
            total_depth = sum(depth_weights)

            # Get weighted average of sand, clay, and silt content across all depths
            avg_sand = 0
            avg_clay = 0
            avg_silt = 0

            for depth, weight in zip(depths, depth_weights):
                sand_key = f"soil.sand_{depth}_mean"
                clay_key = f"soil.clay_{depth}_mean"
                silt_key = f"soil.silt_{depth}_mean"

                if sand_key in texture_results and clay_key in texture_results and silt_key in texture_results:
                    avg_sand += texture_results[sand_key] * weight / total_depth
                    avg_clay += texture_results[clay_key] * weight / total_depth
                    avg_silt += texture_results[silt_key] * weight / total_depth

            # Convert from g/kg to fraction (divide by 10)
            avg_sand / 1000
            avg_clay / 1000
            avg_silt / 1000

            # Calculate hydraulic properties using pedotransfer functions

            # Porosity (saturated water content)
            porosity = 0.46 - 0.0026 * avg_clay
            results["soil.porosity"] = porosity

            # Field capacity (-33 kPa matric potential)
            field_capacity = 0.2576 - 0.002 * avg_sand + 0.0036 * avg_clay + 0.0299 * avg_silt
            results["soil.field_capacity"] = field_capacity

            # Wilting point (-1500 kPa matric potential)
            wilting_point = 0.026 + 0.005 * avg_clay + 0.0158 * avg_silt
            results["soil.wilting_point"] = wilting_point

            # Available water capacity
            results["soil.available_water_capacity"] = field_capacity - wilting_point

            # Saturated hydraulic conductivity (mm/h)
            ksat = 10 * (2.54 * (2.778 * (10**-6)) * 10**(3.0 * porosity - 8.5))
            results["soil.ksat"] = ksat

        else:
            # For distributed catchment
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self._get_config_value(
                lambda: self.config.paths.catchment_hruid,
                dict_key='CATCHMENT_SHP_HRUID',
            )

            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"

                # Calculate weighted average properties across all depths
                depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
                depth_weights = [5, 10, 15, 30, 40, 100]  # Thickness of each layer in cm
                total_depth = sum(depth_weights)

                # Get weighted average of sand, clay, and silt content across all depths
                avg_sand = 0
                avg_clay = 0
                avg_silt = 0

                for depth, weight in zip(depths, depth_weights):
                    sand_key = f"{prefix}soil.sand_{depth}_mean"
                    clay_key = f"{prefix}soil.clay_{depth}_mean"
                    silt_key = f"{prefix}soil.silt_{depth}_mean"

                    if sand_key in texture_results and clay_key in texture_results and silt_key in texture_results:
                        avg_sand += texture_results[sand_key] * weight / total_depth
                        avg_clay += texture_results[clay_key] * weight / total_depth
                        avg_silt += texture_results[silt_key] * weight / total_depth

                # Convert from g/kg to fraction (divide by 10)
                avg_sand / 1000
                avg_clay / 1000
                avg_silt / 1000

                # Calculate hydraulic properties using pedotransfer functions

                # Porosity (saturated water content)
                porosity = 0.46 - 0.0026 * avg_clay
                results[f"{prefix}soil.porosity"] = porosity

                # Field capacity (-33 kPa matric potential)
                field_capacity = 0.2576 - 0.002 * avg_sand + 0.0036 * avg_clay + 0.0299 * avg_silt
                results[f"{prefix}soil.field_capacity"] = field_capacity

                # Wilting point (-1500 kPa matric potential)
                wilting_point = 0.026 + 0.005 * avg_clay + 0.0158 * avg_silt
                results[f"{prefix}soil.wilting_point"] = wilting_point

                # Available water capacity
                results[f"{prefix}soil.available_water_capacity"] = field_capacity - wilting_point

                # Saturated hydraulic conductivity (mm/h)
                ksat = 10 * (2.54 * (2.778 * (10**-6)) * 10**(3.0 * porosity - 8.5))
                results[f"{prefix}soil.ksat"] = ksat

        return results

    def _calculate_usda_texture_classes(self, results: Dict[str, Any]) -> None:
        """
        Calculate USDA soil texture classes based on sand, silt, and clay percentages.
        Updates the results dictionary in place.

        Args:
            results: Dictionary to update with texture class information
        """
        # USDA soil texture classes
        usda_classes = {
            'clay': (0, 45, 0, 40, 40, 100),  # Sand%, Silt%, Clay% ranges
            'silty_clay': (0, 20, 40, 60, 40, 60),
            'sandy_clay': (45, 65, 0, 20, 35, 55),
            'clay_loam': (20, 45, 15, 53, 27, 40),
            'silty_clay_loam': (0, 20, 40, 73, 27, 40),
            'sandy_clay_loam': (45, 80, 0, 28, 20, 35),
            'loam': (23, 52, 28, 50, 7, 27),
            'silty_loam': (0, 50, 50, 88, 0, 27),
            'sandy_loam': (50, 80, 0, 50, 0, 20),
            'silt': (0, 20, 80, 100, 0, 12),
            'loamy_sand': (70, 90, 0, 30, 0, 15),
            'sand': (85, 100, 0, 15, 0, 10)
        }

        is_lumped = self._is_lumped()

        # Depths to analyze
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']

        if is_lumped:
            # For each depth, determine texture class percentage
            for depth in depths:
                sand_key = f"soil.sand_{depth}_mean"
                clay_key = f"soil.clay_{depth}_mean"
                silt_key = f"soil.silt_{depth}_mean"

                if sand_key in results and clay_key in results and silt_key in results:
                    # Get values and convert to percentages
                    sand_val = results[sand_key] / 10  # g/kg to %
                    clay_val = results[clay_key] / 10  # g/kg to %
                    silt_val = results[silt_key] / 10  # g/kg to %

                    # Normalize to ensure they sum to 100%
                    total = sand_val + clay_val + silt_val
                    if total > 0:
                        sand_pct = (sand_val / total) * 100
                        clay_pct = (clay_val / total) * 100
                        silt_pct = (silt_val / total) * 100

                        # Determine soil texture class
                        for texture_class, (sand_min, sand_max, silt_min, silt_max, clay_min, clay_max) in usda_classes.items():
                            if (sand_min <= sand_pct <= sand_max and
                                silt_min <= silt_pct <= silt_max and
                                clay_min <= clay_pct <= clay_max):

                                # Add texture class to results
                                results[f"soil.texture_class_{depth}"] = texture_class
                                break
        else:
            # For distributed catchment
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self._get_config_value(
                lambda: self.config.paths.catchment_hruid,
                dict_key='CATCHMENT_SHP_HRUID',
            )

            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"

                # For each depth, determine texture class percentage
                for depth in depths:
                    sand_key = f"{prefix}soil.sand_{depth}_mean"
                    clay_key = f"{prefix}soil.clay_{depth}_mean"
                    silt_key = f"{prefix}soil.silt_{depth}_mean"

                    if sand_key in results and clay_key in results and silt_key in results:
                        # Get values and convert to percentages
                        sand_val = results[sand_key] / 10  # g/kg to %
                        clay_val = results[clay_key] / 10  # g/kg to %
                        silt_val = results[silt_key] / 10  # g/kg to %

                        # Normalize to ensure they sum to 100%
                        total = sand_val + clay_val + silt_val
                        if total > 0:
                            sand_pct = (sand_val / total) * 100
                            clay_pct = (clay_val / total) * 100
                            silt_pct = (silt_val / total) * 100

                            # Determine soil texture class
                            for texture_class, (sand_min, sand_max, silt_min, silt_max, clay_min, clay_max) in usda_classes.items():
                                if (sand_min <= sand_pct <= sand_max and
                                    silt_min <= silt_pct <= silt_max and
                                    clay_min <= clay_pct <= clay_max):

                                    # Add texture class to results
                                    results[f"{prefix}soil.texture_class_{depth}"] = texture_class
                                    break

    def _process_pelletier_soil_depth(self) -> Dict[str, Any]:
        """
        Process soil depth attributes from Pelletier dataset.

        Returns:
            Dict[str, Any]: Dictionary of soil depth attributes
        """
        results = {}

        # Define path to Pelletier data (configurable via PELLETIER_DIR)
        pelletier_dir = Path(self._get_config_value(
            lambda: None,
            default=str(self.project_dir / 'data' / 'attributes' / 'soilclass' / 'pelletier'),
            dict_key='PELLETIER_DIR',
        ))

        # Define attributes to process with possible filename patterns
        pelletier_attributes = {
            'regolith_thickness': 'regolith_thickness',
            'soil_thickness': 'soil_thickness',
            'sedimentary_thickness': 'sedimentary_thickness',
            'average_thickness': 'average_thickness',
            'hillslope_valley_average': 'hillslope_valley_average',
        }

        stats = ['mean', 'min', 'max', 'std']

        for attr_key, attribute in pelletier_attributes.items():
            # Search: acquisition naming then legacy naming
            tif_path = self._find_pelletier_file(pelletier_dir, attr_key)

            if tif_path is None:
                continue

            self.logger.info(f"Processing Pelletier {attribute}")

            # Check and set no-data value if needed
            self._check_and_set_nodata_value(tif_path)

            # Calculate zonal statistics
            zonal_out = zonal_stats(
                str(self.catchment_path),
                str(tif_path),
                stats=stats,
                all_touched=True
            )

            # Some tifs may have no data because the variable doesn't exist in the area
            zonal_out = self._check_zonal_stats_outcomes(zonal_out, new_val=0)

            # Get scale and offset
            scale, offset = self._read_scale_and_offset(tif_path)

            # Update results
            is_lumped = self._is_lumped()

            if is_lumped:
                for stat in stats:
                    if zonal_out and len(zonal_out) > 0 and stat in zonal_out[0]:
                        value = zonal_out[0][stat]
                        if value is not None:
                            # Apply scale and offset if they exist
                            if scale is not None:
                                value = value * scale
                            if offset is not None:
                                value = value + offset
                            results[f"soil.{attribute}_{stat}"] = value
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self._get_config_value(
                    lambda: self.config.paths.catchment_hruid,
                    dict_key='CATCHMENT_SHP_HRUID',
                )

                for i, zonal_result in enumerate(zonal_out):
                    if i < len(catchment):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"

                        for stat in stats:
                            if stat in zonal_result and zonal_result[stat] is not None:
                                value = zonal_result[stat]
                                # Apply scale and offset if they exist
                                if scale is not None:
                                    value = value * scale
                                if offset is not None:
                                    value = value + offset
                                results[f"{prefix}soil.{attribute}_{stat}"] = value

        return results

    def _read_scale_and_offset(self, geotiff_path: Path) -> Tuple[Optional[float], Optional[float]]:
        """
        Read scale and offset from a GeoTIFF file.

        Args:
            geotiff_path: Path to the GeoTIFF file

        Returns:
            Tuple of scale and offset values, potentially None if not set
        """
        if gdal is None:
            return None, None
        try:
            dataset = gdal.Open(str(geotiff_path))
            if dataset is None:
                self.logger.warning(f"Could not open GeoTIFF: {geotiff_path}")
                return None, None

            # Get the scale and offset values
            scale = dataset.GetRasterBand(1).GetScale()
            offset = dataset.GetRasterBand(1).GetOffset()

            # Close the dataset
            dataset = None

            return scale, offset
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error reading scale and offset: {str(e)}", exc_info=True)
            return None, None

    def _check_and_set_nodata_value(self, tif: Path, nodata: int = 255) -> None:
        """
        Check and set no-data value for a GeoTIFF if not already set.

        Args:
            tif: Path to the GeoTIFF file
            nodata: No-data value to set
        """
        if gdal is None:
            return
        try:
            # Open the dataset
            ds = gdal.Open(str(tif), gdal.GA_Update)
            if ds is None:
                self.logger.warning(f"Could not open GeoTIFF for no-data setting: {tif}")
                return

            # Get the current no-data value
            band = ds.GetRasterBand(1)
            current_nodata = band.GetNoDataValue()

            # If no no-data value is set but we need one
            if current_nodata is None:
                # Check if the maximum value in the dataset is the no-data value
                data = band.ReadAsArray()
                if data.max() == nodata:
                    # Set the no-data value
                    band.SetNoDataValue(nodata)
                    self.logger.info(f"Set no-data value to {nodata} for {tif}")

            # Close the dataset
            ds = None
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error checking and setting no-data value: {str(e)}", exc_info=True)

    def _check_zonal_stats_outcomes(self, zonal_out: List[Dict], new_val: Union[float, int] = np.nan) -> List[Dict]:
        """
        Check for None values in zonal statistics results and replace with specified value.

        Args:
            zonal_out: List of dictionaries with zonal statistics results
            new_val: Value to replace None with

        Returns:
            Updated zonal statistics results
        """
        for i in range(len(zonal_out)):
            for key, val in zonal_out[i].items():
                if val is None:
                    zonal_out[i][key] = new_val
                    self.logger.debug(f"Replaced None in {key} with {new_val}")

        return zonal_out
