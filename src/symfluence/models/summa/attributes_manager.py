# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Attributes Manager.

This module provides the SummaAttributesManager class for managing HRU attributes
in SUMMA model preprocessing, including calculations of slope, aspect, elevation,
and land/soil classifications.
"""

# Standard library imports
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import geopandas as gpd
import netCDF4 as nc4
import numpy as np
import pandas as pd
import rasterio
import rasterstats
import xarray as xr

# Local imports
from symfluence.core import ConfigurableMixin
from symfluence.core.exceptions import ModelExecutionError

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class SummaAttributesManager(ConfigurableMixin):
    """
    Manager for SUMMA HRU attributes.

    This class handles the creation and population of SUMMA attributes files,
    including calculations of topographic properties (slope, aspect, elevation),
    soil and land class assignments, and HRU connectivity.
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: Any,
        catchment_path: Path,
        catchment_name: str,
        dem_path: Path,
        forcing_summa_path: Path,
        setup_dir: Path,
        project_dir: Path,
        hruId: str,
        gruId: str,
        attribute_name: str,
        forcing_measurement_height: float,
        get_default_path_callback: Any
    ):
        """
        Initialize the SummaAttributesManager.

        Args:
            config: Configuration (typed SymfluenceConfig or dict) containing setup parameters
            logger: Logger object for recording processing information
            catchment_path: Path to catchment shapefile directory
            catchment_name: Name of catchment shapefile
            dem_path: Path to DEM file
            forcing_summa_path: Path to SUMMA forcing files
            setup_dir: Path to setup directory
            project_dir: Path to project directory
            hruId: Column name for HRU ID in shapefile
            gruId: Column name for GRU ID in shapefile
            attribute_name: Name of attributes file to create
            forcing_measurement_height: Height of forcing measurements (m)
            get_default_path_callback: Callback function to get default paths
        """
        # Handle typed config
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)

        self.logger = logger
        self.catchment_path = catchment_path
        self.catchment_name = catchment_name
        self.dem_path = dem_path
        self.forcing_summa_path = forcing_summa_path
        self.setup_dir = setup_dir
        self.project_dir = project_dir
        self.hruId = hruId
        self.gruId = gruId
        self.attribute_name = attribute_name
        self.forcing_measurement_height = forcing_measurement_height
        self._get_default_path = get_default_path_callback

    def create_attributes_file(self):
        """
        Create the attributes file for SUMMA.

        This method performs the following steps:
        1. Load the catchment shapefile
        2. Get HRU order from a forcing file
        3. Create a netCDF file with HRU attributes
        4. Set attribute values for each HRU
        5. Insert soil class, land class, and elevation data
        6. Optionally set up HRU connectivity

        The resulting file provides SUMMA with essential information about each HRU.

        Raises:
            FileNotFoundError: If required input files are not found.
            IOError: If there are issues creating or writing to the attributes file.
            ValueError: If there are inconsistencies in the attribute data.
        """
        self.logger.info("Creating attributes file")

        # Load the catchment shapefile
        shp = gpd.read_file(self.catchment_path / self.catchment_name)

        # Get HRU order from a forcing file
        # SUMMA_input is shared by every experiment on the domain (#339).
        from symfluence.core.modeling.forcing_naming import resolve_forcing_files
        forcing_files = resolve_forcing_files(
            sorted(self.forcing_summa_path.glob('*.nc')),
            self._get_config_value(lambda: self.config.domain.discretization,
                                   dict_key='SUB_GRID_DISCRETIZATION'),
            self.catchment_path / self.catchment_name, self.hruId)
        if not forcing_files:
            self.logger.error("No forcing files found in the SUMMA input directory")
            return
        forcing_file = forcing_files[0]

        with xr.open_dataset(forcing_file) as forc:
            forcing_hruIds = forc['hruId'].values.astype(int)

        # Sort shapefile based on forcing HRU order
        catchment_hruid = self._get_config_value(lambda: self.config.paths.catchment_hruid)
        shp = shp.set_index(catchment_hruid)
        shp.index = shp.index.astype(int)
        available_hru_ids = set(shp.index.astype(int))
        missing_hru_ids = [hru_id for hru_id in forcing_hruIds if hru_id not in available_hru_ids]
        if missing_hru_ids:
            self.logger.warning(
                "Forcing HRU IDs not found in catchment shapefile; filtering missing IDs. "
                "Missing count: %s (showing first 10): %s",
                len(missing_hru_ids),
                missing_hru_ids[:10],
            )
            forcing_hruIds = [hru_id for hru_id in forcing_hruIds if hru_id in available_hru_ids]
        if len(forcing_hruIds) == 0:
            raise ModelExecutionError("No forcing HRU IDs match catchment shapefile HRU IDs.")
        shp = shp.loc[forcing_hruIds].reset_index()

        # Get number of GRUs and HRUs
        catchment_gruid = self._get_config_value(lambda: self.config.paths.catchment_gruid)
        hru_ids = pd.unique(shp[catchment_hruid].values)
        gru_ids = pd.unique(shp[catchment_gruid].values)
        num_hru = len(hru_ids)
        num_gru = len(gru_ids)

        attribute_path = self.setup_dir / self.attribute_name

        with nc4.Dataset(attribute_path, "w", format="NETCDF4") as att:
            # Set attributes
            att.setncattr('Author', "Created by SUMMA workflow scripts")
            att.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

            # Define dimensions
            att.createDimension('hru', num_hru)
            att.createDimension('gru', num_gru)

            # Define variables
            variables = {
                'hruId': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index of hydrological response unit (HRU)'},
                'gruId': {'dtype': 'i4', 'dims': 'gru', 'units': '-', 'long_name': 'Index of grouped response unit (GRU)'},
                'hru2gruId': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index of GRU to which the HRU belongs'},
                'downHRUindex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index of downslope HRU (0 = basin outlet)'},
                'longitude': {'dtype': 'f8', 'dims': 'hru', 'units': 'Decimal degree east', 'long_name': 'Longitude of HRU''s centroid'},
                'latitude': {'dtype': 'f8', 'dims': 'hru', 'units': 'Decimal degree north', 'long_name': 'Latitude of HRU''s centroid'},
                'elevation': {'dtype': 'f8', 'dims': 'hru', 'units': 'm', 'long_name': 'Mean HRU elevation'},
                'HRUarea': {'dtype': 'f8', 'dims': 'hru', 'units': 'm^2', 'long_name': 'Area of HRU'},
                'tan_slope': {'dtype': 'f8', 'dims': 'hru', 'units': 'm m-1', 'long_name': 'Average tangent slope of HRU'},
                'contourLength': {'dtype': 'f8', 'dims': 'hru', 'units': 'm', 'long_name': 'Contour length of HRU'},
                'slopeTypeIndex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index defining slope'},
                'soilTypeIndex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index defining soil type'},
                'vegTypeIndex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index defining vegetation type'},
                'mHeight': {'dtype': 'f8', 'dims': 'hru', 'units': 'm', 'long_name': 'Measurement height above bare ground'},
            }

            for var_name, var_attrs in variables.items():
                var = att.createVariable(var_name, var_attrs['dtype'], var_attrs['dims'], fill_value=False)
                var.setncattr('units', var_attrs['units'])
                var.setncattr('long_name', var_attrs['long_name'])

            # Fill GRU variable
            att['gruId'][:] = gru_ids

            # Fill HRU variables
            catchment_area = self._get_config_value(lambda: self.config.paths.catchment_area)
            catchment_lat = self._get_config_value(lambda: self.config.paths.catchment_lat)
            catchment_lon = self._get_config_value(lambda: self.config.paths.catchment_lon)
            for idx in range(num_hru):
                att['hruId'][idx] = shp.iloc[idx][catchment_hruid]
                att['HRUarea'][idx] = shp.iloc[idx][catchment_area]
                att['latitude'][idx] = shp.iloc[idx][catchment_lat]
                att['longitude'][idx] = shp.iloc[idx][catchment_lon]
                att['hru2gruId'][idx] = shp.iloc[idx][catchment_gruid]

                # Set slope and contour length (will be updated later)
                att['tan_slope'][idx] = 0.1
                att['contourLength'][idx] = 100

                att['slopeTypeIndex'][idx] = 1
                att['mHeight'][idx] = self.forcing_measurement_height
                att['downHRUindex'][idx] = 0
                att['elevation'][idx] = -999
                att['soilTypeIndex'][idx] = -999
                att['vegTypeIndex'][idx] = -999

        self.logger.info(f"Attributes file created at: {attribute_path}")

        self.insert_land_class(attribute_path)
        self.insert_soil_class(attribute_path)
        self.insert_elevation(attribute_path)
        self.insert_aspect(attribute_path)
        self.insert_tan_slope(attribute_path)

    def calculate_slope_and_contour(self, shp, dem_path):
        """
        Calculate average slope and contour length for each HRU using vectorized operations.

        Args:
            shp (gpd.GeoDataFrame): GeoDataFrame containing HRU polygons
            dem_path (Path): Path to the DEM file

        Returns:
            dict: Dictionary with HRU IDs as keys and tuples (slope, contour_length) as values
        """
        self.logger.info("Calculating slope and contour length for each HRU")

        # Calculate contour lengths using vectorized operation
        contour_lengths = np.sqrt(shp.geometry.area)

        # Read DEM once
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform

            # Calculate dx and dy for the entire DEM once
            cell_size_x = transform[0]
            cell_size_y = -transform[4]  # Negative because Y increases downward in pixel space

            # Convert cell sizes from degrees to meters if CRS is geographic
            if src.crs.is_geographic:
                # Get center latitude for conversion
                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2

                # Convert degrees to meters
                meters_per_degree_lat = 111000.0
                meters_per_degree_lon = 111000.0 * np.cos(np.radians(center_lat))

                cell_size_x = cell_size_x * meters_per_degree_lon
                cell_size_y = cell_size_y * meters_per_degree_lat

            # Calculate gradients for entire DEM once
            dy, dx = np.gradient(dem, cell_size_y, cell_size_x)
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))

            # Use zonal_stats to get mean slope for all HRUs at once
            mean_slopes = rasterstats.zonal_stats(
                shp.geometry,
                slope,
                affine=transform,
                stats=['mean'],
                nodata=np.nan
            )

        # Create results dictionary using vectorized operations
        results = {}
        catchment_hruid = self._get_config_value(lambda: self.config.paths.catchment_hruid)
        for idx, row in shp.iterrows():
            hru_id = row[catchment_hruid]
            avg_slope = mean_slopes[idx]['mean']

            if avg_slope is None or np.isnan(avg_slope):
                self.logger.warning(f"No valid slope data found for HRU {hru_id}")
                results[hru_id] = (0.1, 30)  # Default values
            else:
                results[hru_id] = (avg_slope, contour_lengths[idx])

        return results

    def calculate_contour_length(self, hru_dem, hru_geometry, downstream_geometry, transform, hru_id):
        """
        Calculate the length of intersection between an HRU and its downstream neighbor.

        Args:
            hru_dem (numpy.ndarray): DEM data for the HRU
            hru_geometry (shapely.geometry): Geometry of the current HRU
            downstream_geometry (shapely.geometry): Geometry of the downstream HRU
            transform (affine.Affine): Transform for converting pixel to geographic coordinates
            hru_id (int): ID of the current HRU

        Returns:
            float: Length of the intersection between the HRU and its downstream neighbor
        """
        # If there's no downstream HRU (outlet), use the HRU's minimum perimeter length
        if downstream_geometry is None:
            min_dimension = min(hru_geometry.bounds[2] - hru_geometry.bounds[0],
                            hru_geometry.bounds[3] - hru_geometry.bounds[1])
            self.logger.info(f"HRU {hru_id} is an outlet. Using minimum dimension: {min_dimension}")
            return min_dimension

        # Find the intersection between current and downstream HRUs
        intersection = hru_geometry.intersection(downstream_geometry)

        if intersection.is_empty:
            self.logger.warning(f"No intersection found between HRU {hru_id} and its downstream HRU")
            # Use minimum perimeter length as a fallback
            min_dimension = min(hru_geometry.bounds[2] - hru_geometry.bounds[0],
                            hru_geometry.bounds[3] - hru_geometry.bounds[1])
            return min_dimension

        # Calculate the length of the intersection
        contour_length = intersection.length

        self.logger.info(f"Calculated contour length {contour_length:.2f} m for HRU {hru_id}")
        return contour_length

    def insert_aspect(self, attribute_file):
        """
        Calculate and insert aspect data into the attributes file.

        Aspect is calculated from the DEM using gradient analysis and represents
        the compass direction that the slope faces (in degrees from North).

        Args:
            attribute_file (str): Path to the SUMMA attributes NetCDF file
        """
        self.logger.info("Calculating and inserting aspect into attributes file")

        try:
            # Load the catchment shapefile
            shp = gpd.read_file(self.catchment_path / self.catchment_name)

            # Calculate aspect for each HRU using the DEM
            aspect_values = self._calculate_aspect_from_dem(shp)

            # Update the attributes file
            with nc4.Dataset(attribute_file, "r+") as att:
                # Check if aspect variable already exists, if not create it
                if 'aspect' not in att.variables:
                    aspect_var = att.createVariable('aspect', 'f8', 'hru', fill_value=False)
                    aspect_var.setncattr('units', 'degrees')
                    aspect_var.setncattr('long_name', 'Mean aspect of HRU (degrees from North)')

                # Fill aspect values for each HRU
                for idx in range(len(att['hruId'])):
                    hru_id_raw = att['hruId'][idx]
                    # Convert to proper scalar type (handle potential MaskedArray)
                    if hasattr(hru_id_raw, 'item'):
                        hru_id = int(hru_id_raw.item())
                    else:
                        hru_id = int(hru_id_raw)

                    if hru_id in aspect_values:
                        att['aspect'][idx] = aspect_values[hru_id]
                    else:
                        self.logger.warning(f"No aspect data found for HRU {hru_id}, using default value")
                        att['aspect'][idx] = 180.0  # Default to south-facing

            self.logger.info("Successfully inserted aspect data into attributes file")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error inserting aspect data: {str(e)}", exc_info=True)
            # Set default values if calculation fails
            with nc4.Dataset(attribute_file, "r+") as att:
                if 'aspect' not in att.variables:
                    aspect_var = att.createVariable('aspect', 'f8', 'hru', fill_value=False)
                    aspect_var.setncattr('units', 'degrees')
                    aspect_var.setncattr('long_name', 'Mean aspect of HRU (degrees from North)')
                att['aspect'][:] = 180.0  # Default to south-facing
                self.logger.warning("Set all aspect values to default (180 degrees - south-facing)")

    def insert_tan_slope(self, attribute_file):
        """
        Calculate and insert tangent of slope data into the attributes file.

        Tangent of slope is calculated from the DEM using gradient analysis.
        This updates the existing tan_slope values that were set to default in create_attributes_file.

        Args:
            attribute_file (str): Path to the SUMMA attributes NetCDF file
        """
        self.logger.info("Calculating and inserting tangent of slope into attributes file")

        try:
            # Load the catchment shapefile
            shp = gpd.read_file(self.catchment_path / self.catchment_name)

            # Calculate tan_slope for each HRU using the DEM
            tan_slope_values = self._calculate_tan_slope_from_dem(shp)

            # Update the attributes file
            with nc4.Dataset(attribute_file, "r+") as att:
                # tan_slope variable should already exist from create_attributes_file
                # Fill tan_slope values for each HRU
                for idx in range(len(att['hruId'])):
                    hru_id_raw = att['hruId'][idx]
                    # Convert to proper scalar type (handle potential MaskedArray)
                    if hasattr(hru_id_raw, 'item'):
                        hru_id = int(hru_id_raw.item())
                    else:
                        hru_id = int(hru_id_raw)

                    if hru_id in tan_slope_values:
                        att['tan_slope'][idx] = tan_slope_values[hru_id]
                    else:
                        self.logger.warning(f"No slope data found for HRU {hru_id}, using default value")
                        att['tan_slope'][idx] = 0.1  # Default slope

            self.logger.info("Successfully inserted tangent of slope data into attributes file")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error inserting tangent of slope data: {str(e)}", exc_info=True)
            # Set default values if calculation fails
            with nc4.Dataset(attribute_file, "r+") as att:
                att['tan_slope'][:] = 0.1  # Default slope
                self.logger.warning("Set all tan_slope values to default (0.1)")

    def _calculate_aspect_from_dem(self, shp):
        """
        Calculate mean aspect for each HRU from the DEM.

        Args:
            shp (gpd.GeoDataFrame): GeoDataFrame containing HRU polygons

        Returns:
            dict: Dictionary with HRU IDs as keys and aspect values (degrees) as values
        """
        self.logger.info("Calculating aspect from DEM for each HRU")

        results = {}

        try:
            with rasterio.open(self.dem_path) as src:
                dem = src.read(1)
                transform = src.transform

                # Get cell sizes
                cell_size_x = abs(transform[0])  # dx
                cell_size_y = abs(transform[4])  # dy

                # Convert cell sizes from degrees to meters if CRS is geographic
                if src.crs.is_geographic:
                    # Get center latitude for conversion
                    bounds = src.bounds
                    center_lat = (bounds.bottom + bounds.top) / 2

                    # Convert degrees to meters
                    meters_per_degree_lat = 111000.0
                    meters_per_degree_lon = 111000.0 * np.cos(np.radians(center_lat))

                    cell_size_x = cell_size_x * meters_per_degree_lon
                    cell_size_y = cell_size_y * meters_per_degree_lat

                # Calculate gradients
                dy, dx = np.gradient(dem.astype(np.float64), cell_size_y, cell_size_x)

                # Calculate aspect in radians (-π to π)
                aspect_rad = np.arctan2(-dy, dx)  # Note: -dy because aspect is measured from North

                # Convert to degrees (0 to 360, where 0/360 = North, 90 = East, 180 = South, 270 = West)
                aspect_deg = np.degrees(aspect_rad)
                aspect_deg = (90 - aspect_deg) % 360  # Convert from math convention to compass bearing

                # Handle flat areas (where both dx and dy are near zero)
                flat_mask = (np.abs(dx) < 1e-8) & (np.abs(dy) < 1e-8)
                aspect_deg[flat_mask] = -1  # Special value for flat areas

                # Use zonal_stats to get mean aspect for all HRUs at once
                mean_aspects = rasterstats.zonal_stats(
                    shp.geometry,
                    aspect_deg,
                    affine=transform,
                    stats=['mean'],
                    nodata=src.nodata
                )

            # Create results dictionary
            hru_id_col = self._get_config_value(lambda: self.config.paths.catchment_hruid)
            for idx, row in shp.iterrows():
                # Convert HRU ID to proper scalar type (handle MaskedArray)
                hru_id_raw = row[hru_id_col]
                if hasattr(hru_id_raw, 'item'):
                    hru_id = int(hru_id_raw.item())  # Extract scalar from MaskedArray
                else:
                    hru_id = int(hru_id_raw)  # Already a scalar

                mean_aspect = mean_aspects[idx]['mean']

                if mean_aspect is None or np.isnan(mean_aspect):
                    self.logger.warning(f"No valid aspect data found for HRU {hru_id}")
                    results[hru_id] = 180.0  # Default to south-facing
                elif mean_aspect == -1:
                    # Flat area
                    results[hru_id] = 180.0  # Default to south-facing for flat areas
                else:
                    results[hru_id] = float(mean_aspect)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error calculating aspect from DEM: {str(e)}", exc_info=True)
            # Return default values for all HRUs
            hru_id_col = self._get_config_value(lambda: self.config.paths.catchment_hruid)
            for idx, row in shp.iterrows():
                # Convert HRU ID to proper scalar type (handle MaskedArray)
                hru_id_raw = row[hru_id_col]
                if hasattr(hru_id_raw, 'item'):
                    hru_id = int(hru_id_raw.item())
                else:
                    hru_id = int(hru_id_raw)
                results[hru_id] = 180.0

        return results

    def _calculate_tan_slope_from_dem(self, shp):
        """
        Calculate mean tangent of slope for each HRU from the DEM.

        Args:
            shp (gpd.GeoDataFrame): GeoDataFrame containing HRU polygons

        Returns:
            dict: Dictionary with HRU IDs as keys and tan_slope values as values
        """
        self.logger.info("Calculating tangent of slope from DEM for each HRU")

        results = {}
        min_slope = 1e-6
        # Guard against DEM artifacts (e.g., nodata edges) producing unrealistically large values.
        max_tan_slope = float(np.tan(np.radians(80.0)))

        try:
            with rasterio.open(self.dem_path) as src:
                # Read as masked array so nodata does not contaminate gradients.
                dem = src.read(1, masked=True).astype(np.float64).filled(np.nan)
                transform = src.transform

                # Get cell sizes
                cell_size_x = abs(transform[0])  # dx
                cell_size_y = abs(transform[4])  # dy

                # Convert cell sizes from degrees to meters if CRS is geographic
                if src.crs is not None and src.crs.is_geographic:
                    # Get center latitude for conversion
                    bounds = src.bounds
                    center_lat = (bounds.bottom + bounds.top) / 2

                    # Convert degrees to meters
                    meters_per_degree_lat = 111000.0
                    meters_per_degree_lon = 111000.0 * np.cos(np.radians(center_lat))

                    cell_size_x = cell_size_x * meters_per_degree_lon
                    cell_size_y = cell_size_y * meters_per_degree_lat

                    self.logger.debug(f"DEM in geographic coordinates - converted cell sizes to meters: dx={cell_size_x:.2f}m, dy={cell_size_y:.2f}m")

                # Calculate gradients
                dy, dx = np.gradient(dem.astype(np.float64), cell_size_y, cell_size_x)

                # Calculate slope magnitude (rise over run)
                slope_magnitude = np.sqrt(dx*dx + dy*dy)

                # Convert to tangent of slope angle
                # slope_magnitude is already rise/run = tan(slope_angle)
                tan_slope = slope_magnitude

                # Set minimum slope to avoid zero values (SUMMA may have issues with zero slope)
                tan_slope = np.maximum(tan_slope, min_slope)
                tan_slope = np.minimum(tan_slope, max_tan_slope)

                # rasterstats expects a scalar nodata value; use sentinel for invalid cells.
                slope_nodata = -9999.0
                tan_slope_for_stats = np.where(np.isfinite(tan_slope), tan_slope, slope_nodata)

                n_clipped = int(np.sum((tan_slope_for_stats != slope_nodata) & (tan_slope_for_stats >= max_tan_slope)))
                if n_clipped > 0:
                    self.logger.debug(
                        "Clipped %s DEM cells to max tan_slope %.3f to suppress artifacts",
                        n_clipped,
                        max_tan_slope,
                    )

                # Use zonal_stats to get mean tan_slope for all HRUs at once
                mean_tan_slopes = rasterstats.zonal_stats(
                    shp.geometry,
                    tan_slope_for_stats,
                    affine=transform,
                    stats=['mean'],
                    nodata=slope_nodata
                )

            # Create results dictionary
            hru_id_col = self._get_config_value(lambda: self.config.paths.catchment_hruid)
            for idx, row in shp.iterrows():
                # Convert HRU ID to proper scalar type (handle MaskedArray)
                hru_id_raw = row[hru_id_col]
                if hasattr(hru_id_raw, 'item'):
                    hru_id = int(hru_id_raw.item())  # Extract scalar from MaskedArray
                else:
                    hru_id = int(hru_id_raw)  # Already a scalar

                mean_tan_slope = mean_tan_slopes[idx]['mean']

                if mean_tan_slope is None or np.isnan(mean_tan_slope):
                    self.logger.warning(f"No valid slope data found for HRU {hru_id}")
                    results[hru_id] = 0.1  # Default slope
                else:
                    # Ensure minimum slope value
                    results[hru_id] = max(float(mean_tan_slope), min_slope)

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error calculating tan_slope from DEM: {str(e)}", exc_info=True)
            # Return default values for all HRUs
            hru_id_col = self._get_config_value(lambda: self.config.paths.catchment_hruid)
            for idx, row in shp.iterrows():
                # Convert HRU ID to proper scalar type (handle MaskedArray)
                hru_id_raw = row[hru_id_col]
                if hasattr(hru_id_raw, 'item'):
                    hru_id = int(hru_id_raw.item())
                else:
                    hru_id = int(hru_id_raw)
                results[hru_id] = 0.1

        return results

    def insert_soil_class(self, attribute_file):
        """Insert soil class data into the attributes file."""
        self.logger.info("Inserting soil class into attributes file")

        intersect_path = self._get_default_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self._get_config_value(
            lambda: self.config.paths.intersect_soil_name, default='default'
        )
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        intersect_hruId_var = self._get_config_value(lambda: self.config.paths.catchment_hruid)

        try:
            # Prefer the model-ready attributes store (/soil/ fractions keyed by
            # hru_id); fall back to the SoilGrids intersection shapefile.
            store_frac = self._load_store_class_fractions(
                'soil', 'soil_fraction', 'soil_class_name')
            shp = None
            if store_frac is None:
                shp = gpd.read_file(intersect_path / intersect_name)
                # Check and create missing USGS_X columns
                for i in range(13):
                    col_name = f'USGS_{i}'
                    if col_name not in shp.columns:
                        shp[col_name] = 0  # Add the missing column and initialize with 0

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    attribute_hru = att['hruId'][idx]

                    if store_frac is not None:
                        hist = store_frac.get(int(attribute_hru))
                        if hist is None:
                            self.logger.warning(f"No soil class data found for HRU {attribute_hru}, using default class")
                            att['soilTypeIndex'][idx] = 6  # Use a default value (6 = loam)
                            continue
                        tmp_hist = [hist.get(j, 0.0) for j in range(13)]
                        tmp_hist[0] = -1  # Set USGS_0 to -1 to avoid selecting it
                        tmp_sc = int(np.argmax(np.asarray(tmp_hist)))
                    else:
                        assert shp is not None  # read above when the store is unused
                        shp_mask = (shp[intersect_hruId_var].astype(int) == attribute_hru)

                        # Check if there are any matching rows for this HRU
                        if not any(shp_mask):
                            self.logger.warning(f"No soil class data found for HRU {attribute_hru}, using default class")
                            att['soilTypeIndex'][idx] = 6  # Use a default value (6 = loam)
                            continue

                        tmp_hist = []
                        for j in range(13):
                            col_name = f'USGS_{j}'
                            tmp_hist.append(shp[col_name][shp_mask].values[0])

                        tmp_hist[0] = -1  # Set USGS_0 to -1 to avoid selecting it
                        tmp_sc = np.argmax(np.asarray(tmp_hist))

                        if shp[f'USGS_{tmp_sc}'][shp_mask].values[0] != tmp_hist[tmp_sc]:
                            self.logger.warning(f'Index and mode soil class do not match at hru_id {attribute_hru}')
                            tmp_sc = 6  # Use a default value (6 = loam) instead of -999

                    # Ensure soil type index is positive (SUMMA requires this)
                    if tmp_sc <= 0:
                        self.logger.warning(f"Invalid soil class {tmp_sc} for HRU {attribute_hru}, using default class")
                        tmp_sc = 6  # Use a default value (6 = loam)

                    att['soilTypeIndex'][idx] = tmp_sc

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error inserting soil class: {str(e)}", exc_info=True)
            # If the process fails, set all soil types to a default value
            with nc4.Dataset(attribute_file, "r+") as att:
                self.logger.warning("Setting all soil types to default value (6 = loam)")
                att['soilTypeIndex'][:] = 6  # Set all to loam as fallback

    def insert_land_class(self, attribute_file):
        """Insert land class data into the attributes file."""
        self.logger.info("Inserting land class into attributes file")

        intersect_path = self._get_default_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self._get_config_value(
            lambda: self.config.paths.intersect_land_name, default='default'
        )
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'
        intersect_hruId_var = self._get_config_value(lambda: self.config.paths.catchment_hruid)

        try:
            # Prefer the model-ready attributes store (/landcover/ fractions
            # keyed by hru_id); fall back to the land-class intersection shapefile.
            store_frac = self._load_store_class_fractions(
                'landcover', 'land_fraction', 'land_class_name')
            shp = None
            if store_frac is None:
                shp = gpd.read_file(intersect_path / intersect_name)
                # Check and create missing IGBP_X columns
                for i in range(1, 18):
                    col_name = f'IGBP_{i}'
                    if col_name not in shp.columns:
                        shp[col_name] = 0  # Add the missing column and initialize with 0

            is_water = 0

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    attribute_hru = att['hruId'][idx]

                    if store_frac is not None:
                        hist = store_frac.get(int(attribute_hru))
                        if hist is None:
                            self.logger.warning(f"No land class data found for HRU {attribute_hru}, using default class")
                            att['vegTypeIndex'][idx] = 1  # Use a default value (1 = Evergreen Needleleaf)
                            continue
                        tmp_hist = [hist.get(j, 0.0) for j in range(1, 18)]
                        tmp_lc = int(np.argmax(np.asarray(tmp_hist))) + 1
                    else:
                        assert shp is not None  # read above when the store is unused
                        shp_mask = (shp[intersect_hruId_var].astype(int) == attribute_hru)

                        # Check if there are any matching rows for this HRU
                        if not any(shp_mask):
                            self.logger.warning(f"No land class data found for HRU {attribute_hru}, using default class")
                            att['vegTypeIndex'][idx] = 1  # Use a default value (1 = Evergreen Needleleaf)
                            continue

                        tmp_hist = []
                        for j in range(1, 18):
                            col_name = f'IGBP_{j}'
                            tmp_hist.append(shp[col_name][shp_mask].values[0])

                        tmp_lc = np.argmax(np.asarray(tmp_hist)) + 1

                        if shp[f'IGBP_{tmp_lc}'][shp_mask].values[0] != tmp_hist[tmp_lc - 1]:
                            self.logger.warning(f'Index and mode land class do not match at hru_id {attribute_hru}')
                            tmp_lc = 1  # Use a default value (1 = Evergreen Needleleaf) instead of -999

                    if tmp_lc == 17:
                        if any(val > 0 for val in tmp_hist[0:-1]):  # HRU is mostly water but other land classes are present
                            tmp_lc = np.argmax(np.asarray(tmp_hist[0:-1])) + 1  # select 2nd-most common class
                        else:
                            is_water += 1  # HRU is exclusively water

                    # Ensure vegetation type index is positive (SUMMA requires this)
                    if tmp_lc <= 0:
                        self.logger.warning(f"Invalid vegetation class {tmp_lc} for HRU {attribute_hru}, using default class")
                        tmp_lc = 1  # Use a default value (1 = Evergreen Needleleaf)

                    att['vegTypeIndex'][idx] = tmp_lc

                self.logger.info(f"{is_water} HRUs were identified as containing only open water. Note that SUMMA skips hydrologic calculations for such HRUs.")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error inserting land class: {str(e)}", exc_info=True)
            # If the process fails, set all vegetation types to a default value
            with nc4.Dataset(attribute_file, "r+") as att:
                self.logger.warning("Setting all vegetation types to default value (1 = Evergreen Needleleaf)")
                att['vegTypeIndex'][:] = 1  # Set all to Evergreen Needleleaf as fallback

    def insert_elevation(self, attribute_file):
        """Insert elevation data into the attributes file."""
        self.logger.info("Inserting elevation into attributes file")

        intersect_path = self._get_default_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self._get_config_value(
            lambda: self.config.paths.intersect_dem_name, default='default'
        )
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

        intersect_hruId_var = self._get_config_value(lambda: self.config.paths.catchment_hruid)
        elev_column = 'elev_mean'

        connect_hrus = self._get_config_value(
            lambda: self.config.model.summa.connect_hrus, default='no'
        )
        do_downHRUindex = connect_hrus == 'yes'

        # Prefer the model-ready attributes store (terrain/elev_mean keyed by
        # hru_id); fall back to the DEM intersection shapefile when the store is
        # absent or a given HRU is missing. The store's elev_mean is copied from
        # the same shapefile, so per-HRU values are identical — no objective drift.
        store_elev = self._load_store_elevation()
        shp = None if store_elev else gpd.read_file(intersect_path / intersect_name)

        with nc4.Dataset(attribute_file, "r+") as att:
            gru_data: Dict[int, List[Tuple[Any, float]]] = {}
            for idx in range(len(att['hruId'])):
                hru_id = att['hruId'][idx]
                gru_id = att['hru2gruId'][idx]

                elevation = store_elev.get(int(hru_id)) if store_elev else None
                if elevation is None:
                    if shp is None:
                        shp = gpd.read_file(intersect_path / intersect_name)
                    shp_mask = (shp[intersect_hruId_var].astype(int) == hru_id)
                    if any(shp_mask):
                        elevation = float(shp[elev_column][shp_mask].values[0])

                if elevation is not None:
                    att['elevation'][idx] = elevation
                    if do_downHRUindex:
                        gru_data.setdefault(gru_id, []).append((hru_id, elevation))
                else:
                    self.logger.warning(f"No elevation data found for HRU {hru_id}")

            if do_downHRUindex:
                self._set_downHRUindex(att, gru_data)

    def _load_store_elevation(self) -> Optional[Dict[int, float]]:
        """Load per-HRU mean elevation from the model-ready attributes store.

        Returns an ``{hru_id: elevation}`` mapping keyed by integer HRU id, or
        ``None`` when the store/terrain group is unavailable so the caller falls
        back to the DEM intersection shapefile.
        """
        try:
            from symfluence.core.modeling.model_ready import open_canonical_attributes

            domain_name = self.project_dir.name.replace('domain_', '', 1)
            reader = open_canonical_attributes(self.project_dir, domain_name)
            if reader is None:
                return None
            raw = reader.per_hru_values('terrain', 'elev_mean')
            if not raw:
                return None
            mapping: Dict[int, float] = {}
            for key, value in raw.items():
                try:
                    mapping[int(float(key))] = float(value)
                except (ValueError, TypeError):
                    continue
            if mapping:
                self.logger.info(
                    f"Using elevation from model-ready attributes store ({len(mapping)} HRUs)"
                )
                return mapping
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(
                f"Could not read elevation from attributes store: {e}", exc_info=True
            )
            return None

    def _load_store_class_fractions(
        self, group: str, frac_var: str, name_var: str
    ) -> Optional[Dict[int, Dict[int, float]]]:
        """Load per-HRU class fractions from a model-ready attributes group.

        Returns ``{hru_id: {class_index: fraction}}`` for a soil/landcover group
        (e.g. ``group='soil'``, ``frac_var='soil_fraction'``,
        ``name_var='soil_class_name'`` whose names are ``USGS_0``/``USGS_1``/…),
        keyed by integer HRU id. Returns ``None`` when the store/group is
        unavailable so the caller falls back to the intersection shapefile.

        No drift: the store fractions originate from the same intersection
        shapefile, so the dominant-class selection is identical.
        """
        try:
            from symfluence.core.modeling.model_ready import open_canonical_attributes

            domain_name = self.project_dir.name.replace('domain_', '', 1)
            reader = open_canonical_attributes(self.project_dir, domain_name)
            if reader is None or not reader.has_group(group):
                return None
            ids = reader.hru_ids(group)
            if not ids:
                return None
            with reader.group(group) as ds:
                if frac_var not in ds.variables or name_var not in ds.variables:
                    return None
                frac = np.asarray(ds[frac_var].values)
                names = [str(x) for x in np.atleast_1d(ds[name_var].values)]
            if frac.ndim != 2 or frac.shape[0] != len(ids):
                return None
            class_idx: List[Optional[int]] = []
            for nm in names:
                try:
                    class_idx.append(int(nm.split('_')[-1]))
                except ValueError:
                    class_idx.append(None)
            mapping: Dict[int, Dict[int, float]] = {}
            for h, hid in enumerate(ids):
                try:
                    key = int(float(hid))
                except (ValueError, TypeError):
                    continue
                mapping[key] = {
                    ci: float(frac[h, c])
                    for c, ci in enumerate(class_idx) if ci is not None
                }
            if mapping:
                self.logger.info(
                    f"Using {group} class fractions from model-ready attributes "
                    f"store ({len(mapping)} HRUs)"
                )
                return mapping
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(
                f"Could not read {group} fractions from store: {e}", exc_info=True
            )
            return None

    def _set_downHRUindex(self, att, gru_data):
        """Set the downHRUindex based on elevation data or D8 flow direction."""
        # Check if this is grid-based distribute mode
        definition_method = self._get_config_value(
            lambda: self.config.domain.definition_method, default='subset'
        )
        is_grid_distribute = definition_method == 'distribute'

        if is_grid_distribute:
            self._set_downHRUindex_from_d8(att)
        else:
            # Existing elevation-based logic
            for gru_id, hru_list in gru_data.items():
                sorted_hrus = sorted(hru_list, key=lambda x: x[1], reverse=True)
                for i, (hru_id, _) in enumerate(sorted_hrus):
                    idx = np.where(att['hruId'][:] == hru_id)[0][0]
                    if i == len(sorted_hrus) - 1:
                        att['downHRUindex'][idx] = 0  # outlet
                    else:
                        att['downHRUindex'][idx] = sorted_hrus[i+1][0]
                    self.logger.info(f"Set downHRUindex for HRU {hru_id} to {att['downHRUindex'][idx]}")

    def _set_downHRUindex_from_d8(self, att):
        """
        Set downHRUindex from D8 flow direction topology for grid-based modeling.

        Reads D8 connectivity from the grid shapefile and maps to SUMMA HRU indices.

        Args:
            att: NetCDF attributes file handle
        """
        self.logger.info("Setting downHRUindex from D8 flow direction")

        # Load grid shapefile with D8 topology
        domain_name = self._get_config_value(lambda: self.config.domain.name)
        grid_path = self.project_dir / 'shapefiles' / 'river_basins' / f"{domain_name}_riverBasins_distribute.shp"

        if not grid_path.exists():
            self.logger.warning(f"Grid basins not found at {grid_path}, using default connectivity")
            return

        grid_gdf = gpd.read_file(grid_path)

        # Create mapping from GRU_ID to downstream_id
        # Note: shapefile truncates column names to 10 chars, so downstream_id becomes downstream
        if 'downstream_id' in grid_gdf.columns:
            d8_downstream = dict(zip(grid_gdf['GRU_ID'].astype(int), grid_gdf['downstream_id'].astype(int)))
        elif 'downstream' in grid_gdf.columns:
            d8_downstream = dict(zip(grid_gdf['GRU_ID'].astype(int), grid_gdf['downstream'].astype(int)))
        elif 'DSLINKNO' in grid_gdf.columns:
            d8_downstream = dict(zip(grid_gdf['GRU_ID'].astype(int), grid_gdf['DSLINKNO'].astype(int)))
        else:
            self.logger.warning("No D8 topology found in grid shapefile")
            return

        # Get HRU IDs from attributes file
        hru_ids = att['hruId'][:]

        n_set = 0
        n_outlets = 0

        for idx, hru_id_raw in enumerate(hru_ids):
            # Handle potential MaskedArray
            if hasattr(hru_id_raw, 'item'):
                hru_id = int(hru_id_raw.item())
            else:
                hru_id = int(hru_id_raw)

            # Get downstream HRU from D8 topology
            downstream_hru = d8_downstream.get(hru_id, 0)

            if downstream_hru == 0:
                att['downHRUindex'][idx] = 0  # Outlet
                n_outlets += 1
            else:
                # Verify downstream HRU exists in attributes file
                downstream_indices = np.where(att['hruId'][:] == downstream_hru)[0]
                if len(downstream_indices) > 0:
                    att['downHRUindex'][idx] = int(downstream_hru)  # Use HRU ID, not array index
                else:
                    self.logger.warning(f"Downstream HRU {downstream_hru} not found for HRU {hru_id}")
                    att['downHRUindex'][idx] = 0  # Mark as outlet

            n_set += 1

        self.logger.info(f"Set downHRUindex for {n_set} HRUs using D8 flow direction")
        self.logger.info(f"Grid outlets: {n_outlets}")
