# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5 Dataset Handler for SYMFLUENCE

This module provides the ERA5-specific implementation for forcing data processing.
ERA5 uses regular lat/lon grids and typically doesn't require merging operations.
"""

import shutil
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon

from ...utils import VariableStandardizer
from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('era5')
@DatasetRegistry.register('era5_cds')
class ERA5Handler(BaseDatasetHandler):
    """Handler for ERA5 (ECMWF Reanalysis v5) dataset."""

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        ERA5 variable name mapping to standard names.

        Uses centralized VariableStandardizer for consistency across the codebase.

        Returns:
            Dictionary mapping ERA5 variable names to standard names
        """
        standardizer = VariableStandardizer(self.logger)
        return standardizer.get_rename_map('ERA5')

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process ERA5 dataset with variable renaming if needed.

        ERA5 data typically comes in standard units, but this method
        handles any necessary conversions.

        Args:
            ds: Input ERA5 dataset

        Returns:
            Processed dataset with standardized variables
        """
        # Rename variables using mapping
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}

        if existing_vars:
            ds = ds.rename(existing_vars)

        # Normalize longitude from 0-360 to -180/+180 if needed
        # ARCO-ERA5 uses 0-360 convention, but shapefiles use -180/+180
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        if lon_name in ds.coords and float(ds[lon_name].max()) > 180:
            new_lons = ds[lon_name].values.copy()
            new_lons[new_lons > 180] -= 360
            ds = ds.assign_coords({lon_name: new_lons})
            ds = ds.sortby(lon_name)

        # Apply standard CF-compliant attributes (uses centralized definitions)
        # ERA5 precipitation is typically in mm/s, override the default
        ds = self.apply_standard_attributes(ds, overrides={
            'pptrate': {'units': 'mm/s', 'standard_name': 'precipitation_rate'}
        })

        return ds

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        ERA5 uses standard latitude/longitude coordinates.

        Returns:
            Tuple of ('latitude', 'longitude')
        """
        return ('latitude', 'longitude')

    def needs_merging(self) -> bool:
        """
        ERA5 data needs 'merging' to ensure variable names are standardized.
        Even if not combining files, we use this step to rename vars (e.g. tp -> pptrate).
        """
        return True

    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Process raw ERA5 files to ensure variable names are standardized.

        This step copies raw files to the merged path, applying variable renaming
        (e.g., tp -> pptrate) via process_dataset().
        """
        self.logger.info("Processing ERA5 files to standardize variables...")

        all_raw_files = sorted(list(raw_forcing_path.glob('*.nc')))
        if not all_raw_files:
            self.logger.warning(f"No raw ERA5 files found in {raw_forcing_path}")
            return

        # Filter to files whose year range overlaps the configured period
        raw_files = [
            f for f in all_raw_files
            if self._file_overlaps_period(f, start_year, end_year)
        ]
        skipped = len(all_raw_files) - len(raw_files)
        if skipped:
            self.logger.info(
                f"Skipped {skipped} ERA5 file(s) outside configured period "
                f"{start_year}-{end_year}"
            )

        # Create a temp dir for processing to avoid lock/permission issues
        temp_dir = merged_forcing_path.parent / 'temp_processing'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for raw_file in raw_files:
                # Output filename matches input for simplicity in this case
                output_file = merged_forcing_path / raw_file.name

                if output_file.exists():
                    self.logger.debug(f"File already processed: {output_file.name}")
                    continue

                self.logger.debug(f"Processing {raw_file.name} -> {output_file.name}")
                try:
                    # Use load() to read into memory and close file handle immediately
                    with self.open_dataset(raw_file) as ds:
                        # Apply standardization (renaming vars like tp->pptrate)
                        ds_processed = self.process_dataset(ds)
                        ds_processed.load() # Load into memory

                    # Clean attributes and encoding via base class method
                    ds_final = self.clean_variable_attributes(ds_processed)

                    # Write to a file in temp dir first
                    temp_output = temp_dir / raw_file.name
                    if temp_output.exists():
                        temp_output.unlink()

                    ds_final.to_netcdf(temp_output)
                    ds_final.close()

                    # Verify temp file exists
                    if not temp_output.exists():
                        raise FileNotFoundError(f"Failed to create temp file: {temp_output}")

                    self.logger.debug(f"Temp file created: {temp_output} ({temp_output.stat().st_size} bytes)")

                    # Move to final filename (copy then delete to be safe)
                    if output_file.exists():
                        output_file.unlink()

                    shutil.copy2(str(temp_output), str(output_file))
                    temp_output.unlink()

                    self.logger.debug(f"Successfully saved {output_file.name}")

                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.error(f"Failed to process {raw_file.name}: {e}")
                    raise
        finally:
            # Cleanup temp dir
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        self.logger.info("ERA5 file processing complete.")

    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create ERA5 grid shapefile.

        ERA5 uses a regular lat/lon grid (typically 0.25° resolution).

        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to ERA5 data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics

        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating ERA5 shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        try:
            # Find an .nc file in the forcing path
            forcing_files = list(merged_forcing_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No ERA5 forcing files found")

            forcing_file = forcing_files[0]
            self.logger.info(f"Using ERA5 file: {forcing_file}")

            # Set the dimension variable names
            source_name_lat = "latitude"
            source_name_lon = "longitude"

            # Open the file and get the dimensions
            try:
                with self.open_dataset(forcing_file) as src:
                    lat = src[source_name_lat].values
                    lon = src[source_name_lon].values

                self.logger.info(f"ERA5 dimensions: lat={lat.shape}, lon={lon.shape}")

                # Check for empty dimensions
                if lat.size == 0 or lon.size == 0:
                    raise ValueError(
                        f"ERA5 forcing file has empty spatial dimensions (lat: {lat.shape}, lon: {lon.shape}). "
                        f"This typically happens when the bounding box is smaller than ERA5's 0.25° resolution. "
                        f"Please use a larger bounding box (at least 0.5° x 0.5°) or use a higher-resolution "
                        f"forcing dataset like HRRR or CONUS404 for small domains."
                    )
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error reading ERA5 dimensions: {str(e)}")
                raise

            # Find the grid spacing
            try:
                half_dlat = abs(lat[1] - lat[0])/2 if len(lat) > 1 else 0.125
                half_dlon = abs(lon[1] - lon[0])/2 if len(lon) > 1 else 0.125

                self.logger.info(f"ERA5 grid spacings: half_dlat={half_dlat}, half_dlon={half_dlon}")
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error calculating grid spacings: {str(e)}")
                raise

            # Create lists to store the data
            geometries = []
            ids = []
            lats = []
            lons = []

            # Create grid cells
            try:
                self.logger.info("Creating grid cell geometries")
                if len(lat) == 1:
                    self.logger.info("Single latitude value detected, creating row of grid cells")
                    for i, center_lon in enumerate(lon):
                        center_lat = lat[0]
                        vertices = [
                            [float(center_lon)-half_dlon, float(center_lat)-half_dlat],
                            [float(center_lon)-half_dlon, float(center_lat)+half_dlat],
                            [float(center_lon)+half_dlon, float(center_lat)+half_dlat],
                            [float(center_lon)+half_dlon, float(center_lat)-half_dlat],
                            [float(center_lon)-half_dlon, float(center_lat)-half_dlat]
                        ]
                        geometries.append(Polygon(vertices))
                        ids.append(i)
                        lats.append(float(center_lat))
                        lons.append(float(center_lon))
                else:
                    self.logger.info("Multiple latitude values, creating grid")
                    for i, center_lon in enumerate(lon):
                        for j, center_lat in enumerate(lat):
                            vertices = [
                                [float(center_lon)-half_dlon, float(center_lat)-half_dlat],
                                [float(center_lon)-half_dlon, float(center_lat)+half_dlat],
                                [float(center_lon)+half_dlon, float(center_lat)+half_dlat],
                                [float(center_lon)+half_dlon, float(center_lat)-half_dlat],
                                [float(center_lon)-half_dlon, float(center_lat)-half_dlat]
                            ]
                            geometries.append(Polygon(vertices))
                            ids.append(i * len(lat) + j)
                            lats.append(float(center_lat))
                            lons.append(float(center_lon))

                self.logger.info(f"Created {len(geometries)} grid cell geometries")
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error creating grid cell geometries: {str(e)}")
                raise

            # Create the GeoDataFrame
            try:
                self.logger.info("Creating GeoDataFrame")
                gdf = gpd.GeoDataFrame({
                    'geometry': geometries,
                    'ID': ids,
                    self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): lats,
                    self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): lons,
                }, crs='EPSG:4326')

                self.logger.info(f"GeoDataFrame created with {len(gdf)} rows")
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error creating GeoDataFrame: {str(e)}")
                raise

            # Calculate elevation using the safe method
            try:
                self.logger.info("Calculating elevation values using safe method")

                if not Path(dem_path).exists():
                    self.logger.error(f"DEM file not found: {dem_path}")
                    raise FileNotFoundError(f"DEM file not found: {dem_path}")

                elevations = elevation_calculator(gdf, dem_path, batch_size=20)
                gdf['elev_m'] = elevations

                self.logger.info("Elevation calculation complete")
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error calculating elevation: {str(e)}")
                # Continue without elevation data rather than failing completely
                gdf['elev_m'] = -9999
                self.logger.warning("Using default elevation values due to calculation error")

            # Save the shapefile
            try:
                self.logger.info(f"Saving shapefile to: {output_shapefile}")
                gdf.to_file(output_shapefile)
                self.logger.info(f"ERA5 shapefile saved successfully to {output_shapefile}")
                return output_shapefile
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error saving shapefile: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error in create_era5_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
