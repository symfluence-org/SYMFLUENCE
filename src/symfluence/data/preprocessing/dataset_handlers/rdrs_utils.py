# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RDRS Dataset Handler for SYMFLUENCE

This module provides the RDRS-specific implementation for forcing data processing.
It handles RDRS variable mappings, unit conversions, grid structure, and shapefile creation.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon

from symfluence.core.constants import PhysicalConstants, UnitConversion

from ...utils import VariableStandardizer
from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('rdrs')
class RDRSHandler(BaseDatasetHandler):
    """Handler for RDRS (Regional Deterministic Reforecast System) dataset."""

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        RDRS variable name mapping to standard names.

        Combines v2.1 and v3.1 rename maps so this handler can process
        either version. Unit differences are handled by heuristic checks
        in process_dataset().

        Returns:
            Dictionary mapping RDRS variable names to standard names
        """
        standardizer = VariableStandardizer(self.logger)
        combined = standardizer.get_rename_map('RDRS')
        combined.update(standardizer.get_rename_map('RDRS_v3.1'))
        return combined

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process RDRS dataset with variable renaming and unit conversions.

        Unit conversions applied:
        - airpres: mb -> Pa (multiply by 100)
        - airtemp: °C -> K (add 273.15)
        - pptrate: mm/hr -> m/s (divide by 3600)
        - windspd: knots -> m/s (multiply by 0.514444)

        Args:
            ds: Input RDRS dataset

        Returns:
            Processed dataset with standardized variables and units
        """
        # Rename variables, handling cases where multiple source vars map to the same target
        variable_mapping = self.get_variable_mapping()
        # Build rename dict: only include sources present in dataset, skip if target already exists
        rename_dict = {}
        targets_seen = set(ds.variables)  # track existing + already-claimed target names
        for old, new in variable_mapping.items():
            if old in ds.variables and new not in targets_seen:
                rename_dict[old] = new
                targets_seen.add(new)
        ds = ds.rename(rename_dict)

        # Apply unit conversions (must happen before attribute setting)
        if 'surface_air_pressure' in ds:
            # RDRS v2.1 uses mb, but v3.1 might use Pa
            if ds['surface_air_pressure'].max() < 2000: # Probably mb
                ds['surface_air_pressure'] = ds['surface_air_pressure'] * 100

        if 'air_temperature' in ds:
            # RDRS v2.1 uses Celsius, but v3.1 might use Kelvin
            if ds['air_temperature'].max() < 100: # Probably Celsius
                ds['air_temperature'] = ds['air_temperature'] + PhysicalConstants.KELVIN_OFFSET

        if 'precipitation_flux' in ds:
            # RDRS v2.1 uses mm/hr, but v3.1 might use kg/m2/s (which is mm/s)
            # Check if it's already small enough to be mm/s
            if ds['precipitation_flux'].max() > 0.1: # Probably mm/hr
                ds['precipitation_flux'] = ds['precipitation_flux'] / UnitConversion.SECONDS_PER_HOUR

        if 'wind_speed' in ds:
            # RDRS v2.1 uses knots, but v3.1 uses m/s
            if 'UVC' in rename_dict: # v3.1 names
                pass
            else:
                ds['wind_speed'] = ds['wind_speed'] * 0.514444

        # Apply standard CF-compliant attributes (uses centralized definitions)
        # RDRS precipitation is in mm/s (or kg m-2 s-1, which is equivalent) after conversion
        ds = self.apply_standard_attributes(ds, overrides={
            'precipitation_flux': {'units': 'kg m-2 s-1', 'standard_name': 'precipitation_rate'}
        })

        return ds

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        RDRS uses rotated pole coordinates with auxiliary lat/lon.

        Returns:
            Tuple of ('lat', 'lon') for auxiliary coordinates
        """
        return ('lat', 'lon')

    def needs_merging(self) -> bool:
        """RDRS requires merging of daily files into monthly files."""
        return True

    def _detect_consolidated_file(self, raw_forcing_path: Path) -> Optional[Path]:
        """
        Check if raw forcing data is a single consolidated file (from cloud download).

        Returns:
            Path to consolidated file if found, None otherwise
        """
        patterns = [
            f"domain_{self.domain_name}_RDRS_*.nc",
            f"{self.domain_name}_RDRS_*.nc",
            "*RDRS*.nc",
        ]
        for pattern in patterns:
            matches = sorted(raw_forcing_path.glob(pattern))
            if len(matches) == 1:
                return matches[0]
        return None

    def _has_daily_files(self, raw_forcing_path: Path, start_year: int, end_year: int) -> bool:
        """Check if raw forcing data is organized as daily files in year subdirectories."""
        for year in range(start_year - 1, end_year + 1):
            year_folder = raw_forcing_path / str(year)
            if year_folder.exists():
                return True
        return False

    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Merge RDRS forcing data files into monthly files.

        Handles two acquisition formats:
        - Cloud download: single consolidated NetCDF split into monthly files
        - Traditional download: daily files in year subdirectories merged into monthly files

        Args:
            raw_forcing_path: Path to raw RDRS data
            merged_forcing_path: Path where merged monthly files will be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        self.logger.info("Starting to merge RDRS forcing data")
        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        consolidated_file = self._detect_consolidated_file(raw_forcing_path)

        if consolidated_file and not self._has_daily_files(raw_forcing_path, start_year, end_year):
            self._merge_from_consolidated(consolidated_file, merged_forcing_path, start_year, end_year)
        else:
            self._merge_from_daily_files(raw_forcing_path, merged_forcing_path, start_year, end_year)

        self.logger.info("RDRS forcing data merging completed")

    def _merge_from_consolidated(self, consolidated_file: Path, merged_forcing_path: Path,
                                  start_year: int, end_year: int) -> None:
        """
        Split a single consolidated RDRS file (from cloud download) into monthly files.

        Args:
            consolidated_file: Path to the consolidated NetCDF file
            merged_forcing_path: Path where monthly files will be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        self.logger.info(f"Processing consolidated cloud file: {consolidated_file.name}")

        ds = self.open_dataset(consolidated_file)
        ds = self.process_dataset(ds)

        years = range(start_year - 1, end_year + 1)

        for year in years:
            for month in range(1, 13):
                start_time = pd.Timestamp(year, month, 1)
                if month == 12:
                    end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
                else:
                    end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)

                # Select time slice for this month
                monthly_data = ds.sel(time=slice(str(start_time), str(end_time)))

                if monthly_data.sizes['time'] == 0:
                    self.logger.debug(f"No data for {year}-{month:02d}, skipping")
                    continue

                # Ensure complete hourly time series and fill gaps
                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times)
                monthly_data = monthly_data.interpolate_na(dim='time', method='linear')
                monthly_data = monthly_data.ffill(dim='time').bfill(dim='time')

                # Set time encoding and metadata
                monthly_data = self.setup_time_encoding(monthly_data)
                monthly_data = self.add_metadata(
                    monthly_data,
                    'RDRS data split from consolidated file into monthly files and variables standardized'
                )
                monthly_data = self.clean_variable_attributes(monthly_data)

                output_file = merged_forcing_path / f"RDRS_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)
                self.logger.info(f"Saved monthly file: {output_file.name}")

        ds.close()

    def _merge_from_daily_files(self, raw_forcing_path: Path, merged_forcing_path: Path,
                                 start_year: int, end_year: int) -> None:
        """
        Merge daily RDRS files from year subdirectories into monthly files.

        Args:
            raw_forcing_path: Path to raw RDRS data organized by year
            merged_forcing_path: Path where merged monthly files will be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        self.logger.info("Processing daily files from year subdirectories")

        years = range(start_year - 1, end_year + 1)
        file_name_pattern = f"domain_{self.domain_name}_*.nc"

        for year in years:
            self.logger.debug(f"Processing RDRS year {year}")
            year_folder = raw_forcing_path / str(year)

            if not year_folder.exists():
                self.logger.debug(f"Year folder not found: {year_folder}")
                continue

            for month in range(1, 13):
                self.logger.debug(f"Processing RDRS {year}-{month:02d}")

                # Find daily files for this month
                daily_files = list(year_folder.glob(
                    file_name_pattern.replace('*', f'{year}{month:02d}*')
                ))

                # Also look for the last file of the previous month to cover the start of this month
                # (RDRS files starting with YYYYMMDD12 contain data for the first 12 hours of the next day)
                prev_month_date = pd.Timestamp(year, month, 1) - pd.Timedelta(days=1)
                prev_year = prev_month_date.year
                prev_year_folder = raw_forcing_path / str(prev_year)

                if prev_year_folder.exists():
                    prev_pattern = file_name_pattern.replace('*', f"{prev_month_date.strftime('%Y%m%d')}*")
                    prev_files = list(prev_year_folder.glob(prev_pattern))
                    daily_files.extend(prev_files)

                daily_files = sorted(list(set(daily_files)))

                if not daily_files:
                    self.logger.debug(f"No RDRS files found for {year}-{month:02d}")
                    continue

                # Load datasets
                datasets = []
                for file in daily_files:
                    try:
                        ds = self.open_dataset(file)
                        datasets.append(ds)
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.error(f"Error opening RDRS file {file}: {str(e)}")

                if not datasets:
                    self.logger.warning(f"No valid RDRS datasets for {year}-{month:02d}")
                    continue

                # Process each dataset
                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = self.process_dataset(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.error(f"Error processing RDRS dataset: {str(e)}")

                if not processed_datasets:
                    self.logger.warning(f"No processed RDRS datasets for {year}-{month:02d}")
                    continue

                # Concatenate into monthly data
                monthly_data = xr.concat(processed_datasets, dim="time", data_vars='all')
                monthly_data = monthly_data.sortby("time")
                monthly_data = monthly_data.drop_duplicates(dim='time')

                # Set up time range
                start_time = pd.Timestamp(year, month, 1)
                if month == 12:
                    end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
                else:
                    end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)

                # Ensure complete hourly time series and fill gaps
                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times)
                monthly_data = monthly_data.interpolate_na(dim='time', method='linear')
                monthly_data = monthly_data.ffill(dim='time').bfill(dim='time')

                # Set time encoding and metadata
                monthly_data = self.setup_time_encoding(monthly_data)
                monthly_data = self.add_metadata(
                    monthly_data,
                    'RDRS data aggregated to monthly files and variables renamed for SUMMA compatibility'
                )
                monthly_data = self.clean_variable_attributes(monthly_data)

                # Save monthly file
                output_file = merged_forcing_path / f"RDRS_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)

                # Clean up
                for ds in datasets:
                    ds.close()

    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create RDRS grid shapefile.

        RDRS uses a rotated pole grid with auxiliary lat/lon coordinates.

        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to merged RDRS data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics

        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating RDRS grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        try:
            # Find an RDRS file to get grid information
            forcing_file = next(
                (f for f in os.listdir(merged_forcing_path)
                 if f.endswith('.nc') and f.startswith('RDRS_monthly_')),
                None
            )

            if not forcing_file:
                self.logger.error("No RDRS monthly file found")
                raise FileNotFoundError("No RDRS monthly file found")

            # Read grid information
            with self.open_dataset(merged_forcing_path / forcing_file) as ds:
                rlat, rlon = ds.rlat.values, ds.rlon.values
                lat, lon = ds.lat.values, ds.lon.values

            self.logger.info(f"RDRS dimensions: rlat={rlat.shape}, rlon={rlon.shape}")

            # Create grid cells
            geometries, ids, lats, lons = [], [], [], []

            batch_size = 100
            total_cells = len(rlat) * len(rlon)
            num_batches = (total_cells + batch_size - 1) // batch_size

            self.logger.info(f"Creating RDRS grid cells in {num_batches} batches")

            cell_count = 0
            for i in range(len(rlat)):
                for j in range(len(rlon)):
                    # Create grid cell corners
                    [
                        rlat[i], rlat[i],
                        rlat[i+1] if i+1 < len(rlat) else rlat[i],
                        rlat[i+1] if i+1 < len(rlat) else rlat[i]
                    ]
                    [
                        rlon[j],
                        rlon[j+1] if j+1 < len(rlon) else rlon[j],
                        rlon[j+1] if j+1 < len(rlon) else rlon[j],
                        rlon[j]
                    ]

                    # Get actual lat/lon corners
                    lat_corners = [
                        lat[i,j],
                        lat[i, j+1] if j+1 < len(rlon) else lat[i,j],
                        lat[i+1, j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j],
                        lat[i+1, j] if i+1 < len(rlat) else lat[i,j]
                    ]
                    lon_corners = [
                        lon[i,j],
                        lon[i, j+1] if j+1 < len(rlon) else lon[i,j],
                        lon[i+1, j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j],
                        lon[i+1, j] if i+1 < len(rlat) else lon[i,j]
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * len(rlon) + j)
                    lats.append(lat[i,j])
                    lons.append(lon[i,j])

                    cell_count += 1
                    if cell_count % batch_size == 0 or cell_count == total_cells:
                        self.logger.info(f"Created {cell_count}/{total_cells} RDRS grid cells")

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self._get_config_value(lambda: self.config.forcing.shape_lat_name, default='lat'): lats,
                self._get_config_value(lambda: self.config.forcing.shape_lon_name, default='lon'): lons,
            }, crs='EPSG:4326')

            # Calculate elevation
            self.logger.info("Calculating elevation values using safe method")
            elevations = elevation_calculator(gdf, dem_path, batch_size=50)
            gdf['elev_m'] = elevations

            # Remove invalid elevation cells if requested
            if self._get_config_value(lambda: None, default=False, dict_key='REMOVE_INVALID_ELEVATION_CELLS'):
                valid_count = len(gdf)
                gdf = gdf[gdf['elev_m'] != -9999].copy()
                removed_count = valid_count - len(gdf)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} cells with invalid elevation values")

            # Save shapefile
            output_shapefile.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_shapefile)
            self.logger.info(f"RDRS shapefile created and saved to {output_shapefile}")

            return output_shapefile

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error in create_rdrs_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
