# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CASR Dataset Handler for SYMFLUENCE

This module provides the CASR (Canadian Arctic System Reanalysis) specific implementation
for forcing data processing. It handles CASR variable mappings, unit conversions,
grid structure, and shapefile creation.
"""

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


@DatasetRegistry.register('casr')
class CASRHandler(BaseDatasetHandler):
    """Handler for CASR (Canadian Arctic System Reanalysis) dataset."""

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        CASR variable name mapping to standard names.

        Combines v3.1 and v3.2 rename maps so this handler can process
        either version. Unit differences are handled by heuristic checks
        in process_dataset().

        Returns:
            Dictionary mapping CASR variable names to standard names
        """
        standardizer = VariableStandardizer(self.logger)
        combined = standardizer.get_rename_map('CASR_v3.1')
        combined.update(standardizer.get_rename_map('CASR_v3.2'))
        return combined

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CASR dataset with variable renaming and unit conversions.

        Unit conversions applied:
        - airpres: mb -> Pa (multiply by 100)
        - airtemp: °C -> K (add 273.15)
        - pptrate: m/hr -> m/s (multiply by 1000, divide by 3600)
        - windspd: knots -> m/s (multiply by 0.514444)

        Args:
            ds: Input CASR dataset

        Returns:
            Processed dataset with standardized variables and units
        """
        # Rename variables, handling cases where multiple source vars map to the same target
        variable_mapping = self.get_variable_mapping()
        rename_dict = {}
        targets_seen = set(ds.variables)
        for old, new in variable_mapping.items():
            if old in ds.variables and new not in targets_seen:
                rename_dict[old] = new
                targets_seen.add(new)
        ds = ds.rename(rename_dict)

        # Apply unit conversions with clean attributes
        # Use heuristic checks to handle both CaSR v3.1 (non-SI) and v3.2 (SI units)
        if 'airpres' in ds:
            # v3.1 uses mb (typically 800-1100), v3.2 uses Pa (typically 80000-110000)
            if ds['airpres'].max() < 2000:  # Probably mb
                ds['airpres'] = ds['airpres'] * 100
            ds['airpres'].attrs = {}
            ds['airpres'].attrs.update({
                'units': 'Pa',
                'long_name': 'air pressure',
                'standard_name': 'air_pressure'
            })

        if 'airtemp' in ds:
            # v3.1 uses °C (typically -40 to +50), v3.2 uses K (typically 220-330)
            if ds['airtemp'].max() < 100:  # Probably Celsius
                ds['airtemp'] = ds['airtemp'] + PhysicalConstants.KELVIN_OFFSET
            ds['airtemp'].attrs = {}
            ds['airtemp'].attrs.update({
                'units': 'K',
                'long_name': 'air temperature',
                'standard_name': 'air_temperature'
            })

        if 'pptrate' in ds:
            # v3.1 uses m/hr, v3.2 uses kg/m2/s (which is mm/s)
            if ds['pptrate'].max() > 0.1:  # Probably m/hr or mm/hr, not mm/s
                ds['pptrate'] = ds['pptrate'] * 1000 / UnitConversion.SECONDS_PER_HOUR
            ds['pptrate'].attrs = {}
            ds['pptrate'].attrs.update({
                'units': 'mm s-1',
                'long_name': 'precipitation rate',
                'standard_name': 'precipitation_rate'
            })

        if 'windspd' in ds:
            # v3.1 uses knots (typically 0-60), v3.2 uses m/s (typically 0-30)
            # Knots to m/s: * 0.514444, so knots values are ~2x larger
            if ds['windspd'].max() > 50:  # Probably knots
                ds['windspd'] = ds['windspd'] * 0.514444
            ds['windspd'].attrs = {}
            ds['windspd'].attrs.update({
                'units': 'm s-1',
                'long_name': 'wind speed',
                'standard_name': 'wind_speed'
            })

        if 'windspd_u' in ds:
            if ds['windspd_u'].max() > 50:  # Probably knots
                ds['windspd_u'] = ds['windspd_u'] * 0.514444
            ds['windspd_u'].attrs = {}
            ds['windspd_u'].attrs.update({
                'units': 'm s-1',
                'long_name': 'eastward wind',
                'standard_name': 'eastward_wind'
            })

        if 'windspd_v' in ds:
            if ds['windspd_v'].max() > 50:  # Probably knots
                ds['windspd_v'] = ds['windspd_v'] * 0.514444
            ds['windspd_v'].attrs = {}
            ds['windspd_v'].attrs.update({
                'units': 'm s-1',
                'long_name': 'northward wind',
                'standard_name': 'northward_wind'
            })

        # Radiation variables are already in W m**-2, just update attributes
        if 'LWRadAtm' in ds:
            ds['LWRadAtm'].attrs = {}
            ds['LWRadAtm'].attrs.update({
                'long_name': 'downward longwave radiation at the surface',
                'standard_name': 'surface_downwelling_longwave_flux_in_air'
            })

        if 'SWRadAtm' in ds:
            ds['SWRadAtm'].attrs = {}
            ds['SWRadAtm'].attrs.update({
                'long_name': 'downward shortwave radiation at the surface',
                'standard_name': 'surface_downwelling_shortwave_flux_in_air'
            })

        if 'spechum' in ds:
            ds['spechum'].attrs = {}
            ds['spechum'].attrs.update({
                'long_name': 'specific humidity',
                'standard_name': 'specific_humidity'
            })

        return ds

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CASR uses rotated pole coordinates with auxiliary lat/lon.

        Returns:
            Tuple of ('lat', 'lon') for auxiliary coordinates
        """
        return ('lat', 'lon')

    def needs_merging(self) -> bool:
        """CASR requires merging of daily files into monthly files."""
        return True

    def _detect_consolidated_file(self, raw_forcing_path: Path) -> Optional[Path]:
        """
        Check if raw forcing data is a single consolidated file (from OPeNDAP download).

        The RDRSAcquirer (which also handles CaSR v3.2) saves data as
        ``domain_{domain_name}_RDRS_{start}_{end}.nc``.

        Returns:
            Path to consolidated file if found, None otherwise
        """
        patterns = [
            f"domain_{self.domain_name}_RDRS_*.nc",
            f"{self.domain_name}_RDRS_*.nc",
            f"domain_{self.domain_name}_CASR_*.nc",
            f"{self.domain_name}_CASR_*.nc",
            "*RDRS*.nc",
            "*CASR*.nc",
        ]
        for pattern in patterns:
            matches = sorted(raw_forcing_path.glob(pattern))
            if len(matches) == 1:
                return matches[0]
        return None

    def _has_daily_files(self, raw_forcing_path: Path, start_year: int, end_year: int) -> bool:
        """Check if raw forcing data has daily files matching the expected pattern."""
        file_name_pattern = f"domain_{self.domain_name}_*.nc"
        all_files = list(raw_forcing_path.glob(file_name_pattern))
        # Check if any files match the daily date pattern (YYYYMMDD in name)
        for f in all_files:
            name = f.name
            for year in range(start_year - 1, end_year + 1):
                if f"_{year}" in name and any(f"_{year}{m:02d}" in name for m in range(1, 13)):
                    return True
        return False

    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Merge CASR forcing data files into monthly files.

        Handles two acquisition formats:
        - Cloud download (OPeNDAP): single consolidated NetCDF split into monthly files
        - Traditional download: daily files in one directory merged into monthly files

        Args:
            raw_forcing_path: Path to raw CASR data
            merged_forcing_path: Path where merged monthly files will be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        self.logger.info("Starting to merge CASR forcing data")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        # Check for consolidated file from OPeNDAP/cloud download first
        consolidated_file = self._detect_consolidated_file(raw_forcing_path)
        if consolidated_file and not self._has_daily_files(raw_forcing_path, start_year, end_year):
            self._merge_from_consolidated(consolidated_file, merged_forcing_path, start_year, end_year)
            return

        years = range(start_year - 1, end_year + 1)

        # Get all CASR files in the raw_data directory
        file_name_pattern = f"domain_{self.domain_name}_*.nc"
        all_casr_files = sorted(raw_forcing_path.glob(file_name_pattern))

        if not all_casr_files:
            self.logger.warning(f"No CASR files found in {raw_forcing_path}")
            return

        self.logger.info(f"Found {len(all_casr_files)} CASR files in {raw_forcing_path}")

        for year in years:
            self.logger.debug(f"Processing CASR year {year}")

            for month in range(1, 13):
                self.logger.debug(f"Processing CASR {year}-{month:02d}")

                # Look for daily CASR files for this month
                month_pattern = f"domain_{self.domain_name}_{year}{month:02d}"
                daily_files = [f for f in all_casr_files if month_pattern in f.name]

                # Also look for the last file of the previous month to cover the start of this month
                # (CASR files starting with YYYYMMDD12 contain data for the first 12 hours of the next day)
                prev_month_date = pd.Timestamp(year, month, 1) - pd.Timedelta(days=1)
                prev_pattern = f"domain_{self.domain_name}_{prev_month_date.strftime('%Y%m%d')}"
                prev_files = [f for f in all_casr_files if prev_pattern in f.name]

                daily_files = sorted(list(set(daily_files + prev_files)))

                if not daily_files:
                    self.logger.debug(f"No CASR files found for {year}-{month:02d}")
                    continue

                self.logger.debug(f"Found {len(daily_files)} CASR files for {year}-{month:02d}")

                # Load datasets
                datasets = []
                for file in daily_files:
                    try:
                        ds = self.open_dataset(file)
                        ds = ds.drop_duplicates(dim='time')
                        datasets.append(ds)
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.error(f"Error opening CASR file {file}: {str(e)}")

                if not datasets:
                    self.logger.warning(f"No valid CASR datasets for {year}-{month:02d}")
                    continue

                # Process each dataset
                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = self.process_dataset(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.error(f"Error processing CASR dataset: {str(e)}")

                if not processed_datasets:
                    self.logger.warning(f"No processed CASR datasets for {year}-{month:02d}")
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

                # Ensure complete hourly time series and fill gaps (CASR is 3-hourly)
                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times)
                monthly_data = monthly_data.interpolate_na(dim='time', method='linear')
                monthly_data = monthly_data.ffill(dim='time').bfill(dim='time')

                # Set time encoding
                monthly_data = self.setup_time_encoding(monthly_data)

                # Add metadata
                monthly_data = self.add_metadata(
                    monthly_data,
                    'CASR data aggregated to monthly files and variables renamed for SUMMA compatibility'
                )

                # Aggressively clean up variable attributes and encoding
                for var_name in monthly_data.data_vars:
                    var = monthly_data[var_name]
                    var.attrs.clear()
                    var.encoding.clear()

                    # Set clean attributes based on variable name
                    if var_name == 'airpres':
                        var.attrs = {'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'}
                    elif var_name == 'airtemp':
                        var.attrs = {'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'}
                    elif var_name == 'pptrate':
                        var.attrs = {'units': 'mm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'}
                    elif var_name == 'windspd':
                        var.attrs = {'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'}
                    elif var_name == 'windspd_u':
                        var.attrs = {'units': 'm s-1', 'long_name': 'eastward wind', 'standard_name': 'eastward_wind'}
                    elif var_name == 'windspd_v':
                        var.attrs = {'units': 'm s-1', 'long_name': 'northward wind', 'standard_name': 'northward_wind'}
                    elif var_name == 'LWRadAtm':
                        var.attrs = {'units': 'W m-2', 'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air'}
                    elif var_name == 'SWRadAtm':
                        var.attrs = {'units': 'W m-2', 'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'}
                    elif var_name == 'spechum':
                        var.attrs = {'units': 'kg kg-1', 'long_name': 'specific humidity', 'standard_name': 'specific_humidity'}

                    # Consistently set missing values in encoding
                    var.encoding['missing_value'] = -999.0
                    var.encoding['_FillValue'] = -999.0

                # Save monthly file
                output_file = merged_forcing_path / f"CASR_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)
                self.logger.debug(f"Saved CASR monthly file: {output_file}")

                # Clean up
                for ds in datasets:
                    ds.close()

        self.logger.info("CASR forcing data merging completed")

    def _merge_from_consolidated(self, consolidated_file: Path, merged_forcing_path: Path,
                                  start_year: int, end_year: int) -> None:
        """
        Split a single consolidated CASR file (from OPeNDAP download) into monthly files.

        The RDRSAcquirer downloads CaSR v3.2 as a single file. This method splits it
        into monthly files with variable renaming and unit conversions applied.

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

                monthly_data = ds.sel(time=slice(str(start_time), str(end_time)))

                if monthly_data.sizes['time'] == 0:
                    self.logger.debug(f"No data for {year}-{month:02d}, skipping")
                    continue

                # Ensure complete hourly time series and fill gaps (CASR is 3-hourly)
                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times)
                monthly_data = monthly_data.interpolate_na(dim='time', method='linear')
                monthly_data = monthly_data.ffill(dim='time').bfill(dim='time')

                monthly_data = self.setup_time_encoding(monthly_data)
                monthly_data = self.add_metadata(
                    monthly_data,
                    'CASR data split from consolidated file into monthly files and variables standardized'
                )

                # Clean variable attributes
                for var_name in monthly_data.data_vars:
                    var = monthly_data[var_name]
                    var.attrs.clear()
                    var.encoding.clear()

                    if var_name == 'airpres':
                        var.attrs = {'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'}
                    elif var_name == 'airtemp':
                        var.attrs = {'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'}
                    elif var_name == 'pptrate':
                        var.attrs = {'units': 'mm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'}
                    elif var_name == 'windspd':
                        var.attrs = {'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'}
                    elif var_name == 'windspd_u':
                        var.attrs = {'units': 'm s-1', 'long_name': 'eastward wind', 'standard_name': 'eastward_wind'}
                    elif var_name == 'windspd_v':
                        var.attrs = {'units': 'm s-1', 'long_name': 'northward wind', 'standard_name': 'northward_wind'}
                    elif var_name == 'LWRadAtm':
                        var.attrs = {'units': 'W m-2', 'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air'}
                    elif var_name == 'SWRadAtm':
                        var.attrs = {'units': 'W m-2', 'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'}
                    elif var_name == 'spechum':
                        var.attrs = {'units': 'kg kg-1', 'long_name': 'specific humidity', 'standard_name': 'specific_humidity'}

                    var.encoding['missing_value'] = -999.0
                    var.encoding['_FillValue'] = -999.0

                output_file = merged_forcing_path / f"CASR_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)
                self.logger.debug(f"Saved CASR monthly file: {output_file}")

        ds.close()
        self.logger.info("CASR forcing data merging from consolidated file completed")

    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create CASR grid shapefile.

        CASR uses a rotated pole grid similar to RDRS.

        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to merged CASR data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics

        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating CASR grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset, default='unknown')}.shp"

        try:
            # Find a CASR file to get grid information
            casr_files = list(merged_forcing_path.glob('*.nc'))
            if not casr_files:
                raise FileNotFoundError("No CASR files found")
            casr_file = casr_files[0]

            self.logger.info(f"Using CASR file for grid: {casr_file}")

            # Read CASR data - similar structure to RDRS
            with self.open_dataset(casr_file) as ds:
                rlat, rlon = ds.rlat.values, ds.rlon.values
                lat, lon = ds.lat.values, ds.lon.values

            self.logger.info(f"CASR dimensions: rlat={rlat.shape}, rlon={rlon.shape}")

            # Create grid cells
            geometries, ids, lats, lons = [], [], [], []

            batch_size = 100
            total_cells = len(rlat) * len(rlon)
            num_batches = (total_cells + batch_size - 1) // batch_size

            self.logger.info(f"Creating CASR grid cells in {num_batches} batches")

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
                        self.logger.info(f"Created {cell_count}/{total_cells} CASR grid cells")

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
            self.logger.info(f"CASR shapefile created and saved to {output_shapefile}")

            return output_shapefile

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error in create_casr_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
