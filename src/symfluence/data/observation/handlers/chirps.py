# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CHIRPS Precipitation Observation Handler.

Provides acquisition and preprocessing of CHIRPS (Climate Hazards Group InfraRed
Precipitation with Station data) for hydrological model forcing validation.

CHIRPS Overview:
    Data Type: Quasi-global rainfall estimates (satellite + station blended)
    Resolution: 0.05° x 0.05° (~5km)
    Coverage: 50°S to 50°N, global
    Temporal: Daily, pentadal, monthly
    Units: mm/day (daily), mm/pentad, mm/month

Output Format:
    CSV with columns: datetime, precipitation_mm
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('chirps')
class CHIRPSHandler(BaseObservationHandler):
    """
    Handles CHIRPS precipitation data acquisition and processing.

    Provides basin-averaged precipitation time series from CHIRPS
    quasi-global dataset for model forcing comparison or independent validation.
    """

    obs_type = "precipitation"
    source_name = "CHIRPS"

    def acquire(self) -> Path:
        """
        Locate or download CHIRPS data.

        Returns:
            Path to directory containing CHIRPS NetCDF files
        """
        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS')
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        chirps_path = self._get_config_value(
            lambda: None,
            default='default', dict_key='CHIRPS_PATH'
        )
        if isinstance(chirps_path, str) and chirps_path.lower() == 'default':
            chirps_dir = self.project_observations_dir / "precipitation" / "chirps"
        else:
            chirps_dir = Path(chirps_path)

        chirps_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        existing_files = list(chirps_dir.glob("*.nc"))
        if existing_files and not force_download:
            self.logger.info(f"Using existing CHIRPS data: {len(existing_files)} files")
            return chirps_dir

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for CHIRPS precipitation")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('CHIRPS', self.config, self.logger)
            return acquirer.download(chirps_dir)

        return chirps_dir

    def process(self, input_path: Path) -> Path:
        """
        Process CHIRPS NetCDF data to daily precipitation time series.

        Args:
            input_path: Path to directory containing CHIRPS files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing CHIRPS precipitation for domain: {self.domain_name}")

        # Find NetCDF files
        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No CHIRPS NetCDF files found")
            return input_path

        self.logger.info(f"Processing {len(nc_files)} CHIRPS files")

        # Get bounding box for spatial averaging
        lat_min = lat_max = lon_min = lon_max = None
        if self.bbox:
            lat_min = self.bbox.get('lat_min')
            lat_max = self.bbox.get('lat_max')
            lon_min = self.bbox.get('lon_min')
            lon_max = self.bbox.get('lon_max')

        results: List[pd.DataFrame] = []

        for nc_file in sorted(nc_files):
            try:
                ds = self._open_dataset(nc_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to open {nc_file.name}: {e}", exc_info=True)
                continue

            with ds:
                # Find precipitation variable
                precip_var = self._find_precip_variable(ds)
                if precip_var is None:
                    self.logger.warning(f"No precipitation variable found in {nc_file.name}")
                    continue

                precip = ds[precip_var]

                # Identify dimension names
                lat_dim = self._find_lat_dim(precip)
                lon_dim = self._find_lon_dim(precip)

                # Spatial subsetting if bbox available
                if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]) and lat_dim and lon_dim:
                    precip = self._subset_spatial(precip, lat_dim, lon_dim, lat_min, lat_max, lon_min, lon_max)

                # Temporal subsetting
                if 'time' in precip.dims and self.start_date is not None and self.end_date is not None:
                    precip = precip.sel(time=slice(self.start_date, self.end_date))

                # Compute spatial average
                non_time_dims = [d for d in precip.dims if d != 'time']
                if non_time_dims:
                    mean_precip = precip.mean(dim=non_time_dims)
                else:
                    mean_precip = precip

                # Convert to DataFrame
                df = mean_precip.to_dataframe().reset_index()

                # Standardize column names
                if precip_var in df.columns:
                    df = df.rename(columns={precip_var: 'precipitation_mm'})
                elif 'precip' in df.columns:
                    df = df.rename(columns={'precip': 'precipitation_mm'})

                # Handle time column
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df[['datetime', 'precipitation_mm']]

                results.append(df)

        if not results:
            self.logger.warning("No CHIRPS precipitation data could be extracted")
            return input_path

        # Combine all results
        df_combined = pd.concat(results, ignore_index=True)

        # Handle any duplicate times by averaging
        if 'datetime' in df_combined.columns:
            df_combined = df_combined.groupby('datetime').mean().reset_index()
            df_combined = df_combined.sort_values('datetime')

            # Filter to experiment time range
            if self.start_date is not None and self.end_date is not None:
                mask = (df_combined['datetime'] >= self.start_date) & \
                       (df_combined['datetime'] <= self.end_date)
                df_combined = df_combined[mask]

        # Ensure precipitation is non-negative (CHIRPS uses -9999 as fill value)
        if 'precipitation_mm' in df_combined.columns:
            df_combined['precipitation_mm'] = df_combined['precipitation_mm'].where(
                df_combined['precipitation_mm'] >= 0, np.nan
            )
            df_combined = df_combined.dropna(subset=['precipitation_mm'])

        # Save output
        output_dir = self._get_observation_dir('precipitation')
        output_file = output_dir / f"{self.domain_name}_chirps_processed.csv"
        df_combined.to_csv(output_file, index=False)

        # Also save to product-specific location
        product_dir = self.project_observations_dir / "precipitation" / "chirps" / "processed"
        product_dir.mkdir(parents=True, exist_ok=True)
        product_file = product_dir / f"{self.domain_name}_chirps_processed.csv"
        df_combined.to_csv(product_file, index=False)

        self.logger.info(f"CHIRPS processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df_combined)}")
        if 'precipitation_mm' in df_combined.columns and len(df_combined) > 0:
            self.logger.info(f"  Mean precipitation: {df_combined['precipitation_mm'].mean():.2f} mm/day")

        return output_file

    def _find_precip_variable(self, ds: xr.Dataset) -> Optional[str]:
        """Find the precipitation variable in the dataset."""
        # CHIRPS variable names
        candidates = ['precip', 'precipitation', 'pr', 'prcp', 'ppt']

        for var in candidates:
            if var in ds.data_vars:
                return var

        # Fallback: find any variable with 'precip' in name
        for var in ds.data_vars:
            if 'precip' in var.lower() or 'rain' in var.lower():
                return var

        return None

    def _find_lat_dim(self, da: xr.DataArray) -> Optional[str]:
        """Find latitude dimension name."""
        for dim in da.dims:
            dim_str = str(dim)
            if dim_str.lower() in ['lat', 'latitude', 'y']:
                return dim_str
        return None

    def _find_lon_dim(self, da: xr.DataArray) -> Optional[str]:
        """Find longitude dimension name."""
        for dim in da.dims:
            dim_str = str(dim)
            if dim_str.lower() in ['lon', 'longitude', 'x']:
                return dim_str
        return None

    def _subset_spatial(
        self,
        da: xr.DataArray,
        lat_dim: str,
        lon_dim: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> xr.DataArray:
        """Subset data array to bounding box."""
        # Check if lat is descending
        if da[lat_dim][0] > da[lat_dim][-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        # Handle longitude wrapping
        if lon_min <= lon_max:
            lon_slice = slice(lon_min, lon_max)
            return da.sel({lat_dim: lat_slice, lon_dim: lon_slice})
        else:
            # Longitude crosses dateline
            da_west = da.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, 180)})
            da_east = da.sel({lat_dim: lat_slice, lon_dim: slice(-180, lon_max)})
            return xr.concat([da_west, da_east], dim=lon_dim)
