# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SNODAS Snow Observation Handler.

Provides acquisition and preprocessing of NOAA SNODAS (Snow Data Assimilation System)
snow data for hydrological model calibration and validation.

SNODAS Overview:
    Data Type: Assimilated snow analysis (satellite + ground obs)
    Resolution: ~1km (30 arc-second)
    Coverage: CONUS and southern Canada
    Variables: SWE, snow depth, snowmelt runoff, sublimation
    Temporal: Daily
    Units: meters

Output Format:
    CSV with columns: datetime, swe_m (or snow_depth_m, etc.)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('snodas')
@R.observation_handlers.add('snodas_swe')
class SNODASHandler(BaseObservationHandler):
    """
    Handles SNODAS snow data acquisition and processing.

    Provides basin-averaged daily snow water equivalent or snow depth
    time series from NOAA's SNODAS product for model calibration
    and validation.
    """

    obs_type = "snow"
    source_name = "NOAA_SNODAS"

    def acquire(self) -> Path:
        """
        Locate or download SNODAS data.

        Returns:
            Path to directory containing SNODAS files
        """
        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS')
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        snodas_path = self._get_config_value(
            lambda: None,
            default='default', dict_key='SNODAS_PATH'
        )
        if isinstance(snodas_path, str) and snodas_path.lower() == 'default':
            snodas_dir = self.project_observations_dir / "snow" / "snodas"
        else:
            snodas_dir = Path(snodas_path)

        snodas_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        force_download = self._get_config_value(
            lambda: self.config.data.force_download,
            default=False
        )

        existing_files = list(snodas_dir.glob("*.nc")) + list(snodas_dir.glob("*.tar"))
        if existing_files and not force_download:
            self.logger.info(f"Using existing SNODAS data: {len(existing_files)} files")
            return snodas_dir

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for SNODAS snow data")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('SNODAS', self.config, self.logger)
            return acquirer.download(snodas_dir)

        return snodas_dir

    def process(self, input_path: Path) -> Path:
        """
        Process SNODAS data to daily basin-averaged time series.

        Args:
            input_path: Path to directory containing SNODAS files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing SNODAS snow data for domain: {self.domain_name}")

        # Get variable from config
        variable = self._get_config_value(
            lambda: None,
            default='swe', dict_key='SNODAS_VARIABLE'
        )
        if isinstance(variable, str):
            variable = variable.lower()

        # Find NetCDF files (prioritize merged file)
        nc_files = list(input_path.glob(f"*SNODAS*{variable}*.nc"))
        if not nc_files:
            nc_files = list(input_path.glob("*SNODAS*.nc"))
        if not nc_files:
            self.logger.warning("No SNODAS NetCDF files found")
            return input_path

        self.logger.info(f"Processing {len(nc_files)} SNODAS files for variable: {variable}")

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
                # Find snow variable
                snow_var = self._find_snow_variable(ds, variable)
                if snow_var is None:
                    self.logger.warning(f"No {variable} variable found in {nc_file.name}")
                    continue

                snow = ds[snow_var]

                # Identify dimension names
                lat_dim = self._find_lat_dim(snow)
                lon_dim = self._find_lon_dim(snow)

                # Spatial subsetting if bbox available
                if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]) and lat_dim and lon_dim:
                    snow = self._subset_spatial(snow, lat_dim, lon_dim, lat_min, lat_max, lon_min, lon_max)

                # Temporal subsetting
                if 'time' in snow.dims and self.start_date is not None and self.end_date is not None:
                    snow = snow.sel(time=slice(self.start_date, self.end_date))

                # Compute spatial average (excluding NaN)
                non_time_dims = [d for d in snow.dims if d != 'time']
                if non_time_dims:
                    mean_snow = snow.mean(dim=non_time_dims, skipna=True)
                else:
                    mean_snow = snow

                # Convert to DataFrame
                df = mean_snow.to_dataframe().reset_index()

                # Standardize column names
                output_col = f"{variable}_m"
                if snow_var in df.columns:
                    df = df.rename(columns={snow_var: output_col})

                # Handle time column
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df[['datetime', output_col]]

                results.append(df)

        if not results:
            self.logger.warning("No SNODAS snow data could be extracted")
            return input_path

        # Combine all results
        df_combined = pd.concat(results, ignore_index=True)

        # Handle any duplicate times by averaging
        if 'datetime' in df_combined.columns:
            output_col = f"{variable}_m"
            df_combined = df_combined.groupby('datetime').mean().reset_index()
            df_combined = df_combined.sort_values('datetime')

            # Filter to experiment time range
            if self.start_date is not None and self.end_date is not None:
                mask = (df_combined['datetime'] >= self.start_date) & \
                       (df_combined['datetime'] <= self.end_date)
                df_combined = df_combined[mask]

        # Ensure values are non-negative
        output_col = f"{variable}_m"
        if output_col in df_combined.columns:
            df_combined[output_col] = df_combined[output_col].clip(lower=0)

        # Also convert to mm for compatibility with other snow products
        mm_col = f"{variable}_mm"
        if output_col in df_combined.columns:
            df_combined[mm_col] = df_combined[output_col] * 1000

        # Save output
        output_dir = self._get_observation_dir('snow')
        output_file = output_dir / f"{self.domain_name}_snodas_{variable}_processed.csv"
        df_combined.to_csv(output_file, index=False)

        # Also save to product-specific location
        product_dir = self.project_observations_dir / "snow" / "snodas" / "processed"
        product_dir.mkdir(parents=True, exist_ok=True)
        product_file = product_dir / f"{self.domain_name}_snodas_{variable}_processed.csv"
        df_combined.to_csv(product_file, index=False)

        self.logger.info(f"SNODAS processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df_combined)}")
        if output_col in df_combined.columns and len(df_combined) > 0:
            self.logger.info(f"  Mean {variable}: {df_combined[output_col].mean():.4f} m")

        return output_file

    def _find_snow_variable(self, ds: xr.Dataset, target_var: str) -> Optional[str]:
        """Find the snow variable in the dataset."""
        # Direct matches
        if target_var in ds.data_vars:
            return target_var

        # Common variations
        variations = {
            'swe': ['swe', 'SWE', 'snow_water_equivalent', 'SnowWaterEquivalent'],
            'snow_depth': ['snow_depth', 'snowDepth', 'SnowDepth', 'depth'],
            'snowmelt_runoff': ['snowmelt_runoff', 'snowmelt', 'melt', 'runoff'],
            'sublimation': ['sublimation', 'sublim'],
        }

        for var in variations.get(target_var, []):
            if var in ds.data_vars:
                return var

        # Fallback: find any snow-related variable
        for var in ds.data_vars:
            if 'snow' in var.lower() or 'swe' in var.lower():
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

        return da.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})
