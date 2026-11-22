# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GPM IMERG Precipitation Observation Handler.

Provides acquisition and preprocessing of NASA GPM IMERG satellite precipitation
data for hydrological model forcing validation and calibration.

GPM IMERG Overview:
    Data Type: Satellite-derived precipitation
    Resolution: 0.1° x 0.1° (~10km)
    Coverage: Global (60°N to 60°S)
    Temporal: Daily (aggregated from half-hourly)
    Units: mm/day

Output Format:
    CSV with columns: datetime, precipitation_mm
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('gpm_imerg')
@R.observation_handlers.add('gpm')
class GPMIMERGHandler(BaseObservationHandler):
    """
    Handles GPM IMERG precipitation data acquisition and processing.

    Provides basin-averaged daily precipitation time series from NASA's
    GPM IMERG satellite product for comparison with model forcing or
    independent validation.
    """

    obs_type = "precipitation"
    source_name = "NASA_GPM_IMERG"

    def acquire(self) -> Path:
        """
        Locate or download GPM IMERG data.

        Returns:
            Path to directory containing GPM IMERG NetCDF files
        """
        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS')
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        gpm_path = self._get_config_value(
            lambda: None,
            default='default', dict_key='GPM_PATH'
        )
        if isinstance(gpm_path, str) and gpm_path.lower() == 'default':
            gpm_dir = self.project_observations_dir / "precipitation" / "gpm_imerg"
        else:
            gpm_dir = Path(gpm_path)

        gpm_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        existing_files = list(gpm_dir.glob("*.nc")) + list(gpm_dir.glob("*.nc4"))
        if existing_files and not force_download:
            self.logger.info(f"Using existing GPM IMERG data: {len(existing_files)} files")
            return gpm_dir

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for GPM IMERG precipitation")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('GPM_IMERG', self.config, self.logger)
            return acquirer.download(gpm_dir)

        return gpm_dir

    def process(self, input_path: Path) -> Path:
        """
        Process GPM IMERG NetCDF data to daily precipitation time series.

        Args:
            input_path: Path to directory containing GPM IMERG files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing GPM IMERG precipitation for domain: {self.domain_name}")

        # Find NetCDF files
        nc_files = list(input_path.glob("*.nc")) + list(input_path.glob("*.nc4"))
        if not nc_files:
            self.logger.warning("No GPM IMERG NetCDF files found")
            return input_path

        self.logger.info(f"Processing {len(nc_files)} GPM IMERG files")

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

                # Spatial subsetting if bbox available
                if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
                    precip = self._subset_spatial(precip, lat_min, lat_max, lon_min, lon_max)

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
                elif 'precipitation' in df.columns:
                    df = df.rename(columns={'precipitation': 'precipitation_mm'})

                # Handle time column
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df[['datetime', 'precipitation_mm']]
                else:
                    # Try to extract date from filename
                    date = self._extract_date_from_filename(nc_file.name)
                    if date:
                        df['datetime'] = date
                        if 'precipitation_mm' in df.columns:
                            df = df[['datetime', 'precipitation_mm']]

                results.append(df)

        if not results:
            self.logger.warning("No GPM IMERG precipitation data could be extracted")
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

        # Ensure precipitation is non-negative
        if 'precipitation_mm' in df_combined.columns:
            df_combined['precipitation_mm'] = df_combined['precipitation_mm'].clip(lower=0)

        # Save output
        output_dir = self._get_observation_dir('precipitation')
        output_file = output_dir / f"{self.domain_name}_gpm_imerg_processed.csv"
        df_combined.to_csv(output_file, index=False)

        # Also save to legacy location
        legacy_dir = self.project_observations_dir / "precipitation" / "gpm_imerg" / "processed"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_file = legacy_dir / f"{self.domain_name}_gpm_imerg_processed.csv"
        df_combined.to_csv(legacy_file, index=False)

        self.logger.info(f"GPM IMERG processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df_combined)}")
        if 'precipitation_mm' in df_combined.columns:
            self.logger.info(f"  Mean precipitation: {df_combined['precipitation_mm'].mean():.2f} mm/day")

        return output_file

    def _find_precip_variable(self, ds: xr.Dataset) -> Optional[str]:
        """Find the precipitation variable in the dataset."""
        # Priority order for GPM IMERG variables
        candidates = [
            'precipitation',
            'precipitationCal',
            'precipitationUncal',
            'HQprecipitation',
            'IRprecipitation',
            'precip',
        ]

        for var in candidates:
            if var in ds.data_vars:
                return var

        # Fallback: find any variable with 'precip' in name
        for var in ds.data_vars:
            if 'precip' in var.lower():
                return var

        return None

    def _subset_spatial(
        self,
        da: xr.DataArray,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> xr.DataArray:
        """Subset data array to bounding box."""
        # Identify lat/lon dimension names
        lat_dim = None
        lon_dim = None
        for dim in da.dims:
            if dim.lower() in ['lat', 'latitude']:
                lat_dim = dim
            elif dim.lower() in ['lon', 'longitude']:
                lon_dim = dim

        if lat_dim is None or lon_dim is None:
            return da

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

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from GPM IMERG filename."""
        import re

        # GPM naming: 3B-DAY.MS.MRG.3IMERG.YYYYMMDD-...
        match = re.search(r'(\d{8})', filename)
        if match:
            try:
                return pd.to_datetime(match.group(1), format='%Y%m%d')
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Could not parse date from '{match.group(1)}' in filename '{filename}': {e}", exc_info=True)

        return None
