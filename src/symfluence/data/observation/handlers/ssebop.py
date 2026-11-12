# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SSEBop Evapotranspiration Observation Handler.

Provides acquisition and preprocessing of USGS SSEBop (operational Simplified
Surface Energy Balance) evapotranspiration data for model validation.

SSEBop Overview:
    Data Type: Satellite-derived actual evapotranspiration (ETa)
    Resolution: 1km (CONUS), 10km (global)
    Coverage: CONUS daily, global monthly
    Variables: Actual ET
    Units: mm/day

Output Format:
    CSV with columns: datetime, et_mm_day
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('ssebop')
@R.observation_handlers.add('ssebop_et')
class SSEBopHandler(BaseObservationHandler):
    """
    Handles SSEBop ET data acquisition and processing.

    Provides basin-averaged daily/monthly evapotranspiration time series
    from USGS SSEBop product for model calibration and validation.
    """

    obs_type = "et"
    source_name = "USGS_SSEBOP"

    def acquire(self) -> Path:
        """
        Locate or download SSEBop data.

        Returns:
            Path to directory containing SSEBop files
        """
        data_access = self._get_config_value(lambda: self.config.domain.data_access, dict_key='DATA_ACCESS')
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        ssebop_path = self._get_config_value(
            lambda: None,
            default='default', dict_key='SSEBOP_PATH'
        )
        if isinstance(ssebop_path, str) and ssebop_path.lower() == 'default':
            ssebop_dir = self.project_observations_dir / "et" / "ssebop"
        else:
            ssebop_dir = Path(ssebop_path)

        ssebop_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        force_download = self._get_config_value(
            lambda: self.config.data.force_download,
            default=False
        )

        existing_files = list(ssebop_dir.glob("*.nc")) + list(ssebop_dir.glob("*.tif"))
        if existing_files and not force_download:
            self.logger.info(f"Using existing SSEBop data: {len(existing_files)} files")
            return ssebop_dir

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for SSEBop ET")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('SSEBOP', self.config, self.logger)
            return acquirer.download(ssebop_dir)

        return ssebop_dir

    def process(self, input_path: Path) -> Path:
        """
        Process SSEBop data to basin-averaged ET time series.

        Args:
            input_path: Path to directory containing SSEBop files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing SSEBop ET for domain: {self.domain_name}")

        # Find NetCDF files first, then GeoTIFFs
        nc_files = list(input_path.glob("*SSEBop*.nc"))
        if nc_files:
            return self._process_netcdf(nc_files, input_path)
        else:
            tif_files = list(input_path.glob("*.tif"))
            if tif_files:
                return self._process_geotiff(tif_files, input_path)
            else:
                self.logger.warning("No SSEBop files found")
                return input_path

    def _process_netcdf(self, nc_files: List[Path], input_path: Path) -> Path:
        """Process SSEBop NetCDF files."""
        self.logger.info(f"Processing {len(nc_files)} SSEBop NetCDF files")

        # Get bounding box
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
                # Find ET variable
                et_var = self._find_et_variable(ds)
                if et_var is None:
                    continue

                et = ds[et_var]

                # Identify dimensions
                lat_dim = self._find_dim(et, ['lat', 'latitude', 'y'])
                lon_dim = self._find_dim(et, ['lon', 'longitude', 'x'])

                # Spatial subsetting
                if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]) and lat_dim and lon_dim:
                    et = self._subset_spatial(et, lat_dim, lon_dim, lat_min, lat_max, lon_min, lon_max)

                # Temporal subsetting
                if 'time' in et.dims and self.start_date is not None and self.end_date is not None:
                    et = et.sel(time=slice(self.start_date, self.end_date))

                # Compute spatial average
                non_time_dims = [d for d in et.dims if d != 'time']
                if non_time_dims:
                    mean_et = et.mean(dim=non_time_dims, skipna=True)
                else:
                    mean_et = et

                # Convert to DataFrame
                df = mean_et.to_dataframe().reset_index()
                if et_var in df.columns:
                    df = df.rename(columns={et_var: 'et_mm_day'})

                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df[['datetime', 'et_mm_day']]

                results.append(df)

        return self._finalize_output(results, input_path)

    def _process_geotiff(self, tif_files: List[Path], input_path: Path) -> Path:
        """Process SSEBop GeoTIFF files."""
        import rasterio

        self.logger.info(f"Processing {len(tif_files)} SSEBop GeoTIFF files")

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        results = []

        for tif_file in sorted(tif_files):
            try:
                # Extract date from filename
                date = self._extract_date(tif_file.name)
                if date is None:
                    continue

                with rasterio.open(tif_file) as src:
                    data = src.read(1).astype(np.float32)
                    transform = src.transform
                    nodata = src.nodata if src.nodata is not None else -9999

                    # Create coordinate arrays
                    rows, cols = data.shape
                    lons = np.linspace(
                        transform.c,
                        transform.c + cols * transform.a,
                        cols
                    )
                    lats = np.linspace(
                        transform.f,
                        transform.f + rows * transform.e,
                        rows
                    )

                    # Find indices for bounding box
                    lon_mask = (lons >= lon_min) & (lons <= lon_max)
                    lat_mask = (lats >= lat_min) & (lats <= lat_max)

                    if not np.any(lon_mask) or not np.any(lat_mask):
                        continue

                    # Subset data
                    data_sub = data[np.ix_(lat_mask, lon_mask)]

                    # Handle nodata and invalid values
                    data_sub[data_sub == nodata] = np.nan
                    data_sub[data_sub < 0] = np.nan

                    # Scale if needed (CONUS is scaled by 10)
                    if 'modisSSEBop' in tif_file.name:
                        data_sub = data_sub / 10.0

                    # Compute mean
                    mean_et = np.nanmean(data_sub)

                    if not np.isnan(mean_et):
                        results.append({
                            'datetime': date,
                            'et_mm_day': mean_et
                        })

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed to process {tif_file.name}: {e}", exc_info=True)

        if not results:
            self.logger.warning("No SSEBop ET data could be extracted")
            return input_path

        df = pd.DataFrame(results)
        return self._save_output(df, input_path)

    def _find_et_variable(self, ds: xr.Dataset) -> Optional[str]:
        """Find the ET variable in the dataset."""
        candidates = ['et', 'ET', 'eta', 'ETa', 'evapotranspiration']

        for var in candidates:
            if var in ds.data_vars:
                return var

        for var in ds.data_vars:
            if 'et' in var.lower() or 'evap' in var.lower():
                return var

        return None

    def _find_dim(self, da: xr.DataArray, candidates: List[str]) -> Optional[str]:
        """Find dimension matching candidates."""
        for dim in da.dims:
            dim_str = str(dim)
            if dim_str.lower() in [c.lower() for c in candidates]:
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
        if da[lat_dim][0] > da[lat_dim][-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        return da.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})

    def _extract_date(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from SSEBop filename."""
        import re
        from datetime import datetime, timedelta

        # CONUS: det2020001.modisSSEBopETv4.tif
        match = re.search(r'det(\d{7})\.', filename)
        if match:
            year = int(match.group(1)[:4])
            doy = int(match.group(1)[4:])
            return pd.Timestamp(datetime(year, 1, 1) + timedelta(days=doy - 1))

        # Global: m202001eta.tif
        match = re.search(r'm(\d{6})eta\.', filename)
        if match:
            year = int(match.group(1)[:4])
            month = int(match.group(1)[4:])
            return pd.Timestamp(datetime(year, month, 1))

        return None

    def _finalize_output(self, results: List[pd.DataFrame], input_path: Path) -> Path:
        """Finalize and save output from DataFrame list."""
        if not results:
            self.logger.warning("No SSEBop ET data could be extracted")
            return input_path

        df = pd.concat(results, ignore_index=True)
        return self._save_output(df, input_path)

    def _save_output(self, df: pd.DataFrame, input_path: Path) -> Path:
        """Save processed DataFrame to CSV."""
        if 'datetime' in df.columns:
            df = df.groupby('datetime').mean().reset_index()
            df = df.sort_values('datetime')

            if self.start_date is not None and self.end_date is not None:
                mask = (df['datetime'] >= self.start_date) & (df['datetime'] <= self.end_date)
                df = df[mask]

        # Ensure non-negative
        if 'et_mm_day' in df.columns:
            df['et_mm_day'] = df['et_mm_day'].clip(lower=0)

        # Save output
        output_dir = self._get_observation_dir('et')
        output_file = output_dir / f"{self.domain_name}_ssebop_et_processed.csv"
        df.to_csv(output_file, index=False)

        # Also save to product-specific location
        product_dir = self.project_observations_dir / "et" / "ssebop" / "processed"
        product_dir.mkdir(parents=True, exist_ok=True)
        product_file = product_dir / f"{self.domain_name}_ssebop_et_processed.csv"
        df.to_csv(product_file, index=False)

        self.logger.info(f"SSEBop processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df)}")
        if 'et_mm_day' in df.columns and len(df) > 0:
            self.logger.info(f"  Mean ET: {df['et_mm_day'].mean():.2f} mm/day")

        return output_file
