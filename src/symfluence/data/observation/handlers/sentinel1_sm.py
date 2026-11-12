# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Sentinel-1 SAR Soil Moisture Observation Handler

Processes Sentinel-1 SAR-derived soil moisture data for hydrological
modeling. Sentinel-1 provides high-resolution (~1 km) soil moisture
estimates that complement coarser passive microwave products.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('sentinel1_sm')
@R.observation_handlers.add('s1_sm')
class Sentinel1SMHandler(BaseObservationHandler):
    """
    Handles Sentinel-1 SAR soil moisture data processing.

    Processes Sentinel-1 backscatter or derived soil moisture products
    to basin-averaged time series.

    Configuration:
        SENTINEL1_SM_DIR: Directory containing Sentinel-1 data
        SENTINEL1_SM_METHOD: Retrieval method ('change_detection', 'model')
        SENTINEL1_POLARIZATION: Polarization to use ('VV', 'VH')
    """

    obs_type = "soil_moisture"
    source_name = "Sentinel-1"

    def acquire(self) -> Path:
        """Acquire Sentinel-1 SM data via cloud acquisition."""
        s1_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "soil_moisture" / "sentinel1", dict_key='SENTINEL1_SM_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = s1_dir.exists() and (
            any(s1_dir.glob("*.zip")) or
            any(s1_dir.glob("*.nc")) or
            any(s1_dir.glob("*.tif"))
        )

        if not has_files or force_download:
            self.logger.info("Acquiring Sentinel-1 SM data...")
            try:
                from ...acquisition.handlers.sentinel1_sm import Sentinel1SMAcquirer
                acquirer = Sentinel1SMAcquirer(self.config, self.logger)
                acquirer.download(s1_dir)
            except ImportError as e:
                self.logger.warning(f"Sentinel-1 acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"Sentinel-1 acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing Sentinel-1 data in {s1_dir}")

        return s1_dir

    def process(self, input_path: Path) -> Path:
        """
        Process Sentinel-1 data for the current domain.

        Args:
            input_path: Path to Sentinel-1 data directory

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing Sentinel-1 SM for domain: {self.domain_name}")

        # Find data files
        nc_files = list(input_path.glob("*.nc"))
        tif_files = list(input_path.glob("*.tif"))
        zip_files = list(input_path.glob("*.zip"))

        if not nc_files and not tif_files and not zip_files:
            self.logger.error("No Sentinel-1 data files found")
            return input_path

        # Load catchment shapefile
        basin_gdf = self._load_catchment_shapefile()

        # Process files
        results: dict[str, list] = {'datetime': [], 'soil_moisture': [], 'backscatter_vv': []}

        # Process NetCDF files (preprocessed SM)
        for nc_file in nc_files:
            try:
                data = self._process_netcdf(nc_file, basin_gdf)
                if data:
                    results['datetime'].extend(data['datetime'])
                    results['soil_moisture'].extend(data['soil_moisture'])
                    results['backscatter_vv'].extend(data.get('backscatter_vv', [np.nan] * len(data['datetime'])))
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {nc_file.name}: {e}", exc_info=True)

        # Process ZIP files (raw S1 products)
        for zip_file in zip_files:
            try:
                data = self._process_zip(zip_file, basin_gdf)
                if data:
                    results['datetime'].extend(data['datetime'])
                    results['soil_moisture'].extend(data.get('soil_moisture', [np.nan] * len(data['datetime'])))
                    results['backscatter_vv'].extend(data['backscatter_vv'])
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {zip_file.name}: {e}", exc_info=True)

        if not results['datetime']:
            self.logger.warning("No Sentinel-1 data could be processed")
            return input_path

        # Create DataFrame
        df = pd.DataFrame(results)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "soil_moisture" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_sentinel1_sm_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"Sentinel-1 SM processing complete: {output_file}")

        return output_file

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """Load the catchment shapefile for spatial masking.

        Delegates to the shared resolver, which handles the nested
        discretization layout and river_basins fallback. Returns None when
        no basin is found (callers fall back to the bounding box).
        """
        return self._load_catchment_gdf()

    def _process_netcdf(
        self,
        nc_file: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ) -> Optional[dict]:
        """Process NetCDF file containing processed SM data."""
        ds = xr.open_dataset(nc_file)

        # Find soil moisture variable
        sm_var = self._find_variable(ds, ['soil_moisture', 'sm', 'SSM', 'SWI'])
        if not sm_var:
            ds.close()
            return None

        time_dim = self._find_coord(ds, ['time', 'date'])
        lat_name = self._find_coord(ds, ['lat', 'latitude', 'y'])
        lon_name = self._find_coord(ds, ['lon', 'longitude', 'x'])

        results: dict[str, list] = {'datetime': [], 'soil_moisture': []}

        if time_dim:
            time_vals = pd.to_datetime(ds[time_dim].values)
        else:
            time_vals = [self._extract_date_from_filename(nc_file.name)]

        for i, t in enumerate(time_vals):
            if time_dim:
                da = ds[sm_var].isel({time_dim: i})
            else:
                da = ds[sm_var]

            # Extract basin mean
            sm_val = self._extract_basin_mean(da, basin_gdf, lat_name, lon_name)

            results['datetime'].append(t)
            results['soil_moisture'].append(sm_val)

        ds.close()
        return results

    def _process_zip(
        self,
        zip_file: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ) -> Optional[dict]:
        """Process raw Sentinel-1 ZIP file."""
        # Extract date from filename
        date = self._extract_date_from_filename(zip_file.name)
        if not date:
            return None

        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Find measurement files
                measurement_files = [
                    f for f in zf.namelist()
                    if 'measurement' in f and f.endswith('.tiff')
                ]

                if not measurement_files:
                    return None

                # Find VV polarization file (preferred) or use first available
                vv_file = next(
                    (f for f in measurement_files if 'vv' in f.lower()),
                    measurement_files[0]
                )

                # Raw Sentinel-1 SAR processing requires radiometric calibration
                # and terrain correction which is beyond the scope of this handler.
                # Use preprocessed NetCDF or GeoTIFF products instead.
                self.logger.debug(
                    f"Raw ZIP processing not implemented for {vv_file}. "
                    "Use preprocessed NetCDF/GeoTIFF products for Sentinel-1 SM."
                )
                backscatter_vv = np.nan

                return {
                    'datetime': [date],
                    'backscatter_vv': [backscatter_vv],
                }

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to process ZIP {zip_file}: {e}", exc_info=True)
            return None

    def _find_variable(self, ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
        """Find variable name from candidates."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        return None

    def _find_coord(self, ds, candidates: List[str]) -> Optional[str]:
        """Find coordinate name from candidates."""
        for name in candidates:
            if name in ds.coords or name in ds.dims:
                return name
        return None

    def _extract_basin_mean(
        self,
        da: xr.DataArray,
        basin_gdf: Optional[gpd.GeoDataFrame],
        lat_name: Optional[str],
        lon_name: Optional[str]
    ) -> float:
        """Extract basin-averaged value."""
        if basin_gdf is not None and lat_name and lon_name:
            bounds = basin_gdf.total_bounds
            try:
                lat_slice = slice(bounds[1], bounds[3])
                if da[lat_name].values[0] > da[lat_name].values[-1]:
                    lat_slice = slice(bounds[3], bounds[1])

                da = da.sel({
                    lon_name: slice(bounds[0], bounds[2]),
                    lat_name: lat_slice
                })
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Could not subset by basin bounds: {e}")
        elif self.bbox and lat_name and lon_name:
            try:
                lat_slice = slice(self.bbox['lat_min'], self.bbox['lat_max'])
                if da[lat_name].values[0] > da[lat_name].values[-1]:
                    lat_slice = slice(self.bbox['lat_max'], self.bbox['lat_min'])

                da = da.sel({
                    lon_name: slice(self.bbox['lon_min'], self.bbox['lon_max']),
                    lat_name: lat_slice
                })
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Could not subset by bounding box: {e}")

        return float(da.mean(skipna=True).values)

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from Sentinel-1 filename."""
        import re

        # Pattern: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_...
        match = re.search(r'(\d{8})T(\d{6})', filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            return pd.Timestamp(
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
                f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            )

        return None

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed Sentinel-1 SM data."""
        processed_path = (
            self.project_observations_dir / "soil_moisture" / "preprocessed"
            / f"{self.domain_name}_sentinel1_sm_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading Sentinel-1 data: {e}", exc_info=True)
            return None
