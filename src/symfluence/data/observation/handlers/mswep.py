# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MSWEP Precipitation Observation Handler

Processes Multi-Source Weighted-Ensemble Precipitation (MSWEP) data for
use in hydrological modeling validation and calibration. MSWEP provides
high-quality merged precipitation estimates combining gauge, satellite,
and reanalysis data.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('mswep')
class MSWEPHandler(BaseObservationHandler):
    """
    Handles MSWEP precipitation data processing.

    Processes gridded MSWEP precipitation to basin-averaged time series
    suitable for comparison with model precipitation inputs or outputs.

    Configuration:
        MSWEP_DIR: Directory containing MSWEP data
        MSWEP_AGGREGATE: Temporal aggregation ('3hourly', 'daily', 'monthly')
    """

    obs_type = "precipitation"
    source_name = "GLOH2O_MSWEP"

    def acquire(self) -> Path:
        """Acquire MSWEP data via cloud acquisition."""
        mswep_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "precipitation" / "mswep", dict_key='MSWEP_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = mswep_dir.exists() and any(mswep_dir.glob("*.nc"))

        if not has_files or force_download:
            self.logger.info("Acquiring MSWEP precipitation data...")
            try:
                from ...acquisition.handlers.mswep import MSWEPAcquirer
                acquirer = MSWEPAcquirer(self.config, self.logger)
                acquirer.download(mswep_dir)
            except ImportError as e:
                self.logger.warning(f"MSWEP acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"MSWEP acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing MSWEP data in {mswep_dir}")

        return mswep_dir

    def process(self, input_path: Path) -> Path:
        """
        Process MSWEP precipitation data for the current domain.

        Args:
            input_path: Path to MSWEP data directory

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing MSWEP precipitation for domain: {self.domain_name}")

        # Find MSWEP files
        if input_path.is_file():
            nc_files = [input_path]
        else:
            nc_files = sorted(input_path.glob("mswep*.nc"))
            if not nc_files:
                nc_files = sorted(input_path.glob("*.nc"))

        if not nc_files:
            self.logger.error("No MSWEP NetCDF files found")
            return input_path

        # Load catchment shapefile
        basin_gdf = self._load_catchment_shapefile()

        # Process files
        all_data = []
        for nc_file in nc_files:
            try:
                precip = self._extract_basin_precip(nc_file, basin_gdf)
                if precip is not None:
                    all_data.append(precip)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {nc_file.name}: {e}", exc_info=True)

        if not all_data:
            self.logger.warning("No MSWEP data could be processed")
            return input_path

        # Combine all data
        df = pd.concat(all_data)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Aggregate if needed
        aggregate = self._get_config_value(lambda: None, default='daily', dict_key='MSWEP_AGGREGATE')
        if aggregate == 'daily':
            df = df.resample('D').sum()
        elif aggregate == 'monthly':
            df = df.resample('MS').sum()

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "precipitation" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_mswep_precip_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"MSWEP processing complete: {output_file}")

        return output_file

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """Load the catchment shapefile for spatial masking.

        Delegates to the shared resolver, which handles the nested
        discretization layout and river_basins fallback. Returns None when
        no basin is found (callers fall back to the bounding box).
        """
        return self._load_catchment_gdf()

    def _extract_basin_precip(
        self,
        nc_file: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ) -> Optional[pd.DataFrame]:
        """Extract basin-averaged precipitation from NetCDF file."""
        try:
            ds = xr.open_dataset(nc_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to open {nc_file}: {e}", exc_info=True)
            return None

        # Find precipitation variable
        precip_var = None
        for var in ['precipitation', 'precip', 'pr', 'P', 'tp']:
            if var in ds.data_vars:
                precip_var = var
                break

        if precip_var is None:
            # Use first data variable
            precip_var = list(ds.data_vars)[0]

        da = ds[precip_var]

        # Find coordinate names
        lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
        lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])
        time_name = self._find_coord(da, ['time', 'valid_time', 't'])

        if not lat_name or not lon_name:
            self.logger.warning(f"Cannot identify coordinates in {nc_file}")
            ds.close()
            return None

        # Spatial subsetting
        if basin_gdf is not None:
            bounds = basin_gdf.total_bounds
            da = self._subset_to_bounds(da, bounds, lat_name, lon_name)
        elif self.bbox:
            bounds = [
                self.bbox['lon_min'], self.bbox['lat_min'],
                self.bbox['lon_max'], self.bbox['lat_max']
            ]
            da = self._subset_to_bounds(da, bounds, lat_name, lon_name)

        # Compute spatial mean
        spatial_dims = [d for d in da.dims if d != time_name]
        precip_mean = da.mean(dim=spatial_dims, skipna=True)

        # Convert to DataFrame
        if time_name and time_name in precip_mean.dims:
            df = precip_mean.to_dataframe(name='precip_mm').reset_index()
            df = df.set_index(time_name)
            df.index.name = 'datetime'
            df = df[['precip_mm']]
        else:
            # Single timestep - extract from filename
            date = self._extract_date_from_filename(nc_file.name)
            if date:
                df = pd.DataFrame(
                    {'precip_mm': [float(precip_mean.values)]},
                    index=pd.DatetimeIndex([date], name='datetime')
                )
            else:
                ds.close()
                return None

        ds.close()
        return df

    def _find_coord(self, da: xr.DataArray, candidates: List[str]) -> Optional[str]:
        """Find coordinate name from candidates."""
        for name in candidates:
            if name in da.coords or name in da.dims:
                return name
        return None

    def _subset_to_bounds(
        self,
        da: xr.DataArray,
        bounds: List[float],
        lat_name: str,
        lon_name: str
    ) -> xr.DataArray:
        """Subset DataArray to bounding box."""
        lon_min, lat_min, lon_max, lat_max = bounds

        # Handle longitude conventions
        lon_vals = da[lon_name].values
        if lon_vals.max() > 180 and lon_min < 0:
            lon_min = lon_min % 360
            lon_max = lon_max % 360

        # Handle inverted latitude
        lat_vals = da[lat_name].values
        if lat_vals[0] > lat_vals[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        return da.sel({lon_name: slice(lon_min, lon_max), lat_name: lat_slice})

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from MSWEP filename."""
        import re

        # Pattern: mswep_YYYYDDD.nc or mswep_YYYYDDDHH.nc
        match = re.search(r'mswep_(\d{4})(\d{3})(\d{2})?', filename)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            hour = int(match.group(3)) if match.group(3) else 0
            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1, hours=hour)
            return date

        # Pattern: mswep_YYYYMM.nc (monthly)
        match = re.search(r'mswep_(\d{4})(\d{2})\.nc', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return pd.Timestamp(year=year, month=month, day=1)

        return None

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed MSWEP precipitation data."""
        processed_path = (
            self.project_observations_dir / "precipitation" / "preprocessed"
            / f"{self.domain_name}_mswep_precip_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading MSWEP data: {e}", exc_info=True)
            return None
