# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Daymet Climate Observation Handler

Processes Daymet daily surface weather data for use in hydrological
modeling. Daymet provides high-resolution (1km) gridded climate data
across North America.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('daymet')
class DaymetHandler(BaseObservationHandler):
    """
    Handles Daymet climate data processing.

    Processes Daymet gridded climate data to basin-averaged time series
    for temperature, precipitation, and other variables.

    Configuration:
        DAYMET_DIR: Directory containing Daymet data
        DAYMET_VARIABLES: Variables to process
        DAYMET_AGGREGATE: Temporal aggregation ('daily', 'monthly')
    """

    obs_type = "climate"
    source_name = "ORNL_Daymet"

    # Variable mapping
    VARIABLE_MAP = {
        'tmax': 'temp_max_c',
        'tmin': 'temp_min_c',
        'prcp': 'precip_mm',
        'swe': 'swe_mm',
        'vp': 'vapor_pressure_pa',
        'srad': 'shortwave_radiation_wm2',
        'dayl': 'day_length_s',
    }

    def acquire(self) -> Path:
        """Acquire Daymet data via cloud acquisition."""
        daymet_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "climate" / "daymet", dict_key='DAYMET_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = daymet_dir.exists() and (
            any(daymet_dir.glob("*.nc")) or any(daymet_dir.glob("*.csv"))
        )

        if not has_files or force_download:
            self.logger.info("Acquiring Daymet climate data...")
            try:
                from ...acquisition.handlers.daymet import DaymetAcquirer
                acquirer = DaymetAcquirer(self.config, self.logger)
                acquirer.download(daymet_dir)
            except ImportError as e:
                self.logger.warning(f"Daymet acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"Daymet acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing Daymet data in {daymet_dir}")

        return daymet_dir

    def process(self, input_path: Path) -> Path:
        """
        Process Daymet climate data for the current domain.

        Args:
            input_path: Path to Daymet data directory or file

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing Daymet climate for domain: {self.domain_name}")

        # Find data files
        nc_files = list(input_path.glob("daymet*.nc"))
        csv_files = list(input_path.glob("daymet*.csv"))

        if not nc_files and not csv_files:
            self.logger.error("No Daymet data files found")
            return input_path

        # Load catchment shapefile
        basin_gdf = self._load_catchment_shapefile()

        all_data = []

        # Process NetCDF files
        for nc_file in nc_files:
            try:
                df = self._process_netcdf(nc_file, basin_gdf)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {nc_file.name}: {e}", exc_info=True)

        # Process CSV files (single-pixel)
        for csv_file in csv_files:
            try:
                df = self._process_csv(csv_file)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {csv_file.name}: {e}", exc_info=True)

        if not all_data:
            self.logger.warning("No Daymet data could be processed")
            return input_path

        # Combine all data
        df = pd.concat(all_data, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_index()

        # Aggregate if requested
        aggregate = self._get_config_value(lambda: None, default=None, dict_key='DAYMET_AGGREGATE')
        if aggregate == 'monthly':
            # Sum for precip, mean for others
            agg_dict = {}
            for col in df.columns:
                if 'precip' in col or 'swe' in col:
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'
            df = df.resample('MS').agg(agg_dict)

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "climate" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_daymet_climate_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"Daymet processing complete: {output_file}")

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
    ) -> Optional[pd.DataFrame]:
        """Process Daymet NetCDF file."""
        ds = xr.open_dataset(nc_file)

        results = {}

        # Find time dimension
        time_dim = self._find_coord(ds, ['time', 'date'])
        lat_name = self._find_coord(ds, ['lat', 'latitude', 'y'])
        lon_name = self._find_coord(ds, ['lon', 'longitude', 'x'])

        for var_name in ds.data_vars:
            var_name_str = str(var_name)
            if var_name_str in self.VARIABLE_MAP:
                std_name = self.VARIABLE_MAP[var_name_str]
            else:
                std_name = var_name_str

            try:
                da = ds[var_name]

                # Extract basin mean
                if basin_gdf is not None and lat_name and lon_name:
                    bounds = basin_gdf.total_bounds
                    da = self._subset_to_bounds(da, bounds, lat_name, lon_name)

                elif self.bbox and lat_name and lon_name:
                    bounds = [
                        self.bbox['lon_min'], self.bbox['lat_min'],
                        self.bbox['lon_max'], self.bbox['lat_max']
                    ]
                    da = self._subset_to_bounds(da, bounds, lat_name, lon_name)

                # Compute spatial mean
                spatial_dims = [d for d in da.dims if d != time_dim]
                da_mean = da.mean(dim=spatial_dims, skipna=True)

                # Convert to Series
                if time_dim:
                    series = da_mean.to_series()
                    results[std_name] = series

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process variable {var_name}: {e}", exc_info=True)

        ds.close()

        if not results:
            return None

        df = pd.DataFrame(results)
        df.index.name = 'datetime'

        return df

    def _process_csv(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Process Daymet single-pixel CSV file."""
        try:
            # Daymet CSV has 6 header lines before the column names on line 7
            # Lines 0-5: metadata (Latitude, X&Y, Tile, Elevation, Version, Citation)
            # Line 6: column headers (year,yday,prcp,tmax,tmin,...)
            df = pd.read_csv(csv_file, skiprows=6)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
            self.logger.debug(f"Daymet CSV parse with skiprows=6 failed: {e}, trying without skip")
            try:
                df = pd.read_csv(csv_file)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e2:
                self.logger.warning(f"Could not parse {csv_file}: {e2}")
                return None

        # Standardize column names
        column_map = {
            'year': 'year',
            'yday': 'doy',
            'tmax (deg c)': 'temp_max_c',
            'tmin (deg c)': 'temp_min_c',
            'prcp (mm/day)': 'precip_mm',
            'swe (mm)': 'swe_mm',
            'vp (Pa)': 'vapor_pressure_pa',
            'srad (W/m^2)': 'shortwave_radiation_wm2',
            'dayl (s)': 'day_length_s',
        }

        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Create datetime index from year and day-of-year
        if 'year' in df.columns and 'doy' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['year'].astype(str) + df['doy'].astype(str).str.zfill(3),
                format='%Y%j'
            )
            df = df.set_index('datetime')
            df = df.drop(columns=['year', 'doy'], errors='ignore')

        return df

    def _find_coord(self, ds, candidates: List[str]) -> Optional[str]:
        """Find coordinate name from candidates."""
        for name in candidates:
            if name in ds.coords or name in ds.dims:
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

        # Handle inverted latitude
        lat_vals = da[lat_name].values
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        return da.sel({lon_name: slice(lon_min, lon_max), lat_name: lat_slice})

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed Daymet climate data."""
        processed_path = (
            self.project_observations_dir / "climate" / "preprocessed"
            / f"{self.domain_name}_daymet_climate_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading Daymet data: {e}", exc_info=True)
            return None
