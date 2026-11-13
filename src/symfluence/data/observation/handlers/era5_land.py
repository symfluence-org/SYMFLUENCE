# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5-Land Observation Handler

Processes ERA5-Land reanalysis data for use as observation/validation data
in hydrological modeling. Provides basin-averaged time series for multiple
variables including precipitation, temperature, snow, soil moisture, and ET.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler

# Variable name mapping (CDS names to standard names)
VARIABLE_MAPPING = {
    'tp': 'precip_mm',
    'total_precipitation': 'precip_mm',
    't2m': 'temp_k',
    '2m_temperature': 'temp_k',
    'sd': 'snow_depth_m',
    'snow_depth': 'snow_depth_m',
    'sde': 'swe_m',
    'snow_depth_water_equivalent': 'swe_m',
    'swvl1': 'soil_moisture_l1',
    'volumetric_soil_water_layer_1': 'soil_moisture_l1',
    'swvl2': 'soil_moisture_l2',
    'volumetric_soil_water_layer_2': 'soil_moisture_l2',
    'swvl3': 'soil_moisture_l3',
    'volumetric_soil_water_layer_3': 'soil_moisture_l3',
    'swvl4': 'soil_moisture_l4',
    'volumetric_soil_water_layer_4': 'soil_moisture_l4',
    'e': 'et_m',
    'total_evaporation': 'et_m',
    'pev': 'pet_m',
    'potential_evaporation': 'pet_m',
    'sro': 'surface_runoff_m',
    'surface_runoff': 'surface_runoff_m',
    'ssro': 'subsurface_runoff_m',
    'subsurface_runoff': 'subsurface_runoff_m',
}

# Unit conversion factors (to SI units where applicable)
UNIT_CONVERSIONS = {
    'precip_mm': ('m_to_mm', 1000.0),  # m -> mm
    'et_m': ('m_to_mm', 1000.0),  # m -> mm (accumulated)
    'pet_m': ('m_to_mm', 1000.0),  # m -> mm
    'surface_runoff_m': ('m_to_mm', 1000.0),  # m -> mm
    'subsurface_runoff_m': ('m_to_mm', 1000.0),  # m -> mm
    'temp_k': ('k_to_c', -273.15),  # K -> °C (additive)
}


@R.observation_handlers.add('era5_land')
@R.observation_handlers.add('era5land')
class ERA5LandHandler(BaseObservationHandler):
    """
    Handles ERA5-Land reanalysis data processing for hydrological applications.

    Processes gridded ERA5-Land data to basin-averaged time series for
    comparison with model outputs or use as reference data.

    Configuration:
        ERA5_LAND_DIR: Directory containing ERA5-Land data
        ERA5_LAND_VARIABLES: Variables to process (default: all available)
        ERA5_LAND_AGGREGATE: Temporal aggregation ('hourly', 'daily', 'monthly')
        ERA5_LAND_CONVERT_UNITS: Whether to convert to standard units (default: True)
    """

    obs_type = "reanalysis"
    source_name = "ECMWF_ERA5_LAND"

    def acquire(self) -> Path:
        """Acquire ERA5-Land data via cloud acquisition."""
        era5_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "era5_land", dict_key='ERA5_LAND_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = era5_dir.exists() and any(era5_dir.glob("*.nc"))

        if not has_files or force_download:
            self.logger.info("Acquiring ERA5-Land data...")
            try:
                from ...acquisition.handlers.era5_land import ERA5LandAcquirer
                acquirer = ERA5LandAcquirer(self.config, self.logger)
                acquirer.download(era5_dir)
            except ImportError as e:
                self.logger.warning(f"ERA5-Land acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"ERA5-Land acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing ERA5-Land data in {era5_dir}")

        return era5_dir

    def process(self, input_path: Path) -> Path:
        """
        Process ERA5-Land data for the current domain.

        Extracts basin-averaged time series for all available variables
        and saves to standardized CSV format.

        Args:
            input_path: Path to ERA5-Land data directory or file

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing ERA5-Land for domain: {self.domain_name}")

        # Find ERA5-Land files
        if input_path.is_file():
            nc_files = [input_path]
        else:
            nc_files = list(input_path.glob("era5_land*.nc"))
            if not nc_files:
                nc_files = list(input_path.glob("*.nc"))

        if not nc_files:
            self.logger.error("No ERA5-Land NetCDF files found")
            return input_path

        # Load catchment shapefile for masking
        basin_gdf = self._load_catchment_shapefile()

        # Process each file and combine
        all_results = []
        for nc_file in nc_files:
            result = self._process_file(nc_file, basin_gdf)
            if result is not None:
                all_results.append(result)

        if not all_results:
            self.logger.warning("No ERA5-Land data could be processed")
            return input_path

        # Combine results
        df = pd.concat(all_results, axis=1)
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Apply unit conversions if requested
        if self._get_config_value(lambda: None, default=True, dict_key='ERA5_LAND_CONVERT_UNITS'):
            df = self._convert_units(df)

        # Temporal aggregation
        aggregate = self._get_config_value(lambda: None, default='daily', dict_key='ERA5_LAND_AGGREGATE')
        if aggregate == 'daily' and self._is_hourly(df):
            df = self._aggregate_to_daily(df)
        elif aggregate == 'monthly':
            df = df.resample('MS').mean()

        # Save processed data
        output_dir = self.project_observations_dir / "era5_land" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_era5_land_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"ERA5-Land processing complete: {output_file}")

        return output_file

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """Load the catchment shapefile for spatial masking.

        Delegates to the shared resolver, which handles the nested
        discretization layout and river_basins fallback. Returns None when
        no basin is found (callers fall back to the bounding box).
        """
        return self._load_catchment_gdf()

    def _process_file(
        self,
        nc_file: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ) -> Optional[pd.DataFrame]:
        """Process a single ERA5-Land NetCDF file."""
        try:
            ds = xr.open_dataset(nc_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to open {nc_file}: {e}", exc_info=True)
            return None

        results = {}

        for var_name in ds.data_vars:
            var_name_str = str(var_name)
            std_name = VARIABLE_MAPPING.get(var_name_str, var_name_str)

            try:
                da = ds[var_name]

                # Apply spatial subsetting
                if basin_gdf is not None:
                    da = self._extract_basin_mean(da, basin_gdf)
                elif self.bbox:
                    da = self._extract_bbox_mean(da)
                else:
                    # Global mean (not recommended)
                    spatial_dims = [d for d in da.dims if d not in ['time', 'valid_time']]
                    da = da.mean(dim=spatial_dims, skipna=True)

                # Convert to pandas Series
                time_dim = 'valid_time' if 'valid_time' in da.dims else 'time'
                if time_dim in da.dims:
                    series = da.to_series()
                    results[std_name] = series

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process variable {var_name}: {e}", exc_info=True)
                continue

        ds.close()

        if not results:
            return None

        df = pd.DataFrame(results)
        df.index.name = 'datetime'

        return df

    def _extract_basin_mean(
        self,
        da: xr.DataArray,
        basin_gdf: gpd.GeoDataFrame
    ) -> xr.DataArray:
        """Extract basin-averaged values using shapefile mask."""
        # Get bounds
        bounds = basin_gdf.total_bounds  # [minx, miny, maxx, maxy]

        # Find coordinate names
        lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
        lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])

        if not lat_name or not lon_name:
            raise ValueError("Could not identify lat/lon coordinates")

        # Subset to bounding box
        lat_slice = slice(bounds[1], bounds[3])
        lon_vals = da[lon_name].values

        # Handle 0-360 vs -180-180 longitude
        if lon_vals.max() > 180 and bounds[0] < 0:
            lon_min = bounds[0] % 360
            lon_max = bounds[2] % 360
        else:
            lon_min, lon_max = bounds[0], bounds[2]

        # Handle inverted latitude
        if da[lat_name].values[0] > da[lat_name].values[-1]:
            lat_slice = slice(bounds[3], bounds[1])

        subset = da.sel({lon_name: slice(lon_min, lon_max), lat_name: lat_slice})

        # Compute spatial mean
        spatial_dims = [d for d in subset.dims if d not in ['time', 'valid_time']]
        return subset.mean(dim=spatial_dims, skipna=True)

    def _extract_bbox_mean(self, da: xr.DataArray) -> xr.DataArray:
        """Extract bounding box averaged values."""
        lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
        lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])

        if not lat_name or not lon_name:
            spatial_dims = [d for d in da.dims if d not in ['time', 'valid_time']]
            return da.mean(dim=spatial_dims, skipna=True)

        # Handle longitude convention
        lon_vals = da[lon_name].values
        lon_min, lon_max = self.bbox['lon_min'], self.bbox['lon_max']
        if lon_vals.max() > 180 and lon_min < 0:
            lon_min = lon_min % 360
            lon_max = lon_max % 360

        lat_min, lat_max = self.bbox['lat_min'], self.bbox['lat_max']

        # Handle inverted latitude
        if da[lat_name].values[0] > da[lat_name].values[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        subset = da.sel({lon_name: slice(lon_min, lon_max), lat_name: lat_slice})
        spatial_dims = [d for d in subset.dims if d not in ['time', 'valid_time']]
        return subset.mean(dim=spatial_dims, skipna=True)

    def _find_coord(self, da: xr.DataArray, candidates: List[str]) -> Optional[str]:
        """Find coordinate name from candidates."""
        for name in candidates:
            if name in da.coords or name in da.dims:
                return name
        return None

    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply unit conversions to standard SI units."""
        for col in df.columns:
            if col in UNIT_CONVERSIONS:
                conversion_type, factor = UNIT_CONVERSIONS[col]
                if 'to_mm' in conversion_type:
                    df[col] = df[col] * factor
                    # Rename column to reflect units
                    new_name = col.replace('_m', '_mm')
                    df = df.rename(columns={col: new_name})
                elif 'k_to_c' in conversion_type:
                    df[col] = df[col] + factor
                    df = df.rename(columns={col: col.replace('_k', '_c')})

        return df

    def _is_hourly(self, df: pd.DataFrame) -> bool:
        """Check if data is hourly frequency."""
        if len(df) < 2:
            return False
        time_diff = df.index[1] - df.index[0]
        return time_diff <= pd.Timedelta(hours=1)

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate hourly data to daily."""
        # Identify variables that need sum vs mean
        sum_vars = ['precip', 'et', 'pet', 'runoff', 'evaporation']
        # Mean variables used implicitly via else branch below
        _mean_vars = ['temp', 'soil_moisture', 'snow', 'swe']

        result_cols = {}
        for col in df.columns:
            if any(sv in col.lower() for sv in sum_vars):
                # Sum for fluxes
                result_cols[col] = df[col].resample('D').sum()
            else:
                # Mean for states
                result_cols[col] = df[col].resample('D').mean()

        return pd.DataFrame(result_cols)

    def get_variable(self, variable: str) -> Optional[pd.Series]:
        """
        Get a specific variable from processed ERA5-Land data.

        Args:
            variable: Variable name (e.g., 'precip_mm', 'temp_c')

        Returns:
            Time series of the variable or None if not found
        """
        processed_path = (
            self.project_observations_dir / "era5_land" / "preprocessed"
            / f"{self.domain_name}_era5_land_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            if variable in df.columns:
                return df[variable]

            # Try partial match
            matches = [c for c in df.columns if variable in c]
            if matches:
                return df[matches[0]]

            return None
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading ERA5-Land data: {e}", exc_info=True)
            return None
