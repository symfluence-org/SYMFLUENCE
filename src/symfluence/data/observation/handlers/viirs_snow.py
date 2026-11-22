# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIIRS Snow Cover Observation Handler

Processes VIIRS (Visible Infrared Imaging Radiometer Suite) snow cover
data for hydrological modeling. VIIRS is the successor to MODIS with
improved spatial resolution and cloud detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler

# VIIRS NDSI snow cover valid ranges and QA flags
NDSI_VALID_RANGE = (0, 100)  # 0-100% snow cover
NDSI_FILL_VALUES = [200, 201, 211, 237, 239, 250, 251, 252, 253, 254, 255]


@R.observation_handlers.add('viirs_snow')
@R.observation_handlers.add('vnp10')
class VIIRSSnowHandler(BaseObservationHandler):
    """
    Handles VIIRS snow cover data processing.

    Processes daily or 8-day composite VIIRS snow cover data to
    basin-averaged snow cover fraction time series.

    Configuration:
        VIIRS_SNOW_DIR: Directory containing VIIRS snow data
        VIIRS_SNOW_MIN_QUALITY: Minimum quality (0-3, default: 1)
        VIIRS_SNOW_CONVERT_TO_DAILY: Interpolate 8-day to daily (default: True)
    """

    obs_type = "snow_cover"
    source_name = "NASA_VIIRS"

    def acquire(self) -> Path:
        """Acquire VIIRS snow data via cloud acquisition."""
        viirs_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "snow" / "viirs", dict_key='VIIRS_SNOW_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = viirs_dir.exists() and (
            any(viirs_dir.glob("*VNP10*.nc")) or any(viirs_dir.glob("*Snow*.nc"))
        )

        if not has_files or force_download:
            self.logger.info("Acquiring VIIRS snow cover data...")
            try:
                from ...acquisition.handlers.viirs_snow import VIIRSSnowAcquirer
                acquirer = VIIRSSnowAcquirer(self.config, self.logger)
                acquirer.download(viirs_dir)
            except ImportError as e:
                self.logger.warning(f"VIIRS snow acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"VIIRS snow acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing VIIRS snow data in {viirs_dir}")

        return viirs_dir

    def process(self, input_path: Path) -> Path:
        """
        Process VIIRS snow cover data for the current domain.

        Args:
            input_path: Path to VIIRS snow data directory

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing VIIRS snow cover for domain: {self.domain_name}")

        # Find snow files
        nc_files = list(input_path.glob("*VNP10*.nc")) + list(input_path.glob("*Snow*.nc"))
        nc_files += list(input_path.glob("*NDSI*.nc"))

        if not nc_files:
            self.logger.error("No VIIRS snow files found")
            return input_path

        # Load catchment shapefile
        basin_gdf = self._load_catchment_shapefile()

        # Process files
        results: dict[str, list] = {'datetime': [], 'sca': [], 'snow_albedo': []}

        for nc_file in nc_files:
            try:
                data = self._process_netcdf(nc_file, basin_gdf)
                if data:
                    results['datetime'].extend(data['datetime'])
                    results['sca'].extend(data['sca'])
                    results['snow_albedo'].extend(data.get('snow_albedo', [np.nan] * len(data['datetime'])))
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {nc_file.name}: {e}", exc_info=True)

        if not results['datetime']:
            self.logger.warning("No VIIRS snow data could be processed")
            return input_path

        # Create DataFrame
        df = pd.DataFrame(results)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Interpolate to daily if needed (for 8-day composites)
        if self._get_config_value(lambda: None, default=True, dict_key='VIIRS_SNOW_CONVERT_TO_DAILY'):
            df = self._interpolate_to_daily(df)

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "snow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_viirs_snow_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"VIIRS snow processing complete: {output_file}")

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
        """Process NetCDF file containing VIIRS snow data."""
        ds = xr.open_dataset(nc_file)

        # Find snow cover variable
        sca_var = self._find_variable(ds, [
            'CGF_NDSI_Snow_Cover',
            'NDSI_Snow_Cover',
            'snow_cover',
            'SCA',
            'sca'
        ])

        albedo_var = self._find_variable(ds, [
            'Snow_Albedo_Daily_Tile',
            'snow_albedo',
            'albedo'
        ])

        qc_var = self._find_variable(ds, ['Basic_QA', 'QA', 'qc'])

        if not sca_var:
            ds.close()
            return None

        # Get time dimension
        time_dim = self._find_coord(ds, ['time', 'date'])
        lat_name = self._find_coord(ds, ['lat', 'latitude', 'y'])
        lon_name = self._find_coord(ds, ['lon', 'longitude', 'x'])

        results: dict[str, list] = {'datetime': [], 'sca': [], 'snow_albedo': []}

        if time_dim:
            time_vals = pd.to_datetime(ds[time_dim].values)
        else:
            time_vals = [self._extract_date_from_filename(nc_file.name)]

        for i, t in enumerate(time_vals):
            # Extract SCA
            if time_dim:
                da_sca = ds[sca_var].isel({time_dim: i})
            else:
                da_sca = ds[sca_var]

            # Apply QC filter if available
            if qc_var:
                qc_da = ds[qc_var].isel({time_dim: i}) if time_dim else ds[qc_var]
                da_sca = self._apply_qc_filter(da_sca, qc_da)

            # Mask fill values
            da_sca = da_sca.where(~da_sca.isin(NDSI_FILL_VALUES))
            da_sca = da_sca.where((da_sca >= NDSI_VALID_RANGE[0]) & (da_sca <= NDSI_VALID_RANGE[1]))

            # Extract basin mean
            sca_val = self._extract_basin_mean(da_sca, basin_gdf, lat_name, lon_name)

            # Convert to fraction (0-1)
            if sca_val is not None and not np.isnan(sca_val):
                sca_val = sca_val / 100.0

            # Extract albedo if available
            if albedo_var:
                if time_dim:
                    da_albedo = ds[albedo_var].isel({time_dim: i})
                else:
                    da_albedo = ds[albedo_var]
                albedo_val = self._extract_basin_mean(da_albedo, basin_gdf, lat_name, lon_name)
            else:
                albedo_val = np.nan

            results['datetime'].append(t)
            results['sca'].append(sca_val if sca_val is not None else np.nan)
            results['snow_albedo'].append(albedo_val if albedo_val is not None else np.nan)

        ds.close()
        return results

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

    def _apply_qc_filter(self, da: xr.DataArray, qc_da: xr.DataArray) -> xr.DataArray:
        """Apply quality filter based on QC flags."""
        min_quality = self._get_config_value(lambda: None, default=1, dict_key='VIIRS_SNOW_MIN_QUALITY')

        # VIIRS QA: bits 0-1 indicate quality
        # 0 = best, 1 = good, 2 = ok, 3 = poor
        quality_bits = qc_da.values & 0b11
        mask = quality_bits <= min_quality

        return da.where(mask)

    def _extract_basin_mean(
        self,
        da: xr.DataArray,
        basin_gdf: Optional[gpd.GeoDataFrame],
        lat_name: Optional[str],
        lon_name: Optional[str]
    ) -> Optional[float]:
        """Extract basin-averaged value."""
        if basin_gdf is not None and lat_name and lon_name:
            bounds = basin_gdf.total_bounds
            try:
                lat_slice = slice(bounds[1], bounds[3])
                if len(da[lat_name].values) > 1 and da[lat_name].values[0] > da[lat_name].values[-1]:
                    lat_slice = slice(bounds[3], bounds[1])

                da = da.sel({
                    lon_name: slice(bounds[0], bounds[2]),
                    lat_name: lat_slice
                })
            except (KeyError, ValueError, IndexError) as e:
                self.logger.debug(f"Could not subset VIIRS data to basin bounds: {e}")

        elif self.bbox and lat_name and lon_name:
            try:
                lat_slice = slice(self.bbox['lat_min'], self.bbox['lat_max'])
                if len(da[lat_name].values) > 1 and da[lat_name].values[0] > da[lat_name].values[-1]:
                    lat_slice = slice(self.bbox['lat_max'], self.bbox['lat_min'])

                da = da.sel({
                    lon_name: slice(self.bbox['lon_min'], self.bbox['lon_max']),
                    lat_name: lat_slice
                })
            except (KeyError, ValueError, IndexError) as e:
                self.logger.debug(f"Could not subset VIIRS data to bbox: {e}")

        mean_val = float(da.mean(skipna=True).values)
        return mean_val if not np.isnan(mean_val) else None

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from VIIRS filename."""
        import re

        # Pattern: VNP10A1F.AYYYYDDD or similar
        match = re.search(r'\.A(\d{4})(\d{3})\.', filename)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)

        # Pattern: YYYYDDD in filename
        match = re.search(r'(\d{4})(\d{3})', filename)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)

        return None

    def _interpolate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate to daily frequency if needed."""
        if len(df) < 2:
            return df

        # Check if already daily
        time_diff = (df.index[1] - df.index[0]).days
        if time_diff <= 1:
            return df

        # Create daily index
        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df_daily = df.reindex(daily_index)
        df_daily = df_daily.interpolate(method='linear')

        # Clip SCA to valid range
        if 'sca' in df_daily.columns:
            df_daily['sca'] = df_daily['sca'].clip(0, 1)

        df_daily.index.name = 'datetime'
        return df_daily

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed VIIRS snow cover data."""
        processed_path = (
            self.project_observations_dir / "snow" / "preprocessed"
            / f"{self.domain_name}_viirs_snow_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading VIIRS snow data: {e}", exc_info=True)
            return None
