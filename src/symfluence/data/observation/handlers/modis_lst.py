# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS Land Surface Temperature (LST) Observation Handler

Processes MODIS MOD11A1/MYD11A1/MOD11A2/MYD11A2 Land Surface Temperature
data for use in hydrological modeling. LST is critical for:
- Potential evapotranspiration estimation
- Snow/freeze-thaw modeling
- Energy balance calculations
- Model temperature validation
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

# Scale factors and valid ranges
LST_SCALE_FACTOR = 0.02  # Convert to Kelvin
LST_VALID_RANGE = (7500, 65535)  # Valid DN range
LST_FILL_VALUE = 0

# QC bit interpretation for MOD11
QC_GOOD_QUALITY = [0, 1]  # Bits 0-1: 00=good, 01=other quality


@R.observation_handlers.add('modis_lst')
@R.observation_handlers.add('mod11')
class MODISLSTHandler(BaseObservationHandler):
    """
    Handles MODIS Land Surface Temperature data processing.

    Processes daily or 8-day composite LST data to basin-averaged
    temperature time series with quality filtering.

    Configuration:
        MODIS_LST_DIR: Directory containing MODIS LST data
        MODIS_LST_CONVERT_TO_DAILY: Interpolate 8-day to daily (default: True)
        MODIS_LST_MIN_QUALITY: Minimum QC quality (0-3, default: 1)
        MODIS_LST_OUTPUT_UNITS: 'kelvin' or 'celsius' (default: celsius)
    """

    obs_type = "lst"
    source_name = "NASA_MODIS"

    def acquire(self) -> Path:
        """Acquire MODIS LST data via cloud acquisition."""
        lst_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "temperature" / "modis_lst", dict_key='MODIS_LST_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = lst_dir.exists() and (
            any(lst_dir.glob("*LST*.nc")) or any(lst_dir.glob("*LST*.tif"))
        )

        if not has_files or force_download:
            self.logger.info("Acquiring MODIS LST data...")
            try:
                from ...acquisition.handlers.modis_lst import MODISLSTAcquirer
                acquirer = MODISLSTAcquirer(self.config, self.logger)
                acquirer.download(lst_dir)
            except ImportError as e:
                self.logger.warning(f"MODIS LST acquirer not available: {e}")
                raise
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, LookupError) as e:
                self.logger.error(f"MODIS LST acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing MODIS LST data in {lst_dir}")

        return lst_dir

    def process(self, input_path: Path) -> Path:
        """
        Process MODIS LST data for the current domain.

        Args:
            input_path: Path to MODIS LST data directory

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing MODIS LST for domain: {self.domain_name}")

        # Find LST files
        nc_files = list(input_path.glob("*LST*.nc")) + list(input_path.glob("*MOD11*.nc"))
        tif_files = list(input_path.glob("*LST*.tif"))

        if not nc_files and not tif_files:
            self.logger.error("No MODIS LST files found")
            return input_path

        # Load catchment shapefile
        basin_gdf = self._load_catchment_shapefile()

        # Process files
        results: dict[str, list] = {'lst_day_k': [], 'lst_night_k': [], 'datetime': []}

        for nc_file in nc_files:
            try:
                day_lst, night_lst, times = self._process_netcdf(nc_file, basin_gdf)
                if day_lst is not None:
                    results['lst_day_k'].extend(day_lst)
                    results['lst_night_k'].extend(night_lst)
                    results['datetime'].extend(times)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Failed to process {nc_file.name}: {e}")

        for tif_file in tif_files:
            try:
                lst_val, time_val, is_day = self._process_geotiff(tif_file, basin_gdf)
                if lst_val is not None:
                    if is_day:
                        results['lst_day_k'].append(lst_val)
                        results['lst_night_k'].append(np.nan)
                    else:
                        results['lst_day_k'].append(np.nan)
                        results['lst_night_k'].append(lst_val)
                    results['datetime'].append(time_val)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Failed to process {tif_file.name}: {e}")

        if not results['datetime']:
            self.logger.warning("No MODIS LST data could be processed")
            return input_path

        # Create DataFrame
        df = pd.DataFrame(results)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Convert units if requested
        output_units = self._get_config_value(lambda: None, default='celsius', dict_key='MODIS_LST_OUTPUT_UNITS')
        if output_units == 'celsius':
            df['lst_day_c'] = df['lst_day_k'] - 273.15
            df['lst_night_c'] = df['lst_night_k'] - 273.15
            df = df.drop(columns=['lst_day_k', 'lst_night_k'])

        # Interpolate to daily if needed (for 8-day composites)
        if self._get_config_value(lambda: None, default=True, dict_key='MODIS_LST_CONVERT_TO_DAILY'):
            df = self._interpolate_to_daily(df)

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "temperature" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_lst_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"MODIS LST processing complete: {output_file}")

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
    ):
        """Process NetCDF file containing LST data."""
        ds = xr.open_dataset(nc_file)

        # Find LST variables
        day_var = self._find_variable(ds, ['LST_Day_1km', 'LST_Day', 'lst_day'])
        night_var = self._find_variable(ds, ['LST_Night_1km', 'LST_Night', 'lst_night'])

        if not day_var and not night_var:
            ds.close()
            return None, None, None

        # Find QC variables
        qc_day_var = self._find_variable(ds, ['QC_Day', 'qc_day'])
        qc_night_var = self._find_variable(ds, ['QC_Night', 'qc_night'])

        # Get time dimension
        time_dim = self._find_coord(ds, ['time', 'date'])

        day_lst = []
        night_lst = []
        times = []

        if time_dim:
            time_vals = ds[time_dim].values
        else:
            # Single timestep - extract from filename
            time_vals = [self._extract_date_from_filename(nc_file.name)]

        for i, t in enumerate(time_vals):
            # Extract day LST
            if day_var:
                if time_dim:
                    da_day = ds[day_var].isel({time_dim: i})
                else:
                    da_day = ds[day_var]

                # Apply QC filter
                if qc_day_var:
                    qc_da = ds[qc_day_var].isel({time_dim: i}) if time_dim else ds[qc_day_var]
                    da_day = self._apply_qc_filter(da_day, qc_da)

                # Extract basin mean
                day_val = self._extract_basin_mean(da_day, basin_gdf)
                if day_val is not None and day_val > 0:
                    day_val = day_val * LST_SCALE_FACTOR  # Convert to Kelvin
                else:
                    day_val = np.nan
            else:
                day_val = np.nan

            # Extract night LST
            if night_var:
                if time_dim:
                    da_night = ds[night_var].isel({time_dim: i})
                else:
                    da_night = ds[night_var]

                if qc_night_var:
                    qc_da = ds[qc_night_var].isel({time_dim: i}) if time_dim else ds[qc_night_var]
                    da_night = self._apply_qc_filter(da_night, qc_da)

                night_val = self._extract_basin_mean(da_night, basin_gdf)
                if night_val is not None and night_val > 0:
                    night_val = night_val * LST_SCALE_FACTOR
                else:
                    night_val = np.nan
            else:
                night_val = np.nan

            day_lst.append(day_val)
            night_lst.append(night_val)
            times.append(pd.to_datetime(t))

        ds.close()
        return day_lst, night_lst, times

    def _process_geotiff(
        self,
        tif_file: Path,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ):
        """Process GeoTIFF file containing LST data."""
        import rasterio
        from rasterio.mask import mask as rio_mask

        # Determine if day or night from filename
        is_day = 'Day' in tif_file.name or 'day' in tif_file.name

        # Extract date from filename
        time_val = self._extract_date_from_filename(tif_file.name)
        if time_val is None:
            return None, None, None

        with rasterio.open(tif_file) as src:
            # Basin masking using rasterio
            if basin_gdf is not None:
                # Reproject basin to raster CRS if needed
                if basin_gdf.crs != src.crs:
                    basin_gdf = basin_gdf.to_crs(src.crs)

                try:
                    out_image, _ = rio_mask(
                        src,
                        basin_gdf.geometry,
                        crop=True,
                        nodata=np.nan
                    )
                    data = out_image[0]
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.warning(f"Rasterio masking failed, using full extent: {e}")
                    data = src.read(1)
            else:
                data = src.read(1)

            # Apply valid range filter
            data = np.where(
                (data >= LST_VALID_RANGE[0]) & (data <= LST_VALID_RANGE[1]),
                data, np.nan
            )

            lst_val = np.nanmean(data) * LST_SCALE_FACTOR

        return lst_val, time_val, is_day

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
        min_quality = self._get_config_value(lambda: None, default=1, dict_key='MODIS_LST_MIN_QUALITY')

        # Extract quality bits (bits 0-1)
        quality_bits = qc_da.values & 0b11

        # Mask poor quality
        mask = quality_bits <= min_quality
        return da.where(mask)

    def _extract_basin_mean(
        self,
        da: xr.DataArray,
        basin_gdf: Optional[gpd.GeoDataFrame]
    ) -> Optional[float]:
        """Extract basin-averaged value."""
        # Apply valid range filter
        da = da.where((da >= LST_VALID_RANGE[0]) & (da <= LST_VALID_RANGE[1]))

        if basin_gdf is not None:
            bounds = basin_gdf.total_bounds
            lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
            lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])

            if lat_name and lon_name:
                # Subset to bounds
                try:
                    lat_slice = slice(bounds[1], bounds[3])
                    if da[lat_name].values[0] > da[lat_name].values[-1]:
                        lat_slice = slice(bounds[3], bounds[1])

                    da = da.sel({
                        lon_name: slice(bounds[0], bounds[2]),
                        lat_name: lat_slice
                    })
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.debug(f"Could not subset data to shapefile bounds: {e}")

        elif self.bbox:
            lat_name = self._find_coord(da, ['lat', 'latitude', 'y'])
            lon_name = self._find_coord(da, ['lon', 'longitude', 'x'])

            if lat_name and lon_name:
                try:
                    lat_slice = slice(self.bbox['lat_min'], self.bbox['lat_max'])
                    if da[lat_name].values[0] > da[lat_name].values[-1]:
                        lat_slice = slice(self.bbox['lat_max'], self.bbox['lat_min'])

                    da = da.sel({
                        lon_name: slice(self.bbox['lon_min'], self.bbox['lon_max']),
                        lat_name: lat_slice
                    })
                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.debug(f"Could not subset data to bbox: {e}")

        return float(da.mean(skipna=True).values)

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from MODIS filename."""
        import re

        # Pattern: MOD11A1.AYYYYDDD or similar
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
        df_daily.index.name = 'datetime'

        return df_daily

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed MODIS LST data."""
        processed_path = (
            self.project_observations_dir / "temperature" / "preprocessed"
            / f"{self.domain_name}_modis_lst_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error loading MODIS LST data: {e}")
            return None
