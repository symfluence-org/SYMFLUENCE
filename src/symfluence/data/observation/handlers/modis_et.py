# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS MOD16 Evapotranspiration (ET) Observation Handler

Handles MODIS 8-day Evapotranspiration (ET) data processing for calibration.
Supports data from:
1. Cloud acquisition via AppEEARS (MOD16ETAcquirer)
2. Pre-downloaded CSV files (legacy format)
3. Pre-downloaded NetCDF files

The handler processes 8-day composite ET data and converts it to daily
values suitable for comparison with model outputs like SUMMA scalarTotalET.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('modis_et')
@R.observation_handlers.add('mod16')
@R.observation_handlers.add('mod16a2')
class MODISETHandler(BaseObservationHandler):
    """
    Handles MODIS MOD16A2 8-day Evapotranspiration (ET) data.

    Processes 8-day composite ET products into daily values and provides
    basin-averaged time series for calibration against model outputs.

    Configuration:
        MOD16_ET_DIR: Directory containing MOD16 data
        MOD16_CONVERT_TO_DAILY: True (default) - interpolate 8-day to daily
        MOD16_INTERPOLATION_METHOD: 'linear' (default), 'nearest', 'constant'
        MOD16_MIN_VALID_FRACTION: Minimum fraction of valid data (default 0.5)
    """

    obs_type = "et"
    source_name = "NASA_MODIS"
    SOURCE_INFO = {
        'source': 'MODIS MOD16A2',
        'source_doi': '10.5067/MODIS/MOD16A2.061',
        'url': 'https://lpdaac.usgs.gov/products/mod16a2v061/',
    }

    def _resolve_mod16_dir(self) -> Path:
        """Resolve MOD16_ET_DIR, treating 'default' as the project observations path."""
        default_dir = self.project_observations_dir / "et" / "modis"
        configured = self._get_config_value(
            lambda: None,
            default=default_dir,
            dict_key='MOD16_ET_DIR',
        )
        if isinstance(configured, str) and configured.strip().lower() == 'default':
            return default_dir
        return Path(configured)

    def acquire(self) -> Path:
        """
        Acquire MODIS ET data via cloud acquisition.

        Returns path to raw data directory/file.
        """
        # Check if data already exists
        et_dir = self._resolve_mod16_dir()

        # Check for existing processed file
        processed_file = et_dir / "preprocessed" / f"{self.domain_name}_modis_et_processed.csv"
        if processed_file.exists() and not self._get_config_value(
            lambda: self.config.system.force_run_all_steps,
            dict_key='FORCE_RUN_ALL_STEPS',
        ):
            self.logger.info(f"Using existing processed MOD16 ET: {processed_file}")
            return processed_file.parent

        # Check for existing raw data
        raw_nc = et_dir / f"{self.domain_name}_MOD16_ET.nc"
        raw_csv = et_dir / f"{self.domain_name}_MOD16_ET_timeseries.csv"

        if (raw_nc.exists() or raw_csv.exists()) and not self._get_config_value(
            lambda: self.config.system.force_run_all_steps,
            dict_key='FORCE_RUN_ALL_STEPS',
        ):
            self.logger.info(f"Using existing raw MOD16 data in: {et_dir}")
            return et_dir

        # Run cloud acquisition
        try:
            from ...acquisition.handlers.modis_et import MOD16ETAcquirer

            acquirer = MOD16ETAcquirer(self.config, self.logger)
            et_dir.mkdir(parents=True, exist_ok=True)
            result_path = acquirer.download(et_dir)
            self.logger.info(f"MOD16 ET acquisition complete: {result_path}")
            return et_dir

        except ImportError:
            self.logger.warning("MOD16 acquisition handler not available")
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, LookupError) as e:
            self.logger.warning(f"MOD16 acquisition failed: {e}")

        # Fallback: check for manually placed data
        et_dir.mkdir(parents=True, exist_ok=True)
        return et_dir

    def process(self, input_path: Path) -> Path:
        """
        Process MODIS ET data to daily time series.

        Handles:
        1. NetCDF files from cloud acquisition
        2. CSV files from cloud acquisition
        3. Legacy CSV files (ET8D_Basin_*.csv)

        Args:
            input_path: Path to raw data directory or file

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing MODIS ET for domain: {self.domain_name}")

        output_dir = self.project_observations_dir / "et" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_modis_et_processed.csv"

        # Check for existing processed file
        if output_file.exists() and not self._get_config_value(
            lambda: self.config.system.force_run_all_steps,
            dict_key='FORCE_RUN_ALL_STEPS',
        ):
            self.logger.info(f"Using existing processed file: {output_file}")
            return output_file

        # Try different data sources
        df = None

        # 1. Try NetCDF from cloud acquisition
        if input_path.is_dir():
            nc_files = list(input_path.glob("*MOD16*.nc")) + list(input_path.glob("*_ET.nc"))
            if nc_files:
                df = self._process_netcdf(nc_files[0])

        elif input_path.suffix == '.nc':
            df = self._process_netcdf(input_path)

        # 2. Try CSV from cloud acquisition
        if df is None:
            csv_pattern = f"{self.domain_name}_MOD16_ET_timeseries.csv"
            if input_path.is_dir():
                csv_files = list(input_path.glob(csv_pattern))
                if not csv_files:
                    csv_files = list(input_path.glob("*MOD16*timeseries*.csv"))
                if csv_files:
                    df = self._process_timeseries_csv(csv_files[0])

        # 3. Try legacy CSV format
        if df is None:
            if input_path.is_dir():
                legacy_files = list(input_path.glob("ET8D_Basin_*.csv"))
                if legacy_files:
                    df = self._process_legacy_csv(legacy_files[0])

        # 4. Try any CSV in directory
        if df is None and input_path.is_dir():
            any_csv = list(input_path.glob("*.csv"))
            if any_csv:
                df = self._process_generic_csv(any_csv[0])

        if df is None or df.empty:
            self.logger.warning("No MODIS ET data found or processed")
            # Create empty placeholder
            df = pd.DataFrame(columns=['date', 'et_mm_day'])

        # Save processed data
        df.to_csv(output_file)
        self.logger.info(f"MODIS ET processing complete: {output_file}")

        return output_file

    def _process_netcdf(self, nc_path: Path) -> pd.DataFrame:
        """Process NetCDF file from cloud acquisition."""
        self.logger.info(f"Processing NetCDF: {nc_path}")

        try:
            ds = xr.open_dataset(nc_path)

            # Find ET variable
            et_var = None
            for var in ['ET_basin_mean', 'ET', 'et', 'et_mm_day']:
                if var in ds.data_vars:
                    et_var = var
                    break

            if et_var is None:
                self.logger.warning(f"No ET variable found in {nc_path}")
                ds.close()
                return pd.DataFrame()

            da = ds[et_var]

            # If gridded, compute spatial mean
            spatial_dims = [d for d in da.dims if d not in ['time', 'date']]
            if spatial_dims:
                da = da.mean(dim=spatial_dims, skipna=True)

            # Convert to DataFrame
            df = da.to_dataframe().reset_index()

            # Standardize column names
            df = self._standardize_columns(df)

            # Interpolate 8-day to daily if needed
            df = self._interpolate_to_daily(df)

            ds.close()
            return df

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing NetCDF: {e}")
            return pd.DataFrame()

    def _process_timeseries_csv(self, csv_path: Path) -> pd.DataFrame:
        """Process CSV timeseries from cloud acquisition."""
        self.logger.info(f"Processing timeseries CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])
            df = self._standardize_columns(df)
            df = self._interpolate_to_daily(df)
            return df
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing timeseries CSV: {e}")
            return pd.DataFrame()

    def _process_legacy_csv(self, csv_path: Path) -> pd.DataFrame:
        """Process legacy ET8D_Basin_*.csv format."""
        self.logger.info(f"Processing legacy CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])

            # Legacy format has 8-day cumulative values
            if 'mean_et_mm' in df.columns:
                # Convert 8-day sum to daily mean
                df['et_mm_day'] = df['mean_et_mm'] / 8.0
            elif 'et_mm' in df.columns:
                df['et_mm_day'] = df['et_mm'] / 8.0

            df = self._standardize_columns(df)
            df = self._interpolate_to_daily(df)
            return df

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing legacy CSV: {e}")
            return pd.DataFrame()

    def _process_generic_csv(self, csv_path: Path) -> pd.DataFrame:
        """Process generic CSV with ET data."""
        self.logger.info(f"Processing generic CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)

            # Try to find date column
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                df = df.rename(columns={date_cols[0]: 'date'})

            df = self._standardize_columns(df)
            df = self._interpolate_to_daily(df)
            return df

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing generic CSV: {e}")
            return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        # Map various column names to standard names
        column_map = {
            'time': 'date',
            'datetime': 'date',
            'Date': 'date',
            'et': 'et_mm_day',
            'ET': 'et_mm_day',
            'et_mm': 'et_mm_day',
            'ET_500m': 'et_mm_day',
            'ET_basin_mean': 'et_mm_day',
            'mean_et_mm': 'et_8day_mm',
            'et_daily_mm': 'et_mm_day'
        }

        for old, new in column_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # Convert 8-day to daily if needed
        if 'et_8day_mm' in df.columns and 'et_mm_day' not in df.columns:
            df['et_mm_day'] = df['et_8day_mm'] / 8.0

        return df

    def _interpolate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate 8-day composite to daily values."""
        if not self._get_config_value(lambda: None, default=True, dict_key='MOD16_CONVERT_TO_DAILY'):
            return df

        if df.empty or 'et_mm_day' not in df.columns:
            return df

        method = self._get_config_value(lambda: None, default='linear', dict_key='MOD16_INTERPOLATION_METHOD')

        try:
            # Resample to daily frequency
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df = df.set_index('date')
                else:
                    return df

            # Create daily date range
            start = df.index.min()
            end = df.index.max()
            daily_index = pd.date_range(start=start, end=end, freq='D')

            # Reindex and interpolate
            df_daily = df.reindex(daily_index)

            if method == 'constant':
                # Forward fill (constant over 8-day period)
                df_daily = df_daily.ffill()
            elif method == 'nearest':
                df_daily = df_daily.interpolate(method='nearest')
            else:
                # Linear interpolation (default)
                df_daily = df_daily.interpolate(method='linear')

            df_daily.index.name = 'date'
            return df_daily

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.warning(f"Interpolation failed: {e}")
            return df

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get processed ET data ready for calibration.

        Returns DataFrame with columns: date (index), et_mm_day
        """
        processed_path = (
            self.project_observations_dir / "et" / "preprocessed"
            / f"{self.domain_name}_modis_et_processed.csv"
        )

        if not processed_path.exists():
            # Try to process data
            raw_dir = self.project_observations_dir / "et" / "modis"
            if raw_dir.exists():
                processed_path = self.process(raw_dir)
            else:
                self.logger.warning("No MOD16 ET data available")
                return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['date'], index_col='date')
            return df
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error loading processed data: {e}")
            return None
