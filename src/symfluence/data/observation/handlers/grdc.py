# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GRDC (Global Runoff Data Centre) Streamflow Observation Handler

Processes GRDC streamflow data for use in hydrological model calibration
and validation. GRDC is the primary global archive for river discharge data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('grdc')
class GRDCHandler(BaseObservationHandler):
    """
    Handles GRDC streamflow data processing.

    Processes GRDC discharge data to standardized format for model
    calibration and evaluation.

    Configuration:
        GRDC_DATA_DIR: Directory containing GRDC data files
        GRDC_STATION_IDS: Station ID(s) to process
        GRDC_RESAMPLE: Temporal resampling ('daily', 'monthly')
    """

    obs_type = "streamflow"
    source_name = "GRDC"

    def acquire(self) -> Path:
        """Acquire GRDC data via cloud acquisition."""
        grdc_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "streamflow" / "grdc", dict_key='GRDC_DATA_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, default=False)
        has_files = grdc_dir.exists() and any(grdc_dir.glob("grdc_*.csv"))

        if not has_files or force_download:
            self.logger.info("Acquiring GRDC streamflow data...")
            try:
                from ...acquisition.handlers.grdc import GRDCAcquirer
                acquirer = GRDCAcquirer(self.config, self.logger)
                acquirer.download(grdc_dir)
            except ImportError as e:
                self.logger.warning(f"GRDC acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"GRDC acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing GRDC data in {grdc_dir}")

        return grdc_dir

    def process(self, input_path: Path) -> Path:
        """
        Process GRDC streamflow data for the current domain.

        Args:
            input_path: Path to GRDC data directory or file

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing GRDC streamflow for domain: {self.domain_name}")

        # Find GRDC files
        if input_path.is_file():
            csv_files = [input_path]
        else:
            csv_files = list(input_path.glob("grdc_*.csv"))
            if not csv_files:
                csv_files = list(input_path.glob("*.csv"))

        if not csv_files:
            self.logger.error("No GRDC CSV files found")
            return input_path

        # Process files (may have multiple stations)
        all_data = []

        for csv_file in csv_files:
            try:
                df = self._process_file(csv_file)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {csv_file.name}: {e}", exc_info=True)

        if not all_data:
            self.logger.warning("No GRDC data could be processed")
            return input_path

        # Combine if multiple files
        if len(all_data) == 1:
            df = all_data[0]
        else:
            # Average multiple stations if present
            df = pd.concat(all_data, axis=1)
            df['discharge_cms'] = df.filter(like='discharge').mean(axis=1)
            df = df[['discharge_cms']]

        # Resample if requested
        resample = self._get_config_value(lambda: None, default=None, dict_key='GRDC_RESAMPLE')
        if resample == 'monthly':
            df = df.resample('MS').mean()

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_grdc_streamflow_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"GRDC processing complete: {output_file}")

        return output_file

    def _process_file(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Process a single GRDC CSV file."""
        # GRDC files can have various formats
        try:
            # Try standard format first
            df = pd.read_csv(csv_file, comment='#')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
            self.logger.debug(f"Standard CSV parse failed for {csv_file.name}: {e}, trying semicolon separator")
            # Try with semicolon separator (some GRDC exports use this)
            try:
                df = pd.read_csv(csv_file, sep=';', comment='#')
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e2:
                self.logger.warning(f"Could not parse {csv_file}: {e2}")
                return None

        # Standardize column names
        column_map = {
            'YYYY-MM-DD': 'date',
            'date': 'date',
            'Date': 'date',
            'TIME': 'date',
            'calculated': 'discharge_cms',
            'Value': 'discharge_cms',
            'discharge': 'discharge_cms',
            'RUNOFF': 'discharge_cms',
            ' Value': 'discharge_cms',  # Some files have leading space
        }

        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Handle missing date column
        if 'date' not in df.columns:
            # Try to construct from year/month/day columns
            if all(c in df.columns for c in ['YYYY', 'MM', 'DD']):
                df['date'] = pd.to_datetime(
                    df['YYYY'].astype(str) + '-' +
                    df['MM'].astype(str).str.zfill(2) + '-' +
                    df['DD'].astype(str).str.zfill(2)
                )
            elif 'Year' in df.columns and 'Month' in df.columns:
                # Monthly data
                df['date'] = pd.to_datetime(
                    df['Year'].astype(str) + '-' +
                    df['Month'].astype(str).str.zfill(2) + '-01'
                )
            else:
                self.logger.warning(f"Cannot identify date column in {csv_file}")
                return None

        # Parse dates unless already datetime (strings are ``object`` on
        # pandas 2 and ``str`` on pandas 3)
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # Set index
        df = df.set_index('date')
        df.index.name = 'datetime'

        # Handle discharge column
        if 'discharge_cms' not in df.columns:
            # Try to find numeric column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                df['discharge_cms'] = df[numeric_cols[0]]
            else:
                self.logger.warning(f"Cannot identify discharge column in {csv_file}")
                return None

        # Handle missing values (GRDC uses -999 or similar)
        df['discharge_cms'] = df['discharge_cms'].replace([-999, -999.0, -9999], pd.NA)
        df['discharge_cms'] = pd.to_numeric(df['discharge_cms'], errors='coerce')

        # Filter to just discharge column
        df = df[['discharge_cms']]

        return df

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed GRDC streamflow data."""
        processed_path = (
            self.project_observations_dir / "streamflow" / "preprocessed"
            / f"{self.domain_name}_grdc_streamflow_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading GRDC data: {e}", exc_info=True)
            return None
