# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
OpenET Evapotranspiration Observation Handler

Processes OpenET ensemble evapotranspiration data for hydrological
model calibration and validation. OpenET provides high-resolution
(30m) ET estimates using an ensemble of 6 satellite-based ET models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('openet')
class OpenETHandler(BaseObservationHandler):
    """
    Handles OpenET evapotranspiration data processing.

    Processes OpenET ensemble or individual model ET data to
    standardized format for model calibration.

    Configuration:
        OPENET_DIR: Directory containing OpenET data
        OPENET_MODEL: Model to use ('ensemble' or individual model name)
        OPENET_AGGREGATE: Temporal aggregation ('daily', 'monthly')
    """

    obs_type = "et"
    source_name = "OpenET"

    def acquire(self) -> Path:
        """Acquire OpenET data via cloud acquisition."""
        openet_dir = Path(self._get_config_value(lambda: None, default=self.project_observations_dir / "et" / "openet", dict_key='OPENET_DIR'))

        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        has_files = openet_dir.exists() and any(openet_dir.glob("openet_*.csv"))

        if not has_files or force_download:
            self.logger.info("Acquiring OpenET data...")
            try:
                from ...acquisition.handlers.openet import OpenETAcquirer
                acquirer = OpenETAcquirer(self.config, self.logger)
                acquirer.download(openet_dir)
            except ImportError as e:
                self.logger.warning(f"OpenET acquirer not available: {e}")
                raise
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"OpenET acquisition failed: {e}")
                raise
        else:
            self.logger.info(f"Using existing OpenET data in {openet_dir}")

        return openet_dir

    def process(self, input_path: Path) -> Path:
        """
        Process OpenET data for the current domain.

        Args:
            input_path: Path to OpenET data directory or file

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing OpenET for domain: {self.domain_name}")

        # Find OpenET files
        if input_path.is_file():
            csv_files = [input_path]
        else:
            csv_files = list(input_path.glob("openet_*.csv"))
            if not csv_files:
                csv_files = list(input_path.glob("*.csv"))

        if not csv_files:
            self.logger.error("No OpenET CSV files found")
            return input_path

        # Process files
        all_data = []

        for csv_file in csv_files:
            try:
                df = self._process_file(csv_file)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {csv_file.name}: {e}", exc_info=True)

        if not all_data:
            self.logger.warning("No OpenET data could be processed")
            return input_path

        # Combine data
        df = pd.concat(all_data)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Aggregate if requested
        aggregate = self._get_config_value(lambda: None, default=None, dict_key='OPENET_AGGREGATE')
        if aggregate == 'monthly':
            df = df.resample('MS').sum()  # Sum for monthly totals

        # Filter to experiment period
        df = df.loc[self.start_date:self.end_date]

        # Save processed data
        output_dir = self.project_observations_dir / "et" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_openet_et_processed.csv"

        df.to_csv(output_file)
        self.logger.info(f"OpenET processing complete: {output_file}")

        return output_file

    def _process_file(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Process a single OpenET CSV file."""
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Could not parse {csv_file}: {e}", exc_info=True)
            return None

        # Standardize column names
        column_map = {
            'date': 'date',
            'Date': 'date',
            'time': 'date',
            'et_mm': 'et_mm_day',
            'et': 'et_mm_day',
            'ET': 'et_mm_day',
            'value': 'et_mm_day',
            'et_mm/day': 'et_mm_day',
        }

        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.index.name = 'datetime'

        # Ensure ET column exists
        if 'et_mm_day' not in df.columns:
            # Look for any numeric column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                df['et_mm_day'] = df[numeric_cols[0]]
            else:
                return None

        # Keep only ET column
        df = df[['et_mm_day']]

        # Handle missing values
        df['et_mm_day'] = pd.to_numeric(df['et_mm_day'], errors='coerce')

        return df

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed OpenET data."""
        processed_path = (
            self.project_observations_dir / "et" / "preprocessed"
            / f"{self.domain_name}_openet_et_processed.csv"
        )

        if not processed_path.exists():
            return None

        try:
            df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
            return df
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error loading OpenET data: {e}", exc_info=True)
            return None
