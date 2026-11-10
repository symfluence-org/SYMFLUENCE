# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SMHI Observation Handlers

Provides handlers for Swedish Meteorological and Hydrological Institute (SMHI) streamflow data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError, symfluence_error_handler
from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('smhi_streamflow')
class SMHIStreamflowHandler(BaseObservationHandler):
    """
    Handles SMHI streamflow data acquisition and processing.
    """

    obs_type = "streamflow"
    source_name = "SMHI"
    SOURCE_INFO = {
        'source': 'SMHI',
        'url': 'https://vattenwebb.smhi.se/',
    }

    def acquire(self) -> Path:
        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='cloud', dict_key='DATA_ACCESS').lower()
        station_id = self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID')

        if not station_id:
            self.logger.error("Missing STATION_ID in configuration for SMHI streamflow")
            raise ValueError("STATION_ID required for SMHI streamflow acquisition")

        raw_dir = self.project_observations_dir / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"smhi_{station_id}_raw.csv"

        if data_access == 'cloud':
            # Idempotency guard (mirrors the WSC handler): acquire() is invoked
            # twice in a single process_observed_data() run, and the SMHI
            # corrected-archive endpoint returns the full ~73 MB station record
            # on every call. Reuse the already-downloaded raw CSV unless
            # FORCE_DOWNLOAD is set.
            force_download = self._get_config_value(
                lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'
            )
            if raw_file.exists() and not force_download:
                self.logger.info(f"Using existing SMHI data: {raw_file}")
                return raw_file
            return self._download_from_smhi(station_id, raw_file)

        if raw_file.exists():
            return raw_file

        self.logger.warning(f"SMHI raw file not found: {raw_file}")
        return raw_file

    def _download_from_smhi(self, station_id: str, output_path: Path) -> Path:
        self.logger.info(f"Downloading SMHI streamflow data for station {station_id}")

        # SMHI uses parameter 2 for Discharge (Vattenföring)
        parameter_key = 2
        period = 'corrected-archive'
        url = f"https://opendata-download-hydroobs.smhi.se/api/version/latest/parameter/{parameter_key}/station/{station_id}/period/{period}/data.json"

        with symfluence_error_handler("SMHI data download", self.logger, error_type=DataAcquisitionError):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            values = data.get('value', [])
            if not values:
                self.logger.error(f"No data available for SMHI station {station_id}")
                raise DataAcquisitionError(f"No data found for SMHI station {station_id}")

            df = pd.DataFrame(values)
            # SMHI uses milliseconds for date
            df['date'] = pd.to_datetime(df['date'] / 1000, unit='s')
            df = df.rename(columns={'value': 'discharge_m3s', 'quality': 'quality_code'})

            # Save to CSV in a format that ObservedDataProcessor can understand or that process() expects
            df.to_csv(output_path, index=False)

            self.logger.info(f"Successfully downloaded {len(df)} records to {output_path}")
            return output_path

    def process(self, input_path: Path) -> Path:
        """
        Process SMHI data into standard SYMFLUENCE format.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"SMHI raw data file not found: {input_path}")

        self.logger.info(f"Processing SMHI streamflow data from {input_path}")

        df = pd.read_csv(input_path)

        # SMHI download format from _download_from_smhi: date, discharge_m3s, quality_code
        datetime_col = 'date'
        discharge_col = 'discharge_m3s'

        if datetime_col not in df.columns or discharge_col not in df.columns:
            # Fallback to general column finding if format changed
            datetime_col = self._find_col(df.columns, ['date', 'datetime'])
            discharge_col = self._find_col(df.columns, ['value', 'discharge', 'm3s'])

        if not datetime_col or not discharge_col:
            raise DataAcquisitionError(f"Could not identify required columns in SMHI data: {input_path}")

        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df[discharge_col] = pd.to_numeric(df[discharge_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, discharge_col])

        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Standardize naming
        df['discharge_cms'] = df[discharge_col]

        # Resample to target timestep
        # SMHI archive values are daily means, never interpolated.
        resampled = self._daily_streamflow_to_target(df['discharge_cms'])

        # Save processed data
        output_dir = self.project_observations_dir / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"

        resampled.to_csv(output_file, header=True, index_label='datetime')

        self.logger.info(f"SMHI streamflow processing complete: {output_file}")
        return output_file
