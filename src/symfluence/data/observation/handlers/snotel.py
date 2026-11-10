# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SNOTEL Observation Handler

Provides cloud acquisition and processing for SNOTEL (Snow Telemetry) data
via the NRCS AWDB (Air-Water Database) API.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError, symfluence_error_handler
from symfluence.core.registries import R

from ..base import BaseObservationHandler


@R.observation_handlers.add('snotel')
class SNOTELHandler(BaseObservationHandler):
    """
    Handles SNOTEL data acquisition and processing.
    """

    obs_type = "swe"
    source_name = "NRCS_SNOTEL"
    SOURCE_INFO = {
        'source': 'NRCS SNOTEL',
        'url': 'https://wcc.sc.egov.usda.gov/nwcc/site',
    }

    def acquire(self) -> Path:
        """
        Acquire SNOTEL data from NRCS AWDB API.
        """
        download_enabled = self._get_config_value(lambda: self.config.evaluation.snotel.download, default=False, dict_key='DOWNLOAD_SNOTEL')
        if isinstance(download_enabled, str):
            download_enabled = download_enabled.lower() == 'true'

        station_id = self._get_config_value(lambda: self.config.evaluation.snotel.station, dict_key='SNOTEL_STATION') or self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID')

        if not station_id:
            self.logger.error("Missing SNOTEL_STATION in configuration")
            raise ValueError("SNOTEL_STATION required for SNOTEL acquisition")

        raw_dir = self.project_observations_dir / "snow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"snotel_{station_id}_raw.csv"

        if download_enabled:
            return self._download_data(station_id, raw_file)
        else:
            if raw_file.exists():
                return raw_file

            # Check for legacy path format
            snotel_path = self._get_config_value(lambda: self.config.evaluation.snotel.path, dict_key='SNOTEL_PATH')
            if snotel_path and snotel_path != 'default':
                matches = list(Path(snotel_path).rglob(f"*{station_id}*.csv"))
                if matches:
                    return matches[0]

            self.logger.warning(f"SNOTEL raw file not found and download disabled: {raw_file}")
            return raw_file

    def _download_data(self, station_id: str, output_path: Path) -> Path:
        """
        Download SNOTEL data via NRCS Report Generator CSV interface.
        """
        self.logger.info(f"Downloading SNOTEL data for station {station_id}")

        # Determine state (Paradise is 679:WA:SNTL)
        state = self._get_config_value(lambda: None, default='WA', dict_key='SNOTEL_STATE')

        # Construct URL based on the reference notebook logic
        url_base = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/'
        url_station = f"{station_id}:{state}:SNTL%7Cid=%22%22%7Cname/"
        url_params = 'POR_BEGIN,POR_END/WTEQ::value,PREC::value,PRCP::value'
        url = url_base + url_station + url_params

        with symfluence_error_handler("SNOTEL data download", self.logger, error_type=DataAcquisitionError):
            self.logger.info(f"Fetching SNOTEL report: {url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            # Save the raw CSV
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

            self.logger.info(f"Successfully downloaded SNOTEL data to {output_path}")
            return output_path

    def process(self, input_path: Path) -> Path:
        """
        Process SNOTEL raw data into standard SYMFLUENCE format.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"SNOTEL raw data file not found: {input_path}")

        self.logger.info(f"Processing SNOTEL data from {input_path}")

        # Load the data, skipping header lines (NRCS CSV has many comments starting with #)
        df = pd.read_csv(input_path, comment='#')

        # Clean column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # Identify columns
        # The CSV report often has headers like 'Date' and 'Snow Water Equivalent (in)'
        datetime_col = None
        swe_col = None

        for col in df.columns:
            if 'Date' in col: datetime_col = col
            if 'Snow Water Equivalent' in col or 'WTEQ' in col: swe_col = col

        if not datetime_col or not swe_col:
            # Try to find date and value by position if names fail
            # Usually column 0 is date, column 1 is SWE in these reports
            if len(df.columns) >= 2:
                datetime_col = df.columns[0]
                swe_col = df.columns[1]
            else:
                raise DataAcquisitionError(f"Could not identify SNOTEL columns in {input_path}")

        # Clean and convert
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df[swe_col] = pd.to_numeric(df[swe_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, swe_col])

        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Resample to the requested period if needed, but here we just save the full series
        # The E2E test will handle period alignment
        # Keep the source unit but make it explicit in the column name so the
        # evaluator never has to guess it from magnitude (NRCS WTEQ is inches).
        out_col = 'swe_mm' if '(mm)' in str(swe_col).lower() else 'swe_in'
        df[out_col] = df[swe_col]

        # Save to preprocessed
        output_dir = self.project_observations_dir / "snow" / "swe" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_swe_processed.csv"

        df[[out_col]].to_csv(output_file)

        # Also save as generic SNOW for easier evaluator lookup
        snow_dir = self.project_observations_dir / "snow" / "preprocessed"
        snow_dir.mkdir(parents=True, exist_ok=True)
        df[[out_col]].to_csv(snow_dir / f"{self.domain_name}_snow_processed.csv")

        self.logger.info(f"SNOTEL processing complete: {output_file}")
        return output_file
