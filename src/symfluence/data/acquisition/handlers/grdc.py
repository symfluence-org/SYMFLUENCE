# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GRDC (Global Runoff Data Centre) Streamflow Data Acquisition Handler

Provides acquisition for discharge data from the Global Runoff Data Centre,
which maintains the world's largest collection of river discharge data.

GRDC features:
- Over 10,000 stations worldwide
- Daily and monthly discharge data
- Long historical records (some since 1800s)
- Quality-controlled data

Data access requires registration at:
https://portal.grdc.bafg.de/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('GRDC')
class GRDCAcquirer(BaseAcquisitionHandler):
    """
    Handles GRDC streamflow data acquisition.

    Downloads daily or monthly discharge data for specified stations
    from the GRDC data portal.

    Configuration:
        GRDC_STATION_IDS: Station ID(s) to download (required)
        GRDC_USERNAME: GRDC portal username (or env var GRDC_USERNAME)
        GRDC_PASSWORD: GRDC portal password (or env var GRDC_PASSWORD)
        GRDC_RESOLUTION: 'daily' or 'monthly' (default: daily)
    """

    PORTAL_URL = "https://portal.grdc.bafg.de"
    API_URL = "https://portal.grdc.bafg.de/KiWebPortal/rest"

    def download(self, output_dir: Path) -> Path:
        """
        Download GRDC streamflow data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data directory
        """
        self.logger.info("Starting GRDC streamflow data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get station IDs
        station_ids = self._get_station_ids()
        if not station_ids:
            raise ValueError(
                "GRDC_STATION_IDS required. Specify station ID(s) in configuration."
            )

        # Get credentials
        username, password = self._get_credentials()

        # Get resolution
        resolution = self._get_config_value(lambda: None, default='daily', dict_key='GRDC_RESOLUTION')
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        downloaded_files = []

        for station_id in station_ids:
            output_file = output_dir / f"grdc_{station_id}_{resolution}.csv"

            if output_file.exists() and not force_download:
                self.logger.info(f"GRDC station {station_id} data already exists")
                downloaded_files.append(output_file)
                continue

            self.logger.info(f"Downloading GRDC station {station_id}")

            try:
                # Try API download first
                if username and password:
                    df = self._download_via_api(station_id, resolution, username, password)
                else:
                    # Try public access for metadata
                    df = self._download_public(station_id, resolution)

                if df is not None and not df.empty:
                    df.to_csv(output_file, index=False)
                    downloaded_files.append(output_file)
                    self.logger.info(f"Downloaded GRDC station {station_id}: {len(df)} records")
                else:
                    self.logger.warning(f"No data retrieved for station {station_id}")

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Failed to download station {station_id}: {e}", exc_info=True)

        if not downloaded_files:
            raise RuntimeError("Failed to download any GRDC data")

        return output_dir

    def _get_station_ids(self) -> List[str]:
        """Get GRDC station IDs from configuration."""
        station_ids = self._get_config_value(lambda: None, default=None, dict_key='GRDC_STATION_IDS')

        if not station_ids:
            return []

        if isinstance(station_ids, str):
            # Could be comma-separated
            return [s.strip() for s in station_ids.split(',')]

        return list(station_ids)

    def _get_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """Get GRDC credentials from environment or config."""
        username = (
            os.environ.get('GRDC_USERNAME') or
            self._get_config_value(lambda: None, default=None, dict_key='GRDC_USERNAME')
        )
        password = (
            os.environ.get('GRDC_PASSWORD') or
            self._get_config_value(lambda: None, default=None, dict_key='GRDC_PASSWORD')
        )
        return username, password

    def _download_via_api(
        self,
        station_id: str,
        resolution: str,
        username: str,
        password: str
    ) -> Optional[pd.DataFrame]:
        """Download data via GRDC REST API."""
        session = requests.Session()

        # Authenticate
        try:
            auth_response = session.post(
                f"{self.API_URL}/login",
                json={'username': username, 'password': password},
                timeout=60
            )
            auth_response.raise_for_status()
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GRDC API authentication failed: {e}", exc_info=True)
            return None

        # Request data
        try:
            params = {
                'stationId': station_id,
                'resolution': resolution,
                'startDate': self.start_date.strftime('%Y-%m-%d'),
                'endDate': self.end_date.strftime('%Y-%m-%d'),
                'format': 'csv'
            }

            response = session.get(
                f"{self.API_URL}/discharge",
                params=params,
                timeout=120
            )
            response.raise_for_status()

            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            return self._standardize_dataframe(df)

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GRDC API download failed: {e}", exc_info=True)
            return None

    def _download_public(
        self,
        station_id: str,
        resolution: str
    ) -> Optional[pd.DataFrame]:
        """
        Download publicly available GRDC data.

        Note: Full data access requires registration. This method
        provides limited access to sample/public data.
        """
        self.logger.info(
            f"Attempting public GRDC data access for station {station_id}. "
            "Note: Full access requires GRDC registration."
        )

        # GRDC provides some data through OGC WFS
        wfs_url = "https://portal.grdc.bafg.de/geoserver/grdc/wfs"

        try:
            params = {
                'service': 'WFS',
                'version': '2.0.0',
                'request': 'GetFeature',
                'typeName': 'grdc:grdc_stations',
                'outputFormat': 'application/json',
                'CQL_FILTER': f"grdc_no='{station_id}'"
            }

            response = requests.get(wfs_url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            if data.get('features'):
                feature = data['features'][0]
                props = feature.get('properties', {})

                self.logger.info(
                    f"Station {station_id}: {props.get('station', 'Unknown')}, "
                    f"River: {props.get('river', 'Unknown')}, "
                    f"Country: {props.get('country', 'Unknown')}"
                )

                # WFS only provides metadata, not time series
                self.logger.warning(
                    "GRDC data download requires registration. "
                    "Please register at https://portal.grdc.bafg.de/ and provide credentials."
                )

            return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"GRDC public access failed: {e}", exc_info=True)
            return None

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize GRDC dataframe to common format."""
        # Common GRDC column names
        column_map = {
            'YYYY-MM-DD': 'date',
            'date': 'date',
            'Date': 'date',
            'calculated': 'discharge_cms',
            'Value': 'discharge_cms',
            'discharge': 'discharge_cms',
            'Original': 'discharge_original',
            'Flag': 'quality_flag',
        }

        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Filter to date range
        if 'date' in df.columns:
            df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]

        return df
