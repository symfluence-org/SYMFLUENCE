# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
OpenET Data Acquisition Handler

Provides acquisition for OpenET ensemble evapotranspiration data.
OpenET is a collaborative project providing field-scale ET estimates
across the Western US using an ensemble of 6 ET models.

OpenET features:
- 30m spatial resolution (Landsat-based)
- Monthly and daily temporal resolution
- Ensemble of 6 ET models (ALEXI, PT-JPL, SIMS, geeSEBAL, eeMETRIC, SSEBop)
- Coverage: Western United States
- Period: 2016-present

API access: https://openetdata.org/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('OPENET')
@R.acquisition_handlers.add('OpenET')
class OpenETAcquirer(BaseAcquisitionHandler):
    """
    Handles OpenET data acquisition via OpenET API.

    Downloads field-scale evapotranspiration data for specified
    regions within the Western United States.

    Configuration:
        OPENET_API_KEY: API key for OpenET (or env var OPENET_API_KEY)
        OPENET_MODEL: Model to use ('ensemble', 'ALEXI', 'eeMETRIC', etc.)
        OPENET_RESOLUTION: 'monthly' or 'daily' (default: monthly)
        OPENET_UNITS: 'mm' or 'in' (default: mm)
    """

    API_BASE = "https://openet-api.org"

    AVAILABLE_MODELS = [
        'ensemble',
        'ALEXI',
        'PT-JPL',
        'SIMS',
        'geeSEBAL',
        'eeMETRIC',
        'SSEBop',
    ]

    def download(self, output_dir: Path) -> Path:
        """
        Download OpenET evapotranspiration data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        self.logger.info("Starting OpenET data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(
                "OpenET API key required. Set OPENET_API_KEY environment variable "
                "or config setting. Register at https://openetdata.org/"
            )

        # Check bounding box is in Western US
        if self.bbox:
            if self.bbox['lon_max'] > -100:
                self.logger.warning(
                    "OpenET coverage is limited to Western United States. "
                    "Data may not be available for this region."
                )

        # Get configuration
        model = self._get_config_value(lambda: None, default='ensemble', dict_key='OPENET_MODEL')
        resolution = self._get_config_value(lambda: None, default='monthly', dict_key='OPENET_RESOLUTION')
        units = self._get_config_value(lambda: None, default='mm', dict_key='OPENET_UNITS')
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        # Build output filename
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        output_file = output_dir / f"openet_{model}_{start_str}_{end_str}_{resolution}.csv"

        if output_file.exists() and not force_download:
            self.logger.info(f"OpenET file already exists: {output_file}")
            return output_file

        # Request data via API
        df = self._request_timeseries(api_key, model, resolution, units)

        if df is not None and not df.empty:
            df.to_csv(output_file, index=False)
            self.logger.info(f"OpenET download complete: {len(df)} records")
        else:
            raise RuntimeError("Failed to retrieve OpenET data")

        return output_file

    def _get_api_key(self) -> Optional[str]:
        """Get OpenET API key from environment or config."""
        return (
            os.environ.get('OPENET_API_KEY') or
            self._get_config_value(lambda: None, default=None, dict_key='OPENET_API_KEY')
        )

    def _request_timeseries(
        self,
        api_key: str,
        model: str,
        resolution: str,
        units: str
    ) -> Optional[pd.DataFrame]:
        """Request time series data from OpenET API."""
        headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json',
        }

        # Build request payload
        payload = {
            'date_range': [
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d'),
            ],
            'interval': resolution,
            'model': model,
            'units': units,
            'file_format': 'csv',
        }

        # Add geometry
        if self.bbox:
            payload['geometry'] = {
                'type': 'Polygon',
                'coordinates': [[
                    [self.bbox['lon_min'], self.bbox['lat_min']],
                    [self.bbox['lon_max'], self.bbox['lat_min']],
                    [self.bbox['lon_max'], self.bbox['lat_max']],
                    [self.bbox['lon_min'], self.bbox['lat_max']],
                    [self.bbox['lon_min'], self.bbox['lat_min']],
                ]]
            }

        try:
            # Submit request
            self.logger.info(f"Requesting OpenET {model} data ({resolution})")

            response = requests.post(
                f"{self.API_BASE}/raster/timeseries/polygon",
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()

            # Parse response
            data = response.json()

            if 'error' in data:
                self.logger.error(f"OpenET API error: {data['error']}")
                return None

            # Convert to DataFrame
            records = []
            for item in data.get('data', []):
                records.append({
                    'date': item.get('date') or item.get('time'),
                    'et_mm': item.get('et') or item.get('value'),
                })

            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                return df

            return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                self.logger.error("OpenET API authentication failed. Check API key.")
            elif e.response.status_code == 429:
                self.logger.error("OpenET API rate limit exceeded.")
            else:
                self.logger.error(f"OpenET API error: {e}")
            return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"OpenET request failed: {e}", exc_info=True)
            return None
