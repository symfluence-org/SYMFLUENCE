# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS Land Surface Temperature (LST) Data Acquisition Handler

Provides acquisition for MODIS MOD11A1/MYD11A1 (daily) and MOD11A2/MYD11A2
(8-day composite) Land Surface Temperature products via NASA AppEEARS API.

MODIS LST features:
- 1 km spatial resolution
- Daily (MOD11A1/MYD11A1) or 8-day composite (MOD11A2/MYD11A2)
- Day and night temperature retrievals
- Global coverage
- Terra (MOD) since 2000, Aqua (MYD) since 2002
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..earthdata_auth import missing_credentials_message

# AppEEARS API endpoints
APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"


@R.acquisition_handlers.add('MODIS_LST')
@R.acquisition_handlers.add('MOD11')
class MODISLSTAcquirer(BaseAcquisitionHandler):
    """
    Handles MODIS LST data acquisition via NASA AppEEARS.

    Downloads daily or 8-day composite land surface temperature data
    for specified spatial extent and time period.

    Configuration:
        MODIS_LST_PRODUCT: Product ID ('MOD11A1', 'MOD11A2', 'MYD11A1', etc.)
        MODIS_LST_LAYERS: Layers to download (default: LST_Day_1km, LST_Night_1km)
        MODIS_LST_QC: Include QC layers (default: True)
    """

    PRODUCTS = {
        'MOD11A1': 'MOD11A1.061',  # Terra daily
        'MYD11A1': 'MYD11A1.061',  # Aqua daily
        'MOD11A2': 'MOD11A2.061',  # Terra 8-day
        'MYD11A2': 'MYD11A2.061',  # Aqua 8-day
    }

    DEFAULT_LAYERS = [
        'LST_Day_1km',
        'LST_Night_1km',
        'QC_Day',
        'QC_Night',
        'Day_view_time',
        'Night_view_time',
    ]

    def download(self, output_dir: Path) -> Path:
        """
        Download MODIS LST data via AppEEARS.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        self.logger.info("Starting MODIS LST data acquisition via AppEEARS")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing data
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        existing_files = list(output_dir.glob("*LST*.nc")) + list(output_dir.glob("*LST*.tif"))
        if existing_files and not force_download:
            self.logger.info(f"MODIS LST data already exists: {len(existing_files)} files")
            return output_dir

        # Get credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            raise ValueError(missing_credentials_message("MODIS LST"))

        # Get configuration
        product = self._get_config_value(lambda: None, default='MOD11A1', dict_key='MODIS_LST_PRODUCT')
        product_id = self.PRODUCTS.get(product, f"{product}.061")
        layers = self._get_layers()

        # Authenticate
        token = self._get_appeears_token(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        # Submit task
        task_id = self._submit_task(token, product_id, layers)
        if not task_id:
            raise RuntimeError("Failed to submit AppEEARS task")

        # Wait for completion and download
        self._wait_and_download(token, task_id, output_dir)

        self.logger.info(f"MODIS LST download complete: {output_dir}")
        return output_dir

    def _get_layers(self) -> List[str]:
        """Get layers to download."""
        config_layers = self._get_config_value(lambda: None, default=None, dict_key='MODIS_LST_LAYERS')
        if config_layers:
            if isinstance(config_layers, str):
                return [config_layers]
            return list(config_layers)

        # Default layers
        layers = ['LST_Day_1km', 'LST_Night_1km']
        if self._get_config_value(lambda: None, default=True, dict_key='MODIS_LST_QC'):
            layers.extend(['QC_Day', 'QC_Night'])

        return layers

    def _get_appeears_token(self, username: str, password: str) -> Optional[str]:
        """Authenticate with AppEEARS and get token."""
        try:
            response = requests.post(
                f"{APPEEARS_BASE}/login",
                auth=(username, password),
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('token')
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"AppEEARS authentication failed: {e}", exc_info=True)
            return None

    def _submit_task(
        self,
        token: str,
        product_id: str,
        layers: List[str]
    ) -> Optional[str]:
        """Submit AppEEARS task request."""
        headers = {'Authorization': f'Bearer {token}'}

        # Build layer specifications
        layer_specs = []
        for layer in layers:
            layer_specs.append({
                'product': product_id,
                'layer': layer,
            })

        # Build task request
        task_name = f"MODIS_LST_{self.domain_name}_{self.start_date.strftime('%Y%m%d')}"

        task_request = {
            'task_type': 'area',
            'task_name': task_name,
            'params': {
                'dates': [{
                    'startDate': self.start_date.strftime('%m-%d-%Y'),
                    'endDate': self.end_date.strftime('%m-%d-%Y'),
                }],
                'layers': layer_specs,
                'output': {
                    'format': {'type': 'netcdf4'},
                    'projection': 'geographic',
                },
                'geo': self._build_geo_spec(),
            }
        }

        try:
            response = requests.post(
                f"{APPEEARS_BASE}/task",
                headers=headers,
                json=task_request,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            task_id = result.get('task_id')
            self.logger.info(f"AppEEARS task submitted: {task_id}")
            return task_id
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to submit AppEEARS task: {e}", exc_info=True)
            return None

    def _build_geo_spec(self) -> Dict[str, Any]:
        """Build geographic specification for task."""
        if self.bbox:
            return {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [self.bbox['lon_min'], self.bbox['lat_min']],
                            [self.bbox['lon_max'], self.bbox['lat_min']],
                            [self.bbox['lon_max'], self.bbox['lat_max']],
                            [self.bbox['lon_min'], self.bbox['lat_max']],
                            [self.bbox['lon_min'], self.bbox['lat_min']],
                        ]]
                    },
                    'properties': {}
                }]
            }
        else:
            raise ValueError("Bounding box required for MODIS LST acquisition")

    def _wait_and_download(
        self,
        token: str,
        task_id: str,
        output_dir: Path,
        max_wait: int = 7200,  # 2 hours
        poll_interval: int = 30
    ):
        """Wait for task completion and download results."""
        headers = {'Authorization': f'Bearer {token}'}
        elapsed = 0

        while elapsed < max_wait:
            try:
                response = requests.get(
                    f"{APPEEARS_BASE}/task/{task_id}",
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
                status = response.json()

                task_status = status.get('status')
                self.logger.info(f"AppEEARS task status: {task_status}")

                if task_status == 'done':
                    # Download files
                    self._download_results(token, task_id, output_dir)
                    return
                elif task_status in ['error', 'expired']:
                    raise RuntimeError(f"AppEEARS task failed: {status.get('error', 'Unknown error')}")

                time.sleep(poll_interval)
                elapsed += poll_interval

            except requests.RequestException as e:
                self.logger.warning(f"Error checking task status: {e}")
                time.sleep(poll_interval)
                elapsed += poll_interval

        raise RuntimeError(f"AppEEARS task timed out after {max_wait} seconds")

    def _download_results(self, token: str, task_id: str, output_dir: Path):
        """Download completed task results."""
        headers = {'Authorization': f'Bearer {token}'}

        # Get file list
        try:
            response = requests.get(
                f"{APPEEARS_BASE}/bundle/{task_id}",
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            bundle = response.json()
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to get bundle info: {e}", exc_info=True)
            return

        files = bundle.get('files', [])
        self.logger.info(f"Downloading {len(files)} files from AppEEARS")

        for file_info in files:
            file_id = file_info.get('file_id')
            file_name = file_info.get('file_name')

            if not file_name.endswith(('.nc', '.nc4', '.tif')):
                continue

            output_path = output_dir / file_name

            try:
                response = requests.get(
                    f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers=headers,
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.debug(f"Downloaded: {file_name}")

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to download {file_name}: {e}", exc_info=True)
