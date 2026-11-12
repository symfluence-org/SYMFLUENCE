# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
AppEEARS Base Class for MODIS Data Acquisition.

This module provides a base class with shared NASA AppEEARS API functionality
used by MODIS MOD16 (ET), MOD10A1/MYD10A1 (SCA), and other AppEEARS-based handlers.

AppEEARS: https://appeears.earthdatacloud.nasa.gov/
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import xarray as xr

from .base import BaseAcquisitionHandler


class BaseAppEEARSAcquirer(BaseAcquisitionHandler):
    """
    Base class providing NASA AppEEARS API functionality for MODIS data acquisition.

    This base class handles:
    - Earthdata credentials via the inherited ``_get_earthdata_credentials``
    - AppEEARS API authentication (login/logout)
    - Task submission and status polling
    - Result download from completed tasks
    - NetCDF file consolidation

    Subclasses should implement:
    - download(): Main entry point that orchestrates the acquisition workflow
    - _submit_appeears_task(): Product-specific task submission with appropriate layers
    - Product-specific processing logic
    """

    APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    # ===== Authentication =====

    def _appeears_login(self, username: str, password: str) -> Optional[str]:
        """
        Login to AppEEARS and get authentication token.

        Args:
            username: NASA Earthdata username
            password: NASA Earthdata password

        Returns:
            Authentication token, or None if login failed
        """
        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/login",
                auth=(username, password),
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('token')
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError) as e:
            self.logger.error(f"AppEEARS login failed: {e}")
            return None

    def _appeears_logout(self, token: str) -> None:
        """
        Logout from AppEEARS (invalidate token).

        Args:
            token: Authentication token to invalidate
        """
        try:
            requests.post(
                f"{self.APPEEARS_BASE}/logout",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
        except (requests.RequestException, OSError):
            pass  # Logout failures are not critical

    # ===== Task Management =====

    def _wait_for_task(self, token: str, task_id: str, timeout_hours: float = 6) -> bool:
        """
        Wait for an AppEEARS task to complete.

        Args:
            token: Authentication token
            task_id: Task ID to monitor
            timeout_hours: Maximum time to wait (default: 6 hours)

        Returns:
            True if task completed successfully, False if failed or timed out
        """
        self.logger.info(f"Waiting for AppEEARS task {task_id} to complete...")

        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        poll_interval = 30  # seconds

        while (time.time() - start_time) < timeout_seconds:
            try:
                response = requests.get(
                    f"{self.APPEEARS_BASE}/task/{task_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=60
                )
                response.raise_for_status()
                status = response.json()

                task_status = status.get('status', '')
                self.logger.debug(f"Task {task_id} status: {task_status}")

                if task_status == 'done':
                    self.logger.info(f"Task {task_id} completed successfully")
                    return True
                elif task_status in ['error', 'failed']:
                    self.logger.error(f"Task {task_id} failed: {status.get('error', 'Unknown error')}")
                    return False

                time.sleep(poll_interval)

            except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
                self.logger.warning(f"Error checking task status: {e}")
                time.sleep(poll_interval)

        self.logger.error(f"Task {task_id} timed out after {timeout_hours} hours")
        return False

    def _download_task_results(
        self,
        token: str,
        task_id: str,
        output_dir: Path,
        prefix: str
    ) -> List[Path]:
        """
        Download results from a completed AppEEARS task.

        Args:
            token: Authentication token
            task_id: Completed task ID
            output_dir: Directory to save downloaded files
            prefix: Prefix for output filenames

        Returns:
            List of downloaded file paths
        """
        downloaded_files = []

        try:
            # Get file list from task bundle
            response = requests.get(
                f"{self.APPEEARS_BASE}/bundle/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60
            )
            response.raise_for_status()
            files = response.json().get('files', [])

            # Download each NetCDF file
            for file_info in files:
                file_name = file_info.get('file_name', '')
                file_id = file_info.get('file_id')

                # Only download NetCDF files
                if not file_name.endswith('.nc'):
                    continue

                out_path = output_dir / f"{prefix}_{file_name}"

                # Skip if already exists
                if out_path.exists() and not self._get_config_value(
                    lambda: self.config.data.force_download,
                    dict_key='FORCE_DOWNLOAD',
                ):
                    downloaded_files.append(out_path)
                    continue

                self.logger.info(f"Downloading: {file_name}")

                dl_response = requests.get(
                    f"{self.APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    stream=True,
                    timeout=600
                )
                dl_response.raise_for_status()

                with open(out_path, 'wb') as f:
                    for chunk in dl_response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)

                downloaded_files.append(out_path)

        except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            self.logger.error(f"Failed to download task results: {e}")
            raise

        return downloaded_files

    def _consolidate_appeears_output(
        self,
        output_dir: Path,
        prefix: str,
        output_file: Path
    ) -> None:
        """
        Consolidate multiple AppEEARS output files into a single NetCDF.

        Args:
            output_dir: Directory containing downloaded files
            prefix: Filename prefix to match (e.g., product name)
            output_file: Path for consolidated output file
        """
        nc_files = list(output_dir.glob(f"{prefix}_*.nc"))

        if not nc_files:
            self.logger.warning(f"No NetCDF files found for {prefix}")
            return

        if len(nc_files) == 1:
            shutil.copy(nc_files[0], output_file)
            return

        # Merge multiple files along time dimension
        try:
            datasets = [xr.open_dataset(f) for f in sorted(nc_files)]
            merged = xr.concat(datasets, dim='time')
            merged = merged.sortby('time')
            merged.to_netcdf(output_file)

            for ds in datasets:
                ds.close()

            self.logger.info(f"Consolidated {len(nc_files)} files into {output_file}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            self.logger.error(f"Failed to consolidate files: {e}")
            # Fall back to using first file
            if nc_files:
                shutil.copy(nc_files[0], output_file)

    # ===== Helper Methods =====

    def _bbox_to_geojson_polygon(self) -> Dict:
        """
        Convert bounding box to GeoJSON polygon format for AppEEARS requests.

        Returns:
            GeoJSON FeatureCollection with polygon geometry
        """
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        coordinates = [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min]
        ]]

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {}
            }]
        }

    def _format_dates_for_appeears(self) -> Tuple[str, str]:
        """
        Format start and end dates for AppEEARS API (MM-DD-YYYY format).

        Returns:
            Tuple of (start_date, end_date) formatted strings
        """
        start_date = self.start_date.strftime("%m-%d-%Y")
        end_date = self.end_date.strftime("%m-%d-%Y")
        return start_date, end_date

    def _build_area_task_request(
        self,
        task_name: str,
        layers: List[Dict[str, str]],
        output_format: str = "netcdf4",
        projection: str = "geographic"
    ) -> Dict[str, Any]:
        """
        Build a standard AppEEARS area task request.

        Args:
            task_name: Name for the task
            layers: List of layer dicts with 'product' and 'layer' keys
            output_format: Output format ('netcdf4', 'geotiff')
            projection: Output projection ('geographic', 'native')

        Returns:
            Task request dictionary ready for submission
        """
        start_date, end_date = self._format_dates_for_appeears()

        return {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": start_date,
                    "endDate": end_date
                }],
                "layers": layers,
                "geo": self._bbox_to_geojson_polygon(),
                "output": {
                    "format": {
                        "type": output_format
                    },
                    "projection": projection
                }
            }
        }

    def _submit_task_request(self, token: str, task_request: Dict[str, Any]) -> Optional[str]:
        """
        Submit a task request to AppEEARS.

        Args:
            token: Authentication token
            task_request: Task request dictionary

        Returns:
            Task ID if successful, None if failed
        """
        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/task",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=task_request,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            task_id = result.get('task_id')
            self.logger.info(f"Submitted AppEEARS task: {task_id}")
            return task_id
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response: {e.response.text[:500]}")
            return None
