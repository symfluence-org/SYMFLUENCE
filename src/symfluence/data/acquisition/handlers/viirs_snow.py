# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIIRS Snow Cover Data Acquisition Handler

Provides acquisition for VIIRS (Visible Infrared Imaging Radiometer Suite)
snow cover products. VIIRS is the successor to MODIS, providing improved
snow detection capabilities.

VIIRS Snow features:
- 375m (I-band) to 750m (M-band) spatial resolution
- Daily temporal resolution
- Improved cloud masking compared to MODIS
- Available from Suomi NPP (2012) and NOAA-20 (2018)
- Products: VNP10A1F (daily cloud-gap-filled), VNP10A1 (daily)

Default method: earthaccess/CMR (fast, direct download)
Fallback method: AppEEARS (slower, queue-based)

References:
- VNP10A1F: https://nsidc.org/data/vnp10a1f
- earthaccess: https://github.com/nsidc/earthaccess
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import xarray as xr

from symfluence.core.registries import R
from symfluence.data.acquisition.earthaccess_base import BaseEarthaccessAcquirer
from symfluence.data.acquisition.earthdata_auth import missing_credentials_message

APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"


@R.acquisition_handlers.add('VIIRS_SNOW')
@R.acquisition_handlers.add('VNP10')
class VIIRSSnowAcquirer(BaseEarthaccessAcquirer):
    """
    Handles VIIRS snow cover data acquisition.

    Uses earthaccess/CMR for direct downloads (faster than AppEEARS).

    Configuration:
        VIIRS_SNOW_PRODUCT: Product ID ('VNP10A1F', 'VNP10A1', default: VNP10A1F)
        VIIRS_SNOW_LAYERS: Layers to download (default: CGF_NDSI_Snow_Cover)
        VIIRS_SNOW_USE_APPEEARS: False (default) - set True to use AppEEARS
    """

    PRODUCTS = {
        'VNP10A1F': 'VNP10A1F',   # Daily cloud-gap-filled (NPP) - recommended
        'VNP10A1': 'VNP10A1',     # Daily (NPP)
        'VNP10A2F': 'VNP10A2F',   # 8-day composite (NPP)
        'VJ110A1F': 'VJ110A1F',   # Daily cloud-gap-filled (NOAA-20)
    }

    # VIIRS snow value interpretation (same as MODIS)
    VALID_SNOW_RANGE = (0, 100)
    CLOUD_VALUE = 250
    MISSING_VALUES = {200, 201, 211, 237, 239, 250, 254, 255}

    def download(self, output_dir: Path) -> Path:
        """
        Download VIIRS snow cover data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded/processed data
        """
        self.logger.info("Starting VIIRS snow cover data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing processed file
        output_file = output_dir / f"{self.domain_name}_VIIRS_snow.nc"
        force_download = self._get_config_value(lambda: self.config.data.force_download, default=False)

        if output_file.exists() and not force_download:
            self.logger.info(f"Using existing VIIRS snow file: {output_file}")
            return output_file

        # Get credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            raise ValueError(missing_credentials_message("VIIRS snow"))

        # Get product configuration
        product = self._get_config_value(lambda: None, default='VNP10A1F', dict_key='VIIRS_SNOW_PRODUCT')
        product_name = self.PRODUCTS.get(product, product)

        # Check if user wants AppEEARS mode
        use_appeears = self._get_config_value(lambda: None, default=False, dict_key='VIIRS_SNOW_USE_APPEEARS')

        if use_appeears:
            # Legacy AppEEARS mode
            self.logger.info("Using AppEEARS mode (legacy)")
            return self._download_via_appeears(output_dir, product_name, username, password)
        else:
            # Default: earthaccess/CMR mode
            self.logger.info("Using earthaccess/CMR mode (default)")
            try:
                return self._download_via_earthaccess(output_dir, product_name, output_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"earthaccess failed: {e}, trying AppEEARS fallback", exc_info=True)
                return self._download_via_appeears(output_dir, product_name, username, password)

    def _download_via_earthaccess(
        self,
        output_dir: Path,
        product: str,
        output_file: Path
    ) -> Path:
        """Download VIIRS data via earthaccess/CMR."""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Search and download granules (no version filter for VIIRS)
        files = self._download_granules_earthaccess(
            short_name=product,
            output_dir=raw_dir,
            version=None,  # VIIRS doesn't need version in CMR query
            extensions=('.h5', '.hdf')
        )

        if not files:
            raise RuntimeError(f"No {product} files downloaded")

        # Process H5 files into NetCDF
        self._process_viirs_h5_files(files, output_file)

        self.logger.info(f"VIIRS snow acquisition complete: {output_file}")
        return output_file

    def _process_viirs_h5_files(self, h5_files: List[Path], output_file: Path):
        """Process downloaded VIIRS H5 files into a single NetCDF."""
        import h5py

        self.logger.info(f"Processing {len(h5_files)} VIIRS H5 files")

        records = []
        for h5_path in sorted(h5_files):
            try:
                # Extract date from filename (e.g., VNP10A1F.A2017001.h10v03...)
                match = re.search(r'\.A(\d{7})\.', h5_path.name)
                if not match:
                    continue

                year = int(match.group(1)[:4])
                doy = int(match.group(1)[4:])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)

                with h5py.File(h5_path, 'r') as f:
                    # Navigate VIIRS HDF5 structure
                    grids = f.get('HDFEOS/GRIDS')
                    if grids is None:
                        continue

                    # Find grid with snow data
                    grid_name = None
                    for name in grids.keys():
                        if 'Grid' in name or 'IMG' in name:
                            grid_name = name
                            break

                    if grid_name is None:
                        continue

                    data_fields = grids[grid_name].get('Data Fields')
                    if data_fields is None:
                        continue

                    # Get NDSI snow cover
                    sca_var = None
                    for var_name in ['CGF_NDSI_Snow_Cover', 'NDSI_Snow_Cover', 'NDSI']:
                        if var_name in data_fields:
                            sca_var = var_name
                            break

                    if sca_var is None:
                        continue

                    data = data_fields[sca_var][:]

                    # Apply valid range mask
                    data = data.astype(float)
                    for mv in self.MISSING_VALUES:
                        data = np.where(data == mv, np.nan, data)

                    records.append({'date': date, 'data': data})

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Error processing {h5_path.name}: {e}", exc_info=True)
                continue

        if not records:
            raise RuntimeError("No valid data extracted from VIIRS files")

        # Aggregate by date (may have multiple tiles)
        from collections import defaultdict
        daily_data = defaultdict(list)
        for r in records:
            daily_data[r['date']].append(r['data'])

        # Create daily means
        final_records = []
        for date in sorted(daily_data.keys()):
            arrays = daily_data[date]
            if len(arrays) == 1:
                final_records.append({'date': date, 'data': arrays[0]})
            else:
                # Average multiple tiles
                stacked = np.stack(arrays, axis=0)
                final_records.append({'date': date, 'data': np.nanmean(stacked, axis=0)})

        # Create xarray dataset
        times = [r['date'] for r in final_records]
        data_stack = np.stack([r['data'] for r in final_records], axis=0)

        da = xr.DataArray(
            data_stack,
            dims=['time', 'y', 'x'],
            coords={'time': times},
            name='NDSI_Snow_Cover',
            attrs={
                'long_name': 'VIIRS NDSI Snow Cover',
                'units': 'percent',
                'valid_range': [0, 100],
                'source': 'VIIRS via earthaccess'
            }
        )

        ds_out = xr.Dataset({'NDSI_Snow_Cover': da})
        ds_out.attrs['title'] = 'VIIRS Snow Cover'
        ds_out.attrs['created'] = datetime.now().isoformat()
        ds_out.to_netcdf(output_file)

        self.logger.info(f"Processed {len(final_records)} timesteps to {output_file}")

    # ===== AppEEARS Fallback Methods =====

    def _download_via_appeears(
        self,
        output_dir: Path,
        product: str,
        username: str,
        password: str
    ) -> Path:
        """Download VIIRS data via AppEEARS (fallback)."""
        self.logger.info(f"Downloading {product} via AppEEARS")

        # Get token
        token = self._get_appeears_token(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        # Get layers
        layers = self._get_config_value(lambda: None, default=['CGF_NDSI_Snow_Cover'], dict_key='VIIRS_SNOW_LAYERS')
        if isinstance(layers, str):
            layers = [layers]

        product_id = f"{product}.002"

        # Submit task
        task_id = self._submit_appeears_task(token, product_id, layers)
        if not task_id:
            raise RuntimeError("Failed to submit AppEEARS task")

        # Wait and download
        self._wait_and_download(token, task_id, output_dir)

        self.logger.info(f"VIIRS AppEEARS download complete: {output_dir}")
        return output_dir

    def _get_appeears_token(self, username: str, password: str) -> Optional[str]:
        """Authenticate with AppEEARS."""
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

    def _submit_appeears_task(
        self,
        token: str,
        product_id: str,
        layers: List[str]
    ) -> Optional[str]:
        """Submit AppEEARS task."""
        headers = {'Authorization': f'Bearer {token}'}
        layer_specs = [{'product': product_id, 'layer': layer} for layer in layers]

        task_request = {
            'task_type': 'area',
            'task_name': f"VIIRS_Snow_{self.domain_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'params': {
                'dates': [{
                    'startDate': self.start_date.strftime('%m-%d-%Y'),
                    'endDate': self.end_date.strftime('%m-%d-%Y'),
                }],
                'layers': layer_specs,
                'output': {'format': {'type': 'netcdf4'}, 'projection': 'geographic'},
                'geo': {
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
                },
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
            task_id = response.json().get('task_id')
            self.logger.info(f"AppEEARS task submitted: {task_id}")
            return task_id
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to submit task: {e}", exc_info=True)
            return None

    def _wait_and_download(
        self,
        token: str,
        task_id: str,
        output_dir: Path,
        max_wait: int = 7200
    ):
        """Wait for AppEEARS task and download results."""
        headers = {'Authorization': f'Bearer {token}'}
        elapsed = 0

        while elapsed < max_wait:
            try:
                response = requests.get(
                    f"{APPEEARS_BASE}/task/{task_id}",
                    headers=headers,
                    timeout=60
                )
                status = response.json().get('status')
                self.logger.info(f"Task status: {status}")

                if status == 'done':
                    self._download_appeears_results(token, task_id, output_dir)
                    return
                elif status in ['error', 'expired']:
                    raise RuntimeError(f"Task failed: {status}")

                time.sleep(30)
                elapsed += 30
            except requests.RequestException as e:
                self.logger.warning(f"Status check error: {e}")
                time.sleep(30)
                elapsed += 30

        raise RuntimeError(f"Task timed out after {max_wait}s")

    def _download_appeears_results(self, token: str, task_id: str, output_dir: Path):
        """Download AppEEARS results."""
        headers = {'Authorization': f'Bearer {token}'}

        response = requests.get(f"{APPEEARS_BASE}/bundle/{task_id}", headers=headers, timeout=60)
        files = response.json().get('files', [])

        for file_info in files:
            file_name = file_info.get('file_name')
            file_id = file_info.get('file_id')

            if not file_name.endswith(('.nc', '.nc4', '.tif')):
                continue

            output_path = output_dir / file_name
            try:
                dl_response = requests.get(
                    f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers=headers,
                    stream=True,
                    timeout=300
                )
                with open(output_path, 'wb') as f:
                    for chunk in dl_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.logger.debug(f"Downloaded: {file_name}")
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Download failed: {file_name}: {e}", exc_info=True)
