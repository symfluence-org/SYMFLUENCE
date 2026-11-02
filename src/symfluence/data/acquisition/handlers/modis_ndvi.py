# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS NDVI/EVI Data Acquisition Handler

Acquires MOD13A2.061 (Terra) and MYD13A2.061 (Aqua) Vegetation Indices
via NASA earthaccess/CMR (default) or AppEEARS (fallback).

Products provide 16-day composite values at 1km resolution:
- _1_km_16_days_NDVI: Normalized Difference Vegetation Index (-0.2 to 1.0)
- _1_km_16_days_EVI: Enhanced Vegetation Index (-0.2 to 1.0)
- _1_km_16_days_VI_Quality: Quality assessment layer

References:
- MOD13A2: https://lpdaac.usgs.gov/products/mod13a2v061/
- MYD13A2: https://lpdaac.usgs.gov/products/myd13a2v061/

Configuration:
    MODIS_NDVI_PRODUCT: MOD13A2 (default), MYD13A2
    MODIS_NDVI_LAYERS: [NDVI, EVI] (default)
    MODIS_NDVI_QC: True (default) - Apply QC filtering
    MODIS_NDVI_USE_APPEEARS: False (default) - Use earthaccess
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R
from symfluence.data.acquisition.earthaccess_base import BaseEarthaccessAcquirer


@R.acquisition_handlers.add('MODIS_NDVI')
@R.acquisition_handlers.add('MOD13')
class MODISNDVIAcquirer(BaseEarthaccessAcquirer):
    """
    Acquires MODIS NDVI/EVI vegetation index data (MOD13A2/MYD13A2).

    By default uses earthaccess for direct NASA Earthdata Cloud downloads.
    Set MODIS_NDVI_USE_APPEEARS: true for legacy AppEEARS API mode.

    Output:
        {project_dir}/attributes/landcover/modis_ndvi/{domain}_MODIS_NDVI.nc
    """

    PRODUCTS = {
        'MOD13A2': 'MOD13A2.061',  # Terra 16-day 1km
        'MYD13A2': 'MYD13A2.061',  # Aqua 16-day 1km
    }

    DEFAULT_LAYERS = ['NDVI', 'EVI']

    # Vegetation index scaling
    SCALE_FACTOR = 0.0001  # DN to actual VI value (-0.2 to 1.0)
    FILL_VALUE = -3000
    VALID_RANGE = (-2000, 10000)  # DN range

    # QC bit interpretation (bits 0-1: VI quality)
    # 0 = good, 1 = marginal, 2 = snow/ice, 3 = cloudy
    QC_GOOD_VALUES = {0, 1}

    # HDF SDS name mapping
    SDS_NAMES = {
        'NDVI': '1 km 16 days NDVI',
        'EVI': '1 km 16 days EVI',
        'QC': '1 km 16 days VI Quality',
    }

    # AppEEARS URL (for fallback)
    APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    def download(self, output_dir: Path) -> Path:
        """Download MODIS NDVI/EVI products."""
        self.logger.info("Starting MODIS NDVI/EVI acquisition")

        ndvi_dir = self._attribute_dir("landcover") / "modis_ndvi"
        ndvi_dir.mkdir(parents=True, exist_ok=True)

        processed_file = ndvi_dir / f"{self.domain_name}_MODIS_NDVI.nc"

        if processed_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False):
            self.logger.info(f"Using existing MODIS NDVI file: {processed_file}")
            return processed_file

        use_appeears = self._get_config_value(lambda: None, default=False, dict_key='MODIS_NDVI_USE_APPEEARS')

        if use_appeears:
            self.logger.info("Using AppEEARS API (legacy mode)")
            return self._download_via_appeears(ndvi_dir, processed_file)
        else:
            self.logger.info("Using earthaccess/CMR (recommended)")
            return self._download_via_earthaccess(ndvi_dir, processed_file)

    # ===== Earthaccess/CMR Primary Pathway =====

    def _download_via_earthaccess(self, output_dir: Path, processed_file: Path) -> Path:
        """Download and process MODIS NDVI/EVI via earthaccess."""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        product = self._get_config_value(lambda: None, default='MOD13A2', dict_key='MODIS_NDVI_PRODUCT')
        self.logger.info(f"Acquiring {product} via earthaccess")

        product_dir = raw_dir / product.lower()

        granules = self._search_granules_cmr(product, version='061')

        if not granules:
            self.logger.info("No granules with v061, trying v006...")
            granules = self._search_granules_cmr(product, version='006')

        if not granules:
            raise RuntimeError(f"No {product} granules found via CMR")

        urls = self._get_download_urls(granules, extensions=('.hdf',))
        self.logger.info(f"Found {len(urls)} {product} files to download")

        files = self._download_with_earthaccess(urls, product_dir)
        self.logger.info(f"Downloaded {len(files)} {product} files")

        if not files:
            raise RuntimeError(f"No {product} files downloaded")

        self._process_modis_hdf_files(files, processed_file)
        return processed_file

    def _process_modis_hdf_files(self, files: List[Path], output_file: Path):
        """Process downloaded MODIS NDVI/EVI HDF files into NetCDF."""
        from pyhdf.SD import SD, SDC

        self.logger.info(f"Processing {len(files)} MODIS NDVI HDF files...")

        layers = self._get_layers()
        qc_filter = self._get_config_value(lambda: None, default=True, dict_key='MODIS_NDVI_QC')

        results = []
        for i, hdf_path in enumerate(sorted(files)):
            if (i + 1) % 50 == 0:
                self.logger.info(f"  Processing {i+1}/{len(files)}")

            try:
                # Extract date from filename: MOD13A2.A2020001.h10v03.061.*.hdf
                match = re.search(r'\.A(\d{7})\.', hdf_path.name)
                if not match:
                    continue

                year = int(match.group(1)[:4])
                doy = int(match.group(1)[4:])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)

                if date < self.start_date or date > self.end_date:
                    continue

                hdf = SD(str(hdf_path), SDC.READ)
                dataset_names = hdf.datasets().keys()
                record = {'date': date}

                # Read QC data
                qc_data = None
                if qc_filter:
                    qc_sds_name = self._find_sds_name(dataset_names, self.SDS_NAMES['QC'])
                    if qc_sds_name:
                        try:
                            qc_data = hdf.select(qc_sds_name)[:]
                        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError):
                            pass

                # Process NDVI
                if 'NDVI' in layers:
                    ndvi_sds = self._find_sds_name(dataset_names, self.SDS_NAMES['NDVI'])
                    if ndvi_sds:
                        record['ndvi'] = self._extract_layer_mean(
                            hdf, ndvi_sds, qc_data
                        )

                # Process EVI
                if 'EVI' in layers:
                    evi_sds = self._find_sds_name(dataset_names, self.SDS_NAMES['EVI'])
                    if evi_sds:
                        record['evi'] = self._extract_layer_mean(
                            hdf, evi_sds, qc_data
                        )

                hdf.end()

                if 'ndvi' in record or 'evi' in record:
                    results.append(record)

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.debug(f"Error processing {hdf_path.name}: {e}")
                continue

        if not results:
            raise RuntimeError("No valid NDVI/EVI data extracted from HDF files")

        df = pd.DataFrame(results)
        df = df.sort_values('date').drop_duplicates('date', keep='first')
        df = df.set_index('date')

        time_values = pd.to_datetime(df.index).values
        data_vars = {}

        if 'ndvi' in df.columns:
            data_vars['NDVI'] = xr.DataArray(
                df['ndvi'].values, dims=['time'],
                coords={'time': time_values},
                attrs={
                    'long_name': 'Normalized Difference Vegetation Index (domain mean)',
                    'units': 'dimensionless',
                    'valid_range': [-0.2, 1.0],
                }
            )

        if 'evi' in df.columns:
            data_vars['EVI'] = xr.DataArray(
                df['evi'].values, dims=['time'],
                coords={'time': time_values},
                attrs={
                    'long_name': 'Enhanced Vegetation Index (domain mean)',
                    'units': 'dimensionless',
                    'valid_range': [-0.2, 1.0],
                }
            )

        ds = xr.Dataset(data_vars)
        ds.attrs['title'] = 'MODIS Vegetation Indices (NDVI/EVI)'
        product = self._get_config_value(lambda: None, default='MOD13A2', dict_key='MODIS_NDVI_PRODUCT')
        ds.attrs['source'] = f'NASA MODIS {product} v061'
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['domain'] = self.domain_name
        ds.attrs['temporal_resolution'] = '16-day composite'
        ds.attrs['spatial_resolution'] = '1 km'

        ds.to_netcdf(output_file)
        self.logger.info(f"Saved {len(df)} NDVI/EVI records to {output_file}")
        self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        if 'ndvi' in df.columns:
            self.logger.info(f"  NDVI range: {df['ndvi'].min():.3f} - {df['ndvi'].max():.3f}")
        if 'evi' in df.columns:
            self.logger.info(f"  EVI range: {df['evi'].min():.3f} - {df['evi'].max():.3f}")

        # CSV output for observation compatibility
        csv_file = output_file.parent / f"{self.domain_name}_MODIS_NDVI_timeseries.csv"
        df.index.name = 'date'
        df.to_csv(csv_file)
        self.logger.info(f"Created CSV timeseries: {csv_file}")

        return output_file

    def _find_sds_name(self, dataset_names, target: str) -> Optional[str]:
        """Find SDS name matching target in HDF dataset names."""
        for name in dataset_names:
            if target in name:
                return name
        return None

    def _extract_layer_mean(
        self, hdf, sds_name: str, qc_data: Optional[np.ndarray]
    ) -> float:
        """Extract domain spatial mean for a vegetation index layer."""
        sds = hdf.select(sds_name)
        data = sds[:].astype(float)

        # Mask fill values
        data[data == self.FILL_VALUE] = np.nan

        # Apply valid range
        data[(data < self.VALID_RANGE[0]) | (data > self.VALID_RANGE[1])] = np.nan

        # Apply QC filter (bits 0-1)
        if qc_data is not None:
            vi_quality = qc_data & 0b11
            good_quality = np.isin(vi_quality, list(self.QC_GOOD_VALUES))
            data[~good_quality] = np.nan

        # Apply scale factor
        data = data * self.SCALE_FACTOR

        valid_mask = ~np.isnan(data)
        if valid_mask.sum() > 0:
            return float(np.nanmean(data))
        return np.nan

    def _get_layers(self) -> List[str]:
        """Get layers to download."""
        config_layers = self._get_config_value(lambda: None, default=None, dict_key='MODIS_NDVI_LAYERS')
        if config_layers:
            if isinstance(config_layers, str):
                return [config_layers]
            return list(config_layers)
        return list(self.DEFAULT_LAYERS)

    # ===== AppEEARS Fallback =====

    def _download_via_appeears(self, output_dir: Path, processed_file: Path) -> Path:
        """Download via AppEEARS API (legacy fallback)."""
        username, password = self._earthdata_auth().require_credentials(
            "MODIS NDVI via AppEEARS"
        )

        product = self._get_config_value(lambda: None, default='MOD13A2', dict_key='MODIS_NDVI_PRODUCT')
        product_id = self.PRODUCTS.get(product, f"{product}.061")
        layers = self._get_layers()

        appeears_layers = []
        for layer in layers:
            if layer == 'NDVI':
                appeears_layers.append({'product': product_id, 'layer': '_1_km_16_days_NDVI'})
            elif layer == 'EVI':
                appeears_layers.append({'product': product_id, 'layer': '_1_km_16_days_EVI'})

        token = self._appeears_login(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            task_name = f"SYMFLUENCE_{self.domain_name}_NDVI_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            task_request = self._build_appeears_task(token, task_name, appeears_layers)
            task_id = self._submit_appeears_task(token, task_request)
            if not task_id:
                raise RuntimeError("Failed to submit AppEEARS task")

            if not self._wait_for_appeears_task(token, task_id):
                raise RuntimeError(f"AppEEARS task {task_id} did not complete")

            self._download_appeears_results(token, task_id, output_dir)
        finally:
            self._appeears_logout(token)

        self.logger.info(f"MODIS NDVI AppEEARS download complete: {output_dir}")
        return output_dir

    def _appeears_login(self, username: str, password: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/login",
                auth=(username, password), timeout=60
            )
            response.raise_for_status()
            return response.json().get('token')
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"AppEEARS login failed: {e}")
            return None

    def _appeears_logout(self, token: str):
        try:
            requests.post(
                f"{self.APPEEARS_BASE}/logout",
                headers={"Authorization": f"Bearer {token}"}, timeout=30
            )
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError):
            pass

    def _build_appeears_task(self, token: str, task_name: str, layers: list) -> dict:
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])
        return {
            'task_type': 'area',
            'task_name': task_name,
            'params': {
                'dates': [{
                    'startDate': self.start_date.strftime('%m-%d-%Y'),
                    'endDate': self.end_date.strftime('%m-%d-%Y'),
                }],
                'layers': layers,
                'geo': {
                    'type': 'FeatureCollection',
                    'features': [{
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [[
                                [lon_min, lat_min], [lon_max, lat_min],
                                [lon_max, lat_max], [lon_min, lat_max],
                                [lon_min, lat_min]
                            ]],
                        },
                        'properties': {}
                    }]
                },
                'output': {
                    'format': {'type': 'netcdf4'},
                    'projection': 'geographic',
                },
            }
        }

    def _submit_appeears_task(self, token: str, task_request: dict) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/task",
                headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
                json=task_request, timeout=120
            )
            response.raise_for_status()
            task_id = response.json().get('task_id')
            self.logger.info(f"Submitted AppEEARS task: {task_id}")
            return task_id
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            return None

    def _wait_for_appeears_task(self, token: str, task_id: str, timeout: int = 7200) -> bool:
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                response = requests.get(
                    f"{self.APPEEARS_BASE}/task/{task_id}",
                    headers={'Authorization': f'Bearer {token}'}, timeout=60
                )
                response.raise_for_status()
                status = response.json().get('status')
                if status == 'done':
                    return True
                elif status in ('error', 'expired', 'failed'):
                    return False
                time.sleep(30)
            except requests.RequestException:
                time.sleep(30)
        return False

    def _download_appeears_results(self, token: str, task_id: str, output_dir: Path):
        headers = {'Authorization': f'Bearer {token}'}
        try:
            response = requests.get(
                f"{self.APPEEARS_BASE}/bundle/{task_id}",
                headers=headers, timeout=60
            )
            response.raise_for_status()
            files = response.json().get('files', [])
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Failed to get bundle info: {e}")
            return

        for file_info in files:
            file_id = file_info.get('file_id')
            file_name = file_info.get('file_name')
            if not file_name.endswith(('.nc', '.nc4', '.tif')):
                continue
            try:
                resp = requests.get(
                    f"{self.APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers=headers, stream=True, timeout=300
                )
                resp.raise_for_status()
                with open(output_dir / file_name, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Failed to download {file_name}: {e}")
