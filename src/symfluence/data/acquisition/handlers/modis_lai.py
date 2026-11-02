# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS LAI/FPAR Data Acquisition Handler

Acquires MCD15A2H (combined Terra+Aqua), MOD15A2H (Terra), and MYD15A2H (Aqua)
Leaf Area Index (LAI) and Fraction of Photosynthetically Active Radiation (FPAR)
products via NASA earthaccess/CMR (default) or AppEEARS (fallback).

References:
- MCD15A2H: https://lpdaac.usgs.gov/products/mcd15a2hv061/
- MOD15A2H: https://lpdaac.usgs.gov/products/mod15a2hv061/
- earthaccess: https://github.com/nsidc/earthaccess

Products provide 8-day composite values at 500m resolution:
- Lai_500m: Leaf Area Index (m²/m² after scaling by 0.1)
- Fpar_500m: Fraction of PAR (0-1 after scaling by 0.01)
- FparLai_QC: Quality Control flags

Configuration:
    MODIS_LAI_USE_APPEEARS: False (default) - Use earthaccess; True = use AppEEARS
    MODIS_LAI_PRODUCT: MCD15A2H (default), MOD15A2H, MYD15A2H
    MODIS_LAI_LAYERS: [Lai_500m, Fpar_500m] (default)
    MODIS_LAI_QC: True (default) - Apply QC filtering
    MODIS_LAI_CONVERT_TO_DAILY: True (default) - Interpolate 8-day to daily
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


@R.acquisition_handlers.add('MODIS_LAI')
@R.acquisition_handlers.add('MCD15')
class MODISLAIAcquirer(BaseEarthaccessAcquirer):
    """
    Acquires MODIS LAI/FPAR data (MCD15A2H / MOD15A2H / MYD15A2H).

    By default uses earthaccess for direct NASA Earthdata Cloud downloads.
    Set MODIS_LAI_USE_APPEEARS: true for legacy AppEEARS API mode.
    """

    PRODUCTS = {
        'MCD15A2H': 'MCD15A2H.061',  # Combined Terra+Aqua
        'MOD15A2H': 'MOD15A2H.061',  # Terra only
        'MYD15A2H': 'MYD15A2H.061',  # Aqua only
    }

    DEFAULT_LAYERS = ['Lai_500m', 'Fpar_500m', 'FparLai_QC']

    # LAI/FPAR scaling and QC constants
    LAI_SCALE_FACTOR = 0.1    # DN to m²/m²
    FPAR_SCALE_FACTOR = 0.01  # DN to fraction (0-1)
    FILL_VALUE = 255
    LAI_VALID_RANGE = (0, 100)   # Valid DN range
    FPAR_VALID_RANGE = (0, 100)  # Valid DN range

    # FparLai_QC: bits 5-7 = algorithm path
    # 0 = main (RT), 1 = backup (empirical), others = fill/not produced
    # Bit 0 (MODLAND QC): 0 = good, 1 = other
    QC_ALGORITHM_SHIFT = 5
    QC_ALGORITHM_MASK = 0b111
    QC_GOOD_ALGORITHMS = {0, 2}  # Main method and saturation

    # AppEEARS URL (for fallback)
    APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    def download(self, output_dir: Path) -> Path:
        """Download MODIS LAI/FPAR products."""
        self.logger.info("Starting MODIS LAI/FPAR acquisition")

        output_dir.mkdir(parents=True, exist_ok=True)
        processed_file = output_dir / f"{self.domain_name}_MODIS_LAI.nc"

        if processed_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False):
            self.logger.info(f"Using existing MODIS LAI file: {processed_file}")
            return processed_file

        # Check if AppEEARS mode is requested
        use_appeears = self._get_config_value(lambda: None, default=False, dict_key='MODIS_LAI_USE_APPEEARS')

        if use_appeears:
            self.logger.info("Using AppEEARS API (legacy mode)")
            return self._download_via_appeears(output_dir, processed_file)
        else:
            self.logger.info("Using earthaccess/CMR (recommended)")
            return self._download_via_earthaccess(output_dir, processed_file)

    # ===== Earthaccess/CMR Primary Pathway =====

    def _download_via_earthaccess(self, output_dir: Path, processed_file: Path) -> Path:
        """Download and process MODIS LAI/FPAR via earthaccess."""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        product = self._get_config_value(lambda: None, default='MCD15A2H', dict_key='MODIS_LAI_PRODUCT')
        self.logger.info(f"Acquiring {product} via earthaccess")

        product_dir = raw_dir / product.lower()

        # Search CMR
        granules = self._search_granules_cmr(product, version='061')

        if not granules:
            self.logger.info("No granules with v061, trying v006...")
            granules = self._search_granules_cmr(product, version='006')

        if not granules:
            raise RuntimeError(f"No {product} granules found via CMR")

        # Get download URLs
        urls = self._get_download_urls(granules, extensions=('.hdf',))
        self.logger.info(f"Found {len(urls)} {product} files to download")

        # Download
        files = self._download_with_earthaccess(urls, product_dir)
        self.logger.info(f"Downloaded {len(files)} {product} files")

        if not files:
            raise RuntimeError(f"No {product} files downloaded")

        # Process HDF files
        self._process_modis_hdf_files(files, processed_file)

        return processed_file

    def _process_modis_hdf_files(self, files: List[Path], output_file: Path):
        """Process downloaded MODIS LAI/FPAR HDF files into NetCDF."""
        from pyhdf.SD import SD, SDC

        self.logger.info(f"Processing {len(files)} MODIS LAI HDF files...")

        layers = self._get_layers()
        qc_filter = self._get_config_value(lambda: None, default=True, dict_key='MODIS_LAI_QC')

        results = []
        for i, hdf_path in enumerate(sorted(files)):
            if (i + 1) % 50 == 0:
                self.logger.info(f"  Processing {i+1}/{len(files)}")

            try:
                # Extract date from filename: MCD15A2H.A2020001.h10v03.061.*.hdf
                match = re.search(r'\.A(\d{7})\.', hdf_path.name)
                if not match:
                    continue

                year = int(match.group(1)[:4])
                doy = int(match.group(1)[4:])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)

                # Skip dates outside range
                if date < self.start_date or date > self.end_date:
                    continue

                # Open HDF
                hdf = SD(str(hdf_path), SDC.READ)
                dataset_names = hdf.datasets().keys()

                record = {'date': date}

                # Extract LAI
                lai_sds_name = self._find_sds_name(dataset_names, 'Lai_500m')
                fpar_sds_name = self._find_sds_name(dataset_names, 'Fpar_500m')
                qc_sds_name = self._find_sds_name(dataset_names, 'FparLai_QC')

                # Read QC data for filtering
                qc_data = None
                if qc_filter and qc_sds_name:
                    try:
                        qc_sds = hdf.select(qc_sds_name)
                        qc_data = qc_sds[:]
                    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError):
                        pass

                # Process LAI
                if lai_sds_name and 'Lai_500m' in layers:
                    lai_val = self._extract_layer_mean(
                        hdf, lai_sds_name, qc_data,
                        self.LAI_VALID_RANGE, self.LAI_SCALE_FACTOR
                    )
                    record['lai'] = lai_val

                # Process FPAR
                if fpar_sds_name and 'Fpar_500m' in layers:
                    fpar_val = self._extract_layer_mean(
                        hdf, fpar_sds_name, qc_data,
                        self.FPAR_VALID_RANGE, self.FPAR_SCALE_FACTOR
                    )
                    record['fpar'] = fpar_val

                hdf.end()

                # Only add if we got at least one valid value
                if 'lai' in record or 'fpar' in record:
                    results.append(record)

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.debug(f"Error processing {hdf_path.name}: {e}")
                continue

        if not results:
            raise RuntimeError("No valid LAI/FPAR data extracted from HDF files")

        # Create DataFrame and Dataset
        df = pd.DataFrame(results)
        df = df.sort_values('date').drop_duplicates('date', keep='first')
        df = df.set_index('date')

        # Build xarray Dataset
        time_values = pd.to_datetime(df.index).values
        data_vars = {}
        if 'lai' in df.columns:
            data_vars['LAI'] = xr.DataArray(
                df['lai'].values,
                dims=['time'],
                coords={'time': time_values},
                attrs={
                    'long_name': 'Leaf Area Index (domain mean)',
                    'units': 'm2/m2',
                    'scale_factor_applied': self.LAI_SCALE_FACTOR,
                }
            )

        if 'fpar' in df.columns:
            data_vars['FPAR'] = xr.DataArray(
                df['fpar'].values,
                dims=['time'],
                coords={'time': time_values},
                attrs={
                    'long_name': 'Fraction of PAR (domain mean)',
                    'units': 'fraction',
                    'scale_factor_applied': self.FPAR_SCALE_FACTOR,
                }
            )

        ds = xr.Dataset(data_vars)
        ds.attrs['title'] = 'MODIS LAI/FPAR'
        ds.attrs['source'] = f"NASA MODIS {self._get_config_value(lambda: None, default='MCD15A2H', dict_key='MODIS_LAI_PRODUCT')} v061"
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['domain'] = self.domain_name
        ds.attrs['method'] = 'earthaccess'

        ds.to_netcdf(output_file)
        self.logger.info(f"Saved {len(df)} LAI/FPAR records to {output_file}")
        self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        if 'lai' in df.columns:
            self.logger.info(f"  LAI range: {df['lai'].min():.3f} - {df['lai'].max():.3f} m2/m2")
        if 'fpar' in df.columns:
            self.logger.info(f"  FPAR range: {df['fpar'].min():.3f} - {df['fpar'].max():.3f}")

        # Create CSV output for observation handler compatibility
        self._create_csv_output(df, output_file.parent)

        return output_file

    def _find_sds_name(self, dataset_names, target: str) -> Optional[str]:
        """Find SDS name matching target in HDF dataset names."""
        for name in dataset_names:
            if target in name:
                return name
        return None

    def _extract_layer_mean(
        self,
        hdf,
        sds_name: str,
        qc_data: Optional[np.ndarray],
        valid_range: tuple,
        scale_factor: float
    ) -> float:
        """Extract domain spatial mean for a single layer from an HDF SDS."""
        sds = hdf.select(sds_name)
        data = sds[:].astype(float)

        # Mask fill values
        data[data == self.FILL_VALUE] = np.nan

        # Apply valid range
        data[(data < valid_range[0]) | (data > valid_range[1])] = np.nan

        # Apply QC filter
        if qc_data is not None:
            algorithm_bits = (qc_data >> self.QC_ALGORITHM_SHIFT) & self.QC_ALGORITHM_MASK
            good_quality = np.isin(algorithm_bits, list(self.QC_GOOD_ALGORITHMS))
            data[~good_quality] = np.nan

        # Apply scale factor
        data = data * scale_factor

        valid_mask = ~np.isnan(data)
        if valid_mask.sum() > 0:
            return float(np.nanmean(data))
        return np.nan

    def _create_csv_output(self, df: pd.DataFrame, output_dir: Path):
        """Create CSV output for observation handler compatibility."""
        csv_file = output_dir / f"{self.domain_name}_MODIS_LAI_timeseries.csv"

        out_df = df.copy()
        out_df.index.name = 'date'

        out_df.to_csv(csv_file)
        self.logger.info(f"Created CSV timeseries: {csv_file}")

    def _get_layers(self) -> List[str]:
        """Get layers to download."""
        config_layers = self._get_config_value(lambda: None, default=None, dict_key='MODIS_LAI_LAYERS')
        if config_layers:
            if isinstance(config_layers, str):
                return [config_layers]
            return list(config_layers)

        layers = ['Lai_500m', 'Fpar_500m']
        if self._get_config_value(lambda: None, default=True, dict_key='MODIS_LAI_QC'):
            layers.append('FparLai_QC')

        return layers

    # ===== AppEEARS Fallback Methods =====

    def _download_via_appeears(self, output_dir: Path, processed_file: Path) -> Path:
        """Download via AppEEARS API (legacy fallback)."""
        username, password = self._earthdata_auth().require_credentials(
            "MODIS LAI via AppEEARS"
        )

        product = self._get_config_value(lambda: None, default='MCD15A2H', dict_key='MODIS_LAI_PRODUCT')
        product_id = self.PRODUCTS.get(product, f"{product}.061")
        layers = self._get_layers()

        token = self._appeears_login(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            task_id = self._submit_appeears_task(token, product_id, layers)
            if not task_id:
                raise RuntimeError("Failed to submit AppEEARS task")

            if not self._wait_for_appeears_task(token, task_id):
                raise RuntimeError(f"AppEEARS task {task_id} did not complete")

            self._download_appeears_results(token, task_id, output_dir)
        finally:
            self._appeears_logout(token)

        # Check for downloaded files and create consolidated output
        nc_files = list(output_dir.glob("*LAI*.nc")) + list(output_dir.glob("*Lai*.nc"))
        nc_files += list(output_dir.glob("*MCD15*.nc")) + list(output_dir.glob("*MOD15*.nc"))

        if nc_files:
            # Mark as appeears output so observation handler can detect it
            marker = output_dir / ".appeears_output"
            marker.touch()

        self.logger.info(f"MODIS LAI AppEEARS download complete: {output_dir}")
        return output_dir

    def _appeears_login(self, username: str, password: str) -> Optional[str]:
        """Login to AppEEARS and get token."""
        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/login",
                auth=(username, password),
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('token')
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"AppEEARS login failed: {e}")
            return None

    def _appeears_logout(self, token: str):
        """Logout from AppEEARS."""
        try:
            requests.post(
                f"{self.APPEEARS_BASE}/logout",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError):
            pass

    def _submit_appeears_task(
        self,
        token: str,
        product_id: str,
        layers: List[str]
    ) -> Optional[str]:
        """Submit AppEEARS task request."""
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        coordinates = [[
            [lon_min, lat_min], [lon_max, lat_min],
            [lon_max, lat_max], [lon_min, lat_max],
            [lon_min, lat_min]
        ]]

        layer_specs = [{'product': product_id, 'layer': layer} for layer in layers]
        task_name = f"SYMFLUENCE_{self.domain_name}_LAI_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        task_request = {
            'task_type': 'area',
            'task_name': task_name,
            'params': {
                'dates': [{
                    'startDate': self.start_date.strftime('%m-%d-%Y'),
                    'endDate': self.end_date.strftime('%m-%d-%Y'),
                }],
                'layers': layer_specs,
                'geo': {
                    'type': 'FeatureCollection',
                    'features': [{
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': coordinates,
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

        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/task",
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                },
                json=task_request,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            task_id = result.get('task_id')
            self.logger.info(f"Submitted AppEEARS task: {task_id}")
            return task_id
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            return None

    def _wait_for_appeears_task(
        self,
        token: str,
        task_id: str,
        timeout: int = 7200,
        poll_interval: int = 30
    ) -> bool:
        """Wait for AppEEARS task to complete."""
        self.logger.info(f"Waiting for AppEEARS task {task_id} to complete...")
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                response = requests.get(
                    f"{self.APPEEARS_BASE}/task/{task_id}",
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=60
                )
                response.raise_for_status()
                status = response.json()

                task_status = status.get('status')
                self.logger.info(f"AppEEARS task status: {task_status}")

                if task_status == 'done':
                    return True
                elif task_status in ['error', 'expired', 'failed']:
                    self.logger.error(
                        f"AppEEARS task failed: {status.get('error', 'Unknown error')}"
                    )
                    return False

                time.sleep(poll_interval)

            except requests.RequestException as e:
                self.logger.warning(f"Error checking task status: {e}")
                time.sleep(poll_interval)

        self.logger.error(f"AppEEARS task timed out after {timeout} seconds")
        return False

    def _download_appeears_results(self, token: str, task_id: str, output_dir: Path):
        """Download completed AppEEARS task results."""
        headers = {'Authorization': f'Bearer {token}'}

        try:
            response = requests.get(
                f"{self.APPEEARS_BASE}/bundle/{task_id}",
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            bundle = response.json()
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Failed to get bundle info: {e}")
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
                    f"{self.APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers=headers,
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.debug(f"Downloaded: {file_name}")

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Failed to download {file_name}: {e}")
