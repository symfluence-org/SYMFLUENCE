# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS MOD16A2 Evapotranspiration (ET) Acquisition Handler

Acquires MOD16A2 (Terra) and MYD16A2 (Aqua) 8-day composite ET products
via NASA earthaccess/CMR (default) or AppEEARS (fallback).

References:
- MOD16A2: https://lpdaac.usgs.gov/products/mod16a2v061/
- MYD16A2: https://lpdaac.usgs.gov/products/myd16a2v061/
- earthaccess: https://github.com/nsidc/earthaccess

Products provide 8-day composite values at 500m resolution:
- ET_500m: Total Evapotranspiration (kg/m²/8day)
- LE_500m: Average Latent Heat Flux (J/m²/day)
- PET_500m: Total Potential ET (kg/m²/8day)
- ET_QC_500m: Quality Control flags

Configuration:
    MOD16_USE_APPEEARS: False (default) - Use earthaccess; True = use AppEEARS
    MOD16_PRODUCTS: List of products, default ['MOD16A2']
    MOD16_VARIABLE: 'ET_500m' (default), 'LE_500m', 'PET_500m'
    MOD16_MERGE_PRODUCTS: True to merge Terra+Aqua, False for Terra only
    MOD16_CONVERT_TO_DAILY: True (default) - convert 8-day to daily mean
    MOD16_QC_FILTER: True (default) - filter by quality flags
    MOD16_UNITS: 'mm_day' (default), 'kg_m2_8day' (raw)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R
from symfluence.data.acquisition.earthaccess_base import BaseEarthaccessAcquirer


@R.acquisition_handlers.add('MOD16')
@R.acquisition_handlers.add('MODIS_ET')
@R.acquisition_handlers.add('MOD16A2')
class MOD16ETAcquirer(BaseEarthaccessAcquirer):
    """
    Acquires MODIS MOD16A2/MYD16A2 Evapotranspiration data.

    By default uses earthaccess for direct NASA Earthdata Cloud downloads.
    Set MOD16_USE_APPEEARS: true for legacy AppEEARS API mode.
    """

    # MOD16A2 QC flag interpretation (ET_QC_500m)
    # Bits 0-1: MODLAND_QC (00=Good, 01=Other, 10=Marginal, 11=Cloud/NoData)
    QC_GOOD_MASK = 0b00000011
    QC_GOOD_VALUES = {0b00, 0b01}
    FILL_VALUE = 32767
    # Special values for non-vegetated/invalid pixels (should be masked)
    SPECIAL_VALUE_MIN = 32761  # 32761-32766: cloud, not processed, water, missing, barren, ice/snow
    SCALE_FACTOR = 0.1
    DAYS_IN_COMPOSITE = 8

    # MODIS sinusoidal projection constants (for spatial subsetting)
    _MODIS_R = 6371007.181        # Earth radius (m)
    _MODIS_TILE_SIZE = 1111950.519667  # Tile extent in metres
    _MODIS_NPIX = 2400            # Pixels per tile (500 m products)

    # AppEEARS URL (for fallback)
    APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    def download(self, output_dir: Path) -> Path:
        """Download MOD16A2 ET products."""
        self.logger.info("Starting MOD16 ET acquisition")

        output_dir.mkdir(parents=True, exist_ok=True)
        processed_file = output_dir / f"{self.domain_name}_MOD16_ET.nc"

        if processed_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False):
            self.logger.info(f"Using existing MOD16 ET file: {processed_file}")
            return processed_file

        # Check if AppEEARS mode is requested
        use_appeears = self._get_config_value(lambda: None, default=False, dict_key='MOD16_USE_APPEEARS')

        if use_appeears:
            self.logger.info("Using AppEEARS API (legacy mode)")
            return self._download_via_appeears(output_dir, processed_file)
        else:
            self.logger.info("Using earthaccess/CMR (recommended)")
            return self._download_via_earthaccess(output_dir, processed_file)

    def _download_via_earthaccess(self, output_dir: Path, processed_file: Path) -> Path:
        """Download and process MOD16A2 via earthaccess."""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        # Get products
        # Note: Using MOD16A2GF (Gap-Filled) as standard MOD16A2 is not in CMR cloud
        merge_products = self._get_config_value(lambda: None, default=False, dict_key='MOD16_MERGE_PRODUCTS')
        products = ['MOD16A2GF']
        if merge_products:
            products.append('MYD16A2GF')

        all_files = []

        for product in products:
            self.logger.info(f"Acquiring {product} via earthaccess")
            product_dir = raw_dir / product.lower().replace('16a2', '')

            # Search CMR - MOD16A2 uses version '061' (3 digits, different from MOD10A1)
            granules = self._search_granules_cmr(product, version='061')

            if not granules:
                # Try older version '006'
                self.logger.info("No granules with v061, trying v006...")
                granules = self._search_granules_cmr(product, version='006')

            if not granules:
                self.logger.warning(f"No {product} granules found")
                continue

            # Get download URLs
            urls = self._get_download_urls(granules, extensions=('.hdf',))
            self.logger.info(f"Found {len(urls)} {product} files to download")

            # Download
            files = self._download_with_earthaccess(urls, product_dir)
            all_files.extend(files)
            self.logger.info(f"Downloaded {len(files)} {product} files")

        if not all_files:
            raise RuntimeError("No MOD16 ET files downloaded")

        # Process HDF files
        self._process_modis_hdf_files(all_files, processed_file)

        return processed_file

    def _process_modis_hdf_files(self, files: List[Path], output_file: Path):
        """Process downloaded MODIS HDF files into NetCDF."""
        from pyhdf.SD import SD, SDC

        self.logger.info(f"Processing {len(files)} MOD16 HDF files...")

        variable = self._get_config_value(lambda: None, default='ET_500m', dict_key='MOD16_VARIABLE')
        qc_filter = self._get_config_value(lambda: None, default=True, dict_key='MOD16_QC_FILTER')
        convert_to_daily = self._get_config_value(lambda: None, default=True, dict_key='MOD16_CONVERT_TO_DAILY')
        target_units = self._get_config_value(lambda: None, default='mm_day', dict_key='MOD16_UNITS')

        results = []
        for i, hdf_path in enumerate(sorted(files)):
            if (i + 1) % 50 == 0:
                self.logger.info(f"  Processing {i+1}/{len(files)}")

            try:
                # Extract date and tile h/v from filename
                # MOD16A2GF.A2020001.h10v03.061.*.hdf
                match = re.search(r'\.A(\d{7})\.', hdf_path.name)
                if not match:
                    continue

                year = int(match.group(1)[:4])
                doy = int(match.group(1)[4:])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)

                # Skip dates outside range
                if date < self.start_date or date > self.end_date:
                    continue

                # Extract tile h/v for spatial subsetting
                tile_match = re.search(r'\.h(\d{2})v(\d{2})\.', hdf_path.name)
                tile_h = int(tile_match.group(1)) if tile_match else None
                tile_v = int(tile_match.group(2)) if tile_match else None

                # Open HDF
                hdf = SD(str(hdf_path), SDC.READ)

                # Find ET variable — use exact name match first, then substring
                sds_names = list(hdf.datasets().keys())
                et_sds_name = variable if variable in sds_names else None
                qc_sds_name = None
                if not et_sds_name:
                    for name in sds_names:
                        if name == variable or name.endswith(variable):
                            if 'QC' not in name:
                                et_sds_name = name
                                break
                for name in sds_names:
                    if 'ET_QC' in name or name == 'QC_500m':
                        qc_sds_name = name

                if not et_sds_name:
                    hdf.end()
                    continue

                # Read ET data
                et_sds = hdf.select(et_sds_name)
                et_data = et_sds[:].astype(float)

                # Apply fill/special value mask (32761-32767 are all invalid)
                et_data[et_data >= self.SPECIAL_VALUE_MIN] = np.nan

                # Apply scale factor
                et_data = et_data * self.SCALE_FACTOR

                # Apply QC filter if available and enabled
                if qc_filter and qc_sds_name:
                    try:
                        qc_sds = hdf.select(qc_sds_name)
                        qc_data = qc_sds[:]
                        # Good quality: bits 0-1 are 00 or 01
                        good_quality = np.isin(qc_data & self.QC_GOOD_MASK, list(self.QC_GOOD_VALUES))
                        et_data[~good_quality] = np.nan
                    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                        pass

                hdf.end()

                # Spatial subsetting: crop to basin bbox
                if tile_h is not None and tile_v is not None and self.bbox:
                    row_slice, col_slice = self._tile_pixel_window(
                        tile_h, tile_v, self.bbox, et_data.shape[0]
                    )
                    if row_slice is not None:
                        et_data = et_data[row_slice, col_slice]

                # Get domain spatial mean
                valid_mask = ~np.isnan(et_data)
                if valid_mask.sum() > 0:
                    domain_mean = float(np.nanmean(et_data))

                    # Convert units
                    if target_units == 'mm_day' and convert_to_daily:
                        # kg/m²/8day to mm/day (1 kg/m² = 1 mm)
                        domain_mean = domain_mean / self.DAYS_IN_COMPOSITE

                    results.append({
                        'date': date,
                        'et': domain_mean,
                        'valid_pixels': int(valid_mask.sum()),
                        'total_pixels': int(et_data.size)
                    })

            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                AttributeError,
                RuntimeError,
            ) as e:
                self.logger.debug(f"Error processing {hdf_path.name}: {e}")
                continue

        if not results:
            raise RuntimeError("No valid ET data extracted from HDF files")

        # Create DataFrame and Dataset
        df = pd.DataFrame(results)
        df = df.sort_values('date').drop_duplicates('date', keep='first')
        df = df.set_index('date')
        df.index.name = 'time'

        # Convert to xarray
        units = 'mm/day' if target_units == 'mm_day' and convert_to_daily else 'kg/m2/8day'
        long_name = 'Evapotranspiration (daily mean)' if convert_to_daily else 'Evapotranspiration (8-day)'

        time_coords = pd.to_datetime(df.index)
        ds = xr.Dataset({
            'ET': xr.DataArray(
                df['et'].values,
                dims=['time'],
                coords={'time': time_coords},
                attrs={'long_name': long_name, 'units': units}
            ),
            'valid_pixels': xr.DataArray(
                df['valid_pixels'].values,
                dims=['time'],
                coords={'time': time_coords},
                attrs={'long_name': 'Number of valid pixels', 'units': 'count'}
            )
        })

        ds.attrs['title'] = 'MODIS MOD16 Evapotranspiration'
        ds.attrs['source'] = 'NASA MODIS MOD16A2/MYD16A2 v061'
        ds.attrs['variable'] = variable
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['domain'] = self.domain_name
        ds.attrs['method'] = 'earthaccess'

        ds.to_netcdf(output_file)
        self.logger.info(f"Saved {len(df)} ET records to {output_file}")
        self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        self.logger.info(f"  ET range: {df['et'].min():.3f} - {df['et'].max():.3f} {units}")

        # Create CSV output
        self._create_csv_output(df, output_file.parent)

        return output_file

    def _create_csv_output(self, df: pd.DataFrame, output_dir: Path):
        """Create CSV output for observation handler compatibility."""
        csv_file = output_dir / f"{self.domain_name}_MOD16_ET_timeseries.csv"

        out_df = df[['et']].copy()
        out_df.columns = ['et_mm_day']
        out_df.index.name = 'date'

        out_df.to_csv(csv_file)
        self.logger.info(f"Created CSV timeseries: {csv_file}")

    # ===== Spatial subsetting helpers =====

    @staticmethod
    def _latlon_to_sinu(lat: float, lon: float):
        """Convert lat/lon (degrees) to MODIS sinusoidal x, y (metres)."""
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        R = MOD16ETAcquirer._MODIS_R
        return R * lon_r * np.cos(lat_r), R * lat_r

    def _tile_pixel_window(
        self, h: int, v: int, bbox: dict, npix: int = 2400
    ) -> Tuple[Optional[slice], Optional[slice]]:
        """
        Return (row_slice, col_slice) to crop a MODIS tile array to the
        domain bounding box.  Returns (None, None) if the bbox does not
        intersect the tile.
        """
        ts = self._MODIS_TILE_SIZE
        pix = ts / npix  # pixel size in metres

        # Tile extent in sinusoidal metres
        tx0 = (h - 18) * ts
        ty1 = (9 - v) * ts          # top (north)
        tx1 = tx0 + ts
        ty0 = ty1 - ts              # bottom (south)

        # Convert bbox corners to sinusoidal
        lat_min = bbox.get('lat_min', bbox.get('S', 0))
        lat_max = bbox.get('lat_max', bbox.get('N', 0))
        lon_min = bbox.get('lon_min', bbox.get('W', 0))
        lon_max = bbox.get('lon_max', bbox.get('E', 0))

        xs, ys = [], []
        for lat in (lat_min, lat_max):
            for lon in (lon_min, lon_max):
                x, y = self._latlon_to_sinu(lat, lon)
                xs.append(x)
                ys.append(y)

        bx0, bx1 = max(min(xs), tx0), min(max(xs), tx1)
        by0, by1 = max(min(ys), ty0), min(max(ys), ty1)

        if bx0 >= bx1 or by0 >= by1:
            return None, None

        # Rows count downward from top (north)
        r0 = max(0, int((ty1 - by1) / pix))
        r1 = min(npix, int(np.ceil((ty1 - by0) / pix)))
        c0 = max(0, int((bx0 - tx0) / pix))
        c1 = min(npix, int(np.ceil((bx1 - tx0) / pix)))

        if r0 >= r1 or c0 >= c1:
            return None, None

        return slice(r0, r1), slice(c0, c1)

    # ===== AppEEARS Fallback Methods =====

    def _download_via_appeears(self, output_dir: Path, processed_file: Path) -> Path:
        """Download via AppEEARS API (legacy fallback)."""
        username, password = self._earthdata_auth().require_credentials(
            "MOD16 ET via AppEEARS"
        )

        products = self._get_config_value(lambda: None, default=['MOD16A2.061'], dict_key='MOD16_PRODUCTS')
        if isinstance(products, str):
            products = [p.strip() for p in products.split(',')]

        merge_products = self._get_config_value(lambda: None, default=False, dict_key='MOD16_MERGE_PRODUCTS')
        if merge_products and 'MYD16A2.061' not in products:
            products.append('MYD16A2.061')

        variable = self._get_config_value(lambda: None, default='ET_500m', dict_key='MOD16_VARIABLE')

        product_files = {}
        for product in products:
            try:
                product_file = self._download_product_appeears(
                    output_dir, product, variable, username, password
                )
                if product_file and product_file.exists():
                    product_files[product] = product_file
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RuntimeError,
                requests.RequestException,
            ) as e:
                self.logger.warning(f"Failed to download {product}: {e}")

        if not product_files:
            raise RuntimeError("No MOD16 ET products could be downloaded")

        self._process_appeears_products(product_files, processed_file, variable)
        return processed_file

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
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError) as e:
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
        except (requests.RequestException, OSError, ValueError, TypeError):
            pass

    def _download_product_appeears(
        self,
        output_dir: Path,
        product: str,
        variable: str,
        username: str,
        password: str
    ) -> Optional[Path]:
        """Download a product via AppEEARS API."""
        self.logger.info(f"Downloading {product} ({variable}) via AppEEARS")

        parts = product.split('.')
        product_name = parts[0]
        version = parts[1] if len(parts) > 1 else '061'

        output_file = output_dir / f"{self.domain_name}_{product_name}_raw.nc"

        if output_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False):
            return output_file

        token = self._appeears_login(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            task_id = self._submit_appeears_task(token, product_name, version, variable)
            if not task_id:
                raise RuntimeError(f"Failed to submit AppEEARS task for {product}")

            if not self._wait_for_task(token, task_id):
                raise RuntimeError(f"AppEEARS task {task_id} did not complete")

            self._download_task_results(token, task_id, output_dir, product_name)
            self._consolidate_appeears_output(output_dir, product_name, output_file)

            return output_file

        finally:
            self._appeears_logout(token)

    def _submit_appeears_task(
        self,
        token: str,
        product: str,
        version: str,
        variable: str
    ) -> Optional[str]:
        """Submit an AppEEARS area request task."""
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        coordinates = [[
            [lon_min, lat_min], [lon_max, lat_min],
            [lon_max, lat_max], [lon_min, lat_max],
            [lon_min, lat_min]
        ]]

        product_full = f"{product}.{version}"
        task_name = f"SYMFLUENCE_{self.domain_name}_{product}_ET_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        layers = [{"product": product_full, "layer": variable}]
        if self._get_config_value(lambda: None, default=True, dict_key='MOD16_QC_FILTER'):
            layers.append({"product": product_full, "layer": "ET_QC_500m"})

        task_request = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": self.start_date.strftime("%m-%d-%Y"),
                    "endDate": self.end_date.strftime("%m-%d-%Y")
                }],
                "layers": layers,
                "geo": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": coordinates},
                        "properties": {}
                    }]
                },
                "output": {"format": {"type": "netcdf4"}, "projection": "geographic"}
            }
        }

        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/task",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=task_request,
                timeout=120
            )
            response.raise_for_status()
            task_id = response.json().get('task_id')
            self.logger.info(f"Submitted AppEEARS task: {task_id}")
            return task_id
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError) as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            return None

    def _wait_for_task(self, token: str, task_id: str, timeout: int = 21600) -> bool:
        """Wait for AppEEARS task to complete."""
        import time

        start_time = time.time()
        check_interval = 60

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.APPEEARS_BASE}/task/{task_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=60
                )
                response.raise_for_status()
                status = response.json().get('status')

                if status == 'done':
                    return True
                elif status in ['error', 'expired']:
                    self.logger.error(f"AppEEARS task failed with status: {status}")
                    return False

                self.logger.debug(f"Task status: {status}")
                time.sleep(check_interval)

            except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
                self.logger.warning(f"Error checking task status: {e}")
                time.sleep(check_interval)

        self.logger.error(f"AppEEARS task timed out after {timeout}s")
        return False

    def _download_task_results(
        self,
        token: str,
        task_id: str,
        output_dir: Path,
        product: str
    ):
        """Download AppEEARS task results."""
        try:
            response = requests.get(
                f"{self.APPEEARS_BASE}/bundle/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60
            )
            response.raise_for_status()
            files = response.json().get('files', [])

            for file_info in files:
                file_id = file_info.get('file_id')
                filename = file_info.get('file_name', f"{file_id}.nc")

                if not filename.endswith('.nc'):
                    continue

                out_path = output_dir / filename
                if out_path.exists():
                    continue

                dl_response = requests.get(
                    f"{self.APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    stream=True,
                    timeout=300
                )
                dl_response.raise_for_status()

                with open(out_path, 'wb') as f:
                    for chunk in dl_response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)

                self.logger.debug(f"Downloaded: {filename}")

        except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            self.logger.error(f"Error downloading task results: {e}")
            raise

    def _consolidate_appeears_output(
        self,
        output_dir: Path,
        product: str,
        output_file: Path
    ):
        """Consolidate AppEEARS NetCDF files."""
        nc_files = list(output_dir.glob(f"*{product}*.nc"))
        if not nc_files:
            raise RuntimeError(f"No NetCDF files found for {product}")

        datasets = []
        for f in nc_files:
            try:
                ds = xr.open_dataset(f)
                datasets.append(ds)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                self.logger.debug(f"Could not open {f}: {e}")

        if not datasets:
            raise RuntimeError("No valid NetCDF files to consolidate")

        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')
        merged.to_netcdf(output_file)

        for ds in datasets:
            ds.close()

    def _process_appeears_products(
        self,
        product_files: Dict[str, Path],
        output_file: Path,
        variable: str
    ):
        """Process AppEEARS products into final output."""
        convert_to_daily = self._get_config_value(lambda: None, default=True, dict_key='MOD16_CONVERT_TO_DAILY')
        qc_filter = self._get_config_value(lambda: None, default=True, dict_key='MOD16_QC_FILTER')
        target_units = self._get_config_value(lambda: None, default='mm_day', dict_key='MOD16_UNITS')

        datasets = {}
        for product, path in product_files.items():
            try:
                ds = xr.open_dataset(path)
                et_var = None
                for var in ds.data_vars:
                    if variable.lower().replace('_500m', '') in var.lower():
                        et_var = var
                        break
                    if 'et' in var.lower() and 'qc' not in var.lower():
                        et_var = var
                        break
                if et_var:
                    datasets[product] = {'data': ds[et_var], 'ds': ds}
                    for qc_var in ds.data_vars:
                        if 'qc' in qc_var.lower():
                            datasets[product]['qc'] = ds[qc_var]
                            break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                self.logger.warning(f"Failed to open {path}: {e}")

        if not datasets:
            raise RuntimeError("No valid ET datasets to process")

        processed_das = []
        for product, data_dict in datasets.items():
            da = data_dict['data'].copy()

            # Mask fill AND special values (32761-32767: cloud, not processed,
            # water, missing, barren, ice/snow) -- not just the 32767 fill.
            # Raw integers are masked before scaling; already-scaled floats
            # (xarray mask_and_scale) are masked above the scaled valid max.
            if np.issubdtype(da.dtype, np.integer):
                da = da.where(da < self.SPECIAL_VALUE_MIN).astype(float) * self.SCALE_FACTOR
            else:
                da = da.where(da < self.SPECIAL_VALUE_MIN * self.SCALE_FACTOR - 0.05)

            if qc_filter and 'qc' in data_dict:
                qc = data_dict['qc']
                good_quality = (qc & self.QC_GOOD_MASK).isin(list(self.QC_GOOD_VALUES))
                da = da.where(good_quality)

            processed_das.append(da)

        if len(processed_das) > 1:
            stacked = xr.concat(processed_das, dim='product')
            et_merged = stacked.mean(dim='product', skipna=True)
        else:
            et_merged = processed_das[0]

        if target_units == 'mm_day' and convert_to_daily:
            et_merged = et_merged / self.DAYS_IN_COMPOSITE
            units = 'mm/day'
            long_name = 'Evapotranspiration (daily mean from 8-day composite)'
        else:
            units = 'kg/m2/8day'
            long_name = 'Evapotranspiration (8-day composite)'

        lat_dim = 'lat' if 'lat' in et_merged.dims else 'y'
        lon_dim = 'lon' if 'lon' in et_merged.dims else 'x'

        if lat_dim in et_merged.dims and lon_dim in et_merged.dims:
            et_basin_mean = et_merged.mean(dim=[lat_dim, lon_dim], skipna=True)
        else:
            et_basin_mean = et_merged

        ds_out = xr.Dataset({
            'ET': et_merged.rename('ET'),
            'ET_basin_mean': et_basin_mean.rename('ET_basin_mean')
        })

        ds_out['ET'].attrs = {'long_name': long_name, 'units': units, 'source': 'MODIS MOD16A2/MYD16A2', 'variable': variable}
        ds_out['ET_basin_mean'].attrs = {'long_name': f'Basin-averaged {long_name}', 'units': units}

        ds_out.attrs['title'] = 'MODIS MOD16 Evapotranspiration'
        ds_out.attrs['source_products'] = list(product_files.keys())
        ds_out.attrs['created'] = datetime.now().isoformat()
        ds_out.attrs['domain'] = self.domain_name
        ds_out.attrs['method'] = 'appeears'

        ds_out.to_netcdf(output_file)

        for data_dict in datasets.values():
            data_dict['ds'].close()

        self.logger.info(f"Processed MOD16 ET saved: {output_file}")

        df = et_basin_mean.to_dataframe()
        self._create_csv_output(df.rename(columns={df.columns[0]: 'et'}), output_file.parent)
