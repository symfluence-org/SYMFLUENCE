# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS Snow Cover Area (SCA) Acquisition Handler

Acquires MOD10A1 (Terra) and MYD10A1 (Aqua) snow cover products and merges them
into a combined daily product with improved spatial/temporal coverage.

Default method: earthaccess/CMR (fast, direct download)
Fallback method: AppEEARS (slower, queue-based)

References:
- MOD10A1: https://nsidc.org/data/mod10a1
- MYD10A1: https://nsidc.org/data/myd10a1
- earthaccess: https://github.com/nsidc/earthaccess
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R
from symfluence.data.acquisition.earthaccess_base import BaseEarthaccessAcquirer


@R.acquisition_handlers.add('MODIS_SCA')
@R.acquisition_handlers.add('MODIS_SNOW_MERGED')
class MODISSCAAcquirer(BaseEarthaccessAcquirer):
    """
    Acquires MODIS Snow Cover Area data from both Terra (MOD10A1) and Aqua (MYD10A1)
    satellites, then merges products for improved daily coverage.

    Uses earthaccess/CMR for direct downloads (faster than AppEEARS).

    The merge strategy prioritizes:
    1. Cloud-free pixels from either satellite
    2. When both have valid data, uses maximum SCA (conservative for snow detection)
    3. Quality flags for filtering unreliable observations

    Configuration:
        MODIS_SCA_PRODUCTS: List of products, default ['MOD10A1', 'MYD10A1']
        MODIS_SCA_MERGE_STRATEGY: 'max' (default), 'mean', 'terra_priority', 'aqua_priority'
        MODIS_SCA_CLOUD_FILTER: True (default) - filter cloud-covered pixels
        MODIS_SCA_QA_FILTER: True (default) - filter by quality flags
        MODIS_SCA_MIN_VALID_RATIO: 0.1 (default) - minimum fraction of valid pixels
        MODIS_SCA_USE_APPEEARS: False (default) - set True to use AppEEARS instead
        EARTHDATA_USERNAME/EARTHDATA_PASSWORD: NASA Earthdata credentials
    """

    # NDSI Snow Cover value interpretation (MOD10A1/MYD10A1)
    # 0-100: NDSI snow cover percentage
    # 200: missing data
    # 201: no decision
    # 211: night
    # 237: inland water
    # 239: ocean
    # 250: cloud
    # 254: detector saturated
    # 255: fill value
    VALID_SNOW_RANGE = (0, 100)
    CLOUD_VALUE = 250
    MISSING_VALUES = {200, 201, 211, 237, 239, 250, 254, 255}

    def download(self, output_dir: Path) -> Path:
        """Download and merge MOD10A1/MYD10A1 snow cover products."""
        self.logger.info("Starting MODIS SCA acquisition (Terra + Aqua merge)")

        output_dir.mkdir(parents=True, exist_ok=True)
        merged_file = output_dir / f"{self.domain_name}_MODIS_SCA_merged.nc"

        if merged_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing merged SCA file: {merged_file}")
            return merged_file

        # Get products to download (strip version suffix for earthaccess)
        products_config = self._get_config_value(
            lambda: self.config.evaluation.modis_snow.products,
            dict_key='MODIS_SCA_PRODUCTS',
        )
        if isinstance(products_config, str):
            products_config = [p.strip() for p in products_config.split(',')]

        # Normalize product names (remove .061 suffix if present)
        products = [p.split('.')[0] for p in products_config]

        # Check for Earthdata credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            self.logger.warning(
                "Earthdata credentials not found. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD "
                "environment variables or add to ~/.netrc"
            )
            return self._download_via_thredds(output_dir, products[0])

        # Check if user wants to use AppEEARS (legacy mode)
        use_appeears = self._get_config_value(lambda: self.config.evaluation.modis_snow.use_appeears, default=False, dict_key='MODIS_SCA_USE_APPEEARS')

        product_files = {}

        if use_appeears:
            # Legacy AppEEARS mode
            self.logger.info("Using AppEEARS mode (legacy)")
            for product in products:
                try:
                    product_file = self._download_product_appeears(
                        output_dir, product, username, password
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
                    self.logger.warning(f"Failed to download {product} via AppEEARS: {e}")
        else:
            # Default: earthaccess/CMR mode (faster)
            self.logger.info("Using earthaccess/CMR mode (default)")
            for product in products:
                try:
                    raw_dir = output_dir / "raw" / product.lower()
                    raw_dir.mkdir(parents=True, exist_ok=True)

                    # Download via earthaccess
                    files = self._download_granules_earthaccess(
                        short_name=product,
                        output_dir=raw_dir,
                        version='61',  # MODIS Collection 6.1
                        extensions=('.hdf',)
                    )

                    if files:
                        # Process downloaded HDF files into NetCDF
                        product_file = self._process_modis_hdf_files(
                            files, output_dir, product
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
                    self.logger.warning(f"Failed to download {product} via earthaccess: {e}")
                    # Try AppEEARS as fallback
                    self.logger.info(f"Trying AppEEARS fallback for {product}")
                    try:
                        product_file = self._download_product_appeears(
                            output_dir, product, username, password
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
                    ) as e2:
                        self.logger.warning(f"AppEEARS fallback also failed: {e2}")

        if not product_files:
            raise RuntimeError("No MODIS SCA products could be downloaded")

        # Merge products if we have multiple
        if len(product_files) > 1:
            self._merge_products(product_files, merged_file)
        else:
            # Just copy/rename the single product
            single_file = list(product_files.values())[0]
            if single_file != merged_file:
                import shutil
                shutil.copy(single_file, merged_file)

        self.logger.info(f"MODIS SCA acquisition complete: {merged_file}")
        return merged_file

    def _process_modis_hdf_files(
        self,
        hdf_files: List[Path],
        output_dir: Path,
        product: str
    ) -> Optional[Path]:
        """Process downloaded MODIS HDF files into a single NetCDF."""
        import re
        from datetime import timedelta

        if not hdf_files:
            return None

        output_file = output_dir / f"{self.domain_name}_{product}_raw.nc"

        if output_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            return output_file

        self.logger.info(f"Processing {len(hdf_files)} {product} HDF files")

        records: List[Dict[str, Any]] = []
        for hdf_path in sorted(hdf_files):
            try:
                # Extract date from filename (e.g., MOD10A1.A2017001.h10v03...)
                match = re.search(r'\.A(\d{7})\.', hdf_path.name)
                if not match:
                    continue

                year = int(match.group(1)[:4])
                doy = int(match.group(1)[4:])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)

                # Open HDF and extract NDSI snow cover
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='variable .* has multiple fill values')
                    ds = xr.open_dataset(hdf_path, engine='netcdf4')

                # Find the snow cover variable
                sca_var = None
                for var in ds.data_vars:
                    if 'ndsi' in var.lower() or 'snow' in var.lower():
                        sca_var = var
                        break

                if sca_var:
                    data = ds[sca_var].values
                    # Apply valid range mask
                    data = np.where((data >= 0) & (data <= 100), data, np.nan)
                    records.append({'date': date, 'data': data, 'coords': ds.coords})

                ds.close()

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

        if not records:
            self.logger.warning(f"No valid data extracted from {product} files")
            return None

        # Create xarray dataset
        records = sorted(records, key=lambda x: cast(datetime, x['date']))
        times = [cast(datetime, r['date']) for r in records]
        data_stack = np.stack([cast(np.ndarray, r['data']) for r in records], axis=0)

        # Get coordinates from first record
        first_coords = cast(Dict[str, Any], records[0]['coords'])
        lat_key = 'lat' if 'lat' in first_coords else 'y'
        lon_key = 'lon' if 'lon' in first_coords else 'x'

        coords = {'time': times}
        if lat_key in first_coords:
            coords[lat_key] = first_coords[lat_key].values
        if lon_key in first_coords:
            coords[lon_key] = first_coords[lon_key].values

        da = xr.DataArray(
            data_stack,
            dims=['time', lat_key, lon_key],
            coords=coords,
            name='NDSI_Snow_Cover',
            attrs={
                'long_name': 'NDSI Snow Cover',
                'units': 'percent',
                'valid_range': [0, 100],
                'source': f'{product} via earthaccess'
            }
        )

        ds_out = xr.Dataset({'NDSI_Snow_Cover': da})
        ds_out.to_netcdf(output_file)

        self.logger.info(f"Processed {len(records)} timesteps to {output_file}")
        return output_file

    # ===== AppEEARS Fallback Methods =====
    # These are kept for backwards compatibility and fallback when earthaccess fails

    APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    def _appeears_login(self, username: str, password: str) -> Optional[str]:
        """Login to AppEEARS and get authentication token."""
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
        """Logout from AppEEARS."""
        try:
            requests.post(
                f"{self.APPEEARS_BASE}/logout",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
        except (requests.RequestException, OSError, ValueError, TypeError):
            pass

    def _wait_for_task(self, token: str, task_id: str, timeout_hours: float = 6) -> bool:
        """Wait for an AppEEARS task to complete."""
        import time
        self.logger.info(f"Waiting for AppEEARS task {task_id} to complete...")
        start_time = time.time()
        timeout_seconds = timeout_hours * 3600

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

                if task_status == 'done':
                    return True
                elif task_status in ['error', 'failed']:
                    self.logger.error(f"Task failed: {status.get('error', 'Unknown')}")
                    return False

                time.sleep(30)
            except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
                self.logger.warning(f"Error checking task: {e}")
                time.sleep(30)

        self.logger.error(f"Task {task_id} timed out after {timeout_hours} hours")
        return False

    def _download_task_results(self, token: str, task_id: str, output_dir: Path, prefix: str) -> List[Path]:
        """Download results from completed AppEEARS task."""
        downloaded = []
        try:
            response = requests.get(
                f"{self.APPEEARS_BASE}/bundle/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60
            )
            response.raise_for_status()
            files = response.json().get('files', [])

            for file_info in files:
                file_name = file_info.get('file_name', '')
                file_id = file_info.get('file_id')
                if not file_name.endswith('.nc'):
                    continue

                out_path = output_dir / f"{prefix}_{file_name}"
                if out_path.exists():
                    downloaded.append(out_path)
                    continue

                dl_response = requests.get(
                    f"{self.APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    stream=True, timeout=600
                )
                dl_response.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in dl_response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                downloaded.append(out_path)
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            self.logger.error(f"Download failed: {e}")
        return downloaded

    def _consolidate_appeears_output(self, output_dir: Path, prefix: str, output_file: Path) -> None:
        """Consolidate multiple AppEEARS files into single NetCDF."""
        import shutil
        nc_files = list(output_dir.glob(f"{prefix}_*.nc"))
        if not nc_files:
            return
        if len(nc_files) == 1:
            shutil.copy(nc_files[0], output_file)
            return
        try:
            datasets = [xr.open_dataset(f) for f in sorted(nc_files)]
            merged = xr.concat(datasets, dim='time').sortby('time')
            merged.to_netcdf(output_file)
            for ds in datasets:
                ds.close()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            self.logger.error(f"Consolidation failed: {e}")
            if nc_files:
                shutil.copy(nc_files[0], output_file)

    def _generate_date_chunks(self, chunk_years: int = 4) -> List[tuple]:
        """Split the full date range into smaller chunks for AppEEARS."""
        chunks = []
        chunk_start = self.start_date
        while chunk_start < self.end_date:
            chunk_end = min(
                chunk_start.replace(year=chunk_start.year + chunk_years) - pd.Timedelta(days=1),
                self.end_date
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + pd.Timedelta(days=1)
        return chunks

    def _download_product_appeears(
        self,
        output_dir: Path,
        product: str,
        username: str,
        password: str
    ) -> Optional[Path]:
        """Download a single MODIS product via AppEEARS API with time chunking."""
        self.logger.info(f"Downloading {product} via AppEEARS")

        # Parse product name (e.g., 'MOD10A1.061' -> product='MOD10A1', version='061')
        parts = product.split('.')
        product_name = parts[0]
        version = parts[1] if len(parts) > 1 else '061'

        output_file = output_dir / f"{self.domain_name}_{product_name}_raw.nc"

        if output_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing file: {output_file}")
            return output_file

        # Split into time chunks to avoid AppEEARS file limits
        chunks = self._generate_date_chunks(chunk_years=4)
        self.logger.info(f"Splitting {product} request into {len(chunks)} time chunks")

        # Login to AppEEARS
        token = self._appeears_login(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                chunk_label = f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"
                self.logger.info(
                    f"Chunk {i+1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}"
                )

                # Submit task for this chunk
                task_id = self._submit_appeears_task(
                    token, product_name, version,
                    start_date=chunk_start, end_date=chunk_end
                )

                if not task_id:
                    self.logger.warning(f"Failed to submit chunk {i+1}, skipping")
                    continue

                # Wait for task completion
                if not self._wait_for_task(token, task_id):
                    self.logger.warning(f"Chunk {i+1} task did not complete, skipping")
                    continue

                # Download results
                self._download_task_results(
                    token, task_id, output_dir, f"{product_name}_{chunk_label}"
                )

            # Consolidate all chunk files into single NetCDF
            self._consolidate_appeears_output(output_dir, product_name, output_file)

            return output_file

        finally:
            self._appeears_logout(token)

    def _submit_appeears_task(
        self,
        token: str,
        product: str,
        version: str,
        start_date=None,
        end_date=None
    ) -> Optional[str]:
        """Submit an AppEEARS area request task."""
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        # Create GeoJSON polygon for the bounding box
        coordinates = [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min]
        ]]

        # Full product name with version (e.g., MOD10A1.061)
        product_full = f"{product}.{version}"

        task_name = f"SYMFLUENCE_{self.domain_name}_{product}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Format dates for AppEEARS (use overrides if provided)
        req_start = start_date if start_date is not None else self.start_date
        req_end = end_date if end_date is not None else self.end_date
        start_date = req_start.strftime("%m-%d-%Y")
        end_date = req_end.strftime("%m-%d-%Y")

        # Build task request
        task_request = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": start_date,
                    "endDate": end_date
                }],
                "layers": [{
                    "product": product_full,
                    "layer": "NDSI_Snow_Cover"
                }],
                "geo": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coordinates
                        },
                        "properties": {}
                    }]
                },
                "output": {
                    "format": {
                        "type": "netcdf4"
                    },
                    "projection": "geographic"
                }
            }
        }

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

    def _convert_time_to_dates(self, time_values) -> List:
        """Convert time values to date objects, handling cftime."""
        dates = []
        for t in time_values:
            try:
                # Try standard datetime conversion
                if hasattr(t, 'date'):
                    dates.append(t.date() if callable(t.date) else t.date)
                elif hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
                    # cftime object
                    from datetime import date
                    dates.append(date(t.year, t.month, t.day))
                else:
                    # Try pandas conversion
                    dates.append(pd.to_datetime(t).date())
            except (ValueError, TypeError, AttributeError):
                # Last resort: string parsing
                dates.append(pd.to_datetime(str(t)[:10]).date())
        return dates

    def _merge_products(self, product_files: Dict[str, Path], output_file: Path):
        """Merge Terra and Aqua products into combined daily SCA.

        Uses chunked xarray operations to avoid loading full datasets into memory.
        """
        self.logger.info("Merging MOD10A1 and MYD10A1 products")

        merge_strategy = self._get_config_value(lambda: self.config.evaluation.modis_snow.merge_strategy, default='max', dict_key='MODIS_SCA_MERGE_STRATEGY').lower()
        cloud_filter = self._get_config_value(lambda: self.config.evaluation.modis_snow.cloud_filter, default=True, dict_key='MODIS_SCA_CLOUD_FILTER')

        # Open datasets lazily with chunking to avoid OOM
        datasets = {}
        sca_vars = {}
        for product, path in product_files.items():
            try:
                ds = xr.open_dataset(path, chunks={'time': 90})
                sca_var = None
                for var in ds.data_vars:
                    if 'snow' in var.lower() or 'ndsi' in var.lower():
                        sca_var = var
                        break
                if sca_var:
                    datasets[product] = ds
                    sca_vars[product] = sca_var
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                self.logger.warning(f"Failed to open {path}: {e}")

        if not datasets:
            raise RuntimeError("No valid datasets to merge")

        # Check for time dimension
        has_time = any('time' in ds.dims for ds in datasets.values())
        if not has_time:
            first_ds = list(datasets.values())[0]
            first_var = sca_vars[list(sca_vars.keys())[0]]
            ds_out = xr.Dataset({'NDSI_Snow_Cover': first_ds[first_var]})
            ds_out.to_netcdf(output_file)
            for ds in datasets.values():
                ds.close()
            return

        # Build mask of invalid/cloud values
        invalid_vals = set(self.MISSING_VALUES)
        if cloud_filter:
            invalid_vals.add(self.CLOUD_VALUE)

        # Align datasets to common time index and mask invalid values
        aligned_das = []
        for product in datasets:
            da = datasets[product][sca_vars[product]].astype(np.float32)
            # Mask invalid values to NaN
            for mv in invalid_vals:
                da = da.where(da != mv)
            # Normalize time to daily (drop sub-day precision)
            da['time'] = pd.to_datetime(
                [pd.Timestamp(t).normalize() for t in da.time.values]
            )
            aligned_das.append(da)

        # Concatenate along a new 'source' dimension
        combined = xr.concat(aligned_das, dim='source')

        # Apply merge strategy along the source dimension
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            if merge_strategy == 'max':
                merged = combined.max(dim='source')
            elif merge_strategy == 'mean':
                merged = combined.mean(dim='source')
            elif merge_strategy == 'terra_priority':
                merged = combined.isel(source=0).combine_first(
                    combined.isel(source=-1) if combined.sizes['source'] > 1 else combined.isel(source=0)
                )
            elif merge_strategy == 'aqua_priority':
                if combined.sizes['source'] > 1:
                    merged = combined.isel(source=-1).combine_first(combined.isel(source=0))
                else:
                    merged = combined.isel(source=0)
            else:
                merged = combined.max(dim='source')

        # Handle duplicate dates (keep first)
        _, unique_idx = np.unique(merged.time.values, return_index=True)
        merged = merged.isel(time=np.sort(unique_idx))

        merged.name = 'NDSI_Snow_Cover'
        merged.attrs = {
            'long_name': 'NDSI Snow Cover (Merged Terra+Aqua)',
            'units': 'percent',
            'valid_range': [0, 100],
            'merge_strategy': merge_strategy,
            'source_products': list(product_files.keys()),
        }

        ds_out = xr.Dataset({'NDSI_Snow_Cover': merged})
        ds_out.attrs['title'] = 'Merged MODIS Snow Cover Area'
        ds_out.attrs['source'] = 'MOD10A1 + MYD10A1'
        ds_out.attrs['created'] = datetime.now().isoformat()

        self.logger.info("Writing merged dataset to disk...")
        ds_out.to_netcdf(output_file)

        # Cleanup
        for ds in datasets.values():
            ds.close()

        self.logger.info(f"Merged SCA product saved: {output_file}")

    def _download_via_thredds(self, output_dir: Path, product: str) -> Path:
        """Fallback to THREDDS download (legacy, single product)."""
        self.logger.warning("Falling back to THREDDS download (single product only)")

        from . import modis
        # Use existing MODIS snow acquirer as fallback
        legacy_acquirer = modis.MODISSnowAcquirer(self.config, self.logger)
        return legacy_acquirer.download(output_dir)
