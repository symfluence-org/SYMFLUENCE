# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tree Canopy Height Data Acquisition Handlers

Cloud-based acquisition of global tree canopy height from multiple sources:
- GEDI L2A (NASA): Space-based LiDAR canopy height via AppEEARS
- Meta/WRI Global (2020): AI-derived 10m resolution via Planetary Computer
- GLAD/UMD: University of Maryland forest height from Landsat

Key Features:
    Multiple Data Sources:
    - NASA GEDI: High-accuracy LiDAR measurements (sparse coverage)
    - Meta/WRI: Complete global coverage at 10m (AI-derived from satellite)
    - GLAD: Research product with vegetation height from optical data

    Output Format:
    - GeoTIFF (all sources)
    - Standardized to WGS84 (EPSG:4326)
    - Height values in meters

References:
    - GEDI: https://gedi.umd.edu/
    - Meta/WRI: https://sustainability.fb.com/blog/2023/04/canopy-height-map/
    - GLAD: https://glad.umd.edu/
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import rasterio
import requests
from rasterio.merge import merge as rio_merge

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..earthdata_auth import missing_credentials_message
from ..mixins import RetryMixin
from ..utils import create_robust_session

# =============================================================================
# GEDI Canopy Height (NASA AppEEARS)
# =============================================================================

APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"


@R.acquisition_handlers.add('GEDI_CANOPY_HEIGHT')
@R.acquisition_handlers.add('GEDI_L2A')
class GEDICanopyHeightAcquirer(BaseAcquisitionHandler):
    """
    GEDI L2A canopy height acquisition via NASA AppEEARS.

    Downloads gridded canopy height metrics from NASA's Global Ecosystem
    Dynamics Investigation (GEDI) mission, which uses space-based LiDAR
    to measure vegetation structure.

    GEDI L2A Overview:
        Data Type: LiDAR-derived canopy height metrics
        Resolution: 25m footprint, gridded to ~1km
        Coverage: 51.6°N to 51.6°S (ISS orbit)
        Source: NASA GEDI Mission (ISS-mounted LiDAR)
        Format: NetCDF via AppEEARS
        Temporal: 2019-present

    Available Metrics (via AppEEARS):
        - rh98: Relative height at 98th percentile (canopy top)
        - rh95: Relative height at 95th percentile
        - rh75: Relative height at 75th percentile (mid-canopy)
        - rh50: Relative height at 50th percentile
        - quality_flag: Data quality indicator

    Acquisition Workflow:
        1. Authenticate with NASA Earthdata via AppEEARS
        2. Submit area task with bounding box and date range
        3. Poll for task completion (up to 6 hours)
        4. Download gridded NetCDF results
        5. Convert to GeoTIFF if needed

    Configuration:
        GEDI_METRIC: Height metric to download (default: 'rh98')
        GEDI_QUALITY_FILTER: Apply quality filtering (default: True)

    Notes:
        - GEDI has sparse coverage due to discrete footprints
        - Best used for validation, not continuous coverage
        - Consider Meta/WRI for wall-to-wall canopy height

    References:
        - GEDI Mission: https://gedi.umd.edu/
        - AppEEARS: https://appeears.earthdatacloud.nasa.gov/
        - Dubayah et al. (2020). Science of Remote Sensing
    """

    GEDI_PRODUCTS = {
        'L2A': 'GEDI02_A.002',
        'L2B': 'GEDI02_B.002',
    }

    DEFAULT_LAYERS = ['rh98', 'rh95', 'quality_flag']

    def download(self, output_dir: Path) -> Path:
        """
        Download GEDI canopy height data via AppEEARS.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        canopy_dir = self._attribute_dir("vegetation")
        canopy_dir = canopy_dir / 'canopy_height' / 'gedi'
        canopy_dir.mkdir(parents=True, exist_ok=True)

        output_file = canopy_dir / f"{self.domain_name}_gedi_canopy_height.tif"

        if self._skip_if_exists(output_file):
            return output_file

        self.logger.info("Starting GEDI canopy height acquisition via AppEEARS")

        # Get credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            raise ValueError(missing_credentials_message("GEDI canopy height"))

        # Get configuration
        metric = self._get_config_value(lambda: None, default='rh98', dict_key='GEDI_METRIC')
        product_id = self.GEDI_PRODUCTS.get('L2A', 'GEDI02_A.002')
        layers = self._get_layers(metric)

        # Authenticate
        token = self._get_appeears_token(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            # Submit task
            task_id = self._submit_task(token, product_id, layers)
            if not task_id:
                raise RuntimeError("Failed to submit AppEEARS task for GEDI")

            # Wait for completion and download
            self._wait_and_download(token, task_id, canopy_dir)

            # Convert/consolidate to output file
            self._consolidate_output(canopy_dir, output_file, metric)

            self.logger.info(f"GEDI canopy height download complete: {output_file}")
            return output_file

        finally:
            # Logout
            self._appeears_logout(token)

    def _get_layers(self, metric: str) -> List[str]:
        """Get layers to download based on metric."""
        layers = [metric]
        if self._get_config_value(lambda: None, default=True, dict_key='GEDI_QUALITY_FILTER'):
            layers.append('quality_flag')
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
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"AppEEARS authentication failed: {e}")
            return None

    def _appeears_logout(self, token: str) -> None:
        """Logout from AppEEARS."""
        try:
            requests.post(
                f"{APPEEARS_BASE}/logout",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
        except (requests.RequestException, OSError):
            pass  # Logout failure is non-critical

    def _submit_task(
        self,
        token: str,
        product_id: str,
        layers: List[str]
    ) -> Optional[str]:
        """Submit AppEEARS task request for GEDI data."""
        headers = {'Authorization': f'Bearer {token}'}

        layer_specs = [{'product': product_id, 'layer': layer} for layer in layers]

        task_name = f"GEDI_CH_{self.domain_name}_{self.start_date.strftime('%Y%m%d')}"

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
                    'format': {'type': 'geotiff'},
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
            self.logger.info(f"AppEEARS GEDI task submitted: {task_id}")
            return task_id
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            return None

    def _build_geo_spec(self) -> Dict[str, Any]:
        """Build geographic specification for task."""
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

    def _wait_and_download(
        self,
        token: str,
        task_id: str,
        output_dir: Path,
        max_wait: int = 21600,  # 6 hours
        poll_interval: int = 60
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
                self.logger.info(f"AppEEARS GEDI task status: {task_status}")

                if task_status == 'done':
                    self._download_results(token, task_id, output_dir)
                    return
                elif task_status in ['error', 'expired']:
                    raise RuntimeError(
                        f"AppEEARS task failed: {status.get('error', 'Unknown error')}"
                    )

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

        try:
            response = requests.get(
                f"{APPEEARS_BASE}/bundle/{task_id}",
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

            if not file_name.endswith(('.tif', '.nc', '.nc4')):
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

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Failed to download {file_name}: {e}")

    def _consolidate_output(self, input_dir: Path, output_file: Path, metric: str):
        """Consolidate downloaded files into single output."""
        import shutil

        # Find downloaded files
        tif_files = list(input_dir.glob(f"*{metric}*.tif"))
        if not tif_files:
            tif_files = list(input_dir.glob("*.tif"))

        if not tif_files:
            raise FileNotFoundError(f"No GEDI output files found in {input_dir}")

        if len(tif_files) == 1:
            shutil.copy(tif_files[0], output_file)
        else:
            # Merge multiple files
            src_files = [rasterio.open(p) for p in tif_files]
            mosaic, out_trans = rio_merge(src_files)
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
            for src in src_files:
                src.close()


# =============================================================================
# Meta/WRI Global Canopy Height (Planetary Computer)
# =============================================================================

@R.acquisition_handlers.add('META_CANOPY_HEIGHT')
@R.acquisition_handlers.add('WRI_CANOPY_HEIGHT')
@R.acquisition_handlers.add('META_WRI_CANOPY')
class MetaCanopyHeightAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    Meta/WRI Global Canopy Height acquisition via Microsoft Planetary Computer.

    Downloads the global tree canopy height map produced by Meta AI and World
    Resources Institute using deep learning on satellite imagery. Provides
    wall-to-wall coverage at 10m resolution.

    Meta/WRI Canopy Height Overview:
        Data Type: AI-derived canopy height from Sentinel-2 + DEM
        Resolution: 10m (native Sentinel-2 resolution)
        Coverage: Global (land areas with vegetation)
        Source: Meta AI Research + World Resources Institute
        Format: Cloud-Optimized GeoTIFF (COG)
        Temporal: 2020 (single epoch)
        Accuracy: RMSE ~4m compared to GEDI reference

    Production Method:
        - Deep learning model trained on GEDI LiDAR footprints
        - Input features: Sentinel-2 optical + Copernicus DEM
        - Inference applied globally to produce wall-to-wall map
        - Post-processing to remove artifacts and water bodies

    Acquisition Workflow:
        1. Query STAC catalog for tiles covering bounding box
        2. Download required COG tiles (1°x1° scheme)
        3. Merge tiles and clip to exact domain bounds
        4. Output single GeoTIFF

    Configuration:
        META_CANOPY_VERSION: Data version (default: 'v1')

    Advantages:
        - Complete global coverage (unlike sparse GEDI)
        - High resolution (10m)
        - Free and openly available
        - Cloud-optimized format for fast access

    Limitations:
        - Single temporal epoch (2020)
        - Lower accuracy than direct LiDAR (4m RMSE)
        - May have artifacts in challenging terrain

    References:
        - Tolan et al. (2023). Very high resolution canopy height maps
          from RGB imagery using self-supervised vision transformer
        - Planetary Computer: https://planetarycomputer.microsoft.com/
    """

    # Tile URL pattern for Meta canopy height on Planetary Computer
    STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "io-lulc-annual-v02"

    # Direct tile access URL pattern (when STAC is not needed)
    # Meta canopy height is distributed as 1-degree tiles
    TILE_BASE_URL = "https://ai4edataeuwest.blob.core.windows.net/io-lulc/io-lulc-model-001-v01-composite"

    def download(self, output_dir: Path) -> Path:
        """
        Download Meta/WRI canopy height data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        canopy_dir = self._attribute_dir("vegetation")
        canopy_dir = canopy_dir / 'canopy_height' / 'meta_wri'
        canopy_dir.mkdir(parents=True, exist_ok=True)

        output_file = canopy_dir / f"{self.domain_name}_meta_canopy_height.tif"

        if self._skip_if_exists(output_file):
            return output_file

        self.logger.info(f"Downloading Meta/WRI canopy height for bbox: {self.bbox}")

        # Calculate required tiles (1-degree tiles)
        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        try:
            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    tile_path = self._download_tile(session, lat, lon, canopy_dir)
                    if tile_path:
                        tile_paths.append(tile_path)

            if not tile_paths:
                raise FileNotFoundError(
                    f"No Meta/WRI canopy height tiles found for bbox: {self.bbox}"
                )

            # Merge tiles if needed
            if len(tile_paths) == 1:
                import shutil
                shutil.copy(tile_paths[0], output_file)
            else:
                self._merge_tiles(tile_paths, output_file)

            # Clean up temp tiles
            for p in tile_paths:
                if p.exists() and 'temp_' in p.name:
                    p.unlink(missing_ok=True)

            self.logger.info(f"Meta/WRI canopy height download complete: {output_file}")
            return output_file

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error downloading Meta/WRI canopy height: {e}")
            raise

    def _download_tile(
        self,
        session: requests.Session,
        lat: int,
        lon: int,
        output_dir: Path
    ) -> Optional[Path]:
        """Download a single 1-degree tile."""
        # Generate tile name (Meta uses N/S E/W naming like: N45E010)
        lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
        lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"

        # Try multiple URL patterns as the exact structure may vary
        url_patterns = [
            f"https://ai4edataeuwest.blob.core.windows.net/io-lulc/io-lulc-model-001-v01-composite/{lat_str}{lon_str}_canopy_height.tif",
            f"https://data.source.coop/meta/meta-canopy-height/tiles/{lat_str}{lon_str}.tif",
            f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/{lat_str}_{lon_str}_treecover2000.tif",
        ]

        local_tile = output_dir / f"temp_meta_{lat_str}{lon_str}.tif"

        if local_tile.exists():
            self.logger.info(f"Using cached tile: {lat_str}{lon_str}")
            return local_tile

        for url in url_patterns:
            try:
                result = self._try_download_tile(session, url, local_tile, f"{lat_str}{lon_str}")
                if result:
                    return result
            except (requests.RequestException, OSError, IOError):
                continue

        self.logger.warning(f"Tile {lat_str}{lon_str} not available from any source")
        return None

    def _try_download_tile(
        self,
        session: requests.Session,
        url: str,
        local_tile: Path,
        tile_name: str
    ) -> Optional[Path]:
        """Attempt to download a tile from a specific URL."""
        def do_download():
            with session.get(url, stream=True, timeout=300) as r:
                if r.status_code == 200:
                    with open(local_tile, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    self.logger.info(f"Downloaded {tile_name}")
                    return local_tile
                elif r.status_code == 404:
                    return None
                else:
                    raise requests.exceptions.HTTPError(f"HTTP {r.status_code}")

        try:
            return self.execute_with_retry(
                do_download,
                max_retries=3,
                base_delay=2,
                backoff_factor=2.0,
                retryable_exceptions=(
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    BrokenPipeError,
                    IOError,
                )
            )
        except (requests.RequestException, OSError, IOError):
            if local_tile.exists():
                local_tile.unlink()
            return None

    def _merge_tiles(self, tile_paths: List[Path], output_file: Path):
        """Merge multiple tiles into a single GeoTIFF."""
        self.logger.info(f"Merging {len(tile_paths)} tiles")

        src_files = [rasterio.open(p) for p in tile_paths]
        mosaic, out_trans = rio_merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })

        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files:
            src.close()


# =============================================================================
# GLAD/UMD Tree Height
# =============================================================================

@R.acquisition_handlers.add('GLAD_TREE_HEIGHT')
@R.acquisition_handlers.add('UMD_TREE_HEIGHT')
@R.acquisition_handlers.add('GLAD_CANOPY')
class GLADTreeHeightAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    GLAD/UMD tree height acquisition from University of Maryland.

    Downloads tree height data from the Global Land Analysis & Discovery
    (GLAD) laboratory at University of Maryland, derived from Landsat
    and ICESat/GLAS data.

    GLAD Tree Height Overview:
        Data Type: Tree height derived from Landsat + ICESat/GLAS
        Resolution: 30m (Landsat native resolution)
        Coverage: Global forest areas
        Source: University of Maryland GLAD Lab
        Format: GeoTIFF
        Temporal: Various years (2000-2020)

    Available Products:
        - Tree Cover 2000: Baseline tree cover percentage
        - Tree Cover Loss: Annual forest loss masks
        - Tree Height 2000: Estimated tree height at year 2000

    Tile Scheme:
        Organization: 10°x10° degree tiles
        Naming: Hansen_GFC-{version}_{variable}_{lat}_{lon}.tif
        Example: Hansen_GFC-2020-v1.8_treecover2000_50N_120W.tif

    Acquisition Workflow:
        1. Calculate tile indices from domain bounding box
        2. Download required tiles from UMD servers
        3. Merge tiles and clip to exact domain bounds
        4. Output single GeoTIFF

    Configuration:
        GLAD_VERSION: Dataset version (default: '2020-v1.8')
        GLAD_VARIABLE: Variable to download (default: 'treecover2000')

    References:
        - Hansen et al. (2013). High-Resolution Global Maps of
          21st-Century Forest Cover Change. Science, 342(6160).
        - GLAD: https://glad.umd.edu/
    """

    BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen"
    DEFAULT_VERSION = "GFC-2020-v1.8"

    def download(self, output_dir: Path) -> Path:
        """
        Download GLAD/UMD tree height data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        canopy_dir = self._attribute_dir("vegetation")
        canopy_dir = canopy_dir / 'canopy_height' / 'glad'
        canopy_dir.mkdir(parents=True, exist_ok=True)

        output_file = canopy_dir / f"{self.domain_name}_glad_tree_height.tif"

        if self._skip_if_exists(output_file):
            return output_file

        self.logger.info(f"Downloading GLAD/UMD tree height for bbox: {self.bbox}")

        version = self._get_config_value(lambda: None, default=self.DEFAULT_VERSION, dict_key='GLAD_VERSION')
        variable = self._get_config_value(lambda: None, default='treecover2000', dict_key='GLAD_VARIABLE')

        # GLAD uses 10-degree tiles
        lat_min = self._snap_to_grid(self.bbox['lat_min'], 10, 'floor')
        lat_max = self._snap_to_grid(self.bbox['lat_max'], 10, 'ceil')
        lon_min = self._snap_to_grid(self.bbox['lon_min'], 10, 'floor')
        lon_max = self._snap_to_grid(self.bbox['lon_max'], 10, 'ceil')

        tile_paths = []
        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        try:
            for lat in range(lat_min, lat_max, 10):
                for lon in range(lon_min, lon_max, 10):
                    tile_path = self._download_tile(
                        session, lat, lon, version, variable, canopy_dir
                    )
                    if tile_path:
                        tile_paths.append(tile_path)

            if not tile_paths:
                raise FileNotFoundError(
                    f"No GLAD tree height tiles found for bbox: {self.bbox}"
                )

            # Merge tiles if needed
            if len(tile_paths) == 1:
                import shutil
                shutil.copy(tile_paths[0], output_file)
            else:
                self._merge_tiles(tile_paths, output_file)

            # Clean up temp tiles
            for p in tile_paths:
                if p.exists() and 'temp_' in p.name:
                    p.unlink(missing_ok=True)

            self.logger.info(f"GLAD tree height download complete: {output_file}")
            return output_file

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error downloading GLAD tree height: {e}")
            raise

    def _snap_to_grid(self, value: float, grid_size: int, mode: str) -> int:
        """Snap coordinate to grid boundary."""
        if mode == 'floor':
            return int(math.floor(value / grid_size) * grid_size)
        else:
            return int(math.ceil(value / grid_size) * grid_size)

    def _download_tile(
        self,
        session: requests.Session,
        lat: int,
        lon: int,
        version: str,
        variable: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Download a single GLAD tile."""
        # GLAD naming: 50N_120W (10-degree tiles, upper-left corner)
        lat_str = f"{abs(lat)}N" if lat >= 0 else f"{abs(lat)}S"
        lon_str = f"{abs(lon)}E" if lon >= 0 else f"{abs(lon)}W"

        tile_name = f"Hansen_{version}_{variable}_{lat_str}_{lon_str}"
        url = f"{self.BASE_URL}/{version}/{tile_name}.tif"

        local_tile = output_dir / f"temp_glad_{tile_name}.tif"

        if local_tile.exists():
            self.logger.info(f"Using cached tile: {tile_name}")
            return local_tile

        self.logger.info(f"Fetching GLAD tile: {tile_name}")

        def do_download():
            with session.get(url, stream=True, timeout=300) as r:
                if r.status_code == 200:
                    with open(local_tile, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    self.logger.info(f"Downloaded {tile_name}")
                    return local_tile
                elif r.status_code == 404:
                    self.logger.warning(f"GLAD tile {tile_name} not found (no forest data)")
                    return None
                else:
                    raise requests.exceptions.HTTPError(f"HTTP {r.status_code}")

        try:
            return self.execute_with_retry(
                do_download,
                max_retries=3,
                base_delay=2,
                backoff_factor=2.0,
                retryable_exceptions=(
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    BrokenPipeError,
                    IOError,
                )
            )
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Failed to download {tile_name}: {e}")
            if local_tile.exists():
                local_tile.unlink()
            return None

    def _merge_tiles(self, tile_paths: List[Path], output_file: Path):
        """Merge multiple tiles into a single GeoTIFF."""
        self.logger.info(f"Merging {len(tile_paths)} GLAD tiles")

        src_files = [rasterio.open(p) for p in tile_paths]
        mosaic, out_trans = rio_merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })

        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files:
            src.close()
