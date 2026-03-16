# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""DEM Acquisition Handlers

Cloud-based acquisition of global elevation data from multiple sources:
- Copernicus GLO-30 (30m): AWS S3, cloud-optimized GeoTIFF
- Copernicus GLO-90 (90m): AWS S3, cloud-optimized GeoTIFF
- FABDEM (30m): Forest/building-removed variant from Source Cooperative
- NASADEM Local (30m): Local tile discovery and merging
- SRTM GL1 (30m): OpenTopography S3
- ETOPO 2022 (15s/30s/60s): NOAA OPeNDAP
- Mapzen Terrain (~30m): AWS S3 Skadi tiles
- ALOS AW3D30 (30m): Microsoft Planetary Computer STAC

Key Features:
    Tile Management:
    - 1x1 degree tile scheme (standard for global DEMs)
    - Automatic tile merging for domains spanning multiple tiles
    - Local caching to avoid re-downloads

    Retry Logic:
    - Exponential backoff for transient failures
    - Configurable max retries and backoff factors
    - Robust session creation with connection pooling

References:
    - Copernicus DEM: https://registry.opendata.aws/copernicus-dem/
    - Hawker et al. (2022). FABDEM. Scientific Data, 9, 488
    - USGS NASADEM: https://lpdaac.usgs.gov/products/nasadem_hgt/
    - SRTM: https://www.opentopodata.org/datasets/srtm/
    - ETOPO 2022: https://www.ncei.noaa.gov/products/etopo-global-relief-model
    - Mapzen Terrain: https://github.com/tilezen/joerd
    - ALOS AW3D30: https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm
"""

import gzip
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.windows import Window, from_bounds

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session

# =============================================================================
# Shared Tile Download Mixin
# =============================================================================

class _TileDownloadMixin:
    """Shared tile download logic for 1x1 degree tile-based DEM acquirers.

    Provides:
    - _validate_tile(): check a cached tile is readable by rasterio
    - _download_tile_with_retry(): download a single tile with retry/backoff
    - _merge_tiles(): merge multiple tile files into a single GeoTIFF
    """

    def _validate_tile(self, tile_path: Path, tile_name: str) -> bool:
        """Validate a cached tile is readable by rasterio."""
        try:
            with rasterio.open(tile_path) as src:
                src.read(1, window=Window(0, 0, 1, 1))
            return True
        except (rasterio.errors.RasterioError, OSError, ValueError, TypeError):
            self.logger.warning(f"Cached tile {tile_name} is corrupted, will re-download")
            return False

    def _download_tile_with_retry(
        self, session, url: str, local_tile: Path, tile_name: str
    ) -> Path | None:
        """Download a single tile with retry logic using RetryMixin."""

        def do_download():
            try:
                with session.get(url, stream=True, timeout=300) as r:
                    if r.status_code == 200:
                        expected_size = int(r.headers.get('Content-Length', 0))
                        bytes_written = 0
                        with open(local_tile, "wb") as f:
                            for chunk in r.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    bytes_written += len(chunk)
                        if expected_size and bytes_written < expected_size:
                            raise IOError(
                                f"Incomplete download for {tile_name}: "
                                f"got {bytes_written}, expected {expected_size}"
                            )
                        self.logger.info(f"Downloaded {tile_name}")
                        return local_tile
                    elif r.status_code == 404:
                        self.logger.warning(f"Tile {tile_name} not found (404)")
                        return None
                    else:
                        raise requests.exceptions.HTTPError(
                            f"HTTP {r.status_code} for {tile_name}"
                        )
            except (requests.RequestException, OSError, IOError) as download_err:
                if local_tile.exists():
                    local_tile.unlink()
                raise download_err

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
        except (
            requests.RequestException,
            OSError,
            IOError,
            BrokenPipeError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Failed to download {tile_name}: {e}")
            raise

    def _merge_tiles(self, tile_paths: list, out_path: Path, compress: str = 'lzw') -> Path:
        """Merge multiple tile files into a single GeoTIFF."""
        if len(tile_paths) == 1:
            if out_path.exists():
                out_path.unlink()
            tile_paths[0].replace(out_path)
        else:
            self.logger.info(f"Merging {len(tile_paths)} tiles into {out_path}")
            src_files = [rasterio.open(p) for p in tile_paths]
            mosaic, out_trans = rio_merge(src_files)
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": compress,
            })
            # Use BigTIFF when output exceeds ~4 GB (classic TIFF limit)
            est_bytes = mosaic.dtype.itemsize * mosaic.shape[0] * mosaic.shape[1] * mosaic.shape[2]
            if est_bytes > 3_500_000_000:
                out_meta["driver"] = "GTiff"
                out_meta["BIGTIFF"] = "YES"
            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            for src in src_files:
                src.close()
            for p in tile_paths:
                p.unlink(missing_ok=True)
        return out_path


# =============================================================================
# Copernicus DEM Acquirers (GLO-30 and GLO-90)
# =============================================================================

@AcquisitionRegistry.register('COPDEM30')
class CopDEM30Acquirer(BaseAcquisitionHandler, RetryMixin, _TileDownloadMixin):
    """Copernicus DEM GLO-30 acquisition via AWS S3 with tile management.

    Downloads and merges global 30m resolution Digital Elevation Model (DEM)
    from the Copernicus DEM collection hosted on AWS S3. Uses cloud-optimized
    GeoTIFF (COG) format for efficient cloud access with per-tile retry logic.

    Copernicus DEM GLO-30:
        Resolution: 30m (1 arc-second)
        Coverage: Global (90S - 90N, 180W - 180E)
        Format: Cloud-Optimized GeoTIFF (COG)

    Tile Scheme:
        Naming: Copernicus_DSM_COG_10_{LAT}_00_{LON}_00_DEM
        COG code 10 = 10 arc-second posting = ~30m

    References:
        - AWS Public Dataset: https://registry.opendata.aws/copernicus-dem/
    """

    _BASE_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
    _COG_CODE = "10"
    _PRODUCT_NAME = "Copernicus DEM GLO-30"

    def download(self, output_dir: Path) -> Path:
        """Download Copernicus GLO-30 tiles covering the domain bbox and merge them.

        Args:
            output_dir: Base output directory (tiles are written to
                ``attributes/elevation/dem/`` under the project directory).

        Returns:
            Path to the merged domain DEM GeoTIFF.
        """
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading {self._PRODUCT_NAME} for bbox: {self.bbox}")

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            session = create_robust_session(max_retries=5, backoff_factor=2.0)

            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    tile_name = f"Copernicus_DSM_COG_{self._COG_CODE}_{lat_str}_00_{lon_str}_00_DEM"
                    url = f"{self._BASE_URL}/{tile_name}/{tile_name}.tif"

                    local_tile = dem_dir / f"temp_{tile_name}.tif"
                    if local_tile.exists():
                        if not self._validate_tile(local_tile, tile_name):
                            local_tile.unlink()
                        else:
                            self.logger.info(f"Using cached tile: {tile_name}")
                            tile_paths.append(local_tile)
                            continue

                    self.logger.info(f"Fetching tile: {tile_name}")
                    tile_result = self._download_tile_with_retry(
                        session, url, local_tile, tile_name
                    )
                    if tile_result:
                        tile_paths.append(tile_result)

            if not tile_paths:
                raise FileNotFoundError(f"No {self._PRODUCT_NAME} tiles found for bbox: {self.bbox}")

            self._merge_tiles(tile_paths, out_path)

        except (
            requests.RequestException,
            rasterio.errors.RasterioError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error downloading/processing {self._PRODUCT_NAME}: {e}")
            for p in tile_paths:
                if p.exists() and p != out_path:
                    p.unlink(missing_ok=True)
            raise

        return out_path


@AcquisitionRegistry.register('COPDEM90')
class CopDEM90Acquirer(CopDEM30Acquirer):
    """Copernicus DEM GLO-90 acquisition via AWS S3.

    Same as GLO-30 but at 90m (3 arc-second) resolution. Smaller file sizes,
    suitable for larger domains or coarse-resolution modeling.

    Copernicus DEM GLO-90:
        Resolution: 90m (3 arc-seconds)
        Coverage: Global (90S - 90N, 180W - 180E)
        Format: Cloud-Optimized GeoTIFF (COG)

    Tile Scheme:
        Naming: Copernicus_DSM_COG_30_{LAT}_00_{LON}_00_DEM
        COG code 30 = 30 arc-second posting = ~90m

    References:
        - AWS Public Dataset: https://registry.opendata.aws/copernicus-dem/
    """

    _BASE_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
    _COG_CODE = "30"
    _PRODUCT_NAME = "Copernicus DEM GLO-90"


# =============================================================================
# FABDEM (unchanged)
# =============================================================================

@AcquisitionRegistry.register('FABDEM')
class FABDEMAcquirer(BaseAcquisitionHandler):
    """FABDEM acquisition handler for forest/building-removed elevation data.

    Downloads and processes FABDEM (Forest And Buildings removed DEM) v1-2,
    a global 30m elevation model with vegetation and anthropogenic structures
    removed. Useful for hydrological modeling where bare-earth DEM is required.

    FABDEM v1-2 Overview:
        Data Type: Digital Elevation Model with forest/building removal
        Resolution: 30m (1 arc-second)
        Coverage: Global (90S - 90N)
        Source: Hawker et al. (2022), Source Cooperative
        Processing: Copernicus DEM + GEDI + landcover masking
        Format: Cloud-Optimized GeoTIFF (COG)
        Datum: WGS84 (EPSG:4326)
        Units: Meters above sea level

    References:
        - Hawker et al. (2022). A 30m global map of elevation corrected for
          vegetation bias and national boundaries. Scientific Data, 9, 488
        - Source Cooperative: https://source.coop/
    """

    def download(self, output_dir: Path) -> Path:
        """Download FABDEM tiles from Source Cooperative and merge into a domain DEM.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the merged bare-earth DEM GeoTIFF.
        """
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading FABDEM for bbox: {self.bbox}")
        # Source Cooperative (AWS)
        base_url = "https://data.source.coop/c_6_6/fabdem/tiles"

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    # FABDEM format: N46W122_FABDEM_V1-2.tif
                    tile_name = f"{lat_str}{lon_str}_FABDEM_V1-2"
                    url = f"{base_url}/{tile_name}.tif"

                    local_tile = dem_dir / f"temp_fab_{tile_name}.tif"
                    if not local_tile.exists():
                        self.logger.info(f"Fetching FABDEM tile: {tile_name}")
                        with requests.get(url, stream=True, timeout=60) as r:
                            if r.status_code == 200:
                                with open(local_tile, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
                                tile_paths.append(local_tile)
                    else:
                        tile_paths.append(local_tile)

            if not tile_paths:
                raise FileNotFoundError(f"No FABDEM tiles found for bbox: {self.bbox}")

            if len(tile_paths) == 1:
                if out_path.exists(): out_path.unlink()
                tile_paths[0].replace(out_path)
            else:
                src_files = [rasterio.open(p) for p in tile_paths]
                mosaic, out_trans = rio_merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
                with rasterio.open(out_path, "w", **out_meta) as dest: dest.write(mosaic)
                for src in src_files: src.close()
                for p in tile_paths: p.unlink(missing_ok=True)
        except (
            requests.RequestException,
            rasterio.errors.RasterioError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error with FABDEM: {e}")
            raise
        return out_path


# =============================================================================
# NASADEM Local (unchanged)
# =============================================================================

@AcquisitionRegistry.register('NASADEM_LOCAL')
class NASADEMLocalAcquirer(BaseAcquisitionHandler):
    """NASADEM local tile acquisition for pre-downloaded elevation data.

    Discovers and merges NASADEM or compatible local elevation tiles to create
    a domain-specific DEM. Enables offline operation and use of pre-downloaded
    tiles or alternative bare-earth elevation products.

    NASADEM Overview:
        Resolution: 30m (1 arc-second)
        Coverage: Global (+/-60 latitude)
        Format: Flexible (HGT or GeoTIFF)

    References:
        - USGS NASADEM: https://lpdaac.usgs.gov/products/nasadem_hgt/
    """
    def download(self, output_dir: Path) -> Path:
        """Discover local NASADEM tiles and merge them into a domain DEM.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the merged DEM GeoTIFF.

        Raises:
            ValueError: If ``NASADEM_LOCAL_DIR`` is not configured.
            FileNotFoundError: If the local directory or matching tiles are missing.
        """
        local_src_dir_cfg = self._get_config_value(
            lambda: self.config.data.geospatial.nasadem.local_dir
        )
        if not local_src_dir_cfg:
            raise ValueError("NASADEM_LOCAL_DIR must be configured for NASADEM_LOCAL acquirer")
        local_src_dir = Path(local_src_dir_cfg)
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if not local_src_dir.exists():
            raise FileNotFoundError(f"NASADEM_LOCAL_DIR not found: {local_src_dir}")

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        for lat in range(lat_min, lat_max):
            for lon in range(lon_min, lon_max):
                lat_str = f"n{lat:02d}" if lat >= 0 else f"s{-lat:02d}"
                lon_str = f"e{lon:03d}" if lon >= 0 else f"w{-lon:03d}"
                # Common NASADEM format: n46w122.hgt or .tif
                pattern = f"{lat_str}{lon_str}*.tif"
                matches = list(local_src_dir.glob(pattern))
                if not matches:
                    pattern = f"{lat_str}{lon_str}*.hgt"
                    matches = list(local_src_dir.glob(pattern))

                if matches:
                    tile_paths.append(matches[0])

        if not tile_paths:
            raise FileNotFoundError(f"No NASADEM tiles found in {local_src_dir} for bbox {self.bbox}")

        if len(tile_paths) == 1:
            # We don't want to move original files, so we crop/copy
            with rasterio.open(tile_paths[0]) as src:
                win = from_bounds(self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max'], self.bbox['lat_max'], src.transform)
                data = src.read(1, window=win)
                meta = src.meta.copy()
                meta.update({"height": data.shape[0], "width": data.shape[1], "transform": src.window_transform(win)})
            with rasterio.open(out_path, "w", **meta) as dst: dst.write(data, 1)
        else:
            src_files = [rasterio.open(p) for p in tile_paths]
            mosaic, out_trans = rio_merge(src_files)
            out_meta = src_files[0].meta.copy()
            out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
            with rasterio.open(out_path, "w", **out_meta) as dest: dest.write(mosaic)
            for src in src_files: src.close()

        return out_path


# =============================================================================
# SRTM GL1 (30m) via OpenTopography S3
# =============================================================================

@AcquisitionRegistry.register('SRTM')
class SRTMAcquirer(BaseAcquisitionHandler, RetryMixin, _TileDownloadMixin):
    """SRTM GL1 (30m) DEM acquisition via OpenTopography S3.

    Downloads SRTM GL1 1-arc-second tiles from OpenTopography's public S3
    bucket. Coverage: 60N to 56S latitude.

    SRTM GL1:
        Resolution: 30m (1 arc-second)
        Coverage: 60N - 56S latitude
        Format: Cloud-Optimized GeoTIFF (COG)
        Datum: WGS84 (EPSG:4326)

    Tile Scheme:
        URL: https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm/{N46}{W122}.tif
        Naming: {N/S}{lat:02d}{E/W}{lon:03d}.tif (Cloud-Optimized GeoTIFF)

    References:
        - OpenTopography: https://opentopography.org/
        - SRTM: https://www2.jpl.nasa.gov/srtm/
    """

    _BASE_URL = "https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm"

    def download(self, output_dir: Path) -> Path:
        """Download SRTM GL1 tiles from OpenTopography S3 and merge into a domain DEM.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the merged DEM GeoTIFF.
        """
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading SRTM GL1 (30m) for bbox: {self.bbox}")

        # Coverage check: warn if bbox outside 60N-56S
        if self.bbox['lat_max'] > 60 or self.bbox['lat_min'] < -56:
            self.logger.warning(
                "SRTM coverage is 60N to 56S. Tiles outside this range will be unavailable. "
                f"Requested bbox lat range: {self.bbox['lat_min']} to {self.bbox['lat_max']}"
            )

        lat_min = max(math.floor(self.bbox['lat_min']), -56)
        lat_max = min(math.ceil(self.bbox['lat_max']), 60)
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            session = create_robust_session(max_retries=5, backoff_factor=2.0)

            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    tile_name = f"{lat_str}{lon_str}"
                    url = f"{self._BASE_URL}/{tile_name}.tif"

                    local_tile = dem_dir / f"temp_srtm_{tile_name}.tif"
                    if local_tile.exists():
                        if not self._validate_tile(local_tile, tile_name):
                            local_tile.unlink()
                        else:
                            self.logger.info(f"Using cached tile: {tile_name}")
                            tile_paths.append(local_tile)
                            continue

                    self.logger.info(f"Fetching SRTM tile: {tile_name}")
                    tile_result = self._download_tile_with_retry(
                        session, url, local_tile, tile_name
                    )
                    if tile_result:
                        tile_paths.append(local_tile)

            if not tile_paths:
                raise FileNotFoundError(f"No SRTM tiles found for bbox: {self.bbox}")

            self._merge_tiles(tile_paths, out_path)

        except (
            requests.RequestException,
            rasterio.errors.RasterioError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error downloading/processing SRTM: {e}")
            for p in tile_paths:
                if p.exists() and p != out_path:
                    p.unlink(missing_ok=True)
            raise

        return out_path


# =============================================================================
# ETOPO 2022 via NOAA OPeNDAP
# =============================================================================

@AcquisitionRegistry.register('ETOPO2022')
class ETOPO2022Acquirer(BaseAcquisitionHandler):
    """ETOPO 2022 global relief model acquisition via NOAA OPeNDAP.

    Downloads elevation data from the ETOPO 2022 global relief model using
    OPeNDAP subsetting. Supports multiple resolutions and surface/bedrock
    variants.

    ETOPO 2022:
        Resolution: 15 arc-second, 30 arc-second, or 60 arc-second
        Coverage: Global (90S - 90N)
        Format: NetCDF via OPeNDAP, output as GeoTIFF
        Datum: WGS84 (EPSG:4326)
        Includes both land elevation and ocean bathymetry

    Config Keys:
        ETOPO_RESOLUTION: '15s', '30s', or '60s' (default: '60s')
        ETOPO_VARIANT: 'surface' or 'bedrock' (default: 'surface')

    References:
        - NOAA ETOPO: https://www.ncei.noaa.gov/products/etopo-global-relief-model
    """

    _OPENDAP_TEMPLATE = (
        "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/{res}/"
        "{res}_{variant}_elev_netcdf/ETOPO_2022_v1_{res}_N90W180_{variant}.nc"
    )

    def download(self, output_dir: Path) -> Path:
        """Download an ETOPO 2022 subset via OPeNDAP and write as GeoTIFF.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the domain DEM GeoTIFF.
        """
        import xarray as xr

        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        resolution = self._get_config_value(lambda: None, default='60s', dict_key='ETOPO_RESOLUTION')
        variant = self._get_config_value(lambda: None, default='surface', dict_key='ETOPO_VARIANT')

        self.logger.info(
            f"Downloading ETOPO 2022 ({resolution}, {variant}) for bbox: {self.bbox}"
        )

        url = self._OPENDAP_TEMPLATE.format(res=resolution, variant=variant)
        self.logger.info(f"OPeNDAP URL: {url}")

        try:
            ds = xr.open_dataset(url, engine='netcdf4')

            # Determine lat/lon variable names (ETOPO uses 'lat'/'lon')
            lat_var = 'lat' if 'lat' in ds.coords else 'y'
            lon_var = 'lon' if 'lon' in ds.coords else 'x'

            # Handle lat axis direction (ascending vs descending)
            lat_vals = ds[lat_var].values
            lat_ascending = lat_vals[0] < lat_vals[-1]

            if lat_ascending:
                lat_slice = slice(self.bbox['lat_min'], self.bbox['lat_max'])
            else:
                lat_slice = slice(self.bbox['lat_max'], self.bbox['lat_min'])

            lon_slice = slice(self.bbox['lon_min'], self.bbox['lon_max'])

            subset = ds.sel({lat_var: lat_slice, lon_var: lon_slice})

            # Get the elevation variable (typically 'z')
            elev_var = None
            for vname in ['z', 'elevation', 'elev']:
                if vname in subset.data_vars:
                    elev_var = vname
                    break
            if elev_var is None:
                elev_var = list(subset.data_vars)[0]

            data = subset[elev_var].values
            if data.ndim == 2:
                data = data[np.newaxis, :, :]  # Add band dimension

            # Ensure north-up orientation
            lats = subset[lat_var].values
            if lats[0] < lats[-1]:
                data = data[:, ::-1, :]
                lats = lats[::-1]

            lons = subset[lon_var].values
            transform = rio_from_bounds(
                lons.min(), lats.min(), lons.max(), lats.max(),
                data.shape[2], data.shape[1]
            )

            with rasterio.open(
                out_path, 'w', driver='GTiff',
                height=data.shape[1], width=data.shape[2],
                count=1, dtype=data.dtype,
                crs='EPSG:4326', transform=transform,
                compress='lzw',
            ) as dst:
                dst.write(data)

            ds.close()

        except (
            rasterio.errors.RasterioError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error downloading/processing ETOPO 2022: {e}")
            raise

        return out_path


# =============================================================================
# Mapzen Terrain Tiles via AWS S3
# =============================================================================

@AcquisitionRegistry.register('MAPZEN')
class MapzenAcquirer(BaseAcquisitionHandler, RetryMixin, _TileDownloadMixin):
    """Mapzen terrain tile acquisition from AWS S3 Skadi dataset.

    Downloads Mapzen/Tilezen terrain tiles (SRTM-derived, void-filled) from
    the public AWS S3 bucket. Tiles are gzip-compressed HGT files in the
    Skadi directory structure.

    Mapzen Terrain Tiles:
        Resolution: ~30m (1 arc-second)
        Coverage: Global (derived from multiple sources)
        Format: gzip-compressed HGT, output as GeoTIFF
        Datum: WGS84 (EPSG:4326)

    Tile Scheme:
        URL: https://elevation-tiles-prod.s3.amazonaws.com/skadi/{LAT_DIR}/{LAT_DIR}{LON_DIR}.hgt.gz
        Example: .../skadi/N46/N46W122.hgt.gz

    References:
        - Tilezen Joerd: https://github.com/tilezen/joerd
        - AWS Registry: https://registry.opendata.aws/terrain-tiles/
    """

    _BASE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"

    def download(self, output_dir: Path) -> Path:
        """Download Mapzen/Skadi .hgt.gz tiles, decompress, and merge into a domain DEM.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the merged DEM GeoTIFF.
        """
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading Mapzen terrain tiles for bbox: {self.bbox}")

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            session = create_robust_session(max_retries=5, backoff_factor=2.0)

            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_dir = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_dir = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    tile_name = f"{lat_dir}{lon_dir}"
                    url = f"{self._BASE_URL}/{lat_dir}/{tile_name}.hgt.gz"

                    local_tile = dem_dir / f"temp_mapzen_{tile_name}.tif"
                    if local_tile.exists():
                        if not self._validate_tile(local_tile, tile_name):
                            local_tile.unlink()
                        else:
                            self.logger.info(f"Using cached tile: {tile_name}")
                            tile_paths.append(local_tile)
                            continue

                    # Download .hgt.gz, decompress, convert to GeoTIFF
                    # HGT filename must match N##E###.hgt pattern for rasterio SRTM driver
                    gz_path = dem_dir / f"temp_mapzen_{tile_name}.hgt.gz"
                    hgt_path = dem_dir / f"{tile_name}.hgt"

                    self.logger.info(f"Fetching Mapzen tile: {tile_name}")
                    tile_result = self._download_tile_with_retry(
                        session, url, gz_path, tile_name
                    )
                    if tile_result:
                        # Decompress gzip
                        with gzip.open(gz_path, 'rb') as gz_in:
                            with open(hgt_path, 'wb') as hgt_out:
                                hgt_out.write(gz_in.read())
                        gz_path.unlink(missing_ok=True)

                        # Convert HGT to GeoTIFF
                        with rasterio.open(hgt_path) as src:
                            meta = src.meta.copy()
                            meta.update({"driver": "GTiff", "compress": "lzw"})
                            with rasterio.open(local_tile, "w", **meta) as dst:
                                dst.write(src.read())
                        hgt_path.unlink(missing_ok=True)
                        tile_paths.append(local_tile)

            if not tile_paths:
                raise FileNotFoundError(f"No Mapzen tiles found for bbox: {self.bbox}")

            self._merge_tiles(tile_paths, out_path)

        except (
            requests.RequestException,
            rasterio.errors.RasterioError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error downloading/processing Mapzen tiles: {e}")
            for p in tile_paths:
                if p.exists() and p != out_path:
                    p.unlink(missing_ok=True)
            raise

        return out_path


# =============================================================================
# ALOS AW3D30 via Microsoft Planetary Computer STAC
# =============================================================================

@AcquisitionRegistry.register('ALOS')
class ALOSAcquirer(BaseAcquisitionHandler, RetryMixin, _TileDownloadMixin):
    """ALOS AW3D30 DEM acquisition via Microsoft Planetary Computer STAC API.

    Downloads ALOS World 3D 30m DEM tiles from Microsoft Planetary Computer
    using the STAC API for spatial search and signed URL access.

    ALOS AW3D30:
        Resolution: 30m (1 arc-second)
        Coverage: Global (approx. 82N to 82S)
        Format: Cloud-Optimized GeoTIFF (COG)
        Datum: WGS84 (EPSG:4326)
        Source: JAXA

    Dependencies:
        Requires optional packages: planetary-computer, pystac-client
        Install with: pip install symfluence[alos]

    References:
        - JAXA ALOS: https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm
        - Planetary Computer: https://planetarycomputer.microsoft.com/dataset/alos-dem
    """

    _STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    _COLLECTION = "alos-dem"

    def download(self, output_dir: Path) -> Path:
        """Download ALOS AW3D30 tiles via Planetary Computer STAC and merge into a domain DEM.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the merged DEM GeoTIFF.

        Raises:
            ImportError: If ``planetary-computer`` or ``pystac-client`` are not installed.
        """
        try:
            import planetary_computer
            import pystac_client
        except ImportError:
            raise ImportError(
                "ALOS DEM acquisition requires 'planetary-computer' and 'pystac-client' packages. "
                "Install with: pip install planetary-computer pystac-client "
                "or: pip install symfluence[alos]"
            ) from None

        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading ALOS AW3D30 (30m) for bbox: {self.bbox}")

        bbox_tuple = (
            self.bbox['lon_min'], self.bbox['lat_min'],
            self.bbox['lon_max'], self.bbox['lat_max'],
        )

        try:
            catalog = pystac_client.Client.open(
                self._STAC_API_URL,
                modifier=planetary_computer.sign_inplace,
            )

            search = catalog.search(
                collections=[self._COLLECTION],
                bbox=bbox_tuple,
                max_items=100,
            )

            items = list(search.items())
            if not items:
                raise FileNotFoundError(
                    f"No ALOS DEM tiles found for bbox: {self.bbox}"
                )

            self.logger.info(f"Found {len(items)} ALOS tiles covering bbox")

            session = create_robust_session(max_retries=5, backoff_factor=2.0)
            tile_paths = []

            for item in items:
                # Get the data asset (typically 'data' or 'elevation')
                asset_key = 'data' if 'data' in item.assets else list(item.assets.keys())[0]
                asset = item.assets[asset_key]
                signed_url = asset.href

                tile_name = item.id
                local_tile = dem_dir / f"temp_alos_{tile_name}.tif"

                if local_tile.exists():
                    if not self._validate_tile(local_tile, tile_name):
                        local_tile.unlink()
                    else:
                        self.logger.info(f"Using cached tile: {tile_name}")
                        tile_paths.append(local_tile)
                        continue

                self.logger.info(f"Fetching ALOS tile: {tile_name}")
                tile_result = self._download_tile_with_retry(
                    session, signed_url, local_tile, tile_name
                )
                if tile_result:
                    tile_paths.append(tile_result)

            if not tile_paths:
                raise FileNotFoundError(
                    f"No ALOS DEM tiles could be downloaded for bbox: {self.bbox}"
                )

            self._merge_tiles(tile_paths, out_path)

        except ImportError:
            raise
        except (
            requests.RequestException,
            rasterio.errors.RasterioError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as e:
            self.logger.error(f"Error downloading/processing ALOS DEM: {e}")
            raise

        return out_path
