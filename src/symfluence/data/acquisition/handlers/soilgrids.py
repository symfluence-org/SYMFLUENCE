# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SoilGrids Acquisition Handler

Cloud-based acquisition of global soil classification data with dual-source strategy:
- Primary: HydroShare (cached, GeoTIFF pre-converted)
- Fallback: OGC WCS service (authoritative but slower)

Soil Classification:
    World Reference Base (WRB) system:
    - 28 soil groups (e.g., Acrisols, Cambisols, Ferralsols)
    - Integer codes 1-28 for classification
    - Applicable globally for 0-5cm or 5-15cm depth

Key Features:
    Dual-Source Strategy:
    - Primary fast/cached source (HydroShare)
    - Fallback authoritative source (SoilGrids WCS)
    - Graceful fallback on primary failure

    Caching:
    - Global cache directories for HydroShare downloads
    - Per-domain output files in project_dir/attributes/soilclass/

    Retry Logic:
    - Exponential backoff for transient failures
    - Configurable max retries and backoff factors
    - Robust session creation with connection pooling

References:
    - Poggio et al. (2021). SoilGrids 2.0: Producing soil class predictions
      Scientific Data, 8, 128
    - ISRIC SoilGrids: https://www.soilgrids.org/
    - HydroShare: https://www.hydroshare.org/
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import rasterio
import requests
from rasterio.windows import from_bounds

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..utils import create_robust_session, download_file_resumable


@R.acquisition_handlers.add('SOILGRIDS')
class SoilGridsAcquirer(BaseAcquisitionHandler, RetryMixin):
    """SoilGrids v2 soil classification acquisition with dual-source strategy.

    Downloads global soil class raster data (World Reference Base classification)
    using intelligent source selection with automatic fallback:

    Acquisition Strategy:
        1. Primary Source: HydroShare (recommended for production)
           - Pre-converted GeoTIFF format (no format conversion needed)
           - Globally cached, reducing server load
           - Better availability and error handling
           - Download: ZIP archive → extract → crop to domain

        2. Fallback Source: SoilGrids OGC WCS service
           - Authoritative ISRIC SoilGrids v2 database
           - On-demand subsetting via WCS parameters
           - Slower and less reliable (server-side processing)
           - Direct GeoTIFF output, no extraction needed

    Soil Classification:
        World Reference Base (WRB) system:
        - 28 soil groups (e.g., Acrisols, Cambisols, Ferralsols)
        - Integer codes 1-28 for classification
        - Applicable globally for 0-5cm or 5-15cm depth

    Output:
        GeoTIFF file: domain_{domain_name}_soil_classes.tif
        - Variable: WRB soil class codes (integers 1-28)
        - Resolution: 250m global
        - Compressed: LZW compression

    Configuration:
        Primary source (HydroShare):
        - Automatic detection from config
        - No explicit configuration needed

        Fallback source (WCS):
        - config.data.geospatial.soilgrids.layer: WRB layer name (default: 'wrb_0-5cm_mode')
        - config.data.geospatial.soilgrids.coverage_id: WCS coverage ID
        - config.data.geospatial.soilgrids.wcs_map: WCS service map path

    Error Handling:
        - HydroShare failures: Log warning, attempt WCS fallback
        - WCS failures: Log error with snippet of response (helps debug)
        - WCS HTML error detection: Checks Content-Type and first bytes
        - Partial downloads: Verified with Content-Length check

    References:
        - Poggio et al. (2021). SoilGrids 2.0: Producing soil class predictions
          Scientific Data, 8, 128
        - ISRIC SoilGrids: https://www.soilgrids.org/
        - HydroShare: https://www.hydroshare.org/
    """

    def download(self, output_dir: Path) -> Path:
        soil_dir = self._attribute_dir("soilclass")
        out_p = soil_dir / f"domain_{self.domain_name}_soil_classes.tif"

        if self._skip_if_exists(out_p):
            return out_p

        # Try HydroShare first (more reliable, cached globally)
        try:
            self.logger.info("Acquiring soil class data from HydroShare (primary source)")
            return self._download_hydroshare_soilclasses(out_p)
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"HydroShare download failed, trying SoilGrids WCS: {exc}", exc_info=True)

        # Fallback to SoilGrids WCS
        try:
            return self._download_soilgrids_wcs(out_p)
        except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Both soil data sources failed. Last error: {exc}")
            raise

    def _download_soilgrids_wcs(self, out_p: Path) -> Path:
        """Download soil data from SoilGrids WCS service (fallback source).

        Uses OGC Web Coverage Service (WCS) to request soil class raster data
        directly from ISRIC's SoilGrids v2 service. This is the authoritative
        source but slower and less reliable than HydroShare.

        WCS Parameters:
            SERVICE: WCS (Web Coverage Service)
            VERSION: 2.0.1 (OGC standard version)
            REQUEST: GetCoverage (retrieve gridded data)
            COVERAGEID: WRB soil classification layer identifier
            FORMAT: GEOTIFF_INT16 (16-bit integer GeoTIFF)
            SUBSETTINGCRS: EPSG:4326 (WGS84 lat/lon)
            SUBSET: Latitude and Longitude bounds to spatial window

        Args:
            out_p: Output GeoTIFF file path

        Returns:
            Path: Output file path

        Raises:
            ValueError: If WCS returns HTML error or unexpected content
            requests.HTTPError: If HTTP request fails
        """
        self.logger.info("Acquiring soil class data from SoilGrids WCS (fallback)")

        layer = self._get_config_value(
            lambda: self.config.data.geospatial.soilgrids.layer,
            dict_key='SOILGRIDS_LAYER',
        )
        wcs_map = self._get_config_value(
            lambda: self.config.data.geospatial.soilgrids.wcs_map,
            dict_key='SOILGRIDS_WCS_MAP',
        )
        coverage = self._get_config_value(
            lambda: self.config.data.geospatial.soilgrids.coverage_id, default=layer
        ) or layer

        params = [
            ("map", wcs_map), ("SERVICE", "WCS"), ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"), ("COVERAGEID", coverage),
            ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("SUBSET", f"Lat({self.bbox['lat_min']},{self.bbox['lat_max']})"),
            ("SUBSET", f"Lon({self.bbox['lon_min']},{self.bbox['lon_max']})")
        ]

        resp = requests.get("https://maps.isric.org/mapserv", params=params, stream=True, timeout=60)
        resp.raise_for_status()

        content_type = (resp.headers.get("Content-Type") or "").lower()
        chunks = resp.iter_content(chunk_size=65536)
        first_chunk = next(chunks, b"")

        if "text/html" in content_type or first_chunk.lstrip().startswith(b"<"):
            snippet = first_chunk[:200].decode("utf-8", errors="ignore")
            raise ValueError(f"SoilGrids WCS returned HTML response: {snippet}")

        if not first_chunk.startswith((b"II*\x00", b"MM\x00*")):
            snippet = first_chunk[:200].decode("utf-8", errors="ignore")
            raise ValueError(f"SoilGrids WCS returned unexpected content: {snippet}")

        with open(out_p, "wb") as f:
            f.write(first_chunk)
            for chunk in chunks:
                f.write(chunk)

        self.logger.info(f"✓ Soil data acquired from SoilGrids WCS: {out_p}")
        return out_p

    def _download_hydroshare_soilclasses(self, out_p: Path) -> Path:
        """Download soil data from HydroShare (preferred primary source).

        HydroShare is a data and model repository where pre-processed SoilGrids
        data is cached and served globally. This approach is preferred because:
        - Pre-converted to GeoTIFF (no format conversion overhead)
        - Globally mirrored (faster, more reliable than ISRIC)
        - Better progress reporting and error handling
        - Reduces load on ISRIC servers

        The download uses RetryMixin for exponential backoff on transient failures.
        Downloads are cached locally to avoid re-downloading for multiple domains.

        Args:
            out_p: Output GeoTIFF file path

        Returns:
            Path: Output file path

        Side Effects:
            - Creates local cache directory for zip archives
            - Extracts GeoTIFF files from downloaded archive
            - Subsets extracted raster to domain bounding box via rasterio
        """
        cache_dir_cfg = self._get_config_value(
            lambda: self.config.data.geospatial.soilgrids.hs_cache_dir,
            dict_key='SOILGRIDS_HS_CACHE_DIR',
        )
        if cache_dir_cfg == 'default':
            # Data-dir-level cache: the HydroShare archive is global (~1 GB,
            # up to an hour to fetch), so every domain shares one download.
            cache_dir = self.data_dir / "cache" / "soilgrids"
        else:
            cache_dir = Path(cache_dir_cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)

        resource_id = self._get_config_value(
            lambda: self.config.data.geospatial.soilgrids.hs_resource_id,
            dict_key='SOILGRIDS_HS_RESOURCE_ID',
        )
        hs_url = self._get_config_value(
            lambda: self.config.data.geospatial.soilgrids.hs_api_url,
            default=f"https://www.hydroshare.org/hsapi/resource/{resource_id}/"
        ) or f"https://www.hydroshare.org/hsapi/resource/{resource_id}/"

        zip_path = cache_dir / f"soilgrids_{resource_id}.zip"

        # Download with retry logic using mixin
        if not zip_path.exists() or zip_path.stat().st_size == 0 or self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            # PID-unique temp: concurrent workflows share this cache dir, so two
            # writers on one .part file would corrupt both downloads. Pass it as
            # part_path so download_file_resumable resumes from it across
            # mid-stream interruptions (HydroShare's S3 backend honours Range
            # requests) instead of re-fetching the ~415 MB archive from byte zero
            # on every network blip — a hard repro blocker on slow/flaky links.
            tmp_path = zip_path.with_suffix(f".zip.part.{os.getpid()}")
            self.logger.info("Downloading soil data from HydroShare...")
            session = create_robust_session(max_retries=0)
            download_file_resumable(
                hs_url,
                zip_path,
                session=session,
                timeout=600,
                part_path=tmp_path,
                logger_=self.logger,
            )

        # Extract and crop to domain
        tif_name = "data/contents/usda_mode_soilclass_250m_ll.tif"
        cached_tif = cache_dir / "usda_mode_soilclass_250m_ll.tif"

        if not cached_tif.exists():
            self.logger.info("Extracting soil data from archive...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(f"{resource_id}/{tif_name}") as src, open(cached_tif, "wb") as dst:
                    dst.write(src.read())

        with rasterio.open(cached_tif) as src:
            win = from_bounds(
                self.bbox["lon_min"],
                self.bbox["lat_min"],
                self.bbox["lon_max"],
                self.bbox["lat_max"],
                src.transform,
            )
            out_d = src.read(1, window=win)  # type: ignore[call-arg]
            meta = src.meta.copy()
            meta.update({
                "height": out_d.shape[0],
                "width": out_d.shape[1],
                "transform": src.window_transform(win),
                "compress": "lzw",
            })

        with rasterio.open(out_p, "w", **meta) as dst:
            dst.write(out_d, 1)  # type: ignore[call-arg]

        self.logger.info(f"✓ Soil data acquired from HydroShare: {out_p}")
        return out_p
