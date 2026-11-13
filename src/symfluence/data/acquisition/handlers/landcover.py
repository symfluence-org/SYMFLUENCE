# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Land Cover Acquisition Handlers

Cloud-based acquisition of global and regional land cover datasets:
- MODIS MCD12Q1 v061: Global land cover (500m)
- USGS NLCD: USA-only land cover (30m)

Key Features:
    Multi-Year Processing:
    - MODIS supports year range with mode calculation
    - Produces single-year consensus classification
    - Useful for filtering year-to-year noise

    Caching:
    - Global cache directories for multi-year MODIS downloads
    - Per-domain output files in project_dir/attributes/landclass/

    Subsetting:
    - Automatic clipping to domain bounding box
    - Window-based rasterio operations (memory-efficient)
    - WCS parameter-based subsetting for remote sources

Data Sources:
    MODIS Landcover:
    - Zenodo: https://zenodo.org/records/8367523
    - Variable: MCD12Q1 v061 classification
    - Resolution: 500m
    - Multi-year: Average mode across years

    USGS NLCD:
    - WCS endpoint: USGS/MRLC server
    - Coverage: USA only
    - Resolution: 30m
    - Classes: Anderson classification system

References:
    - Friedl et al. (2019). MODIS Collection 6 land cover product
      Remote Sensing of Environment, 224, 400-414
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.windows import Window, from_bounds

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..utils import create_robust_session, download_file_resumable


@R.acquisition_handlers.add('MODIS_LANDCOVER')
class MODISLandcoverAcquirer(BaseAcquisitionHandler):
    """MODIS MCD12Q1 land cover acquisition with multi-year support.

    Downloads and processes global MODIS Collection 6 land cover classification
    (MCD12Q1 v061) from Zenodo archive with optional multi-year mode averaging.
    Enables flexible source selection (local file, Zenodo archive) and temporal
    aggregation for robust land cover datasets.

    MODIS Land Cover (MCD12Q1):
        Data Type: Land cover classification (International Geosphere-Biosphere Programme)
        Resolution: 500m
        Version: Collection 6.1 (v061)
        Temporal Resolution: Annual (January 1 - December 31)
        Spatial Coverage: Global (90°S - 90°N)

    Acquisition Modes:
        1. Local File (fastest):
           - Uses existing local/remote GeoTIFF file
           - Configured via LANDCOVER_LOCAL_FILE
           - Automatic VirtualFileSystem (VSI) handling for HTTP URLs

        2. Zenodo Archive (production):
           - Multi-file download from Zenodo (https://zenodo.org/records/8367523)
           - Automatic caching to avoid re-downloads
           - Mode calculation for multi-year datasets

    Multi-Year Processing:
        Configuration options (in precedence order):
        1. Explicit years list: data.geospatial.modis_landcover.years = [2018, 2019, 2020]
        2. Year range: data.geospatial.modis_landcover.start_year + end_year
        3. Single year: data.geospatial.modis_landcover.year
        4. Default: 2019

        Mode Calculation:
        - Stacks multi-year data into 3D array
        - Computes mode (most frequent class) along time axis
        - Produces single-year consensus classification
        - Useful for filtering year-to-year noise

    Output:
        GeoTIFF file: domain_{domain_name}_land_classes.tif (or custom name)
        - Variable: IGBP land cover classes (1-17)
        - Resolution: 500m
        - Compressed: LZW compression
        - Custom naming: Configurable via LAND_CLASS_NAME

    Configuration:
        Source Selection:
        - LANDCOVER_LOCAL_FILE: Path/URL to existing GeoTIFF
        - If set, uses local file instead of Zenodo

        Multi-Year:
        - data.geospatial.modis_landcover.years: List of years [int]
        - data.geospatial.modis_landcover.start_year/end_year: Year range
        - data.geospatial.modis_landcover.year: Single year

        Zenodo Source:
        - data.geospatial.modis_landcover.base_url: Zenodo URL (default provided)
        - data.geospatial.modis_landcover.cache_dir: Cache directory ('default' or path)

    Subsetting Strategy:
        Window-Based Clipping:
        - Uses rasterio window operations (memory-efficient)
        - Clips to domain bounding box coordinates
        - Preserves georeferencing and projection

        Data Validation:
        - Checks for NaN values in output
        - Verifies output contains valid land cover classes

    References:
        - Friedl et al. (2019). MODIS Collection 6 global land cover
          Remote Sensing of Environment, 224, 400-414
        - MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid (MCD12Q1)
          NASA DAAC: https://lpdaac.usgs.gov/
        - Zenodo Archive: https://zenodo.org/records/8367523
    """
    def _download_with_size_check(self, url: str, dest_path: Path) -> None:
        """Download a file with atomic rename and Content-Length verification.

        Args:
            url: Remote URL to download.
            dest_path: Final destination path (a ``.part`` suffix is used during transfer).

        Raises:
            IOError: If the downloaded size does not match the Content-Length header.
        """
        # PID-unique temp name: concurrent workflows may share the cache dir,
        # and two writers on one .part file would corrupt both downloads.
        # download_file_resumable resumes from this partial file across
        # mid-stream interruptions (Zenodo honours Range requests) instead of
        # re-fetching the ~130 MB tile from byte zero on every network blip.
        tmp_path = dest_path.with_suffix(dest_path.suffix + f".part.{os.getpid()}")
        session = create_robust_session(max_retries=0)
        download_file_resumable(
            url,
            dest_path,
            session=session,
            chunk_size=1024 * 1024,
            timeout=60,
            part_path=tmp_path,
            logger_=self.logger,
        )

    def download(self, output_dir: Path) -> Path:
        """Download MODIS land cover data from a local file or Zenodo archive.

        Supports single-year or multi-year mode averaging.  When multiple years
        are requested the per-pixel mode (most frequent class) is computed to
        produce a temporally robust classification.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the land cover GeoTIFF.
        """
        lc_dir = self._attribute_dir("landclass")
        land_name = self._get_config_value(lambda: self.config.domain.land_class_name, dict_key='LAND_CLASS_NAME')
        if land_name == "default":
            land_name = f"domain_{self.domain_name}_land_classes.tif"
        out_path = lc_dir / land_name

        src_p = self._get_config_value(lambda: None, default=None, dict_key='LANDCOVER_LOCAL_FILE')
        if src_p:
            url = f"/vsicurl/{src_p}" if str(src_p).startswith("http") else src_p
            with rasterio.open(url) as src:
                win = from_bounds(self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max'], self.bbox['lat_max'], src.transform)
                # Ensure window covers at least 1 pixel (handles point domains)
                col_off, row_off, win_width, win_height = (
                    int(win.col_off), int(win.row_off),
                    max(1, int(np.ceil(win.width))),
                    max(1, int(np.ceil(win.height)))
                )
                win = Window(col_off, row_off, win_width, win_height)
                out_d = src.read(1, window=win, boundless=True, fill_value=255)
                meta = src.meta.copy()
                meta.update({"height": out_d.shape[0], "width": out_d.shape[1], "transform": src.window_transform(win)})
            with rasterio.open(out_path, "w", **meta) as dst: dst.write(out_d, 1)
            return out_path
        # Multi-year Zenodo fallback logic follows Fire-Engine-Framework legacy fetcher
        self.logger.info("Fetching MODIS Land Cover (MCD12Q1 v061) from Zenodo")
        self.logger.info(f"MODIS landcover bbox: {self.bbox}")

        years = self._get_config_value(
            lambda: self.config.data.geospatial.modis_landcover.years
        )
        if isinstance(years, (list, tuple)):
            years = [int(y) for y in years]
        else:
            start_year = self._get_config_value(
                lambda: self.config.data.geospatial.modis_landcover.start_year
            )
            end_year = self._get_config_value(
                lambda: self.config.data.geospatial.modis_landcover.end_year
            )
            if start_year and end_year:
                years = list(range(int(start_year), int(end_year) + 1))
            else:
                landcover_year = self._get_config_value(
                    lambda: self.config.data.geospatial.modis_landcover.year
                )
                if landcover_year:
                    years = [int(landcover_year)]
                else:
                    years = [2019]

        base_url = self._get_config_value(
            lambda: self.config.data.geospatial.modis_landcover.base_url,
            dict_key='MODIS_LANDCOVER_BASE_URL',
        )
        cache_dir_cfg = self._get_config_value(
            lambda: self.config.data.geospatial.modis_landcover.cache_dir,
            dict_key='MODIS_LANDCOVER_CACHE_DIR',
        )
        if cache_dir_cfg == 'default':
            # Data-dir-level cache: the ~130 MB/year Zenodo tiles are global,
            # so every domain shares one download instead of re-fetching.
            cache_dir = self.data_dir / "cache" / "modis_landcover"
        else:
            cache_dir = Path(cache_dir_cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = []
        out_meta = None

        for year in years:
            fname = (
                "lc_mcd12q1v061.t1_c_500m_s_"
                f"{year}0101_{year}1231_go_epsg.4326_v20230818.tif"
            )
            url = f"{base_url}/{fname}"
            local_tmp = cache_dir / fname

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    if attempt:
                        # Zenodo truncates transfers under concurrent load;
                        # back off before re-downloading.
                        time.sleep(15 * attempt)
                    if not local_tmp.exists():
                        self._download_with_size_check(url, local_tmp)

                    with rasterio.open(local_tmp) as src:
                        win = from_bounds(
                            self.bbox["lon_min"],
                            self.bbox["lat_min"],
                            self.bbox["lon_max"],
                            self.bbox["lat_max"],
                            src.transform,
                        )
                        # Ensure window covers at least 1 pixel (handles point domains)
                        col_off, row_off, win_width, win_height = (
                            int(win.col_off), int(win.row_off),
                            max(1, int(np.ceil(win.width))),
                            max(1, int(np.ceil(win.height)))
                        )
                        win = Window(col_off, row_off, win_width, win_height)
                        data = src.read(1, window=win, boundless=True, fill_value=255)
                        if out_meta is None:
                            out_transform = src.window_transform(win)
                            out_meta = src.meta.copy()
                            out_meta.update({
                                "driver": "GTiff",
                                "height": data.shape[0],
                                "width": data.shape[1],
                                "transform": out_transform,
                                "compress": "lzw",
                                "nodata": 255,
                            })

                        if data.shape == (out_meta["height"], out_meta["width"]):
                            arrays.append(data)
                        else:
                            self.logger.warning(
                                f"MODIS landcover shape mismatch for {year}: {data.shape}"
                            )
                    break
                except rasterio.errors.RasterioIOError as exc:
                    self.logger.warning(
                        f"MODIS landcover read failed for {year} "
                        f"(attempt {attempt + 1}/{max_attempts}): {exc}"
                    )
                    local_tmp.unlink(missing_ok=True)
                except Exception as exc:  # noqa: BLE001 — must-not-raise contract
                    self.logger.warning(
                        f"MODIS landcover download failed for {year} "
                        f"(attempt {attempt + 1}/{max_attempts}): {exc}"
                    )
                    local_tmp.unlink(missing_ok=True)
                    if attempt == max_attempts - 1:
                        raise

        if not arrays:
            raise FileNotFoundError("No MODIS land cover data processed from Zenodo.")

        stack = np.stack(arrays, axis=0)

        def calc_mode(arr):
            valid = arr[arr != 255]
            if valid.size == 0:
                return 255
            vals, counts = np.unique(valid, return_counts=True)
            return vals[np.argmax(counts)]

        mode_data = np.apply_along_axis(calc_mode, 0, stack).astype("uint8")

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mode_data, 1)
        return out_path


@R.acquisition_handlers.add('USGS_NLCD')
class USGSLandcoverAcquirer(BaseAcquisitionHandler):
    """USGS National Land Cover Database (NLCD) acquisition handler.

    Downloads land cover data from the MRLC WCS endpoint for the conterminous
    United States.  Coverage is limited to CONUS; requests outside this extent
    will fail.

    NLCD:
        Resolution: 30m
        Coverage: Conterminous US
        Classification: Anderson Level II
        Default year: 2019

    Config Keys:
        data.geospatial.nlcd.coverage_id: WCS coverage identifier
            (default: ``'NLCD_2019_Land_Cover_L48'``).
    """

    def download(self, output_dir: Path) -> Path:
        """Download NLCD land cover via WCS and write as GeoTIFF.

        Args:
            output_dir: Base output directory (unused; output written to project tree).

        Returns:
            Path to the clipped land cover GeoTIFF.
        """
        lc_dir = self._attribute_dir("landclass")
        land_name = self._get_config_value(lambda: self.config.domain.land_class_name, dict_key='LAND_CLASS_NAME')
        if land_name == "default":
            land_name = f"domain_{self.domain_name}_land_classes.tif"
        out_path = lc_dir / land_name

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading USGS NLCD for bbox: {self.bbox}")

        # MRLC WCS Endpoint
        wcs_url = "https://www.mrlc.gov/geoserver/NLCD_Land_Cover/wcs"

        # Use 2019 data by default
        coverage_id = self._get_config_value(
            lambda: self.config.data.geospatial.nlcd.coverage_id,
            dict_key='NLCD_COVERAGE_ID',
        )

        # WCS 2.0.1 Params
        # Note: MRLC WCS can be picky about CRS. Requesting output in 4326.
        params = [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageid", coverage_id),
            ("subsettingcrs", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("outputcrs", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("subset", f"Lat({self.bbox['lat_min']},{self.bbox['lat_max']})"),
            ("subset", f"Long({self.bbox['lon_min']},{self.bbox['lon_max']})"),
            ("format", "image/geotiff")
        ]

        try:
            self.logger.info(f"Requesting NLCD coverage {coverage_id}")
            resp = requests.get(wcs_url, params=params, stream=True, timeout=120)
            resp.raise_for_status()

            # Check for XML error response
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "xml" in content_type:
                # Read a bit to see error
                snippet = next(resp.iter_content(2048)).decode("utf-8", errors="ignore")
                raise ValueError(f"NLCD WCS returned XML error: {snippet}")

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            self.logger.info(f"✓ NLCD acquired: {out_path}")
            return out_path

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"NLCD acquisition failed: {e}")
            if out_path.exists():
                out_path.unlink()
            raise
