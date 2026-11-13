# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Global Lakes and Wetlands Database (GLWD) v2.0 Acquisition Handler

Cloud-based acquisition of the GLWD v2.0 global wetland and surface water
extent dataset from Figshare.

Provides gridded wetland/waterbody type and extent at 15 arcsecond (~500m)
resolution globally with 33 classes including lakes, rivers, marshes, bogs,
fens, swamps, and other inundated areas.

Data Source:
    Figshare: https://doi.org/10.6084/m9.figshare.28519994
    No authentication required, CC-BY 4.0 license

References:
    Lehner, B., et al. (2025). An updated global lakes and wetlands
    dataset (GLWD v2.0). Earth Syst. Sci. Data, 17, 2277-2310.

Configuration:
    GLWD_FORMAT: 'tif' (default) or 'gdb' - Download format
    GLWD_PRODUCT: 'combined' (default), 'area_pct', 'area_ha'
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional

import rasterio
from rasterio.windows import from_bounds

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..utils import create_robust_session, download_file_streaming

# Figshare direct file download URLs for GLWD v2.0
# Use ndownloader.figshare.com with file IDs (article-level URLs blocked by WAF)
_DOWNLOAD_URLS = {
    'combined_tif': 'https://ndownloader.figshare.com/files/54001814',    # combined_classes_tif.zip (925 MB)
    'area_pct_tif': 'https://ndownloader.figshare.com/files/54001775',    # area_by_class_pct_tif.zip (1.6 GB)
    'area_ha_tif': 'https://ndownloader.figshare.com/files/54001757',     # area_by_class_ha_tif.zip (1.6 GB)
    'combined_gdb': 'https://ndownloader.figshare.com/files/54001748',    # combined_classes_gdb.zip (918 MB)
    'area_pct_gdb': 'https://ndownloader.figshare.com/files/54001727',    # area_by_class_pct_gdb.zip (1.6 GB)
    'area_ha_gdb': 'https://ndownloader.figshare.com/files/54001703',     # area_by_class_ha_gdb.zip (1.6 GB)
}


@R.acquisition_handlers.add('GLWD')
@R.acquisition_handlers.add('GLWD_V2')
@R.acquisition_handlers.add('WETLANDS')
class GLWDAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    GLWD v2.0 global wetland database acquisition.

    Downloads combined wetland type/extent GeoTIFFs from Figshare,
    subsets to domain bounding box.

    Output:
        {project_dir}/attributes/water/glwd/
            domain_{name}_glwd_dominant_type.tif
            domain_{name}_glwd_wetland_extent.tif
    """

    def download(self, output_dir: Path) -> Path:
        glwd_dir = self._attribute_dir("water") / "glwd"
        glwd_dir.mkdir(parents=True, exist_ok=True)

        out_type = glwd_dir / f"domain_{self.domain_name}_glwd_dominant_type.tif"
        out_extent = glwd_dir / f"domain_{self.domain_name}_glwd_wetland_extent.tif"

        # Check if both outputs exist
        if out_type.exists() and out_extent.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            self.logger.info(f"GLWD data already exists: {glwd_dir}")
            return glwd_dir

        self.logger.info(f"Acquiring GLWD v2.0 for bbox: {self.bbox}")

        session = create_robust_session(max_retries=3, backoff_factor=2.0)
        cache_dir = glwd_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download and extract combined classes (GeoTIFF format)
        fmt = self._get_config_value(lambda: None, default='tif', dict_key='GLWD_FORMAT')
        product = self._get_config_value(lambda: None, default='combined', dict_key='GLWD_PRODUCT')
        url_key = f"{product}_{fmt}"

        url = _DOWNLOAD_URLS.get(url_key, _DOWNLOAD_URLS['combined_tif'])

        self._ensure_archive_downloaded(session, cache_dir, url, url_key)

        # Find and subset dominant type
        type_src = self._find_source_tif(
            cache_dir, ['*dominant*type*.tif', '*GLWD*type*.tif', '*class*.tif']
        )
        if type_src and not self._skip_if_exists(out_type):
            self._subset_to_bbox(type_src, out_type)
            self.logger.info(f"Saved dominant type: {out_type}")

        # Find and subset wetland extent
        extent_src = self._find_source_tif(
            cache_dir, ['*extent*.tif', '*GLWD*pct*.tif', '*total*.tif', '*combined*.tif']
        )
        if extent_src and not self._skip_if_exists(out_extent):
            self._subset_to_bbox(extent_src, out_extent)
            self.logger.info(f"Saved wetland extent: {out_extent}")

        self.logger.info(f"GLWD v2.0 acquisition complete: {glwd_dir}")
        return glwd_dir

    def _ensure_archive_downloaded(self, session, cache_dir: Path, url: str, key: str):
        """Download and extract archive if not cached."""
        existing_tifs = list(cache_dir.rglob("*.tif"))
        if existing_tifs:
            return

        zip_path = cache_dir / f"GLWD_v2_{key}.zip"
        if not zip_path.exists():
            self.logger.info("Downloading GLWD v2.0 from Figshare...")
            download_file_streaming(url, zip_path, session=session, timeout=600)

        self.logger.info("Extracting archive...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                from symfluence.core.archive_extraction import safe_zip_extract
                safe_zip_extract(zf, cache_dir)
        except zipfile.BadZipFile:
            self.logger.error("Downloaded file is not a valid zip")
            zip_path.unlink(missing_ok=True)
            raise

    def _find_source_tif(self, cache_dir: Path, patterns: List[str]) -> Optional[Path]:
        """Find a GeoTIFF matching one of the patterns."""
        for pattern in patterns:
            matches = list(cache_dir.rglob(pattern))
            if matches:
                return matches[0]
        # Fallback: return any .tif
        all_tifs = list(cache_dir.rglob("*.tif"))
        return all_tifs[0] if all_tifs else None

    def _subset_to_bbox(self, src_path: Path, dst_path: Path):
        """Subset a GeoTIFF to the domain bounding box."""
        with rasterio.open(src_path) as src:
            window = from_bounds(
                self.bbox['lon_min'], self.bbox['lat_min'],
                self.bbox['lon_max'], self.bbox['lat_max'],
                transform=src.transform
            )
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )

            data = src.read(1, window=window)
            transform = src.window_transform(window)

            meta = src.meta.copy()
            meta.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': transform,
                'compress': 'lzw',
            })

            with rasterio.open(dst_path, 'w', **meta) as dst:
                dst.write(data, 1)
