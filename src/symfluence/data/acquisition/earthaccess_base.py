# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Earthaccess Base Class for NASA Earthdata Cloud Acquisition.

This module provides a base class with shared NASA Earthdata Cloud functionality
using earthaccess library and direct CMR API access. This approach is faster and
more reliable than AppEEARS for tiled products like MODIS and VIIRS.

Benefits over AppEEARS:
- No queue waiting (direct download)
- Resume capability (skips existing files)
- Parallel downloads
- More reliable for large requests

References:
- earthaccess: https://github.com/nsidc/earthaccess
- CMR API: https://cmr.earthdata.nasa.gov/search/
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .base import BaseAcquisitionHandler


class BaseEarthaccessAcquirer(BaseAcquisitionHandler):
    """
    Base class providing NASA Earthdata Cloud functionality via earthaccess/CMR.

    This base class handles:
    - Earthdata sessions via ``earthdata_auth.EarthdataAuth`` (earthaccess first)
    - CMR (Common Metadata Repository) granule search
    - Direct file downloads via earthaccess authenticated sessions
    - Parallel download management

    Subclasses should implement:
    - download(): Main entry point that orchestrates the acquisition workflow
    - Product-specific processing logic

    Attributes:
        CMR_URL: NASA CMR granule search endpoint
    """

    CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

    # ===== CMR Search Methods =====

    def _search_granules_cmr(
        self,
        short_name: str,
        version: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Dict[str, float]] = None,
        page_size: int = 2000
    ) -> List[Dict]:
        """
        Search for granules using NASA CMR API.

        Args:
            short_name: Product short name (e.g., 'MOD10A1')
            version: Product version (e.g., '61'). If None, returns all versions.
            start_date: Start date in ISO format (defaults to self.start_date)
            end_date: End date in ISO format (defaults to self.end_date)
            bbox: Bounding box dict with lon_min, lon_max, lat_min, lat_max
            page_size: Number of results per page

        Returns:
            List of granule entries from CMR
        """
        if start_date is None:
            start_date = self.start_date.strftime('%Y-%m-%d')
        if end_date is None:
            end_date = self.end_date.strftime('%Y-%m-%d')
        if bbox is None:
            bbox = self.bbox

        self.logger.info(f"Searching CMR for {short_name} granules...")

        all_granules = []
        page = 1

        while True:
            params = {
                'short_name': short_name,
                'temporal': f'{start_date}T00:00:00Z,{end_date}T23:59:59Z',
                'bounding_box': f'{bbox["lon_min"]},{bbox["lat_min"]},{bbox["lon_max"]},{bbox["lat_max"]}',
                'page_size': page_size,
                'page_num': page,
            }
            if version:
                params['version'] = version

            try:
                response = requests.get(self.CMR_URL, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                entries = data.get('feed', {}).get('entry', [])
                if not entries:
                    break

                all_granules.extend(entries)
                self.logger.debug(f"  Page {page}: {len(entries)} granules (total: {len(all_granules)})")

                if len(entries) < page_size:
                    break
                page += 1

            except (
                requests.RequestException,
                OSError,
                ValueError,
                TypeError,
                KeyError,
            ) as e:
                self.logger.warning(f"CMR search error on page {page}: {e}")
                break

        self.logger.info(f"Found {len(all_granules)} granules for {short_name}")
        return all_granules

    def _get_download_urls(
        self,
        granules: List[Dict],
        extensions: Tuple[str, ...] = ('.hdf', '.h5', '.nc', '.nc4')
    ) -> List[str]:
        """
        Extract download URLs from CMR granule entries.

        Args:
            granules: List of CMR granule entries
            extensions: File extensions to include

        Returns:
            List of download URLs
        """
        urls = []
        for g in granules:
            for link in g.get('links', []):
                href = link.get('href', '')
                if any(href.endswith(ext) for ext in extensions) and 'http' in href:
                    urls.append(href)
                    break
        return urls

    # ===== Download Methods =====

    def _download_with_earthaccess(
        self,
        urls: List[str],
        output_dir: Path,
        max_workers: int = 4,
        skip_existing: bool = True
    ) -> List[Path]:
        """
        Download files using earthaccess authenticated session.

        Args:
            urls: List of download URLs
            output_dir: Directory to save files
            max_workers: Number of parallel downloads (not used currently, sequential)
            skip_existing: Skip files that already exist

        Returns:
            List of downloaded file paths
        """
        if not urls:
            return []

        # earthaccess session; without earthaccess (or if its login fails),
        # a token / username-password session from the same lookup chain.
        session = self._earthdata_auth().session(
            prefer_earthaccess=True, purpose="NASA Earthdata Cloud downloads",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading {len(urls)} files to {output_dir}")

        downloaded = []
        for i, url in enumerate(urls):
            filename = url.split('/')[-1]
            output_path = output_dir / filename

            if skip_existing and output_path.exists():
                downloaded.append(output_path)
                continue

            if (i + 1) % 50 == 0:
                self.logger.info(f"  Downloading {i+1}/{len(urls)}: {filename}")

            try:
                response = session.get(url, timeout=120)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                downloaded.append(output_path)
            except (
                requests.RequestException,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                self.logger.warning(f"Failed to download {filename}: {e}")

        self.logger.info(f"Downloaded {len(downloaded)} files")
        return downloaded

    def _download_granules_earthaccess(
        self,
        short_name: str,
        output_dir: Path,
        version: Optional[str] = None,
        extensions: Tuple[str, ...] = ('.hdf', '.h5', '.nc', '.nc4')
    ) -> List[Path]:
        """
        Search and download granules for a product.

        Convenience method that combines CMR search and earthaccess download.

        Args:
            short_name: Product short name
            output_dir: Directory to save files
            version: Product version (optional)
            extensions: File extensions to download

        Returns:
            List of downloaded file paths
        """
        # Search for granules
        granules = self._search_granules_cmr(short_name, version=version)

        if not granules:
            self.logger.warning(f"No granules found for {short_name}")
            return []

        # Get download URLs
        urls = self._get_download_urls(granules, extensions=extensions)

        if not urls:
            self.logger.warning(f"No download URLs found for {short_name}")
            return []

        # Download
        return self._download_with_earthaccess(urls, output_dir)

    # ===== Utility Methods =====

    def _count_available_granules(
        self,
        short_name: str,
        version: Optional[str] = None
    ) -> int:
        """
        Count available granules without downloading metadata.

        Args:
            short_name: Product short name
            version: Product version

        Returns:
            Number of available granules
        """
        params = {
            'short_name': short_name,
            'temporal': f'{self.start_date.strftime("%Y-%m-%d")}T00:00:00Z,'
                       f'{self.end_date.strftime("%Y-%m-%d")}T23:59:59Z',
            'bounding_box': f'{self.bbox["lon_min"]},{self.bbox["lat_min"]},'
                           f'{self.bbox["lon_max"]},{self.bbox["lat_max"]}',
            'page_size': 1,
        }
        if version:
            params['version'] = version

        try:
            response = requests.get(self.CMR_URL, params=params, timeout=30)
            return int(response.headers.get('CMR-Hits', 0))
        except (requests.RequestException, OSError, ValueError, TypeError):
            return 0
