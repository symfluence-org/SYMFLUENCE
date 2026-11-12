# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""JRC Global Surface Water Acquisition Handler

Provides cloud acquisition for JRC (Joint Research Centre) Global Surface Water
data derived from Landsat imagery.

JRC Global Surface Water Overview:
    Data Type: Surface water extent and dynamics
    Resolution: 30m (Landsat-based)
    Coverage: Global
    Variables:
        - occurrence: Water occurrence (% of time)
        - recurrence: Inter-annual water recurrence
        - seasonality: Intra-annual water seasonality
        - change: Water extent change
        - transitions: Water state transitions
        - extent: Maximum water extent
    Record: 1984-present (yearly updates)
    Source: European Commission Joint Research Centre

Data Access:
    Primary: Google Cloud Storage (public bucket)
    Alternative: Direct download from JRC portal
    Format: GeoTIFF

URL Pattern:
    gs://global-surface-water/downloads3/
    https://storage.googleapis.com/global-surface-water/downloads2021/
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('JRC_WATER')
@R.acquisition_handlers.add('JRC_GSW')
class JRCWaterAcquirer(BaseAcquisitionHandler):
    """
    Acquires JRC Global Surface Water data from Google Cloud Storage.
    No authentication required - publicly available.
    """

    # Google Cloud Storage public bucket
    GCS_BASE = "https://storage.googleapis.com/global-surface-water/downloads2021"

    # Available datasets/layers
    DATASETS = {
        'occurrence': {
            'path': 'occurrence',
            'description': 'Water occurrence (% of valid observations)',
            'value_range': (0, 100),
        },
        'recurrence': {
            'path': 'recurrence',
            'description': 'Inter-annual water recurrence',
            'value_range': (0, 100),
        },
        'seasonality': {
            'path': 'seasonality',
            'description': 'Number of months water present per year',
            'value_range': (0, 12),
        },
        'change': {
            'path': 'change',
            'description': 'Change in water occurrence',
            'value_range': (-100, 100),
        },
        'transitions': {
            'path': 'transitions',
            'description': 'Water state transitions',
            'value_range': (0, 10),
        },
        'extent': {
            'path': 'extent',
            'description': 'Maximum water extent',
            'value_range': (0, 1),
        },
    }

    # Tile grid parameters (10x10 degree tiles)
    TILE_SIZE = 10  # degrees

    def download(self, output_dir: Path) -> Path:
        """
        Download JRC Global Surface Water data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to output directory containing downloaded files
        """
        self.logger.info("Starting JRC Global Surface Water acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset type from config
        dataset = self._get_config_value(
            lambda: None,
            default='occurrence',
            dict_key='JRC_WATER_DATASET'
        )
        if isinstance(dataset, str):
            dataset = dataset.lower()
        if dataset not in self.DATASETS:
            self.logger.warning(f"Unknown JRC dataset '{dataset}', defaulting to 'occurrence'")
            dataset = 'occurrence'

        # Determine tiles needed for bounding box
        tiles = self._get_tiles_for_bbox()

        self.logger.info(f"Downloading JRC {dataset} for {len(tiles)} tiles")

        downloaded_files: List[Path] = []
        session = requests.Session()

        for tile in tiles:
            try:
                out_file = self._download_tile(session, tile, dataset, output_dir)
                if out_file:
                    downloaded_files.append(out_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to download JRC tile {tile}: {e}", exc_info=True)

        if not downloaded_files:
            raise RuntimeError("No JRC Global Surface Water data could be downloaded")

        # Merge tiles and clip to bbox
        self._merge_tiles(downloaded_files, output_dir, dataset)

        return output_dir

    def _get_tiles_for_bbox(self) -> List[Tuple[int, int]]:
        """
        Determine which 10x10 degree tiles are needed for the bounding box.

        Returns:
            List of (lon, lat) tile corner coordinates
        """
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        tiles = []

        # JRC tiles are named by their SW corner, in 10-degree increments
        lon_start = int(np.floor(lon_min / self.TILE_SIZE) * self.TILE_SIZE)
        lon_end = int(np.ceil(lon_max / self.TILE_SIZE) * self.TILE_SIZE)
        lat_start = int(np.floor(lat_min / self.TILE_SIZE) * self.TILE_SIZE)
        lat_end = int(np.ceil(lat_max / self.TILE_SIZE) * self.TILE_SIZE)

        for lon in range(lon_start, lon_end, self.TILE_SIZE):
            for lat in range(lat_start, lat_end, self.TILE_SIZE):
                tiles.append((lon, lat))

        return tiles

    def _tile_to_filename(self, lon: int, lat: int) -> str:
        """
        Convert tile coordinates to JRC filename format.

        JRC uses format like: occurrence_100W_40Nv1_4_2021.tif
        """
        # Longitude: E for positive, W for negative
        lon_dir = 'E' if lon >= 0 else 'W'
        lon_val = abs(lon)

        # Latitude: N for positive, S for negative
        lat_dir = 'N' if lat >= 0 else 'S'
        lat_val = abs(lat)

        return f"{lon_val}{lon_dir}_{lat_val}{lat_dir}"

    def _download_tile(
        self,
        session: requests.Session,
        tile: Tuple[int, int],
        dataset: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Download a single JRC tile."""
        lon, lat = tile
        tile_name = self._tile_to_filename(lon, lat)
        dataset_info = self.DATASETS[dataset]

        # Try different version patterns
        versions = ['v1_4_2021', 'v1_3_2020', 'v1_2_2019', 'v1_1_2018']

        for version in versions:
            filename = f"{dataset}_{tile_name}{version}.tif"
            url = f"{self.GCS_BASE}/{dataset_info['path']}/{filename}"
            out_file = output_dir / filename

            if out_file.exists() and not self._get_config_value(
                lambda: self.config.data.force_download,
                dict_key='FORCE_DOWNLOAD',
            ):
                return out_file

            self.logger.debug(f"Trying: {url}")

            try:
                response = session.get(url, stream=True, timeout=300)

                if response.status_code == 404:
                    continue

                response.raise_for_status()

                tmp_file = out_file.with_suffix('.tif.part')
                with open(tmp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                tmp_file.replace(out_file)

                self.logger.info(f"Downloaded: {filename}")
                return out_file

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed {version}: {e}", exc_info=True)
                continue

        self.logger.warning(f"Could not find JRC tile for {tile_name}")
        return None

    def _merge_tiles(
        self,
        files: List[Path],
        output_dir: Path,
        dataset: str
    ) -> Path:
        """
        Merge downloaded tiles and clip to bounding box.

        Args:
            files: List of downloaded GeoTIFF files
            output_dir: Output directory
            dataset: Dataset name

        Returns:
            Path to merged output file
        """
        import rasterio
        from rasterio.mask import mask
        from rasterio.merge import merge
        from shapely.geometry import box

        out_file = output_dir / f"{self.domain_name}_JRC_{dataset}_merged.tif"

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return out_file

        if len(files) == 0:
            raise RuntimeError("No JRC tiles to merge")

        # Open all source files
        src_files = [rasterio.open(f) for f in files]

        try:
            # Merge tiles
            mosaic, out_transform = merge(src_files)

            # Update metadata
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })

            # Write merged file
            merged_tmp = out_file.with_suffix('.tmp.tif')
            with rasterio.open(merged_tmp, "w", **out_meta) as dest:
                dest.write(mosaic)

            # Clip to bounding box
            lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
            lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])
            bbox_geom = [box(lon_min, lat_min, lon_max, lat_max)]

            with rasterio.open(merged_tmp) as src:
                clipped, clipped_transform = mask(src, bbox_geom, crop=True)
                clip_meta = src.meta.copy()
                clip_meta.update({
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform
                })

                with rasterio.open(out_file, "w", **clip_meta) as dest:
                    dest.write(clipped)

            # Clean up temp file
            merged_tmp.unlink()

            self.logger.info(f"Merged and clipped JRC {dataset} to {out_file}")

        finally:
            for src in src_files:
                src.close()

        return out_file
