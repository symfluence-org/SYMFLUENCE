# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GLCLU 2020 Land Cover / Land Use Acquisition Handler

Cloud-based acquisition of the Global Land Cover and Land Use (GLCLU)
dataset from the GLAD laboratory at the University of Maryland.

Available layers (each as 10°×10° tiles at 30 m):
    Forest_extent_2020      — binary forest mask
    Forest_height_2020      — canopy height (m)
    Forest_extent_2000      — baseline forest mask
    Forest_height_2000      — baseline canopy height
    Built-up_change_2000_2020 — urban expansion
    Snow-Ice_2020           — permanent snow/ice mask

Data Source:
    https://glad.umd.edu/users/Potapov/GLCLUC2020/
    Licensed under CC-BY 4.0

Tile naming:
    {NN}N_{NNN}W.tif  (lower-left corner of 10° tile)
    Example: 50N_120W.tif covers 50-60°N, 120-110°W

References:
    Hansen, M.C., et al. (2022). Global land use extent and dispersion
    within natural land cover using Landsat data. Environ. Res. Lett.,
    17, 034050.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds

from symfluence.core.registries import R
from symfluence.data.acquisition.utils import atomic_write

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..utils import create_robust_session

_BASE_URL = "https://glad.umd.edu/users/Potapov/GLCLUC2020"

_LAYERS = {
    'forest_extent_2020': 'Forest_extent_2020',
    'forest_height_2020': 'Forest_height_2020',
    'forest_extent_2000': 'Forest_extent_2000',
    'forest_height_2000': 'Forest_height_2000',
    'built_up_change': 'Built-up_change_2000_2020',
    'snow_ice_2020': 'Snow-Ice_2020',
}

_DEFAULT_LAYERS = ['forest_extent_2020']


@R.acquisition_handlers.add('GLCLU_2019')
@R.acquisition_handlers.add('GLCLU')
class GLCLUAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    GLAD GLCLU 2020 land cover acquisition.

    Downloads 10°×10° tiles from UMD, mosaics them, and clips to the
    domain bounding box.

    Output:
        {project_dir}/attributes/landclass/glclu/
            domain_{name}_glclu_{layer}.tif
    """

    def download(self, output_dir: Path) -> Path:
        glclu_dir = self._attribute_dir("landclass") / "glclu"
        glclu_dir.mkdir(parents=True, exist_ok=True)

        layers = self._get_config_value(
            lambda: None,
            default=_DEFAULT_LAYERS,
            dict_key='GLCLU_LAYERS',
        )
        if isinstance(layers, str):
            layers = [layers]
        layers = [l for l in layers if l in _LAYERS]
        if not layers:
            layers = _DEFAULT_LAYERS

        session = create_robust_session(max_retries=5, backoff_factor=2.0)
        cache_dir = glclu_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for layer_key in layers:
            layer_subdir = _LAYERS[layer_key]
            output_file = (
                glclu_dir / f"domain_{self.domain_name}_glclu_{layer_key}.tif"
            )

            if self._skip_if_exists(output_file):
                continue

            self.logger.info(
                f"Acquiring GLCLU {layer_key} for bbox: {self.bbox}"
            )

            tile_paths = self._download_layer_tiles(
                session, layer_subdir, cache_dir
            )

            if not tile_paths:
                self.logger.warning(
                    f"No GLCLU {layer_key} tiles found for bbox"
                )
                continue

            if len(tile_paths) == 1:
                self._subset_to_bbox(tile_paths[0], output_file)
            else:
                self._merge_and_clip(tile_paths, output_file)

            self.logger.info(f"GLCLU {layer_key}: {output_file}")

        return glclu_dir

    def _download_layer_tiles(
        self, session, layer_subdir: str, cache_dir: Path
    ) -> List[Path]:
        """Download all tiles for a layer that intersect the bbox."""
        lat_min = self._snap_to_grid(self.bbox['lat_min'], 10, 'floor')
        lat_max = self._snap_to_grid(self.bbox['lat_max'], 10, 'ceil')
        lon_min = self._snap_to_grid(self.bbox['lon_min'], 10, 'floor')
        lon_max = self._snap_to_grid(self.bbox['lon_max'], 10, 'ceil')

        tile_paths: List[Path] = []
        layer_cache = cache_dir / layer_subdir
        layer_cache.mkdir(parents=True, exist_ok=True)

        for lat in range(lat_min, lat_max, 10):
            for lon in range(lon_min, lon_max, 10):
                tile_path = self._download_tile(
                    session, lat, lon, layer_subdir, layer_cache
                )
                if tile_path:
                    tile_paths.append(tile_path)

        return tile_paths

    def _snap_to_grid(
        self, value: float, grid_size: int, mode: str
    ) -> int:
        if mode == 'floor':
            return int(math.floor(value / grid_size) * grid_size)
        return int(math.ceil(value / grid_size) * grid_size)

    def _tile_name(self, lat: int, lon: int, layer_subdir: str) -> str:
        """Generate the tile filename.

        GLAD tiles are named by the *top* edge of the tile, not the
        lower-left corner.  A tile labeled ``50N_120W`` covers
        40-50°N, 120-110°W.  So for a grid cell starting at lat=50,
        we need the tile labeled (lat+10)N.
        """
        tile_lat = lat + 10
        lat_str = f"{abs(tile_lat):02d}N" if tile_lat >= 0 else f"{abs(tile_lat):02d}S"
        lon_str = f"{abs(lon):03d}E" if lon >= 0 else f"{abs(lon):03d}W"

        prefix = ""
        if "height_2020" in layer_subdir.lower():
            prefix = "2020_"
        elif "height_2000" in layer_subdir.lower():
            prefix = "2000_"
        elif "change" in layer_subdir.lower():
            prefix = "Change_"

        return f"{prefix}{lat_str}_{lon_str}.tif"

    def _download_tile(
        self, session, lat: int, lon: int,
        layer_subdir: str, cache_dir: Path,
    ) -> Optional[Path]:
        tile_name = self._tile_name(lat, lon, layer_subdir)
        url = f"{_BASE_URL}/{layer_subdir}/{tile_name}"
        local_tile = cache_dir / tile_name

        if local_tile.exists() and local_tile.stat().st_size > 1000:
            return local_tile

        self.logger.info(f"  Fetching GLCLU tile: {tile_name}")

        def do_download():
            with session.get(
                url, stream=True, timeout=600, allow_redirects=True
            ) as r:
                if r.status_code == 200:
                    with atomic_write(local_tile) as _part, open(_part, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    return local_tile
                elif r.status_code == 404:
                    self.logger.info(
                        f"  GLCLU tile {tile_name} not available"
                    )
                    return None
                else:
                    raise IOError(f"HTTP {r.status_code} for {tile_name}")

        try:
            return self.execute_with_retry(
                do_download,
                max_retries=3,
                base_delay=2,
                backoff_factor=2.0,
                retryable_exceptions=(IOError, ConnectionError, OSError),
            )
        except (OSError, IOError, RuntimeError) as e:
            self.logger.warning(f"  Failed to download {tile_name}: {e}")
            if local_tile.exists():
                local_tile.unlink()
            return None

    def _subset_to_bbox(self, src_path: Path, dst_path: Path):
        with rasterio.open(src_path) as src:
            # Clamp bbox to tile bounds to avoid empty intersections
            tb = src.bounds
            clamp_lon_min = max(self.bbox['lon_min'], tb.left)
            clamp_lat_min = max(self.bbox['lat_min'], tb.bottom)
            clamp_lon_max = min(self.bbox['lon_max'], tb.right)
            clamp_lat_max = min(self.bbox['lat_max'], tb.top)

            if clamp_lon_min >= clamp_lon_max or clamp_lat_min >= clamp_lat_max:
                raise ValueError(
                    f"Bbox does not overlap tile {src_path.name}"
                )

            window = from_bounds(
                clamp_lon_min, clamp_lat_min,
                clamp_lon_max, clamp_lat_max,
                transform=src.transform,
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

    def _merge_and_clip(
        self, tile_paths: List[Path], output_file: Path
    ):
        self.logger.info(f"  Merging {len(tile_paths)} GLCLU tiles")

        src_files = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, out_trans = rio_merge(src_files)
            meta = src_files[0].meta.copy()
            meta.update({
                'height': mosaic.shape[1],
                'width': mosaic.shape[2],
                'transform': out_trans,
                'compress': 'lzw',
            })

            tmp_merged = output_file.parent / f"_merged_{output_file.name}"
            with rasterio.open(tmp_merged, 'w', **meta) as dst:
                dst.write(mosaic)
        finally:
            for src in src_files:
                src.close()

        self._subset_to_bbox(tmp_merged, output_file)
        tmp_merged.unlink(missing_ok=True)
