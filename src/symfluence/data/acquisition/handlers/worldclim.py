# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WorldClim 2.1 Climate Data Acquisition Handler

Cloud-based acquisition of WorldClim v2.1 monthly climate normals at
30 arcsecond (~1 km) resolution (Fick & Hijmans, 2017).

Variables:
    prec  — precipitation (mm)
    tavg  — mean temperature (°C)
    tmin  — minimum temperature (°C)
    tmax  — maximum temperature (°C)
    srad  — solar radiation (kJ m⁻² day⁻¹)
    wind  — wind speed (m s⁻¹)
    vapr  — water vapour pressure (kPa)

Each variable has 12 monthly GeoTIFF files (global, 30 arcsecond).
Files are downloaded, then subsetted to the domain bounding box.

Data Source:
    https://www.worldclim.org/data/worldclim21.html
    Available under CC-BY-SA 4.0 (derived statistics redistributable,
    raw grids are not)

References:
    Fick, S.E. and Hijmans, R.J. (2017). WorldClim 2: new 1-km spatial
    resolution climate surfaces for global climate land areas. Int. J.
    Climatol., 37, 4302-4315.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..utils import create_robust_session, download_file_streaming

_MIRRORS = [
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base",
    "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base",
]

_VARIABLES = {
    'prec': 'Precipitation (mm)',
    'tavg': 'Mean temperature (°C)',
    'tmin': 'Minimum temperature (°C)',
    'tmax': 'Maximum temperature (°C)',
    'srad': 'Solar radiation (kJ m⁻² day⁻¹)',
    'wind': 'Wind speed (m s⁻¹)',
    'vapr': 'Water vapour pressure (kPa)',
}

_DEFAULT_VARIABLES = list(_VARIABLES.keys())


@R.acquisition_handlers.add('WORLDCLIM')
@R.acquisition_handlers.add('WORLDCLIM_V21')
class WorldClimAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    WorldClim v2.1 monthly climate normals acquisition.

    Downloads global GeoTIFF archives from the UC Davis mirror and
    subsets each monthly raster to the domain bounding box.

    Output:
        {project_dir}/attributes/climate/worldclim/
            wc2.1_30s_{variable}_{month:02d}.tif  (bbox-clipped)
    """

    def download(self, output_dir: Path) -> Path:
        worldclim_dir = self._attribute_dir("climate") / "worldclim"
        worldclim_dir.mkdir(parents=True, exist_ok=True)

        variables = self._get_config_value(
            lambda: None,
            default=_DEFAULT_VARIABLES,
            dict_key='WORLDCLIM_VARIABLES',
        )
        variables = [v for v in variables if v in _VARIABLES]

        if not variables:
            raise ValueError(
                f"No valid WorldClim variables. "
                f"Choose from: {list(_VARIABLES.keys())}"
            )

        self.logger.info(
            f"Acquiring WorldClim 2.1 for bbox: {self.bbox}, "
            f"variables: {variables}"
        )

        session = create_robust_session(max_retries=3, backoff_factor=2.0)
        cache_dir = worldclim_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        total_files = 0
        for i, var in enumerate(variables):
            self.logger.info(
                f"[{i+1}/{len(variables)}] WorldClim {var}: "
                f"{_VARIABLES[var]}"
            )

            zip_name = f"wc2.1_30s_{var}.zip"
            zip_path = cache_dir / zip_name

            if not zip_path.exists():
                downloaded = False
                for mirror in _MIRRORS:
                    url = f"{mirror}/{zip_name}"
                    try:
                        self.logger.info(f"  Trying {mirror.split('/')[2]}")
                        download_file_streaming(
                            url, zip_path, session=session, timeout=1200
                        )
                        downloaded = True
                        break
                    except Exception as e:  # noqa: BLE001
                        self.logger.warning(f"  Mirror failed: {e}")
                        zip_path.unlink(missing_ok=True)
                        continue
                if not downloaded:
                    raise RuntimeError(
                        f"All WorldClim mirrors unreachable for {zip_name}. "
                        f"Download manually from https://www.worldclim.org/"
                        f"data/worldclim21.html and place zip files in "
                        f"{cache_dir}"
                    )

            extract_dir = cache_dir / var
            if not extract_dir.exists():
                # Extract into a sibling temp dir and rename, so an interrupted
                # extraction is redone next time instead of looking complete.
                self.logger.info(f"  Extracting {zip_name}")
                tmp_dir = extract_dir.with_name(extract_dir.name + ".partial")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    from symfluence.core.archive_extraction import safe_zip_extract
                    safe_zip_extract(zf, tmp_dir)
                tmp_dir.replace(extract_dir)

            for month in range(1, 13):
                out_name = f"wc2.1_30s_{var}_{month:02d}.tif"
                out_path = worldclim_dir / out_name

                if self._skip_if_exists(out_path):
                    total_files += 1
                    continue

                src_name = f"wc2.1_30s_{var}_{month:02d}.tif"
                src_path = self._find_tif(extract_dir, src_name)

                if src_path is None:
                    self.logger.warning(
                        f"  Missing: {src_name} in archive"
                    )
                    continue

                self._subset_to_bbox(src_path, out_path)
                total_files += 1

        if total_files == 0:
            raise RuntimeError("No WorldClim files could be processed")

        self.logger.info(
            f"WorldClim acquisition complete: {total_files} files "
            f"in {worldclim_dir}"
        )
        return worldclim_dir

    def _find_tif(self, search_dir: Path, filename: str) -> Path | None:
        """Find a TIF file in a directory tree (zip may have subdirs)."""
        matches = list(search_dir.rglob(filename))
        if matches:
            return matches[0]
        return None

    def _subset_to_bbox(self, src_path: Path, dst_path: Path):
        """Subset a global GeoTIFF to the domain bounding box."""
        with rasterio.open(src_path) as src:
            window = from_bounds(
                self.bbox['lon_min'], self.bbox['lat_min'],
                self.bbox['lon_max'], self.bbox['lat_max'],
                transform=src.transform,
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
                'dtype': 'float32',
            })

            with rasterio.open(dst_path, 'w', **meta) as dst:
                dst.write(data.astype(np.float32), 1)
