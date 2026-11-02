# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Pelletier Soil/Regolith Depth Data Acquisition Handler

Cloud-based acquisition of the global soil and regolith depth dataset from
Pelletier et al. (2016) via NASA Earthdata Cloud (ORNL DAAC).

Provides 6 GeoTIFF grids at ~1km (30 arcsecond) resolution:
- Upland hillslope soil thickness (m)
- Upland hillslope regolith thickness (m)
- Lowland sedimentary deposit thickness (m)
- Average soil and sedimentary-deposit thickness (m)
- Hillslope and valley-bottom average thickness (m)
- Land cover mask

Data Source:
    ORNL DAAC: https://doi.org/10.3334/ORNLDAAC/1304
    Requires NASA Earthdata Login (via earthaccess, .netrc, or env vars)

References:
    Pelletier, J.D., et al. (2016). A gridded global data set of soil,
    intact regolith, and sedimentary deposit thicknesses for regional and
    global land surface modeling. J. Adv. Model. Earth Syst., 8, 41-65.

Configuration:
    PELLETIER_VARIABLES: List of variables to download
        (default: all grids)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin

# Cloud-hosted ORNL DAAC (replaces legacy daac.ornl.gov which is intermittent)
_ORNL_CLOUD_BASE = (
    "https://data.ornldaac.earthdata.nasa.gov/protected/global_soil/"
    "Global_Soil_Regolith_Sediment/data"
)

# Legacy URL kept as fallback
_ORNL_LEGACY_BASE = (
    "https://daac.ornl.gov/daacdata/global_soil/"
    "Global_Soil_Regolith_Sediment/data"
)

_PELLETIER_FILES = {
    'soil_thickness': {
        'filename': 'upland_hill-slope_soil_thickness.tif',
        'description': 'Upland hillslope soil thickness (m)',
    },
    'regolith_thickness': {
        'filename': 'upland_hill-slope_regolith_thickness.tif',
        'description': 'Upland hillslope regolith thickness (m)',
    },
    'sedimentary_thickness': {
        'filename': 'upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif',
        'description': 'Valley-bottom and lowland sedimentary deposit thickness (m)',
    },
    'average_thickness': {
        'filename': 'average_soil_and_sedimentary-deposit_thickness.tif',
        'description': 'Average soil and sedimentary-deposit thickness (m)',
    },
    'hillslope_valley_average': {
        'filename': 'hill-slope_valley-bottom.tif',
        'description': 'Hillslope and valley-bottom average thickness (m)',
    },
}

_DEFAULT_VARIABLES = list(_PELLETIER_FILES.keys())


@R.acquisition_handlers.add('PELLETIER')
@R.acquisition_handlers.add('PELLETIER_SOIL_DEPTH')
@R.acquisition_handlers.add('SOIL_REGOLITH_DEPTH')
class PelletierAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    Pelletier et al. (2016) global soil/regolith depth acquisition.

    Downloads global GeoTIFF grids from NASA Earthdata Cloud (ORNL DAAC)
    and subsets to the domain bounding box.

    Output:
        {project_dir}/attributes/soilclass/pelletier/
            domain_{name}_pelletier_{variable}.tif
    """

    def download(self, output_dir: Path) -> Path:
        pelletier_dir = self._attribute_dir("soilclass") / "pelletier"
        pelletier_dir.mkdir(parents=True, exist_ok=True)

        variables = self._get_config_value(lambda: None, default=_DEFAULT_VARIABLES, dict_key='PELLETIER_VARIABLES')
        variables = [v for v in variables if v in _PELLETIER_FILES]

        if not variables:
            raise ValueError(
                f"No valid Pelletier variables. Choose from: {list(_PELLETIER_FILES.keys())}"
            )

        self.logger.info(
            f"Acquiring Pelletier soil/regolith depth for bbox: {self.bbox}, "
            f"variables: {variables}"
        )

        session = self._create_earthdata_session()

        cache_dir = pelletier_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}
        for i, var in enumerate(variables):
            info = _PELLETIER_FILES[var]
            out_path = pelletier_dir / f"domain_{self.domain_name}_pelletier_{var}.tif"

            if self._skip_if_exists(out_path):
                output_paths[var] = out_path
                continue

            self.logger.info(f"[{i+1}/{len(variables)}] Downloading Pelletier {var}")

            try:
                cached_file = cache_dir / info['filename']
                if not cached_file.exists():
                    url = f"{_ORNL_CLOUD_BASE}/{info['filename']}"
                    self.logger.info(f"  Downloading from NASA Earthdata Cloud: {info['filename']}")
                    self._download_with_session(session, url, cached_file)

                self._subset_to_bbox(cached_file, out_path)
                output_paths[var] = out_path
                self.logger.info(f"  Saved: {out_path}")

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to download Pelletier {var}: {e}", exc_info=True)
                continue

        if not output_paths:
            raise RuntimeError(
                "No Pelletier soil depth data could be downloaded. "
                "Check NASA Earthdata credentials and network connectivity."
            )

        self.logger.info(
            f"Pelletier soil/regolith acquisition complete: {len(output_paths)} files"
        )
        return pelletier_dir

    def _create_earthdata_session(self):
        """Authenticated session for NASA Earthdata Cloud.

        earthaccess first, then a retrying token / username-password session;
        raises ``EarthdataAuthError`` when no credentials are found.
        """
        return self._earthdata_auth().session(
            prefer_earthaccess=True, required=True, purpose="Pelletier soil/regolith data",
            max_retries=3, backoff_factor=2.0,
        )

    def _download_with_session(self, session, url: str, dst: Path):
        """Download a file using the authenticated session with redirect handling."""
        with session.get(url, stream=True, timeout=600, allow_redirects=True) as r:
            r.raise_for_status()
            with open(dst, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

    def _subset_to_bbox(self, src_path: Path, dst_path: Path):
        """Subset a global GeoTIFF to the domain bounding box."""
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
                'dtype': 'float32',
            })

            with rasterio.open(dst_path, 'w', **meta) as dst:
                dst.write(data.astype(np.float32), 1)
