# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HydroSHEDS / HydroBASINS Acquisition Handler

Downloads global catchment boundary data from the HydroSHEDS project
(Lehner & Grill, 2013). HydroBASINS provides Pfafstetter-coded basin
delineations at 12 hierarchical levels organized by continental region.

Workflow:
    1. Map pour point to HydroSHEDS continental region(s) via hardcoded bboxes
    2. Download HydroBASINS shapefile at configured Pfafstetter level
    3. HydroBASINS provides both catchment polygons AND river topology
       (via HYBAS_ID / NEXT_DOWN columns)

Source:
    https://www.hydrosheds.org/products/hydrobasins
    https://data.hydrosheds.org/file/HydroBASINS/standard/

Column Convention:
    HYBAS_ID — unique basin identifier
    NEXT_DOWN — downstream basin HYBAS_ID (0 = outlet/ocean)
    NEXT_SINK — downstream sink basin
    MAIN_BAS — main basin HYBAS_ID
    SUB_AREA — sub-basin area (km²)
    UP_AREA — total upstream area (km²)
    PFAF_ID — Pfafstetter code
    SORT — topological sort order

References:
    - Lehner, B. & Grill, G. (2013). Global river hydrography and network
      routing: baseline data and new approaches to study the world's large
      river systems. Hydrological Processes, 27(15), 2171-2186.
    - Lehner, B. (2014). HydroBASINS Technical Documentation. WWF.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List

import requests

from symfluence.core.registries import R
from symfluence.data.acquisition.utils import atomic_write

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..utils import create_robust_session

# HydroBASINS download URL template (standard version, excludes endorheic)
_HYDROBASINS_URL = (
    "https://data.hydrosheds.org/file/HydroBASINS/standard/"
    "hybas_{region}_lev{level:02d}_v1c.zip"
)

# Continental regions with approximate bounding boxes
# Format: {code: (lat_min, lon_min, lat_max, lon_max)}
_HYDROSHEDS_REGION_BBOXES = {
    'af': (-35.0, -20.0, 38.0, 55.0),     # Africa
    'ar': (51.0, -180.0, 84.0, 180.0),    # Arctic (Greenland, Iceland, etc.)
    'as': (1.0, 57.0, 56.0, 180.0),       # Central-South Asia
    'au': (-56.0, 95.0, 0.0, 180.0),      # Australia-Oceania
    'eu': (12.0, -25.0, 72.0, 70.0),      # Europe-Middle East
    'na': (5.0, -138.0, 63.0, -52.0),     # North America
    'sa': (-56.0, -93.0, 15.0, -32.0),    # South America
    'si': (45.0, 57.0, 84.0, 180.0),      # Siberia
}


@R.acquisition_handlers.add('HYDROSHEDS')
@R.acquisition_handlers.add('HYDROBASINS')
class HydroSHEDSAcquirer(BaseAcquisitionHandler, RetryMixin):
    """HydroSHEDS / HydroBASINS acquisition from hydrosheds.org.

    Downloads HydroBASINS data at a configurable Pfafstetter level. The
    downloaded data provides both catchment polygons and river network
    topology (via HYBAS_ID / NEXT_DOWN columns), so it serves as both
    the basins and rivers input for the subsetter.

    Config Keys:
        HYDROSHEDS_LEVEL: Pfafstetter level 1-12 (default: 12, finest)

    Output Files:
        hybas_{region}_lev{level}_v1c.shp — basin polygons with topology
    """

    def download(self, output_dir: Path) -> Path:
        """Download HydroBASINS data for the domain.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the directory containing downloaded geofabric files
        """
        level = int(self._get_config_value(lambda: None, default=12, dict_key='HYDROSHEDS_LEVEL'))
        if not 1 <= level <= 12:
            raise ValueError(
                f"HYDROSHEDS_LEVEL must be 1-12, got {level}"
            )

        geofabric_dir = self._attribute_dir("geofabric") / "hydrosheds"
        geofabric_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = self._get_pour_point_coords()
        self.logger.info(
            f"Downloading HydroBASINS (level {level}) for "
            f"pour point ({lat:.4f}, {lon:.4f})"
        )

        # Find matching continental region(s)
        regions = self._find_regions(lat, lon)
        if not regions:
            raise ValueError(
                f"Pour point ({lat}, {lon}) does not fall within any "
                "HydroSHEDS continental region"
            )
        self.logger.info(f"Matched HydroSHEDS region(s): {regions}")

        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        for region in regions:
            marker = geofabric_dir / f"hybas_{region}_lev{level:02d}_v1c.shp"
            if self._skip_if_exists(marker):
                continue

            url = _HYDROBASINS_URL.format(region=region, level=level)
            self._download_and_extract(
                session, url, geofabric_dir,
                f"HydroBASINS {region} level {level}"
            )

        self.logger.info(f"HydroBASINS data downloaded to: {geofabric_dir}")
        return geofabric_dir

    def _get_pour_point_coords(self) -> tuple:
        """Extract pour point lat/lon from config.

        Returns:
            Tuple of (lat, lon)
        """
        pour_point_str = self._get_config_value(lambda: self.config.domain.pour_point_coords, default=None)
        if pour_point_str:
            parts = str(pour_point_str).replace('/', ',').split(',')
            return float(parts[0].strip()), float(parts[1].strip())

        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
        return lat, lon

    def _find_regions(self, lat: float, lon: float) -> List[str]:
        """Find HydroSHEDS continental region(s) for the pour point.

        Args:
            lat: Latitude of pour point
            lon: Longitude of pour point

        Returns:
            List of matching region codes
        """
        matching = []
        for code, (lat_min, lon_min, lat_max, lon_max) in _HYDROSHEDS_REGION_BBOXES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                matching.append(code)

        # Also check full domain bbox for large domains
        if self.bbox:
            for code, (lat_min, lon_min, lat_max, lon_max) in _HYDROSHEDS_REGION_BBOXES.items():
                if code not in matching:
                    if (self.bbox['lat_min'] <= lat_max and
                            self.bbox['lat_max'] >= lat_min and
                            self.bbox['lon_min'] <= lon_max and
                            self.bbox['lon_max'] >= lon_min):
                        matching.append(code)

        return sorted(set(matching))

    def _download_and_extract(
        self, session, url: str, output_dir: Path, description: str
    ):
        """Download a zip file and extract its contents.

        Args:
            session: requests.Session
            url: URL to download
            output_dir: Directory to extract files into
            description: Human-readable description for logging
        """
        def do_download():
            self.logger.info(f"Downloading {description} from {url}")
            zip_path = output_dir / "temp_hydrosheds.zip"
            try:
                with session.get(url, stream=True, timeout=600) as resp:
                    resp.raise_for_status()
                    with atomic_write(zip_path) as _part, open(_part, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)

                self.logger.info(f"Extracting {description}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    from symfluence.core.archive_extraction import safe_zip_extract
                    safe_zip_extract(zf, output_dir)

                self.logger.info(f"Downloaded and extracted {description}")
            finally:
                if zip_path.exists():
                    zip_path.unlink()

        self.execute_with_retry(
            do_download,
            max_retries=3,
            base_delay=5,
            backoff_factor=2.0,
            retryable_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                IOError,
            ),
        )
