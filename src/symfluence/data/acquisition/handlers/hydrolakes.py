# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HydroLAKES Data Acquisition Handler.

Acquires lake polygon data from the HydroLAKES v1.0 global database
(Messager et al., 2016) and clips to the domain bounding box.

HydroLAKES provides ~1.4 million lakes globally with attributes
including surface area, shoreline length, depth, volume, elevation,
residence time, and upstream drainage area.

Data source: https://www.hydrosheds.org/products/hydrolakes
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import box
from urllib3.util.retry import Retry

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


def _create_session(max_retries: int = 3, backoff_factor: float = 1.0):
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# HydroLAKES v1.0 download URL (publicly available, no auth required)
HYDROLAKES_URL = (
    "https://data.hydrosheds.org/file/hydrolakes/"
    "HydroLAKES_polys_v10_shp.zip"
)

# Key attribute columns to retain (reduces file size)
KEEP_COLUMNS = [
    "Hylak_id",    # Unique lake identifier
    "Lake_name",   # Lake name (if available)
    "Lake_type",   # 1=lake, 2=reservoir, 3=lake control
    "Lake_area",   # Surface area (km²)
    "Shore_len",   # Shoreline length (km)
    "Shore_dev",   # Shoreline development ratio
    "Vol_total",   # Total volume (mcm = 10⁶ m³)
    "Depth_avg",   # Average depth (m)
    "Dis_avg",     # Average discharge (m³/s)
    "Res_time",    # Residence time (days)
    "Elevation",   # Lake surface elevation (m a.s.l.)
    "Wshd_area",   # Upstream watershed area (km²)
    "Pour_long",   # Pour point longitude
    "Pour_lat",    # Pour point latitude
    "geometry",
]


@R.acquisition_handlers.add("HYDROLAKES")
@R.acquisition_handlers.add("HYDROLAKES_V10")
class HydroLAKESAcquirer(BaseAcquisitionHandler):
    """
    HydroLAKES global lake database acquisition handler.

    Downloads lake polygons from HydroLAKES v1.0, clips to the
    domain bounding box, and saves as a GeoPackage with key
    hydrological attributes.

    Output:
        {project_dir}/attributes/lakes/domain_{name}_hydrolakes.gpkg
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Any = None,
    ):
        super().__init__(config, logger, reporting_manager)
        self.session = _create_session()

    def download(self, output_dir: Path) -> Path:
        """
        Download and process HydroLAKES data for the domain.

        Returns:
            Path to the lakes attribute directory.
        """
        lake_dir = self._attribute_dir("lakes")
        lake_dir.mkdir(parents=True, exist_ok=True)

        out_gpkg = lake_dir / f"domain_{self.domain_name}_hydrolakes.gpkg"

        if out_gpkg.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            self.logger.debug(f"HydroLAKES data already exists: {out_gpkg}")
            return lake_dir

        self.logger.info("Starting HydroLAKES v1.0 acquisition")

        # 1. Download (or use cached) global shapefile
        global_shp = self._get_global_shapefile(lake_dir)
        if global_shp is None:
            self.logger.error(
                "Could not obtain HydroLAKES data. "
                "Download manually from https://www.hydrosheds.org/products/hydrolakes "
                f"and place the shapefile in {lake_dir / 'cache'}"
            )
            return lake_dir

        # 2. Read and clip to domain bbox (with buffer for lakes on boundary)
        buf_deg = 0.1
        domain_box = box(
            self.bbox["lon_min"] - buf_deg,
            self.bbox["lat_min"] - buf_deg,
            self.bbox["lon_max"] + buf_deg,
            self.bbox["lat_max"] + buf_deg,
        )

        self.logger.debug("Reading and clipping HydroLAKES to domain bbox")
        lakes_gdf = gpd.read_file(global_shp, bbox=domain_box)

        if lakes_gdf is None or len(lakes_gdf) == 0:
            self.logger.warning("No lakes found in domain bounding box")
            return lake_dir

        # 3. Filter to columns of interest (keep what's available)
        available_cols = [c for c in KEEP_COLUMNS if c in lakes_gdf.columns]
        lakes_gdf = lakes_gdf[available_cols].copy()

        # Ensure EPSG:4326
        if lakes_gdf.crs is None:
            lakes_gdf = lakes_gdf.set_crs(epsg=4326)
        else:
            lakes_gdf = lakes_gdf.to_crs(epsg=4326)

        # 4. Save clipped GeoPackage
        lakes_gdf.to_file(out_gpkg, driver="GPKG")

        # Log summary
        n_lakes = len(lakes_gdf)
        total_area = lakes_gdf["Lake_area"].sum() if "Lake_area" in lakes_gdf.columns else 0
        named = lakes_gdf["Lake_name"].notna().sum() if "Lake_name" in lakes_gdf.columns else 0
        self.logger.info(
            f"HydroLAKES clipped: {n_lakes} lakes "
            f"(total area {total_area:.1f} km², {named} named) -> {out_gpkg}"
        )

        return lake_dir

    def _get_global_shapefile(self, lake_dir: Path) -> Optional[Path]:
        """Download or locate cached global HydroLAKES shapefile."""
        cache_dir = lake_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing shapefile in cache
        for shp in cache_dir.glob("HydroLAKES_polys_v10*.shp"):
            self.logger.debug(f"Using cached HydroLAKES: {shp}")
            return shp

        # Also check common user-download locations
        for search_dir in [lake_dir, cache_dir.parent]:
            for shp in search_dir.glob("**/HydroLAKES*.shp"):
                self.logger.debug(f"Found local HydroLAKES: {shp}")
                return shp

        # Download from HydroSHEDS
        zip_path = cache_dir / "HydroLAKES_polys_v10_shp.zip"
        self.logger.info(
            f"Downloading HydroLAKES v1.0 (~470 MB, may take several minutes) from {HYDROLAKES_URL}"
        )

        try:
            resp = self.session.get(HYDROLAKES_URL, stream=True, timeout=600)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (50 << 20) == 0:
                            pct = downloaded / total * 100
                            self.logger.debug(f"Download progress: {pct:.0f}%")

            self.logger.debug("Download complete, extracting")

            # Extract shapefile components
            with zipfile.ZipFile(zip_path, "r") as zf:
                from symfluence.core.archive_extraction import safe_zip_extract
                safe_zip_extract(zf, cache_dir)

            # Remove zip to save space
            zip_path.unlink(missing_ok=True)

            # Find extracted shapefile
            for shp in cache_dir.rglob("HydroLAKES_polys_v10*.shp"):
                self.logger.debug(f"Extracted HydroLAKES shapefile: {shp}")
                return shp

            self.logger.error("No shapefile found in extracted archive")
            return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to download HydroLAKES: {e}", exc_info=True)
            self.logger.info(
                "Manual download: visit https://www.hydrosheds.org/products/hydrolakes\n"
                f"  Place the extracted shapefile in: {cache_dir}"
            )
            return None
