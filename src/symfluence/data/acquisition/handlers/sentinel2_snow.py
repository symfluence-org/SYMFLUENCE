# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Sentinel-2 Snow Cover Acquisition Handler.

Acquires Sentinel-2 L2A snow cover fraction using the Scene Classification
Layer (SCL) via the Microsoft Planetary Computer STAC API. No authentication
is required.

Snow detection uses SCL class 11 (snow/ice), excluding cloud-contaminated
pixels (classes 8, 9, 10). Output is a domain-averaged daily snow cover
fraction time series.

Data source:
  - Microsoft Planetary Computer: Sentinel-2 L2A (free, public access)
  - Temporal coverage: 2015-06-23 onwards (Sentinel-2A launch)
  - Native SCL resolution: 20 meters

Dependencies:
  - pystac-client, planetary-computer, rasterio

References:
  - https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a
  - https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# Planetary Computer STAC API endpoint
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# SCL classification values
SCL_SNOW = 11
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_THIN_CIRRUS = 10


@R.acquisition_handlers.add('SENTINEL2_SNOW')
@R.acquisition_handlers.add('S2_SNOW')
@R.acquisition_handlers.add('SENTINEL2_SCF')
class Sentinel2SnowAcquirer(BaseAcquisitionHandler):
    """
    Acquires Sentinel-2 L2A snow cover fraction via Planetary Computer.

    Configuration:
        S2_SNOW_MAX_CLOUD_COVER: Maximum scene cloud cover % (default: 50)
        S2_SNOW_MIN_LAND_PIXELS: Minimum land pixels for valid record (default: 100)
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process Sentinel-2 snow cover data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{self.domain_name}_Sentinel2_snow.nc"
        force = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        if output_file.exists() and not force:
            self.logger.info(f"Using existing Sentinel-2 snow file: {output_file}")
            return output_file

        max_cloud = self._get_config_value(lambda: None, default=50, dict_key='S2_SNOW_MAX_CLOUD_COVER')
        min_land = self._get_config_value(lambda: None, default=100, dict_key='S2_SNOW_MIN_LAND_PIXELS')

        # Sentinel-2A launched 2015-06-23
        start = max(self.start_date, pd.Timestamp('2015-06-23'))
        if start > self.end_date:
            raise ValueError(
                f"Sentinel-2 data starts 2015-06-23, but experiment ends {self.end_date}. "
                "No Sentinel-2 data available for this period."
            )
        if start > self.start_date:
            self.logger.warning(
                f"Sentinel-2 data starts 2015-06-23, adjusting start from {self.start_date.date()}"
            )

        # Search for scenes
        items = self._search_scenes(start, self.end_date, max_cloud)

        if not items:
            raise RuntimeError(
                "No Sentinel-2 scenes found for the domain and time period. "
                "Check bounding box coordinates and date range."
            )

        # Extract snow cover from each scene
        results = self._process_scenes(items, min_land)

        if not results:
            raise RuntimeError("No valid snow data extracted from Sentinel-2 scenes")

        # Create output dataset
        self._save_dataset(results, output_file)

        return output_file

    def _search_scenes(self, start: pd.Timestamp, end: pd.Timestamp, max_cloud: int) -> List:
        """Search for Sentinel-2 L2A scenes via Planetary Computer STAC."""
        import planetary_computer as pc
        from pystac_client import Client

        self.logger.info("Connecting to Microsoft Planetary Computer STAC...")
        client = Client.open(PC_STAC_URL, modifier=pc.sign_inplace)

        bbox_tuple = (
            self.bbox['lon_min'], self.bbox['lat_min'],
            self.bbox['lon_max'], self.bbox['lat_max']
        )

        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        self.logger.info(
            f"Searching Sentinel-2 L2A: {start_str} to {end_str}, "
            f"cloud <{max_cloud}%"
        )

        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox_tuple,
            datetime=f"{start_str}/{end_str}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )

        items = list(search.items())
        self.logger.info(f"Found {len(items)} Sentinel-2 scenes")
        return items

    def _process_scenes(self, items: List, min_land: int) -> List[Dict]:
        """Extract snow cover from each scene."""
        import planetary_computer as pc
        import rasterio
        from rasterio.crs import CRS
        from rasterio.windows import from_bounds

        os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.tiff')

        sorted_items = sorted(items, key=lambda x: x.datetime)
        total = len(sorted_items)
        log_interval = max(1, total // 20)

        results = []
        failed = 0

        bbox_tuple = (
            self.bbox['lon_min'], self.bbox['lat_min'],
            self.bbox['lon_max'], self.bbox['lat_max']
        )

        for i, item in enumerate(sorted_items):
            if (i + 1) % log_interval == 0 or i == 0:
                self.logger.info(f"Processing scene {i+1}/{total}: {item.id}")

            try:
                scl_asset = item.assets.get('SCL')
                if not scl_asset:
                    continue

                signed_href = pc.sign(scl_asset.href)
                vsi_href = f'/vsicurl/{signed_href}' if signed_href.startswith('http') else signed_href

                date = pd.to_datetime(
                    item.datetime.strftime('%Y-%m-%d') if item.datetime else item.id[:10]
                )

                with rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',
                    GDAL_HTTP_MULTIRANGE='YES',
                    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES'
                ):
                    with rasterio.open(vsi_href) as src:
                        if src.crs != CRS.from_epsg(4326):
                            from rasterio.warp import transform_bounds
                            src_bounds = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox_tuple)
                        else:
                            src_bounds = bbox_tuple

                        window = from_bounds(*src_bounds, src.transform)
                        scl_data = src.read(1, window=window)

                total_valid = np.sum((scl_data > 0) & (scl_data <= 11))
                snow_pixels = np.sum(scl_data == SCL_SNOW)
                cloud_pixels = np.sum(
                    (scl_data == SCL_CLOUD_MEDIUM) |
                    (scl_data == SCL_CLOUD_HIGH) |
                    (scl_data == SCL_THIN_CIRRUS)
                )
                land_pixels = total_valid - cloud_pixels

                if land_pixels < min_land:
                    continue

                results.append({
                    'date': date,
                    'snow_fraction': snow_pixels / land_pixels,
                    'cloud_fraction': cloud_pixels / total_valid if total_valid > 0 else np.nan,
                    'snow_pixels': int(snow_pixels),
                    'land_pixels': int(land_pixels),
                    'total_pixels': int(total_valid),
                })

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                failed += 1
                self.logger.debug(f"Error processing {item.id}: {e}", exc_info=True)

        self.logger.info(
            f"Extracted snow data from {len(results)}/{total} scenes "
            f"({failed} errors)"
        )
        return results

    def _save_dataset(self, results: List[Dict], output_file: Path):
        """Create output NetCDF dataset."""
        df = pd.DataFrame(results)
        df = df.sort_values('date').drop_duplicates('date', keep='first')
        df = df.set_index('date')

        time_coord = pd.to_datetime(df.index)

        ds = xr.Dataset({
            'snow_fraction': xr.DataArray(
                df['snow_fraction'].values, dims=['time'],
                coords={'time': time_coord},
                attrs={
                    'long_name': 'Snow covered area fraction from SCL',
                    'units': '1',
                    'valid_range': [0, 1],
                    'source': 'Sentinel-2 L2A Scene Classification Layer',
                }
            ),
            'cloud_fraction': xr.DataArray(
                df['cloud_fraction'].values, dims=['time'],
                coords={'time': time_coord},
                attrs={'long_name': 'Cloud covered fraction', 'units': '1'}
            ),
            'snow_pixels': xr.DataArray(
                df['snow_pixels'].values, dims=['time'],
                coords={'time': time_coord},
                attrs={'long_name': 'Number of snow pixels', 'units': 'count'}
            ),
        })

        ds.attrs['title'] = 'Sentinel-2 Snow Cover'
        ds.attrs['source'] = 'Copernicus Sentinel-2 L2A via Microsoft Planetary Computer'
        ds.attrs['method'] = 'Scene Classification Layer (SCL class 11)'
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['domain'] = self.domain_name
        ds.attrs['bbox'] = (
            f"{self.bbox['lon_min']},{self.bbox['lat_min']},"
            f"{self.bbox['lon_max']},{self.bbox['lat_max']}"
        )

        ds.to_netcdf(output_file)

        self.logger.info(f"Saved {len(df)} records to {output_file}")
        self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        self.logger.info(
            f"  Snow fraction: {df['snow_fraction'].min():.3f} - "
            f"{df['snow_fraction'].max():.3f}"
        )
