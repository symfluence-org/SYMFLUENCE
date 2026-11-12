# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODIS Data Acquisition Handler

Provides cloud acquisition for MODIS products (Snow Cover, ET, etc.) via THREDDS/NCSS.
"""
from __future__ import annotations

from pathlib import Path

import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('MODIS_SNOW')
class MODISSnowAcquirer(BaseAcquisitionHandler):
    """
    Acquires MODIS Snow Cover (MOD10A1/MYD10A1) data via THREDDS NCSS.
    """

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Starting MODIS Snow Cover acquisition via THREDDS")

        # Configuration
        product = self._get_config_value(
            lambda: self.config.evaluation.modis_snow.product,
            dict_key='MODIS_SNOW_PRODUCT',
        )
        thredds_base = self._get_config_value(lambda: None, default="https://ds.nccs.nasa.gov/thredds/ncss/grid", dict_key='MODIS_THREDDS_BASE')

        # BBox
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        # Date range
        start_date = self.start_date.strftime("%Y-%m-%dT12:00:00Z")
        end_date = self.end_date.strftime("%Y-%m-%dT12:00:00Z")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_nc = output_dir / f"{self.domain_name}_{product}_raw.nc"

        if out_nc.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return out_nc

        # Construct NCSS query
        # Note: This path is an example and depends on the specific THREDDS server structure
        # For NCCS, we might need a more specific path if mirrored
        dataset_path = self._get_config_value(lambda: None, default=f"MODIS/{product}/aggregated.ncml", dict_key='MODIS_THREDDS_PATH')

        params = {
            "var": "NDSI_Snow_Cover",  # Standard MOD10A1 variable
            "north": lat_max,
            "south": lat_min,
            "west": lon_min,
            "east": lon_max,
            "horizStride": 1,
            "time_start": start_date,
            "time_end": end_date,
            "accept": "netcdf4"
        }

        url = f"{thredds_base}/{dataset_path}"
        self.logger.info(f"Querying THREDDS: {url}")

        try:
            response = requests.get(url, params=params, stream=True, timeout=600)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'html' in content_type or 'xml' in content_type:
                # Likely an error page
                snippet = response.text[:500]
                self.logger.error(f"THREDDS returned an error page instead of data: {snippet}")
                raise ValueError("THREDDS NCSS request failed: received HTML/XML instead of NetCDF")

            with open(out_nc, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)

            self.logger.info(f"Successfully downloaded MODIS snow data to {out_nc}")
            return out_nc

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"THREDDS download failed for MODIS: {e}")
            # Fallback or re-raise
            raise
