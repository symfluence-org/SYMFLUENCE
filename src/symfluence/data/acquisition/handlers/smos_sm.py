# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SMOS Soil Moisture Acquisition Handler

Provides cloud acquisition for SMOS (Soil Moisture and Ocean Salinity) data:
- Uses Copernicus Climate Data Store (CDS) API
- Downloads passive-only sensor type from satellite-soil-moisture dataset
- Supports monthly temporal chunking for large requests

SMOS SM Overview:
    Data Type: Satellite-derived soil moisture (L-band passive microwave)
    Resolution: ~25 km (ISEA 4H9 grid)
    Sensing Depth: ~0-5 cm
    Coverage: Global
    Available: 2010-present
    Source: ESA SMOS via Copernicus CDS (passive-only retrieval)

Requirements:
    - cdsapi package installed
    - CDS API key configured (~/.cdsapirc)
"""
from __future__ import annotations

import calendar
import shutil
from pathlib import Path
from typing import Optional

from symfluence.core.registries import R

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('SMOS')
@R.acquisition_handlers.add('SMOS_SM')
class SMOSSMAcquirer(BaseAcquisitionHandler):
    """
    Acquires SMOS Soil Moisture data via Copernicus CDS.

    Downloads the passive-only sensor type from the CDS satellite-soil-moisture
    dataset, which yields the SMOS-dominated retrieval (2010-2014) and
    SMOS+SMAP blend (2015+).
    """

    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi required for SMOS SM acquisition")

        self.logger.info("Starting SMOS Soil Moisture acquisition via CDS")

        output_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = output_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        self._output_dir = output_dir

        # Generate year-month list
        ym_range = self._generate_year_month_list()

        # Download each month sequentially (CDS rate limits)
        downloads = []
        for ym in ym_range:
            try:
                result = self._download_month_chunk(ym)
                if result:
                    downloads.append(result)
            except Exception as exc:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to download SMOS {ym[0]}-{ym[1]:02d}: {exc}", exc_info=True)

        # Extract all downloaded archives
        for out_file in downloads:
            if out_file and out_file.exists():
                try:
                    shutil.unpack_archive(str(out_file), extract_dir)
                except Exception as exc:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(
                        f"Failed to extract SMOS archive {out_file.name}: {exc}"
                    , exc_info=True)

        self.logger.info(f"Extracted SMOS SM data to {extract_dir}")
        return extract_dir

    def _generate_year_month_list(self):
        """Generate list of (year, month) tuples for the download period."""
        result = []
        current_year = self.start_date.year
        current_month = self.start_date.month

        while (current_year, current_month) <= (self.end_date.year, self.end_date.month):
            result.append((current_year, current_month))
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1

        return result

    def _download_month_chunk(self, year_month: tuple) -> Optional[Path]:
        """Download a single month of SMOS SM data from CDS."""
        year, month = year_month
        days_in_month = calendar.monthrange(year, month)[1]

        out_file_zip = self._output_dir / f"smos_cds_{year}_{month:02d}.zip"
        out_file_tgz = self._output_dir / f"smos_cds_{year}_{month:02d}.tar.gz"

        # Check for existing file
        force = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        if out_file_zip.exists() and not force:
            return out_file_zip
        if out_file_tgz.exists() and not force:
            return out_file_tgz

        # Build CDS request
        request = {
            'variable': [self._get_config_value(lambda: None, default='surface_soil_moisture_volumetric', dict_key='SMOS_SM_VARIABLE')],
            'type_of_sensor': ['passive'],
            'time_aggregation': [self._get_config_value(lambda: None, default='daily', dict_key='SMOS_SM_TIME_AGGREGATION')],
            'type_of_record': [self._get_config_value(lambda: None, default='cdr', dict_key='SMOS_SM_RECORD_TYPE')],
            'version': [self._get_config_value(lambda: None, default='v202505', dict_key='SMOS_SM_VERSION')],
            'year': [str(year)],
            'month': [f"{month:02d}"],
            'day': [f"{d:02d}" for d in range(1, days_in_month + 1)],
        }

        self.logger.info(f"Requesting SMOS SM for {year}-{month:02d}...")

        c = cdsapi.Client()
        c.retrieve('satellite-soil-moisture', request, str(out_file_zip))

        return out_file_zip
