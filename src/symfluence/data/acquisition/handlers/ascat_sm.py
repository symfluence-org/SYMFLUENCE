# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ASCAT Soil Moisture Acquisition Handler

Provides cloud acquisition for ASCAT (Advanced Scatterometer) soil moisture data:
- Uses Copernicus Climate Data Store (CDS) API
- Downloads active-only sensor type from satellite-soil-moisture dataset
- Supports monthly temporal chunking for large requests

ASCAT SM Overview:
    Data Type: Satellite-derived soil moisture (C-band active microwave)
    Resolution: ~25 km (resampled to 0.25 deg grid)
    Sensing Depth: ~0-2 cm (surface scattering)
    Coverage: Global
    Available: 2007-present (MetOp-A), 2013-present (MetOp-B)
    Source: EUMETSAT ASCAT via Copernicus CDS (active-only retrieval)

Note:
    ASCAT natively provides degree of saturation (0-100%), not volumetric SM.
    The CDS satellite-soil-moisture dataset can provide either:
    - surface_soil_moisture_saturation: degree of saturation (0-1)
    - surface_soil_moisture_volumetric: estimated volumetric (using porosity)
    The observation handler converts saturation to volumetric if needed.

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


@R.acquisition_handlers.add('ASCAT')
@R.acquisition_handlers.add('ASCAT_SM')
class ASCATSMAcquirer(BaseAcquisitionHandler):
    """
    Acquires ASCAT Soil Moisture data via Copernicus CDS.

    Downloads the active-only sensor type from the CDS satellite-soil-moisture
    dataset, which yields ASCAT backscatter-derived soil moisture using the
    TU Wien change detection algorithm.
    """

    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi required for ASCAT SM acquisition")

        self.logger.info("Starting ASCAT Soil Moisture acquisition via CDS")

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
            except Exception as exc:  # noqa: BLE001 — data acquisition resilience
                self.logger.warning(f"Failed to download ASCAT {ym[0]}-{ym[1]:02d}: {exc}")

        # Extract all downloaded archives
        for out_file in downloads:
            if out_file and out_file.exists():
                try:
                    shutil.unpack_archive(str(out_file), extract_dir)
                except Exception as exc:  # noqa: BLE001 — data acquisition resilience
                    self.logger.warning(
                        f"Failed to extract ASCAT archive {out_file.name}: {exc}"
                    )

        self.logger.info(f"Extracted ASCAT SM data to {extract_dir}")
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
        """Download a single month of ASCAT SM data from CDS."""
        year, month = year_month
        days_in_month = calendar.monthrange(year, month)[1]

        out_file_zip = self._output_dir / f"ascat_cds_{year}_{month:02d}.zip"
        out_file_tgz = self._output_dir / f"ascat_cds_{year}_{month:02d}.tar.gz"

        # Check for existing file
        force = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        if out_file_zip.exists() and not force:
            return out_file_zip
        if out_file_tgz.exists() and not force:
            return out_file_tgz

        # Build CDS request — default to saturation variable for ASCAT
        request = {
            'variable': [self._get_config_value(lambda: None, default='surface_soil_moisture_saturation', dict_key='ASCAT_SM_VARIABLE')],
            'type_of_sensor': ['active'],
            'time_aggregation': [self._get_config_value(lambda: None, default='daily', dict_key='ASCAT_SM_TIME_AGGREGATION')],
            'type_of_record': [self._get_config_value(lambda: None, default='cdr', dict_key='ASCAT_SM_RECORD_TYPE')],
            'version': [self._get_config_value(lambda: None, default='v202505', dict_key='ASCAT_SM_VERSION')],
            'year': [str(year)],
            'month': [f"{month:02d}"],
            'day': [f"{d:02d}" for d in range(1, days_in_month + 1)],
        }

        self.logger.info(f"Requesting ASCAT SM for {year}-{month:02d}...")

        c = cdsapi.Client()
        c.retrieve('satellite-soil-moisture', request, str(out_file_zip))

        return out_file_zip
