# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ESA CCI Soil Moisture Acquisition Handler

Provides cloud acquisition for ESA CCI (Climate Change Initiative) Soil Moisture data:
- Uses Copernicus Climate Data Store (CDS) API
- Supports monthly temporal chunking for large requests
- Includes retry logic for CDS rate limits

ESA CCI SM Overview:
    Data Type: Satellite-derived soil moisture (merged active/passive sensors)
    Resolution: 0.25° (global)
    Coverage: Global
    Source: ESA CCI via Copernicus CDS

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
from ..mixins import ChunkedDownloadMixin, RetryMixin


@R.acquisition_handlers.add('ESA_CCI_SM')
class ESACCISMAcquirer(BaseAcquisitionHandler, RetryMixin, ChunkedDownloadMixin):
    """
    Acquires ESA CCI Soil Moisture data via Copernicus CDS.

    Uses RetryMixin for CDS API retries and ChunkedDownloadMixin for monthly chunking.
    """

    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi required for ESA CCI SM acquisition")

        self.logger.info("Starting ESA CCI Soil Moisture acquisition via CDS")

        output_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = output_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        # Store output_dir for use in download function
        self._output_dir = output_dir

        # Generate year-month list using mixin method
        ym_range = self.generate_year_month_list(self.start_date, self.end_date)

        # Download chunks (sequential for CDS rate limits)
        downloads = self.download_chunks_parallel(
            chunks=ym_range,
            download_func=self._download_month_chunk,
            max_workers=1,  # CDS has strict rate limits
            desc="ESA CCI SM download",
            fail_fast=False  # Continue even if some months fail
        )

        # Extract all downloaded archives
        for out_file in downloads:
            if out_file and out_file.exists():
                try:
                    shutil.unpack_archive(str(out_file), extract_dir)
                except Exception as exc:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Failed to extract ESA CCI SM archive {out_file.name}: {exc}", exc_info=True)

        self.logger.info(f"Extracted ESA CCI SM data to {extract_dir}")
        return extract_dir

    def _download_month_chunk(self, year_month: tuple) -> Optional[Path]:
        """Download a single month of ESA CCI SM data using RetryMixin."""
        year, month = year_month
        days_in_month = calendar.monthrange(year, month)[1]

        out_file_zip = self._output_dir / f"{self.domain_name}_ESA_CCI_SM_{year}{month:02d}.zip"
        out_file_tgz = self._output_dir / f"{self.domain_name}_ESA_CCI_SM_{year}{month:02d}.tar.gz"

        # Check for existing file
        if out_file_zip.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return out_file_zip
        if out_file_tgz.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return out_file_tgz

        # Build CDS request
        request = {
            'variable': [self._get_config_value(lambda: None, default='surface_soil_moisture_volumetric', dict_key='ESA_CCI_SM_VARIABLE')],
            'type_of_sensor': [self._get_config_value(lambda: None, default='combined', dict_key='ESA_CCI_SM_SENSOR')],
            'time_aggregation': [self._get_config_value(lambda: None, default='daily', dict_key='ESA_CCI_SM_TIME_AGGREGATION')],
            'type_of_record': [self._get_config_value(lambda: None, default='cdr', dict_key='ESA_CCI_SM_RECORD_TYPE')],
            'version': [self._get_config_value(lambda: None, default='v202312', dict_key='ESA_CCI_SM_VERSION')],
            'year': [str(year)],
            'month': [f"{month:02d}"],
            'day': [f"{d:02d}" for d in range(1, days_in_month + 1)],
        }

        self.logger.info(f"Requesting ESA CCI SM for {year}-{month:02d}...")

        def do_retrieve():
            c = cdsapi.Client()
            c.retrieve('satellite-soil-moisture', request, str(out_file_zip))

        self.execute_with_retry(
            do_retrieve,
            max_retries=3,
            base_delay=60,
            retry_condition=self.is_retryable_cds_error
        )

        return out_file_zip
