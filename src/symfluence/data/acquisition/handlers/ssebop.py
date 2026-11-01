# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SSEBop Evapotranspiration Acquisition Handler

Provides cloud acquisition for USGS SSEBop (operational Simplified Surface
Energy Balance) evapotranspiration data.

SSEBop Overview:
    Data Type: Satellite-derived actual evapotranspiration
    Resolution: 1km (CONUS), 10km (global)
    Coverage: CONUS (daily), Global (monthly)
    Variables: Actual ET (ETa)
    Temporal: Daily (CONUS 2000-present), monthly (global)
    Source: USGS EROS

Data Access:
    Primary: USGS FEWS NET Data Portal
    Alternative: USGS Earth Explorer
    Format: GeoTIFF

URL Patterns:
    CONUS: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/
    Global: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('SSEBOP')
class SSEBopAcquirer(BaseAcquisitionHandler):
    """
    Acquires SSEBop ET data from USGS FEWS NET portal.
    No authentication required - publicly available.
    """

    # USGS FEWS NET base URLs
    CONUS_BASE = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads"
    GLOBAL_BASE = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/monthly/eta/downloads"

    def download(self, output_dir: Path) -> Path:
        """
        Download SSEBop ET data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to output directory containing downloaded files
        """
        self.logger.info("Starting SSEBop ET acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine product type (conus or global)
        product = self._get_config_value(
            lambda: None,
            default='conus',
            dict_key='SSEBOP_PRODUCT'
        )
        if isinstance(product, str):
            product = product.lower()

        if product == 'global':
            return self._download_global(output_dir)
        else:
            return self._download_conus(output_dir)

    def _download_conus(self, output_dir: Path) -> Path:
        """Download CONUS daily SSEBop data."""
        # Generate list of dates
        dates = []
        current = self.start_date
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)

        self.logger.info(f"Downloading SSEBop CONUS daily for {len(dates)} days")

        downloaded_files: List[Path] = []
        session = requests.Session()

        for date in dates:
            try:
                out_file = self._download_conus_date(session, date, output_dir)
                if out_file:
                    downloaded_files.append(out_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed to download SSEBop for {date.strftime('%Y-%m-%d')}: {e}", exc_info=True)

        if not downloaded_files:
            raise RuntimeError("No SSEBop CONUS data could be downloaded")

        # Merge files into single NetCDF
        self._merge_to_netcdf(downloaded_files, output_dir, 'conus_daily')

        return output_dir

    def _download_conus_date(
        self,
        session: requests.Session,
        date: datetime,
        output_dir: Path
    ) -> Optional[Path]:
        """Download SSEBop CONUS for a single date."""
        import io
        import zipfile

        year = date.strftime('%Y')
        doy = date.strftime('%j')
        date_str = date.strftime('%Y%m%d')

        # SSEBop CONUS filename: det{YYYYDOY}.modisSSEBopETactual.zip (flat directory)
        zip_name = f"det{year}{doy}.modisSSEBopETactual.zip"
        url = f"{self.CONUS_BASE}/{zip_name}"

        # Output as extracted TIF
        out_file = output_dir / f"det{year}{doy}.modisSSEBopETactual.tif"

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            default=False,
            dict_key='FORCE_DOWNLOAD'
        ):
            return out_file

        self.logger.debug(f"Downloading: {zip_name}")

        try:
            response = session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Download ZIP into memory and extract the TIF
            zip_bytes = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_bytes) as zf:
                tif_names = [n for n in zf.namelist() if n.endswith('.tif')]
                if not tif_names:
                    self.logger.debug(f"No TIF in ZIP for {date_str}")
                    return None
                zf.extract(tif_names[0], output_dir)
                extracted = output_dir / tif_names[0]
                if extracted != out_file:
                    extracted.rename(out_file)

            return out_file

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.debug(f"SSEBop file not found for {date_str}")
            return None

    def _download_global(self, output_dir: Path) -> Path:
        """Download global monthly SSEBop data."""
        # Generate list of months
        months = []
        current = self.start_date.replace(day=1)
        end = self.end_date.replace(day=1)
        while current <= end:
            months.append(current)
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        self.logger.info(f"Downloading SSEBop global monthly for {len(months)} months")

        downloaded_files: List[Path] = []
        session = requests.Session()

        for month in months:
            try:
                out_file = self._download_global_month(session, month, output_dir)
                if out_file:
                    downloaded_files.append(out_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed to download SSEBop for {month.strftime('%Y-%m')}: {e}", exc_info=True)

        if not downloaded_files:
            raise RuntimeError("No SSEBop global data could be downloaded")

        # Merge files
        self._merge_to_netcdf(downloaded_files, output_dir, 'global_monthly')

        return output_dir

    def _download_global_month(
        self,
        session: requests.Session,
        month: datetime,
        output_dir: Path
    ) -> Optional[Path]:
        """Download SSEBop global for a single month."""
        year = month.strftime('%Y')
        month_num = month.strftime('%m')

        # Global monthly filename pattern varies
        filename = f"m{year}{month_num}eta.tif"
        url = f"{self.GLOBAL_BASE}/{year}/{filename}"

        out_file = output_dir / filename

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            default=False,
            dict_key='FORCE_DOWNLOAD'
        ):
            return out_file

        self.logger.debug(f"Downloading: {filename}")

        try:
            response = session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            tmp_file = out_file.with_suffix('.tif.part')
            with open(tmp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            tmp_file.replace(out_file)

            return out_file

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.debug(f"SSEBop file not found for {month.strftime('%Y-%m')}")
            return None

    def _merge_to_netcdf(
        self,
        files: List[Path],
        output_dir: Path,
        product_type: str
    ) -> Path:
        """
        Merge GeoTIFF files and clip to bounding box, save as NetCDF.

        Args:
            files: List of downloaded GeoTIFF files
            output_dir: Output directory
            product_type: Product type identifier

        Returns:
            Path to merged NetCDF file
        """
        import rasterio
        import xarray as xr

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        out_file = output_dir / f"{self.domain_name}_SSEBop_{product_type}_raw.nc"

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            default=False,
            dict_key='FORCE_DOWNLOAD'
        ):
            return out_file

        datasets = []

        for f in sorted(files):
            try:
                # Extract date from filename
                date = self._extract_date_from_filename(f.name)
                if date is None:
                    continue

                with rasterio.open(f) as src:
                    # Read and get bounds
                    data = src.read(1).astype(np.float32)
                    transform = src.transform
                    nodata = src.nodata if src.nodata is not None else -9999

                    # Create coordinate arrays
                    rows, cols = data.shape
                    lons = np.linspace(
                        transform.c,
                        transform.c + cols * transform.a,
                        cols
                    )
                    lats = np.linspace(
                        transform.f,
                        transform.f + rows * transform.e,
                        rows
                    )

                    # Handle nodata
                    data[data == nodata] = np.nan

                    # SSEBop stores ET in mm (scaled by 10 for CONUS)
                    # Apply scale factor if needed
                    if 'conus' in product_type:
                        data = data / 10.0  # Convert to mm/day

                    # Create DataArray
                    da = xr.DataArray(
                        data[np.newaxis, :, :],
                        dims=['time', 'lat', 'lon'],
                        coords={
                            'time': [date],
                            'lat': lats,
                            'lon': lons
                        },
                        name='et'
                    )

                    # Subset to bounding box
                    da = da.sel(
                        lat=slice(max(lat_min, lats.min()), min(lat_max, lats.max())),
                        lon=slice(max(lon_min, lons.min()), min(lon_max, lons.max()))
                    )

                    if da.size > 0:
                        datasets.append(da)

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.debug(f"Failed to process {f.name}: {e}", exc_info=True)

        if not datasets:
            raise RuntimeError("No SSEBop data could be processed")

        # Merge along time
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')

        # Create dataset and save
        ds = merged.to_dataset(name='et')
        ds['et'].attrs['units'] = 'mm/day'
        ds['et'].attrs['long_name'] = 'Actual Evapotranspiration'
        ds.to_netcdf(out_file)

        self.logger.info(f"Saved SSEBop ET data to {out_file}")
        return out_file

    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from SSEBop filename."""
        import re

        # CONUS: det2020001.modisSSEBopETv4.tif (YYYYDOY)
        match = re.search(r'det(\d{7})\.', filename)
        if match:
            year = int(match.group(1)[:4])
            doy = int(match.group(1)[4:])
            return datetime(year, 1, 1) + timedelta(days=doy - 1)

        # Global: m202001eta.tif (YYYYMM)
        match = re.search(r'm(\d{6})eta\.', filename)
        if match:
            year = int(match.group(1)[:4])
            month = int(match.group(1)[4:])
            return datetime(year, month, 1)

        return None
