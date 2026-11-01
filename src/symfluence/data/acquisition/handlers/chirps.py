# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CHIRPS Precipitation Acquisition Handler

Provides cloud acquisition for CHIRPS (Climate Hazards Group InfraRed Precipitation
with Station data) precipitation data.

CHIRPS Overview:
    Data Type: Quasi-global rainfall estimates
    Resolution: 0.05° x 0.05° (~5km)
    Coverage: 50°S to 50°N, global
    Temporal: Daily, pentadal, monthly
    Record: 1981-present
    Source: UC Santa Barbara Climate Hazards Group

Data Access:
    Primary: UCSB CHG data server via HTTPS
    Alternative: IRI Data Library (Columbia)
    Format: GeoTIFF or NetCDF

URL Structure:
    https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('CHIRPS')
class CHIRPSAcquirer(BaseAcquisitionHandler):
    """
    Acquires CHIRPS precipitation data from UCSB CHG data server.
    No authentication required - publicly available.
    """

    # UCSB CHG data server
    CHG_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"

    # Product configurations
    PRODUCTS = {
        'daily': {
            'path': 'global_daily/netcdf/p05',
            'filename_pattern': 'chirps-v2.0.{year}.days_p05.nc',
        },
        'pentad': {
            'path': 'global_pentad/netcdf',
            'filename_pattern': 'chirps-v2.0.{year}.pentads.nc',
        },
        'monthly': {
            'path': 'global_monthly/netcdf',
            'filename_pattern': 'chirps-v2.0.{year}.months.nc',
        },
    }

    def download(self, output_dir: Path) -> Path:
        """
        Download CHIRPS precipitation data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to output directory containing downloaded files
        """
        self.logger.info("Starting CHIRPS precipitation acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get product type from config
        product_type = self._get_config_value(
            lambda: None,
            default='daily',
            dict_key='CHIRPS_PRODUCT'
        )
        if isinstance(product_type, str):
            product_type = product_type.lower()
        if product_type not in self.PRODUCTS:
            self.logger.warning(f"Unknown CHIRPS product '{product_type}', defaulting to 'daily'")
            product_type = 'daily'

        product = self.PRODUCTS[product_type]

        # Determine years to download
        start_year = self.start_date.year
        end_year = self.end_date.year
        years = list(range(start_year, end_year + 1))

        self.logger.info(f"Downloading CHIRPS {product_type} data for years {start_year}-{end_year}")

        downloaded_files: List[Path] = []
        session = requests.Session()

        for year in years:
            filename = product['filename_pattern'].format(year=year)
            url = f"{self.CHG_BASE}/{product['path']}/{filename}"
            out_file = output_dir / filename

            if self._skip_if_exists(out_file):
                downloaded_files.append(out_file)
                continue

            self.logger.info(f"Downloading: {filename}")
            try:
                response = session.get(url, stream=True, timeout=600)
                response.raise_for_status()

                tmp_file = out_file.with_suffix(out_file.suffix + ".part")
                with open(tmp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                tmp_file.replace(out_file)
                downloaded_files.append(out_file)
                self.logger.info(f"Downloaded: {filename}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    self.logger.warning(f"CHIRPS file not found for {year} (may not be available yet)")
                else:
                    self.logger.error(f"Failed to download CHIRPS for {year}: {e}")
            except Exception as e:  # noqa: BLE001 — data acquisition resilience
                self.logger.error(f"Failed to download CHIRPS for {year}: {e}")

        if not downloaded_files:
            raise RuntimeError("No CHIRPS data could be downloaded for the requested period")

        # Subset and merge files to bounding box
        self._subset_and_merge(downloaded_files, output_dir, product_type)

        return output_dir

    def _subset_and_merge(
        self,
        files: List[Path],
        output_dir: Path,
        product_type: str
    ) -> Path:
        """
        Subset downloaded files to bounding box and merge into single file.

        Args:
            files: List of downloaded CHIRPS NetCDF files
            output_dir: Output directory
            product_type: Product type (daily, pentad, monthly)

        Returns:
            Path to merged output file
        """
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        out_file = output_dir / f"{self.domain_name}_CHIRPS_{product_type}_raw.nc"

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            default=False,
            dict_key='FORCE_DOWNLOAD'
        ):
            return out_file

        datasets = []
        for f in sorted(files):
            try:
                ds = xr.open_dataset(f)

                # CHIRPS uses 'latitude' and 'longitude' as dimension names
                lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
                lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'

                # Check latitude order
                if ds[lat_dim][0] > ds[lat_dim][-1]:
                    lat_slice = slice(lat_max, lat_min)
                else:
                    lat_slice = slice(lat_min, lat_max)

                # Subset spatially
                ds_sub = ds.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})

                # Subset temporally
                if 'time' in ds_sub.dims:
                    ds_sub = ds_sub.sel(time=slice(self.start_date, self.end_date))

                datasets.append(ds_sub)
            except Exception as e:  # noqa: BLE001 — data acquisition resilience
                self.logger.warning(f"Failed to process {f.name}: {e}")
                continue

        if not datasets:
            raise RuntimeError("No CHIRPS data could be processed")

        # Merge along time dimension
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')

        # Save merged file
        merged.to_netcdf(out_file)
        self.logger.info(f"Saved merged CHIRPS data to {out_file}")

        # Clean up individual datasets
        for ds in datasets:
            ds.close()

        return out_file
