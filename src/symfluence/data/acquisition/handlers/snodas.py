# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SNODAS Snow Data Acquisition Handler

Provides cloud acquisition for NOAA SNODAS (SNOw Data Assimilation System) data.

SNODAS Overview:
    Data Type: Snow analysis (assimilated satellite + ground observations)
    Resolution: ~1km (30 arc-second)
    Coverage: CONUS and southern Canada
    Variables: SWE, snow depth, snow melt runoff, sublimation, sublimation from blowing snow
    Temporal: Daily
    Record: 2003-present
    Source: NOAA NOHRSC / NSIDC

Data Access:
    Primary: NSIDC via HTTPS
    Format: Masked binary (requires unpacking) or NetCDF via THREDDS
    URL: https://nsidc.org/data/g02158

Variables:
    - swe: Snow Water Equivalent (m)
    - snow_depth: Snow depth (m)
    - snowmelt_runoff: Snowmelt runoff at base of snow pack (m)
    - sublimation: Sublimation from the snow pack (m)
"""
from __future__ import annotations

import gzip
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('SNODAS')
class SNODASAcquirer(BaseAcquisitionHandler):
    """
    Acquires SNODAS snow data from NSIDC.
    No authentication required - publicly available.
    """

    # NSIDC FTP/HTTPS server
    NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02158"

    # SNODAS grid parameters (from metadata)
    # Note: SNODAS unmasked files use a larger grid than the masked CONUS product
    NROWS = 4096
    NCOLS = 8192
    CELLSIZE = 0.00833333333333333  # ~1km
    XLLCORNER = -130.516666666661  # Minimum x-axis coordinate
    YLLCORNER = 24.0999999999990   # Minimum y-axis coordinate

    # Variable codes in SNODAS files
    VARIABLES = {
        'swe': {
            'code': '1034',
            'scale': 1.0,  # Already in meters
            'description': 'Snow Water Equivalent',
        },
        'snow_depth': {
            'code': '1036',
            'scale': 1.0,
            'description': 'Snow Depth',
        },
        'snowmelt_runoff': {
            'code': '1044',
            'scale': 1.0,
            'description': 'Snowmelt Runoff at Base of Snow Pack',
        },
        'sublimation': {
            'code': '1050',
            'scale': 1.0,
            'description': 'Sublimation from Snow Pack',
        },
    }

    def download(self, output_dir: Path) -> Path:
        """
        Download SNODAS snow data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to output directory containing downloaded files
        """
        self.logger.info("Starting SNODAS snow data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get variable from config
        variable = self._get_config_value(
            lambda: None,
            default='swe',
            dict_key='SNODAS_VARIABLE'
        )
        if isinstance(variable, str):
            variable = variable.lower()
        if variable not in self.VARIABLES:
            self.logger.warning(f"Unknown SNODAS variable '{variable}', defaulting to 'swe'")
            variable = 'swe'

        # Generate list of dates
        dates = []
        current = self.start_date
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)

        self.logger.info(f"Downloading SNODAS {variable} for {len(dates)} days")

        downloaded_files: List[Path] = []
        session = requests.Session()

        for date in dates:
            try:
                out_file = self._download_date(session, date, variable, output_dir)
                if out_file:
                    downloaded_files.append(out_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to download SNODAS for {date.strftime('%Y-%m-%d')}: {e}", exc_info=True)

        if not downloaded_files:
            raise RuntimeError("No SNODAS data could be downloaded for the requested period")

        # Merge into single NetCDF
        self._merge_to_netcdf(downloaded_files, output_dir, variable)

        return output_dir

    def _download_date(
        self,
        session: requests.Session,
        date: datetime,
        variable: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Download SNODAS data for a single date."""
        year = date.strftime('%Y')
        month = date.strftime('%m')
        month_name = date.strftime('%b')
        date_str = date.strftime('%Y%m%d')

        # SNODAS files are in tar.gz format
        # URL pattern: /unmasked/YYYY/MM_Mon/SNODAS_unmasked_YYYYMMDD.tar
        filename = f"SNODAS_unmasked_{date_str}.tar"
        url = f"{self.NSIDC_BASE}/unmasked/{year}/{month}_{month_name}/{filename}"

        tar_file = output_dir / filename
        if tar_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return tar_file

        self.logger.debug(f"Downloading: {filename}")

        try:
            response = session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            tmp_file = tar_file.with_suffix('.tar.part')
            with open(tmp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            tmp_file.replace(tar_file)

            return tar_file

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.debug(f"SNODAS file not found for {date_str}")
            else:
                self.logger.warning(f"HTTP error downloading SNODAS for {date_str}: {e}")
            return None

    def _merge_to_netcdf(
        self,
        tar_files: List[Path],
        output_dir: Path,
        variable: str
    ) -> Path:
        """
        Extract and merge SNODAS tar files into a single NetCDF.

        Args:
            tar_files: List of downloaded tar files
            output_dir: Output directory
            variable: Variable to extract (swe, snow_depth, etc.)

        Returns:
            Path to merged NetCDF file
        """
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        out_file = output_dir / f"{self.domain_name}_SNODAS_{variable}_raw.nc"

        if out_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return out_file

        var_code = self.VARIABLES[variable]['code']
        scale = self.VARIABLES[variable]['scale']

        # Calculate grid indices for bounding box
        col_min = max(0, int((lon_min - self.XLLCORNER) / self.CELLSIZE))
        col_max = min(self.NCOLS, int((lon_max - self.XLLCORNER) / self.CELLSIZE) + 1)
        row_max = max(0, self.NROWS - int((lat_min - self.YLLCORNER) / self.CELLSIZE))
        row_min = max(0, self.NROWS - int((lat_max - self.YLLCORNER) / self.CELLSIZE))

        # Generate coordinate arrays for subset
        lons = np.linspace(
            self.XLLCORNER + col_min * self.CELLSIZE,
            self.XLLCORNER + col_max * self.CELLSIZE,
            col_max - col_min
        )
        lats = np.linspace(
            self.YLLCORNER + (self.NROWS - row_max) * self.CELLSIZE,
            self.YLLCORNER + (self.NROWS - row_min) * self.CELLSIZE,
            row_max - row_min
        )

        datasets = []

        for tar_path in sorted(tar_files):
            try:
                # Extract date from filename
                date_str = tar_path.stem.split('_')[-1]
                date = datetime.strptime(date_str, '%Y%m%d')

                # Extract and read the specific variable file
                data = self._extract_variable(tar_path, var_code, row_min, row_max, col_min, col_max)
                if data is None:
                    continue

                # Apply scale factor
                data = data * scale  # type: ignore[union-attr]

                # Create DataArray
                da = xr.DataArray(
                    data[np.newaxis, :, :],  # type: ignore[index]
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': [date],
                        'lat': lats,
                        'lon': lons
                    },
                    name=variable
                )
                datasets.append(da)

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to process {tar_path.name}: {e}", exc_info=True)

        if not datasets:
            raise RuntimeError("No SNODAS data could be processed")

        # Merge along time
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')

        # Create dataset and save
        ds = merged.to_dataset(name=variable)
        ds[variable].attrs['units'] = 'm'
        ds[variable].attrs['long_name'] = self.VARIABLES[variable]['description']
        ds.to_netcdf(out_file)

        self.logger.info(f"Saved SNODAS {variable} data to {out_file}")
        return out_file

    def _extract_variable(
        self,
        tar_path: Path,
        var_code: str,
        row_min: int,
        row_max: int,
        col_min: int,
        col_max: int
    ) -> Optional[np.ndarray]:
        """
        Extract a specific variable from SNODAS tar file.

        SNODAS files are in a custom binary format with gzip compression.
        Filenames follow pattern: zz_ssmv1XXXXtS__T0001TTNATSYYYYMMDD05HP001.dat.gz
        where XXXX is the variable code (e.g., 1034 for SWE).
        """
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Find the file matching the variable code
                # Pattern: ssmv1XXXX where XXXX is the code (e.g., ssmv11034 for SWE)
                search_pattern = f'ssmv1{var_code}'

                for member in tar.getmembers():
                    # Look for the data file with matching variable code
                    if search_pattern in member.name and member.name.endswith('.dat.gz'):
                        self.logger.debug(f"Found SNODAS file: {member.name}")

                        # Extract gzipped data file
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        # Decompress and read binary data
                        with gzip.GzipFile(fileobj=f) as gz:
                            # SNODAS uses big-endian 2-byte integers
                            raw_data = np.frombuffer(gz.read(), dtype='>i2')

                        # Reshape to grid
                        try:
                            data = raw_data.reshape(self.NROWS, self.NCOLS)
                        except ValueError:
                            # Some files may have different dimensions
                            self.logger.debug(f"Grid size mismatch for {member.name}, trying alternative dimensions")
                            continue

                        # Subset to bounding box
                        data_subset = data[row_min:row_max, col_min:col_max].astype(np.float32)

                        # Handle no-data values (-9999)
                        data_subset[data_subset == -9999] = np.nan

                        # Convert to meters (SNODAS stores in mm or scaled units)
                        data_subset = data_subset / 1000.0

                        return data_subset

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.debug(f"Error extracting from {tar_path.name}: {e}", exc_info=True)

        return None
