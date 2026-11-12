# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IMS (Interactive Multisensor Snow and Ice Mapping System) Acquisition Handler

Provides acquisition for NOAA IMS daily snow cover products. IMS combines
multiple satellite sensors (visible + microwave) to produce cloud-free
daily snow/ice maps of the Northern Hemisphere.

IMS Snow features:
- Cloud-free (multi-sensor fusion)
- Daily temporal resolution
- Available resolutions: 1km (2014+), 4km (2004+), 24km (1997+)
- Binary snow/no-snow classification
- Covers full Northern Hemisphere

Value codes:
  0 = Outside Northern Hemisphere
  1 = Open Water
  2 = Land (no snow)
  3 = Sea Ice
  4 = Snow-covered Land

Data access via NSIDC HTTPS.

References:
- Product: https://nsidc.org/data/g02156
- Documentation: https://nsidc.org/sites/default/files/g02156-v001-userguide_1_1.pdf
"""
from __future__ import annotations

import gzip
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# NSIDC data endpoint
IMS_BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02156"

# IMS grid parameters (4km polar stereographic)
IMS_GRIDS = {
    '1km': {
        'ncols': 24576, 'nrows': 24576, 'cell_size': 1000,
        'x_ll': -12288000, 'y_ll': -12288000,
        'start_year': 2014
    },
    '4km': {
        'ncols': 6144, 'nrows': 6144, 'cell_size': 4000,
        'x_ll': -12288000, 'y_ll': -12288000,
        'start_year': 2004
    },
    '24km': {
        'ncols': 1024, 'nrows': 1024, 'cell_size': 24000,
        'x_ll': -12288000, 'y_ll': -12288000,
        'start_year': 1997
    }
}


@R.acquisition_handlers.add('IMS_SNOW')
@R.acquisition_handlers.add('IMS')
class IMSSnowAcquirer(BaseAcquisitionHandler):
    """
    Handles IMS snow cover data acquisition from NSIDC.

    IMS is a multi-sensor fusion product providing cloud-free daily snow maps.
    This is complementary to MODIS/VIIRS which have cloud gaps.

    Configuration:
        IMS_SNOW_RESOLUTION: '4km' (default), '1km', or '24km'
        IMS_SNOW_DOWNLOAD_RAW: False (default) - whether to keep raw ASCII files
    """

    # IMS value codes
    VALUE_OUTSIDE = 0
    VALUE_WATER = 1
    VALUE_LAND = 2
    VALUE_SEA_ICE = 3
    VALUE_SNOW = 4

    def download(self, output_dir: Path) -> Path:
        """
        Download IMS snow cover data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to processed NetCDF file
        """
        self.logger.info("Starting IMS snow cover acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get configuration
        resolution = self._get_config_value(lambda: None, default='4km', dict_key='IMS_SNOW_RESOLUTION')
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        # Output file
        output_file = output_dir / f"{self.domain_name}_IMS_snow_{resolution}.nc"

        if output_file.exists() and not force_download:
            self.logger.info(f"Using existing IMS snow file: {output_file}")
            return output_file

        # Validate resolution
        if resolution not in IMS_GRIDS:
            raise ValueError(f"Invalid IMS resolution: {resolution}. Must be one of {list(IMS_GRIDS.keys())}")

        grid_params = IMS_GRIDS[resolution]

        # Check date range
        start_year = max(self.start_date.year, grid_params['start_year'])
        if start_year > self.start_date.year:
            self.logger.warning(
                f"IMS {resolution} data starts {grid_params['start_year']}, "
                f"adjusting start from {self.start_date.year}"
            )

        # Calculate pixel bounds for domain
        bbox_pixels = self._get_bbox_pixels(resolution)
        row_start, row_end, col_start, col_end = bbox_pixels
        self.logger.info(f"Domain pixels: rows {row_start}-{row_end}, cols {col_start}-{col_end}")

        # Process each year
        all_results = []
        for year in range(start_year, self.end_date.year + 1):
            year_results = self._process_year(year, resolution, bbox_pixels)
            all_results.extend(year_results)
            self.logger.info(f"Year {year}: {len(year_results)} daily records")

        if not all_results:
            raise RuntimeError("No IMS data collected")

        # Create xarray dataset
        self._create_output_dataset(all_results, output_file, resolution)

        self.logger.info(f"IMS acquisition complete: {output_file}")
        return output_file

    def _latlon_to_ims_pixel(self, lat: float, lon: float, resolution: str) -> Tuple[int, int]:
        """Convert lat/lon to IMS grid pixel coordinates."""
        grid = IMS_GRIDS[resolution]

        # Earth radius (meters)
        R = 6371228.0

        # IMS polar stereographic parameters
        lat_ts = 60.0  # latitude of true scale
        lon_0 = -80.0  # central meridian

        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        lat_ts_rad = math.radians(lat_ts)
        lon_0_rad = math.radians(lon_0)

        # Scale factor
        k0 = (1 + math.sin(lat_ts_rad)) / 2

        # Polar stereographic projection
        t = math.tan(math.pi/4 - lat_rad/2)
        rho = 2 * R * k0 * t

        x = rho * math.sin(lon_rad - lon_0_rad)
        y = -rho * math.cos(lon_rad - lon_0_rad)

        # Convert to pixel coordinates
        col = int((x - grid['x_ll']) / grid['cell_size'])
        row = int((y - grid['y_ll']) / grid['cell_size'])

        return row, col

    def _get_bbox_pixels(self, resolution: str) -> Tuple[int, int, int, int]:
        """Get pixel bounds for the domain bounding box."""
        grid = IMS_GRIDS[resolution]

        # Get corner pixels
        row_min, col_min = self._latlon_to_ims_pixel(
            self.bbox['lat_max'], self.bbox['lon_min'], resolution
        )
        row_max, col_max = self._latlon_to_ims_pixel(
            self.bbox['lat_min'], self.bbox['lon_max'], resolution
        )

        # Ensure proper ordering and add buffer
        buffer = 5
        row_start = max(0, min(row_min, row_max) - buffer)
        row_end = min(grid['nrows'], max(row_min, row_max) + buffer)
        col_start = max(0, min(col_min, col_max) - buffer)
        col_end = min(grid['ncols'], max(col_min, col_max) + buffer)

        return row_start, row_end, col_start, col_end

    def _list_available_files(self, year: int, resolution: str) -> List[str]:
        """List available IMS files for a given year."""
        url = f"{IMS_BASE_URL}/{resolution}/{year}/"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            files = re.findall(r'href="([^"]+\.asc\.gz)"', response.text)
            return sorted(files)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Could not list files for {year}: {e}", exc_info=True)
            return []

    def _download_and_parse(self, year: int, doy: int, resolution: str) -> Optional[np.ndarray]:
        """Download and parse a single IMS file."""
        grid = IMS_GRIDS[resolution]

        # Try different filename patterns
        patterns = [
            f"ims{year}{doy:03d}_00UTC_{resolution}_v1.3.asc.gz",
            f"ims{year}{doy:03d}_{resolution}_v1.3.asc.gz",
            f"ims{year}{doy:03d}_{resolution}.asc.gz",
        ]

        url_base = f"{IMS_BASE_URL}/{resolution}/{year}/"
        response = None

        for pattern in patterns:
            try:
                response = requests.get(url_base + pattern, timeout=120)
                if response.status_code == 200:
                    break
            except Exception:  # noqa: BLE001 — preprocessing resilience
                continue

        if response is None or response.status_code != 200:
            return None

        try:
            # Decompress
            content = gzip.decompress(response.content).decode('utf-8', errors='ignore')
            lines = content.split('\n')

            # Find data start
            data_start = 0
            for i, line in enumerate(lines):
                if 'Data set starts here' in line:
                    data_start = i + 1
                    break

            # Parse grid data
            grid_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = [int(c) for c in line if c.isdigit()]
                    if len(row) == grid['ncols']:
                        grid_data.append(row)
                except ValueError:
                    continue

            if len(grid_data) == grid['nrows']:
                return np.array(grid_data, dtype=np.uint8)
            return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.debug(f"Error parsing IMS file: {e}", exc_info=True)
            return None

    def _extract_domain_stats(
        self,
        grid: np.ndarray,
        bbox_pixels: Tuple[int, int, int, int]
    ) -> Dict:
        """Extract snow statistics for the domain."""
        row_start, row_end, col_start, col_end = bbox_pixels
        subset = grid[row_start:row_end, col_start:col_end]

        total_pixels = subset.size
        land_pixels = np.sum((subset == self.VALUE_LAND) | (subset == self.VALUE_SNOW))
        snow_pixels = np.sum(subset == self.VALUE_SNOW)
        water_pixels = np.sum(subset == self.VALUE_WATER)

        snow_fraction = snow_pixels / land_pixels if land_pixels > 0 else np.nan

        return {
            'snow_fraction': snow_fraction,
            'snow_pixels': int(snow_pixels),
            'land_pixels': int(land_pixels),
            'water_pixels': int(water_pixels),
            'total_pixels': int(total_pixels),
        }

    def _process_year(
        self,
        year: int,
        resolution: str,
        bbox_pixels: Tuple[int, int, int, int]
    ) -> List[Dict]:
        """Process all files for a year."""
        results: List[Dict] = []
        files = self._list_available_files(year, resolution)

        if not files:
            self.logger.warning(f"No files found for {year}")
            return results

        self.logger.info(f"Processing {year}: {len(files)} files")

        for i, filename in enumerate(files):
            # Extract DOY from filename
            match = re.search(r'ims(\d{4})(\d{3})', filename)
            if not match:
                continue

            file_year = int(match.group(1))
            doy = int(match.group(2))
            date = datetime(file_year, 1, 1) + timedelta(days=doy - 1)

            # Skip dates outside our range
            if date < self.start_date or date > self.end_date:
                continue

            if (i + 1) % 30 == 0:
                self.logger.info(f"  {date.strftime('%Y-%m-%d')} ({i+1}/{len(files)})")

            # Download and parse
            grid = self._download_and_parse(year, doy, resolution)
            if grid is None:
                continue

            # Extract stats
            stats = self._extract_domain_stats(grid, bbox_pixels)
            stats['date'] = date
            stats['doy'] = doy
            results.append(stats)

        return results

    def _create_output_dataset(
        self,
        results: List[Dict],
        output_file: Path,
        resolution: str
    ):
        """Create output NetCDF dataset."""
        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Create xarray dataset
        ds = xr.Dataset({
            'snow_fraction': xr.DataArray(
                df['snow_fraction'].values,
                dims=['time'],
                coords={'time': df.index.values},
                attrs={
                    'long_name': 'Fraction of land covered by snow',
                    'units': '1',
                    'valid_range': [0, 1]
                }
            ),
            'snow_pixels': xr.DataArray(
                df['snow_pixels'].values,
                dims=['time'],
                coords={'time': df.index.values},
                attrs={'long_name': 'Number of snow-covered pixels', 'units': 'count'}
            ),
            'land_pixels': xr.DataArray(
                df['land_pixels'].values,
                dims=['time'],
                coords={'time': df.index.values},
                attrs={'long_name': 'Number of land pixels', 'units': 'count'}
            ),
        })

        ds.attrs['title'] = 'IMS Snow Cover'
        ds.attrs['source'] = f'NOAA IMS {resolution} (G02156)'
        ds.attrs['institution'] = 'NOAA/NSIDC'
        ds.attrs['references'] = 'https://nsidc.org/data/g02156'
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['domain'] = self.domain_name
        ds.attrs['bbox'] = f"{self.bbox['lon_min']},{self.bbox['lat_min']},{self.bbox['lon_max']},{self.bbox['lat_max']}"

        ds.to_netcdf(output_file)

        self.logger.info(f"Saved {len(df)} records to {output_file}")
        self.logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        self.logger.info(f"  Snow fraction: {df['snow_fraction'].min():.3f} - {df['snow_fraction'].max():.3f}")
