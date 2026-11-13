# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5-Land Data Acquisition Handler

Provides cloud acquisition for ERA5-Land reanalysis data via the Copernicus
Climate Data Store (CDS) API. ERA5-Land is a high-resolution (9km) land
component of the ERA5 climate reanalysis.

Supports multiple hydrological variables:
- Precipitation (total_precipitation)
- Temperature (2m_temperature)
- Snow (snow_depth, snow_depth_water_equivalent)
- Soil moisture (volumetric_soil_water_layer_1-4)
- Evaporation (total_evaporation, potential_evaporation)
- Runoff (surface_runoff, subsurface_runoff)
"""
from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import SpatialSubsetMixin

# ERA5-Land is served on a 0.1-deg regular lat/lon grid.
ERA5_LAND_GRID_RESOLUTION_DEG = 0.1

# Variable mapping for ERA5-Land
ERA5_LAND_VARIABLES = {
    'precipitation': ['total_precipitation'],
    'temperature': ['2m_temperature'],
    'snow': ['snow_depth', 'snow_depth_water_equivalent'],
    'soil_moisture': [
        'volumetric_soil_water_layer_1',
        'volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3',
        'volumetric_soil_water_layer_4',
    ],
    'evaporation': ['total_evaporation', 'potential_evaporation'],
    'runoff': ['surface_runoff', 'subsurface_runoff'],
    'radiation': [
        'surface_net_solar_radiation',
        'surface_net_thermal_radiation',
    ],
}

# All available variables for comprehensive download
ALL_ERA5_LAND_VARIABLES = [
    'total_precipitation',
    '2m_temperature',
    'snow_depth',
    'snow_depth_water_equivalent',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'total_evaporation',
    'potential_evaporation',
    'surface_runoff',
    'subsurface_runoff',
    'skin_temperature',
    'soil_temperature_level_1',
]


@R.acquisition_handlers.add('ERA5_LAND')
@R.acquisition_handlers.add('ERA5-LAND')
class ERA5LandAcquirer(BaseAcquisitionHandler, SpatialSubsetMixin):
    """
    Handles ERA5-Land data acquisition from Copernicus CDS.

    Downloads hourly or daily ERA5-Land reanalysis data for specified
    bounding box and time period. Supports variable selection for
    targeted downloads.

    Configuration:
        ERA5_LAND_VARIABLES: List of variables to download (default: all)
        ERA5_LAND_FREQUENCY: 'hourly' or 'daily' (default: daily)
        ERA5_LAND_FORMAT: 'netcdf' or 'grib' (default: netcdf)
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download ERA5-Land data from CDS.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded NetCDF file(s)
        """
        self.logger.info("Starting ERA5-Land data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get configuration
        variables = self._get_variables()
        frequency = self._get_config_value(lambda: None, default='daily', dict_key='ERA5_LAND_FREQUENCY')
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')

        # Build output filename
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        output_file = output_dir / f"era5_land_{start_str}_{end_str}_{frequency}.nc"

        if output_file.exists() and not force_download:
            self.logger.info(f"ERA5-Land file already exists: {output_file}")
            return output_file

        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                "cdsapi package required for ERA5-Land download. "
                "Install with: pip install cdsapi"
            ) from None

        # Initialize CDS client
        client = cdsapi.Client()

        # Build request parameters
        request = self._build_request(variables, frequency)

        self.logger.info(f"Requesting ERA5-Land data: {len(variables)} variables, "
                        f"{self.start_date} to {self.end_date}")

        # Download by year-month chunks for large requests
        if self._request_is_large():
            self._download_chunked(client, request, output_dir, frequency)
            # Merge chunks into single file
            output_file = self._merge_chunks(output_dir, output_file)
        else:
            client.retrieve('reanalysis-era5-land', request, str(output_file))

        self.logger.info(f"ERA5-Land download complete: {output_file}")
        return output_file

    def _get_variables(self) -> List[str]:
        """Get list of variables to download."""
        config_vars = self._get_config_value(lambda: None, default=None, dict_key='ERA5_LAND_VARIABLES')

        if config_vars:
            if isinstance(config_vars, str):
                # Check if it's a category name
                if config_vars in ERA5_LAND_VARIABLES:
                    return ERA5_LAND_VARIABLES[config_vars]
                return [config_vars]
            return list(config_vars)

        # Default: core hydrological variables
        return [
            'total_precipitation',
            '2m_temperature',
            'snow_depth_water_equivalent',
            'volumetric_soil_water_layer_1',
            'total_evaporation',
        ]

    def _build_request(self, variables: List[str], frequency: str) -> Dict[str, Any]:
        """Build CDS API request dictionary."""
        # Generate specific date lists for the requested period
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')

        years = sorted(set(str(d.year) for d in date_range))
        months = sorted(set(f"{d.month:02d}" for d in date_range))
        days = sorted(set(f"{d.day:02d}" for d in date_range))

        request = {
            'variable': variables,
            'year': years,
            'month': months,
            'day': days,
            'format': self._get_config_value(lambda: None, default='netcdf', dict_key='ERA5_LAND_FORMAT'),
        }

        # Add time for hourly data
        if frequency == 'hourly':
            request['time'] = [f"{h:02d}:00" for h in range(24)]
        else:
            # For daily aggregation, request fewer hours to reduce request size
            # CDS will provide instantaneous values at these times
            request['time'] = ['00:00', '06:00', '12:00', '18:00']

        # Add spatial subsetting if bbox available. Snap to ERA5-Land's
        # 0.1-deg grid so a point/sub-cell domain still encloses at least one
        # node — a raw sub-cell rectangle makes MARS reject the crop with
        # "non-empty area crop/mask (to at least one point)".
        if self.bbox:
            request['area'] = self.bbox_to_cds_area(
                snap_resolution=ERA5_LAND_GRID_RESOLUTION_DEG
            )

        return request

    def _request_is_large(self) -> bool:
        """Check if request spans multiple years."""
        return (self.end_date.year - self.start_date.year) > 1

    def _download_chunked(
        self,
        client,
        base_request: Dict[str, Any],
        output_dir: Path,
        frequency: str
    ):
        """Download data in year-month chunks."""
        current = self.start_date

        while current <= self.end_date:
            year = current.year
            month = current.month

            # Adjust end day for partial months
            _, last_day = calendar.monthrange(year, month)
            if year == self.end_date.year and month == self.end_date.month:
                last_day = min(last_day, self.end_date.day)

            start_day = 1
            if year == self.start_date.year and month == self.start_date.month:
                start_day = self.start_date.day

            chunk_request = base_request.copy()
            chunk_request['year'] = [str(year)]
            chunk_request['month'] = [f"{month:02d}"]
            chunk_request['day'] = [f"{d:02d}" for d in range(start_day, last_day + 1)]

            chunk_file = output_dir / f"era5_land_{year}{month:02d}.nc"

            if not chunk_file.exists():
                self.logger.info(f"Downloading ERA5-Land {year}-{month:02d}")
                client.retrieve('reanalysis-era5-land', chunk_request, str(chunk_file))

            # Move to next month
            if month == 12:
                current = datetime(year + 1, 1, 1)
            else:
                current = datetime(year, month + 1, 1)

    def _merge_chunks(self, output_dir: Path, output_file: Path) -> Path:
        """Merge downloaded chunks into single file."""
        chunk_files = sorted(output_dir.glob("era5_land_*.nc"))
        chunk_files = [f for f in chunk_files if f != output_file]

        if len(chunk_files) == 1:
            chunk_files[0].replace(output_file)
            return output_file

        self.logger.info(f"Merging {len(chunk_files)} ERA5-Land chunks")

        datasets = [xr.open_dataset(f) for f in chunk_files]
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')
        merged.to_netcdf(output_file)

        # Close and cleanup
        for ds in datasets:
            ds.close()

        # Remove chunk files
        for f in chunk_files:
            f.unlink()

        return output_file
