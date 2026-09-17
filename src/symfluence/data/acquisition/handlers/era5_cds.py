# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5 CDS Data Acquisition Handler for SYMFLUENCE.

Uses mixins for:
- RetryMixin: Exponential backoff retry for CDS API calls
- ChunkedDownloadMixin: Monthly temporal chunking and parallel downloads
- SpatialSubsetMixin: Bbox format conversion for CDS
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from symfluence.core.registries import R

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..mixins import ChunkedDownloadMixin, RetryMixin, SpatialSubsetMixin
from .era5_processing import era5_to_summa_schema

# ERA5 reanalysis single-levels are served on a 0.25-deg regular lat/lon grid.
ERA5_GRID_RESOLUTION_DEG = 0.25


@R.acquisition_handlers.add('ERA5_CDS')
class ERA5CDSAcquirer(BaseAcquisitionHandler, RetryMixin, ChunkedDownloadMixin, SpatialSubsetMixin):
    """
    ERA5 data acquisition handler using the Copernicus Climate Data Store (CDS) API.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process ERA5 data from CDS in monthly chunks."""
        if not HAS_CDSAPI:
            raise ImportError(
                "cdsapi package is required for ERA5 CDS downloads. "
                "Install it with 'pip install cdsapi'."
            )

        self.logger.info(f"Downloading ERA5 data from CDS for {self.domain_name}...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store output_dir for use in download function
        self._output_dir = output_dir

        # Generate year-month list using mixin method
        ym_range = self.generate_year_month_list(self.start_date, self.end_date)

        # Download chunks using mixin (sequential for CDS rate limits)
        chunk_files = self.download_chunks_parallel(
            chunks=ym_range,
            download_func=self._download_month_chunk,
            max_workers=1,  # CDS has strict rate limits
            desc="ERA5 CDS download",
            fail_fast=True
        )

        if not chunk_files:
            raise RuntimeError("No ERA5 data downloaded")

        # Merge chunks using mixin method
        final_f = output_dir / f"domain_{self.domain_name}_ERA5_CDS_{self.start_date.year}_{self.end_date.year}.nc"

        return self.merge_netcdf_chunks(
            chunk_files=chunk_files,
            output_file=final_f,
            time_slice=(self.start_date, self.end_date),
            cleanup=True
        )

    def _download_month_chunk(self, year_month: tuple) -> Optional[Path]:
        """Download a single month chunk (called by download_chunks_parallel)."""
        year, month = year_month
        cached = self._cached_month_chunk(year, month, self._output_dir)
        if cached is not None:
            return cached
        cached = self._month_from_merged_record(year, month, self._output_dir)
        if cached is not None:
            return cached
        return self._download_and_process_month(year, month, self._output_dir)

    def _month_from_merged_record(self, year: int, month: int, output_dir: Path) -> Optional[Path]:
        """Reuse a complete month when extending an existing domain record.

        Successful downloads remove temporary monthly chunks. Recover them
        from the merged record only when all requested hours and the exact
        requested CDS grid are present; incomplete or different-area records
        must go back to CDS.
        """
        start = max(pd.Timestamp(year=year, month=month, day=1), self.start_date)
        end = min(start.normalize().replace(day=1) + pd.offsets.MonthBegin(1)
                  - pd.Timedelta(hours=1), self.end_date)
        expected = pd.date_range(start, end, freq='h')
        north, west, south, east = self.bbox_to_cds_area(snap_resolution=ERA5_GRID_RESOLUTION_DEG)
        import numpy as np
        expected_lat = np.arange(south, north + 0.01, ERA5_GRID_RESOLUTION_DEG)
        expected_lon = np.arange(west, east + 0.01, ERA5_GRID_RESOLUTION_DEG)
        required = {'air_temperature', 'surface_air_pressure', 'wind_speed',
                    'specific_humidity', 'precipitation_flux',
                    'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux'}
        for path in sorted(output_dir.glob(f'domain_{self.domain_name}_ERA5_CDS_*.nc')):
            try:
                with xr.open_dataset(path) as ds:
                    if not required.issubset(ds.data_vars) or not {'time', 'latitude', 'longitude'}.issubset(ds.coords):
                        continue
                    lat, lon = np.sort(ds.latitude.values), np.sort(ds.longitude.values)
                    if (lat.shape != expected_lat.shape or lon.shape != expected_lon.shape
                            or not np.allclose(lat, expected_lat) or not np.allclose(lon, expected_lon)):
                        continue
                    subset = ds.sel(time=slice(start, end))
                    if not pd.DatetimeIndex(subset.time.values).equals(expected):
                        continue
                    subset = subset.load()
                chunk = output_dir / f'{self.domain_name}_era5_cds_processed_{year}{month:02d}_temp.nc'
                subset.to_netcdf(chunk)
                self.logger.info(f'Reusing {year}-{month:02d} from {path.name}')
                return chunk
            except (OSError, ValueError, KeyError):
                self.logger.debug(f'Cannot reuse month from {path.name}')
        return None

    def _cached_month_chunk(self, year: int, month: int, output_dir: Path) -> Optional[Path]:
        """Return an already-downloaded month chunk, or None to fetch it.

        A CDS request spends most of its wall-clock queued server-side (an hour
        is not unusual), so re-requesting months that are already on disk makes
        any interruption cost the whole multi-hour download. The ARCO pathway
        already skips chunks it has; mirror that here, including its validity
        check, so a partially-completed CDS download resumes instead of
        restarting from the first month.
        """
        chunk_file = output_dir / (
            f"{self.domain_name}_era5_cds_processed_{year}{month:02d}_temp.nc"
        )
        if not chunk_file.exists():
            return None
        try:
            with xr.open_dataset(chunk_file) as existing:
                if 'time' in existing.dims and existing.sizes['time'] > 0:
                    self.logger.debug(
                        f"ERA5 CDS chunk {chunk_file.name} already downloaded, skipping")
                    return chunk_file
        except (OSError, ValueError, KeyError):
            self.logger.debug(
                f"Existing chunk {chunk_file.name} is unreadable, re-downloading")
        return None

    def _download_and_process_month(self, year: int, month: int, output_dir: Path) -> Path:
        """Download and process a single month of ERA5 data (executed in thread)."""
        # Create a thread-local CDS client
        try:
            c = cdsapi.Client()
        except (
            ImportError,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            AttributeError,
            PermissionError,
        ) as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            self.logger.error("Ensure ~/.cdsapirc exists or CDSAPI_URL/CDSAPI_KEY env vars are set.")
            raise

        self.logger.info(f"Downloading ERA5 for {year}-{month:02d}...")

        # Build temporal parameters for this month
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)

        # Restrict to actual requested date range
        month_start = max(month_start, self.start_date)
        month_end = min(month_end, self.end_date)

        dates = pd.date_range(month_start, month_end, freq='h')
        days = sorted(list(set([f"{d.day:02d}" for d in dates])))
        times = sorted(list(set([f"{d.hour:02d}:00" for d in dates])))

        # Bounding box for CDS using mixin method. Snap to ERA5's 0.25-deg
        # grid so point/sub-cell domains still enclose at least one node —
        # a raw sub-cell rectangle makes MARS reject the crop.
        area = self.bbox_to_cds_area(snap_resolution=ERA5_GRID_RESOLUTION_DEG)

        # Temp files for this month (analysis + forecast, like CARRA/CERRA)
        analysis_file = output_dir / f"{self.domain_name}_era5_analysis_{year}{month:02d}_temp.nc"
        forecast_file = output_dir / f"{self.domain_name}_era5_forecast_{year}{month:02d}_temp.nc"

        try:
            # Request 1: Analysis variables (instantaneous)
            analysis_vars = [
                '2m_temperature',
                'surface_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '2m_dewpoint_temperature'
            ]

            analysis_request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': analysis_vars,
                'year': [str(year)],
                'month': [f"{month:02d}"],
                'day': days,
                'time': times,
                'area': area,
            }

            # Request 2: Accumulated variables (precipitation, radiation)
            # ERA5 has these in 'reanalysis' product, splitting to reduce request size
            forecast_vars = [
                'total_precipitation',
                'surface_solar_radiation_downwards',
                'surface_thermal_radiation_downwards'
            ]

            forecast_request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': forecast_vars,
                'year': [str(year)],
                'month': [f"{month:02d}"],
                'day': days,
                'time': times,
                'area': area,
            }

            # Download both products using mixin's retry logic
            self.logger.debug(f"Downloading ERA5 analysis data for {year}-{month:02d}")
            self.execute_with_retry(
                lambda: c.retrieve('reanalysis-era5-single-levels', analysis_request, str(analysis_file)),
                max_retries=3,
                base_delay=60,
                retry_condition=self.is_retryable_cds_error
            )

            self.logger.debug(f"Downloading ERA5 forecast data for {year}-{month:02d}")
            self.execute_with_retry(
                lambda: c.retrieve('reanalysis-era5-single-levels', forecast_request, str(forecast_file)),
                max_retries=3,
                base_delay=60,
                retry_condition=self.is_retryable_cds_error
            )

            # Process and merge datasets (like CARRA/CERRA)
            ds_chunk = self._process_and_merge_datasets(analysis_file, forecast_file)

            # Save chunk to disk (with HPC filesystem fallback)
            from .era5 import _safe_to_netcdf
            chunk_file = output_dir / f"{self.domain_name}_era5_cds_processed_{year}{month:02d}_temp.nc"
            _safe_to_netcdf(ds_chunk, chunk_file, logger=self.logger)

            self.logger.debug(f"Processed ERA5 chunk for {year}-{month:02d}")
            return chunk_file

        finally:
            # Cleanup raw downloads for this month
            if analysis_file.exists():
                analysis_file.unlink()
            if forecast_file.exists():
                forecast_file.unlink()

    def _process_and_merge_datasets(self, analysis_file: Path, forecast_file: Path) -> xr.Dataset:
        """Process and merge ERA5 analysis and forecast files (similar to CARRA/CERRA)."""
        with xr.open_dataset(analysis_file, engine='netcdf4') as dsa, \
             xr.open_dataset(forecast_file, engine='netcdf4') as dsf:

            # Handle dimension standardization
            if 'valid_time' in dsa.dims:
                dsa = dsa.rename({'valid_time': 'time'})
            if 'valid_time' in dsf.dims:
                dsf = dsf.rename({'valid_time': 'time'})

            # Handle expver dimension if present
            for ds_name, ds in [('analysis', dsa), ('forecast', dsf)]:
                if 'expver' in ds.dims:
                    if ds.sizes['expver'] > 1:
                        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
                    else:
                        ds = ds.isel(expver=0)
                    if ds_name == 'analysis':
                        dsa = ds
                    else:
                        dsf = ds

            # Handle ensemble members if present (forecast file)
            if 'number' in dsf.dims:
                self.logger.debug("Ensemble data detected. Selecting first member.")
                dsf = dsf.isel(number=0)

            # Sort by time
            dsa = dsa.sortby('time')
            dsf = dsf.sortby('time')

            self.logger.debug(f"Analysis variables: {list(dsa.data_vars)}")
            self.logger.debug(f"Forecast variables: {list(dsf.data_vars)}")

            # Merge analysis and forecast (inner join on time)
            dsm = xr.merge([dsa, dsf], join='inner')
            self.logger.debug(f"Merged variables: {list(dsm.data_vars)}")

            # Now process variables (rename, convert units, derive, etc.)
            dsm = era5_to_summa_schema(dsm, source='cds', logger=self.logger)

            return dsm.load()
