# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GlobSnow SWE Acquisition Handler

Cloud-based acquisition of ESA/FMI GlobSnow Northern Hemisphere Snow Water
Equivalent (SWE) data.

GlobSnow Overview:
    Data Type: Snow Water Equivalent (SWE) estimates
    Resolution: 25km (EASE-Grid 2.0, Northern Hemisphere)
    Coverage: Northern Hemisphere (land areas, excl. glaciers/mountains)
    Variables: SWE (mm), SWE uncertainty, status flags
    Temporal: Daily (v3.0), 1979-2018
    Source: ESA GlobSnow / Finnish Meteorological Institute (FMI)

Data Access:
    Primary: GlobSnow v3.0 archive (HTTP, no auth)
    Fallback: NSIDC (Earthdata auth required)

Projection:
    EASE-Grid 2.0 (Northern Hemisphere, EPSG:6931)
    Reprojected to WGS84 (EPSG:4326) on output

References:
    Luojus, K., et al. (2021). GlobSnow v3.0 Northern Hemisphere snow water
    equivalent dataset. Scientific Data, 8, 163.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import ChunkedDownloadMixin, RetryMixin, SpatialSubsetMixin
from ..utils import create_robust_session, download_file_streaming

# GlobSnow v3.0 archive URL
_V3_BASE_URL = "https://www.globsnow.info/swe/archive_v3.0/L3A_daily_SWE/NetCDF4"

# NSIDC v2.0 fallback
_NSIDC_BASE_URL = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0595_esa_globsnow_swe_v2"
)

# Version-specific URL patterns
_URL_PATTERNS = {
    'v3.0': "{base}/{date}_northern_hemisphere_swe_0.25grid.nc",
    'v2.0': "{base}/{year}/GlobSnow_v2.0_SWE_L3A_{date}.nc",
}


@R.acquisition_handlers.add('GLOBSNOW')
class GlobSnowAcquirer(
    BaseAcquisitionHandler, RetryMixin, ChunkedDownloadMixin, SpatialSubsetMixin
):
    """GlobSnow SWE acquisition from ESA/FMI archive.

    Downloads Northern Hemisphere Snow Water Equivalent data from the GlobSnow
    project. Data is provided on the EASE-Grid 2.0 projection and is
    reprojected to WGS84 for consistency with other datasets.

    Acquisition Strategy:
        1. Generate monthly temporal chunks
        2. For each month, download daily NetCDF files
        3. Reproject from EASE-Grid to WGS84
        4. Subset spatially to domain bounding box
        5. Merge daily files into monthly chunks
        6. Merge monthly chunks into final output

    Configuration:
        GLOBSNOW_VERSION: Data version
            (default: 'v3.0', options: 'v3.0', 'v2.0')
        GLOBSNOW_TEMPORAL_AGG: Temporal aggregation
            (default: 'daily', options: 'daily', 'monthly')

    Output:
        NetCDF file: domain_{domain_name}_globsnow_swe_{start}_{end}.nc
        Variables: swe (mm)

    References:
        Luojus et al. (2021). GlobSnow v3.0. Scientific Data, 8, 163.
    """

    def download(self, output_dir: Path) -> Path:
        snow_dir = self._attribute_dir("snow")
        snow_dir.mkdir(parents=True, exist_ok=True)

        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        out_path = snow_dir / f"domain_{self.domain_name}_globsnow_swe_{start_str}_{end_str}.nc"

        if self._skip_if_exists(out_path):
            return out_path

        version = self._get_config_value(lambda: None, default='v3.0', dict_key='GLOBSNOW_VERSION')
        temporal_agg = self._get_config_value(lambda: None, default='daily', dict_key='GLOBSNOW_TEMPORAL_AGG')

        # Northern Hemisphere check
        if self.bbox['lat_min'] < 0:
            self.logger.warning(
                "GlobSnow covers Northern Hemisphere only. "
                f"Domain extends to {self.bbox['lat_min']}N — southern pixels will be missing."
            )

        self.logger.info(
            f"Acquiring GlobSnow SWE ({version}) for bbox: {self.bbox}, "
            f"period: {start_str}-{end_str}"
        )

        # Set up session (with auth for NSIDC v2.0 fallback)
        session = create_robust_session(max_retries=5, backoff_factor=2.0)
        use_nsidc = False

        if version == 'v2.0':
            username, password = self._earthdata_auth().require_credentials(
                "GlobSnow v2.0 (NSIDC; or set GLOBSNOW_VERSION: v3.0)"
            )
            session.auth = (username, password)
            use_nsidc = True

        # Generate monthly chunks
        chunks = self.generate_temporal_chunks(self.start_date, self.end_date, freq='MS')
        cache_dir = snow_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def process_month(chunk):
            chunk_start, chunk_end = chunk
            month_str = chunk_start.strftime('%Y%m')
            chunk_path = cache_dir / f"globsnow_chunk_{month_str}.nc"

            if chunk_path.exists():
                # Validate cached chunk has actual data (not a corrupt leftover)
                try:
                    with xr.open_dataset(chunk_path) as _check:
                        if 'swe' in _check.data_vars and 'time' in _check.coords:
                            self.logger.info(f"Using cached chunk: {month_str}")
                            return chunk_path
                    self.logger.warning(f"Corrupt chunk {month_str}, reprocessing")
                    chunk_path.unlink()
                except Exception:  # noqa: BLE001 — corrupt cache validation
                    self.logger.warning(f"Unreadable chunk {month_str}, reprocessing")
                    chunk_path.unlink(missing_ok=True)

            dates = pd.date_range(chunk_start, chunk_end, freq='D')
            daily_datasets = []

            for date in dates:
                nc_path = self._download_daily_file(
                    session, date, version, use_nsidc, cache_dir
                )
                if nc_path is None:
                    continue

                try:
                    ds = xr.open_dataset(nc_path)

                    # Reproject EASE-Grid to WGS84 if needed
                    ds_wgs84 = self._reproject_ease_to_wgs84(ds)

                    # Drop scalar variables (causes chunking errors in encoding)
                    ds_wgs84 = ds_wgs84.drop_vars(
                        [v for v in ds_wgs84.data_vars if ds_wgs84[v].ndim == 0],
                        errors='ignore'
                    )
                    # Drop pre-existing time coord/dim to avoid expand_dims conflict
                    if 'time' in ds_wgs84.coords or 'time' in ds_wgs84.dims:
                        ds_wgs84 = ds_wgs84.drop_vars('time', errors='ignore')

                    # Spatial subset — detect grid structure
                    ds_sub = self._spatial_subset(ds_wgs84)

                    # Add time coordinate
                    ds_sub = ds_sub.expand_dims(
                        time=[pd.Timestamp(date)]
                    )

                    daily_datasets.append(ds_sub)
                    ds.close()
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(
                        f"Error processing GlobSnow {date.strftime('%Y-%m-%d')}: {e}\n"
                        f"{traceback.format_exc()}"
                    , exc_info=True)
                    continue

            if not daily_datasets:
                self.logger.warning(f"No GlobSnow data for month {month_str}")
                return None

            month_ds = xr.concat(daily_datasets, dim='time')

            # Monthly aggregation if requested
            if temporal_agg == 'monthly':
                month_ds = month_ds.resample(time='MS').mean()

            # Write chunk — use simple encoding to avoid chunksizes issues
            comp = {'zlib': True, 'complevel': 1}
            encoding = {
                var: comp for var in month_ds.data_vars
                if month_ds[var].ndim > 0
            }
            month_ds.to_netcdf(chunk_path, encoding=encoding)
            self.logger.info(f"Saved GlobSnow chunk: {month_str}")
            return chunk_path

        # Process months
        chunk_files = []
        for chunk in chunks:
            result = process_month(chunk)
            if result:
                chunk_files.append(result)

        if not chunk_files:
            raise RuntimeError("No GlobSnow SWE data could be downloaded")

        # Merge all chunks
        final_path = self.merge_netcdf_chunks(
            chunk_files, out_path,
            time_slice=(self.start_date, self.end_date),
            cleanup=True,
        )

        self.logger.info(f"GlobSnow acquisition complete: {final_path}")
        return final_path

    def _spatial_subset(self, ds: xr.Dataset) -> xr.Dataset:
        """Spatially subset GlobSnow data, handling both regular and EASE grids.

        Regular lat/lon grid (0.25grid files): use standard sel() slicing.
        EASE-Grid with 2D lat/lon: use numpy boolean mask approach.
        """
        bbox = self.bbox
        lat_min = min(bbox['lat_min'], bbox['lat_max'])
        lat_max = max(bbox['lat_min'], bbox['lat_max'])
        lon_min = min(bbox['lon_min'], bbox['lon_max'])
        lon_max = max(bbox['lon_min'], bbox['lon_max'])

        # Check if lat/lon are 1D dimension coordinates (regular grid)
        has_1d_lat = 'lat' in ds.coords and ds['lat'].ndim == 1
        has_1d_lon = 'lon' in ds.coords and ds['lon'].ndim == 1

        if has_1d_lat and has_1d_lon:
            # Regular grid — use efficient sel() slicing
            lat_vals = ds['lat'].values
            if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
                ds = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
            else:
                ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            return ds

        # 2D coordinate grid (EASE-Grid reprojected) — use numpy mask
        if 'lat' in ds.coords and 'lon' in ds.coords and ds['lat'].ndim == 2:
            # Detect actual grid dimension names
            grid_dims = ds['lat'].dims  # e.g., ('y', 'x') or ('rows', 'cols')
            return self.subset_numpy_mask(
                ds, bbox,
                lat_name='lat', lon_name='lon',
                grid_dims=grid_dims
            )

        # Fallback: return as-is with warning
        self.logger.warning(
            f"Cannot determine GlobSnow grid structure for subsetting. "
            f"Coords: {list(ds.coords)}, Dims: {list(ds.dims)}"
        )
        return ds

    def _download_daily_file(
        self, session, date: pd.Timestamp, version: str, use_nsidc: bool,
        cache_dir: Path
    ) -> Path | None:
        """Download a single daily GlobSnow NetCDF file."""
        date_str = date.strftime('%Y%m%d')
        local_path = cache_dir / f"globsnow_{version}_{date_str}.nc"

        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path

        # Build URL based on version
        if version == 'v3.0':
            base = _V3_BASE_URL
            url = _URL_PATTERNS['v3.0'].format(base=base, date=date_str)
        else:
            base = _NSIDC_BASE_URL
            url = _URL_PATTERNS['v2.0'].format(
                base=base, year=date.year, date=date_str
            )

        try:
            self.execute_with_retry(
                lambda u=url, p=local_path: download_file_streaming(
                    u, p, session=session, timeout=300
                ),
                max_retries=3,
                base_delay=5.0,
                backoff_factor=2.0,
            )
            return local_path
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            # Try NSIDC fallback for v3.0 failures
            if version == 'v3.0' and not use_nsidc:
                try:
                    username, password = self._get_earthdata_credentials()
                    if username and password:
                        session.auth = (username, password)
                        fallback_url = _URL_PATTERNS['v2.0'].format(
                            base=_NSIDC_BASE_URL, year=date.year, date=date_str
                        )
                        download_file_streaming(
                            fallback_url, local_path, session=session, timeout=300
                        )
                        return local_path
                except Exception:  # noqa: BLE001 — NSIDC fallback resilience
                    pass

            self.logger.warning(
                f"Failed to download GlobSnow for {date_str}: {e}"
            , exc_info=True)
            return None

    _cached_lats = None
    _cached_lons = None
    _cached_grid_key = None

    def _reproject_ease_to_wgs84(self, ds: xr.Dataset) -> xr.Dataset:
        """Reproject EASE-Grid dataset to WGS84 lat/lon if needed.

        If the dataset already has lat/lon coordinates (some GlobSnow files
        include both projected and geographic coords), use those directly.
        Otherwise, compute the reprojection (cached for repeated calls with
        the same grid).
        """
        # Check if actual geographic lat/lon already present
        has_lat = any(
            n in ds.coords and ds[n].ndim == 1 and float(ds[n].max()) <= 90
            for n in ('lat', 'latitude')
        )
        has_lon = any(
            n in ds.coords and ds[n].ndim == 1 and float(ds[n].max()) <= 360
            for n in ('lon', 'longitude')
        )
        if has_lat and has_lon:
            return ds

        # Try to find projected coordinates
        try:
            from pyproj import Transformer

            # Find x/y coordinates
            x_name = None
            y_name = None
            for name in ds.coords:
                if name.lower() in ('x', 'cols', 'col'):
                    x_name = name
                elif name.lower() in ('y', 'rows', 'row'):
                    y_name = name

            if x_name is None or y_name is None:
                self.logger.warning(
                    "Cannot identify EASE-Grid coordinates; returning dataset as-is"
                )
                return ds

            # Cache the reprojected lat/lon for repeated calls with same grid
            grid_key = (ds.sizes.get(y_name), ds.sizes.get(x_name))
            if self._cached_lats is None or self._cached_grid_key != grid_key:
                self.logger.info(
                    f"Computing EASE-Grid → WGS84 reprojection for {grid_key} grid"
                )
                transformer = Transformer.from_crs(
                    "EPSG:6931", "EPSG:4326", always_xy=True
                )
                x_vals = ds[x_name].values
                y_vals = ds[y_name].values
                xx, yy = np.meshgrid(x_vals, y_vals)
                GlobSnowAcquirer._cached_lons, GlobSnowAcquirer._cached_lats = (
                    transformer.transform(xx, yy)
                )
                GlobSnowAcquirer._cached_grid_key = grid_key

            # Add lat/lon as 2D coordinates
            ds = ds.assign_coords(
                lat=((y_name, x_name), self._cached_lats),
                lon=((y_name, x_name), self._cached_lons),
            )
            return ds

        except ImportError:
            self.logger.warning(
                "pyproj not available for EASE-Grid reprojection. "
                "Install with: pip install pyproj"
            )
            return ds
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"EASE-Grid reprojection failed: {e}", exc_info=True)
            return ds
