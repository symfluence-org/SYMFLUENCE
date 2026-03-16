# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NOAA AORC atmospheric data acquisition from cloud storage.

Provides automated download and processing of Analysis of Record for Calibration
(AORC) forcing data with bounding box subsetting and multi-year support.

Two cloud sources are supported:

1. **Lat-lon gridded AORC v1.1** (primary)
   - S3 bucket: ``noaa-nws-aorc-v1-1-1km``
   - One Zarr archive per year with latitude/longitude coordinates
   - Known issue: NaN cells outside CONUS coverage in the rectangular grid

2. **NWM retrospective v3.0 forcing** (fallback)
   - S3 bucket: ``noaa-nwm-retrospective-3-0-pds``
   - Eight per-variable Zarr stores in Lambert Conformal Conic projection
   - No NaN values (land-only grid)
   - Coverage: Feb 1979 – Jan 2023
"""

from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

# NWM Lambert Conformal Conic projection string (from Zarr metadata)
_NWM_PROJ4 = (
    '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 '
    '+lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 '
    '+x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext +no_defs'
)

# NWM variable Zarr stores → AORC-compatible variable names
_NWM_VAR_MAP = {
    't2d':    'TMP_2maboveground',
    'q2d':    'SPFH_2maboveground',
    'psfc':   'PRES_surface',
    'swdown': 'DSWRF_surface',
    'lwdown': 'DLWRF_surface',
    'precip': 'APCP_surface',
    'u2d':    'UGRD_10maboveground',
    'v2d':    'VGRD_10maboveground',
}

# Uppercase data-var names inside each Zarr store
_NWM_ZARR_VARNAMES = {
    't2d': 'T2D', 'q2d': 'Q2D', 'psfc': 'PSFC',
    'swdown': 'SWDOWN', 'lwdown': 'LWDOWN', 'precip': 'RAINRATE',
    'u2d': 'U2D', 'v2d': 'V2D',
}

# Latest year available in NWM retrospective v3.0 (Feb 1979 – Jan 2023)
_NWM_MAX_YEAR = 2023


@AcquisitionRegistry.register('AORC')
class AORCAcquirer(BaseAcquisitionHandler):
    """
    Download and process NOAA AORC forcing data from cloud storage.

    Tries the lat-lon gridded source first; falls back to NWM projected
    Zarr if the primary source fails (e.g. year not available, S3 error).
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download AORC data, trying lat-lon grid first, NWM projected as fallback.

        Args:
            output_dir: Directory to save downloaded NetCDF file

        Returns:
            Path to downloaded NetCDF file
        """
        try:
            return self._download_latlon_grid(output_dir)
        except Exception as primary_err:  # noqa: BLE001
            self.logger.warning(
                f"Primary AORC source (lat-lon grid) failed: {primary_err}. "
                f"Falling back to NWM retrospective v3.0 projected Zarr."
            )
            try:
                return self._download_nwm_projected(output_dir)
            except Exception as fallback_err:  # noqa: BLE001
                self.logger.error(
                    f"NWM fallback also failed: {fallback_err}. "
                    f"Original error from primary source: {primary_err}"
                )
                raise primary_err from fallback_err

    # ------------------------------------------------------------------
    # Primary source: lat-lon gridded AORC v1.1
    # ------------------------------------------------------------------

    def _download_latlon_grid(self, output_dir: Path) -> Path:
        """Download from the lat-lon gridded AORC v1.1 Zarr archives."""
        self.logger.info("Downloading AORC data from lat-lon grid source (noaa-nws-aorc-v1-1-1km)")
        fs = s3fs.S3FileSystem(anon=True)
        years = range(self.start_date.year, self.end_date.year + 1)
        datasets = []
        for year in years:
            try:
                store = s3fs.S3Map(f'noaa-nws-aorc-v1-1-1km/{year}.zarr', s3=fs)
                ds = xr.open_zarr(store)
                lon1, lon2 = sorted([self.bbox['lon_min'], self.bbox['lon_max']])
                # Convert to 0-360 if dataset uses that convention
                if float(ds['longitude'].max()) > 180.0:
                    lon_min, lon_max = (lon1 + 360.0) % 360.0, (lon2 + 360.0) % 360.0
                else:
                    lon_min, lon_max = lon1, lon2

                # Check if this is a point-scale domain (bbox smaller than grid resolution)
                bbox_lat_range = abs(self.bbox['lat_max'] - self.bbox['lat_min'])
                bbox_lon_range = abs(lon_max - lon_min)
                # AORC resolution is ~0.008-0.01 degrees; threshold at 0.01 degrees
                is_point_scale = (bbox_lat_range < 0.01) or (bbox_lon_range < 0.01)

                if is_point_scale:
                    lat_center = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
                    lon_center = (lon_min + lon_max) / 2
                    self.logger.info(f"Point-scale domain detected (bbox: {bbox_lat_range:.4f}° x {bbox_lon_range:.4f}°). Using nearest-neighbor selection at ({lat_center:.4f}, {lon_center:.4f})")
                    lat_idx = abs(ds['latitude'] - lat_center).argmin().values
                    lon_idx = abs(ds['longitude'] - lon_center).argmin().values
                    ds_subset = ds.isel(latitude=slice(lat_idx, lat_idx+1), longitude=slice(lon_idx, lon_idx+1))
                    self.logger.info(f"Selected nearest cell: lat={ds_subset.latitude.values[0]:.4f}, lon={ds_subset.longitude.values[0]:.4f}")
                else:
                    self.logger.info(f"Watershed-scale domain detected (bbox: {bbox_lat_range:.4f}° x {bbox_lon_range:.4f}°). Using slice-based subsetting")
                    ds_subset = ds.sel(latitude=slice(self.bbox['lat_min'], self.bbox['lat_max']), longitude=slice(lon_min, lon_max))

                ds_subset = ds_subset.sel(time=slice(max(self.start_date, pd.Timestamp(f'{year}-01-01')), min(self.end_date, pd.Timestamp(f'{year}-12-31 23:59:59'))))
                if len(ds_subset.time) > 0: datasets.append(ds_subset)
            except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
                self.logger.error(f"Error processing year {year}: {e}")
                raise
        if not datasets: raise ValueError("No data extracted for the specified time period")
        ds_combined = xr.concat(datasets, dim='time')
        ds_combined.attrs.update({'source': 'NOAA AORC v1.1 (lat-lon grid)', 'bbox': str(self.bbox)})
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_AORC_{self.start_date.year}-{self.end_date.year}.nc"
        for var in ds_combined.data_vars:
            ds_combined[var].encoding.clear()
        for coord in ds_combined.coords:
            ds_combined[coord].encoding.clear()
        ds_combined.load().to_netcdf(output_file, engine="h5netcdf")
        self.logger.info(f"AORC lat-lon grid download complete: {output_file}")
        return output_file

    # ------------------------------------------------------------------
    # Fallback source: NWM retrospective v3.0 projected Zarr
    # ------------------------------------------------------------------

    def _download_nwm_projected(self, output_dir: Path) -> Path:
        """Download from NWM retrospective v3.0 per-variable Zarr stores.

        The NWM data uses Lambert Conformal Conic projection (x/y in metres).
        This method:
        1. Converts the lat/lon bounding box to LCC x/y
        2. Opens all 8 variable Zarr stores and subsets spatially/temporally
        3. Renames variables to AORC-compatible names
        4. Converts precip from rate (mm/s) to hourly accumulation (kg/m²)
        5. Adds latitude/longitude coordinates via inverse projection
        6. Saves as NetCDF with the same filename convention
        """
        import pyproj

        self.logger.info(
            "Downloading AORC data from NWM retrospective v3.0 "
            "(noaa-nwm-retrospective-3-0-pds)"
        )

        if self.end_date.year > _NWM_MAX_YEAR:
            self.logger.warning(
                f"NWM retrospective v3.0 ends in Jan {_NWM_MAX_YEAR}. "
                f"Data after that date will not be available."
            )

        # --- Coordinate transformation ---
        wgs84 = pyproj.Proj('EPSG:4326')
        nwm_proj = pyproj.Proj(_NWM_PROJ4)
        to_lcc = pyproj.Transformer.from_proj(wgs84, nwm_proj, always_xy=True)
        to_wgs = pyproj.Transformer.from_proj(nwm_proj, wgs84, always_xy=True)

        lon_min = min(self.bbox['lon_min'], self.bbox['lon_max'])
        lon_max = max(self.bbox['lon_min'], self.bbox['lon_max'])
        lat_min = min(self.bbox['lat_min'], self.bbox['lat_max'])
        lat_max = max(self.bbox['lat_min'], self.bbox['lat_max'])

        x_min, y_min = to_lcc.transform(lon_min, lat_min)
        x_max, y_max = to_lcc.transform(lon_max, lat_max)
        # Ensure min < max after projection
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        # Add a small buffer (~2 grid cells = 2000m) to avoid edge clipping
        x_min -= 2000
        x_max += 2000
        y_min -= 2000
        y_max += 2000

        self.logger.info(
            f"Bounding box in LCC: x=[{x_min:.0f}, {x_max:.0f}], "
            f"y=[{y_min:.0f}, {y_max:.0f}]"
        )

        # --- Open and subset each variable Zarr store ---
        fs = s3fs.S3FileSystem(anon=True)
        bucket = 'noaa-nwm-retrospective-3-0-pds/CONUS/zarr/forcing'
        var_arrays = {}

        for zarr_name, aorc_name in _NWM_VAR_MAP.items():
            self.logger.info(f"Opening NWM Zarr: {zarr_name}.zarr")
            store = s3fs.S3Map(f'{bucket}/{zarr_name}.zarr', s3=fs)
            ds_var = xr.open_zarr(store)

            # Spatial subset using x/y
            ds_var = ds_var.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

            # Temporal subset
            ds_var = ds_var.sel(
                time=slice(self.start_date, self.end_date)
            )

            if ds_var.sizes['time'] == 0:
                raise ValueError(
                    f"No NWM data for variable {zarr_name} in "
                    f"time range {self.start_date} to {self.end_date}"
                )

            # Extract the data variable (uppercase name inside Zarr)
            internal_name = _NWM_ZARR_VARNAMES[zarr_name]
            data = ds_var[internal_name]

            # Convert precipitation from rate (mm/s) to hourly accumulation (kg/m²)
            # so the AORC preprocessing handler treats it identically to the
            # lat-lon source (which provides accumulated precipitation).
            if zarr_name == 'precip':
                data = data * 3600.0  # mm/s × 3600s = mm = kg/m²
                data.attrs['units'] = 'kg/m^2'
                data.attrs['long_name'] = 'Total precipitation (hourly accumulation)'

            var_arrays[aorc_name] = data
            self.logger.info(
                f"  {zarr_name} -> {aorc_name}: "
                f"shape={dict(data.sizes)}"
            )

        # --- Merge all variables into a single dataset ---
        ds_merged = xr.Dataset(var_arrays)

        # Drop the CRS variable if present (not needed in output)
        if 'crs' in ds_merged:
            ds_merged = ds_merged.drop_vars('crs')

        # --- Add latitude/longitude coordinates via inverse projection ---
        x_vals = ds_merged.x.values
        y_vals = ds_merged.y.values
        xx, yy = np.meshgrid(x_vals, y_vals)
        lons, lats = to_wgs.transform(xx, yy)

        # Store as 1D if the grid is regular in lat/lon (it won't be exactly,
        # but for compatibility with the AORC handler which expects 1D lat/lon,
        # we take the center column for longitude and center row for latitude)
        # Instead, store 2D lat/lon as auxiliary coordinates
        ds_merged = ds_merged.assign_coords({
            'latitude': (('y', 'x'), lats),
            'longitude': (('y', 'x'), lons),
        })

        ds_merged.attrs.update({
            'source': 'NOAA NWM retrospective v3.0 (projected AORC)',
            'bbox': str(self.bbox),
            'projection': _NWM_PROJ4,
        })

        # --- Write to NetCDF ---
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir
            / f"{self.domain_name}_AORC_{self.start_date.year}-{self.end_date.year}.nc"
        )

        # Clear Zarr encoding to avoid HDF5 errors
        for var in ds_merged.data_vars:
            ds_merged[var].encoding.clear()
        for coord in ds_merged.coords:
            ds_merged[coord].encoding.clear()

        ds_merged.load().to_netcdf(output_file, engine="h5netcdf")
        self.logger.info(f"NWM AORC download complete: {output_file}")
        return output_file
