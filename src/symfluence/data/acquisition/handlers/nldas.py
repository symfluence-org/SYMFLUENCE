# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NLDAS-2 Forcing Data Acquisition Handler

Cloud-based acquisition of NLDAS-2 (North American Land Data Assimilation System
Phase 2) forcing data via NASA GES DISC OPeNDAP.

NLDAS-2 Overview:
    Data Type: Land surface forcing analysis
    Resolution: 0.125deg (~13 km), hourly
    Coverage: Central North America (25-53N, 125-67W)
    Period: 1979-present
    Variables: Temperature, humidity, pressure, wind, radiation, precipitation
    Source: NASA GSFC / GES DISC

Data Access:
    OPeNDAP: https://hydro1.gesdisc.eosdis.nasa.gov/opendap/NLDAS/
    Requires NASA Earthdata authentication

References:
    Xia, Y., et al. (2012). Continental-scale water and energy flux analysis and
    validation for the North American Land Data Assimilation System project phase 2
    (NLDAS-2). J. Geophys. Res., 117, D03109.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins import ChunkedDownloadMixin, RetryMixin, SpatialSubsetMixin

# OPeNDAP base URL for NLDAS-2 at GES DISC
_OPENDAP_BASE = "https://hydro1.gesdisc.eosdis.nasa.gov/opendap/NLDAS/NLDAS_FORA0125_H.2.0"

# Core forcing variables for hydrological modelling.
# NOTE: These are the NLDAS-2 v2.0 NetCDF variable names (ALMA convention).
# The legacy GRIB-era names (TMP2m, SPFH2m, PRESsfc, ...) do NOT exist in the
# v2.0 NetCDF files served at NLDAS_FORA0125_H.2.0 — requesting them returns
# HTTP 400 from the OPeNDAP server.
_DEFAULT_VARIABLES = [
    'Tair',       # 2-m air temperature (K)
    'Qair',       # 2-m specific humidity (kg/kg)
    'PSurf',      # Surface pressure (Pa)
    'Wind_E',     # 10-m eastward wind (m/s)
    'Wind_N',     # 10-m northward wind (m/s)
    'LWdown',     # Downward longwave radiation (W/m2)
    'SWdown',     # Downward shortwave radiation (W/m2)
    'Rainf',      # Precipitation hourly total (kg/m2)
]

_EXTENDED_VARIABLES = _DEFAULT_VARIABLES + [
    'CAPE',         # Convective available potential energy (J/kg)
    'PotEvap',      # Potential evaporation (kg/m2)
    'CRainf_frac',  # Convective fraction (fraction)
]

# NLDAS-2 grid parameters (0.125-degree CONUS grid)
_LAT_MIN, _LAT_STEP, _NLAT = 25.0625, 0.125, 224   # 25.0625 to 52.9375
_LON_MIN, _LON_STEP, _NLON = -124.9375, 0.125, 464  # -124.9375 to -67.0625

# Coverage bounds (for validation)
_COVERAGE_LAT = (25.0, 53.0)
_COVERAGE_LON = (-125.0, -67.0)


def _build_opendap_url(date: pd.Timestamp, hour: int) -> str:
    """Build OPeNDAP URL for a single NLDAS-2 hourly file."""
    doy = date.day_of_year
    ymd = date.strftime('%Y%m%d')
    return (
        f"{_OPENDAP_BASE}/{date.year}/{doy:03d}/"
        f"NLDAS_FORA0125_H.A{ymd}.{hour:02d}00.020.nc"
    )


def _compute_grid_indices(bbox):
    """Compute NLDAS-2 grid indices for a bounding box.

    Returns (lat_start, lat_end), (lon_start, lon_end) as integer indices
    into the NLDAS-2 0.125-degree grid.
    """
    lat_min = min(bbox['lat_min'], bbox['lat_max'])
    lat_max = max(bbox['lat_min'], bbox['lat_max'])
    lon_min = min(bbox['lon_min'], bbox['lon_max'])
    lon_max = max(bbox['lon_min'], bbox['lon_max'])

    lat_min = max(lat_min, _COVERAGE_LAT[0])
    lat_max = min(lat_max, _COVERAGE_LAT[1])
    lon_min = max(lon_min, _COVERAGE_LON[0])
    lon_max = min(lon_max, _COVERAGE_LON[1])

    lat_start = max(0, int(math.floor((lat_min - _LAT_MIN) / _LAT_STEP)))
    lat_end = min(_NLAT - 1, int(math.ceil((lat_max - _LAT_MIN) / _LAT_STEP)))
    lon_start = max(0, int(math.floor((lon_min - _LON_MIN) / _LON_STEP)))
    lon_end = min(_NLON - 1, int(math.ceil((lon_max - _LON_MIN) / _LON_STEP)))

    return (lat_start, lat_end), (lon_start, lon_end)


def _build_subset_url(base_url, variables, lat_indices, lon_indices):
    """Build OPeNDAP subset URL with .nc4 extension for NetCDF4 download.

    Each NLDAS-2 hourly file has a single timestep, so time index is [0].
    """
    lat_s, lat_e = lat_indices
    lon_s, lon_e = lon_indices

    constraints = []
    for var in variables:
        constraints.append(f"{var}[0][{lat_s}:{lat_e}][{lon_s}:{lon_e}]")

    constraints.append("time[0]")
    constraints.append(f"lat[{lat_s}:{lat_e}]")
    constraints.append(f"lon[{lon_s}:{lon_e}]")

    return f"{base_url}.nc4?{','.join(constraints)}"


@R.acquisition_handlers.add('NLDAS')
@R.acquisition_handlers.add('NLDAS2')
@R.acquisition_handlers.add('NLDAS-2')
class NLDASAcquirer(
    BaseAcquisitionHandler, RetryMixin, ChunkedDownloadMixin, SpatialSubsetMixin
):
    """NLDAS-2 forcing data acquisition via GES DISC OPeNDAP.

    Downloads hourly NLDAS-2 forcing data from NASA GES DISC using OPeNDAP
    server-side subsetting over authenticated HTTPS. Data covers the
    continental US and parts of Canada/Mexico at 0.125-degree resolution.

    Acquisition Strategy:
        1. Authenticate via Earthdata token or ~/.netrc
        2. Compute NLDAS-2 grid indices for the domain bounding box
        3. Generate monthly temporal chunks
        4. For each day, download hourly files via OPeNDAP (server-side spatial crop)
        5. Merge hourly files into daily, then daily into monthly chunks
        6. Merge chunks into final output

    Configuration:
        NLDAS_VARIABLES: List of variables (default: core forcing vars)
        NLDAS_TEMPORAL_STEP_HOURS: Download every Nth hour (1, 3, 6; default: 1)

    Output:
        NetCDF file: domain_{name}_nldas2_{start}_{end}.nc
    """

    def _get_authenticated_session(self) -> requests.Session:
        """HTTPS session authenticated for NASA Earthdata GES DISC.

        Bearer token first, then username/password (config, env, .netrc,
        earthaccess login); raises ``EarthdataAuthError`` if neither exists.
        """
        return self._earthdata_auth().session(
            origin_auth=False, required=True, purpose="GES DISC (NLDAS-2)",
        )

    def _validate_coverage(self):
        """Check that the domain bounding box overlaps NLDAS-2 coverage."""
        if not self.bbox:
            return
        lat_min = min(self.bbox['lat_min'], self.bbox['lat_max'])
        lat_max = max(self.bbox['lat_min'], self.bbox['lat_max'])
        lon_min = min(self.bbox['lon_min'], self.bbox['lon_max'])
        lon_max = max(self.bbox['lon_min'], self.bbox['lon_max'])

        if (lat_max < _COVERAGE_LAT[0] or lat_min > _COVERAGE_LAT[1] or
                lon_max < _COVERAGE_LON[0] or lon_min > _COVERAGE_LON[1]):
            raise ValueError(
                f"Domain bbox ({lat_min}-{lat_max}N, {lon_min}-{lon_max}E) "
                f"is outside NLDAS-2 coverage "
                f"({_COVERAGE_LAT[0]}-{_COVERAGE_LAT[1]}N, "
                f"{_COVERAGE_LON[0]}-{_COVERAGE_LON[1]}E). "
                "NLDAS-2 covers central North America only."
            )

    def _download_subset_file(self, session, subset_url):
        """Download a subsetted OPeNDAP NetCDF4 file via HTTPS.

        Returns an xarray Dataset loaded into memory, or None on failure.
        """
        def _fetch():
            resp = session.get(subset_url, timeout=120)
            content_type = resp.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                if 'authorization' in resp.url.lower() or 'error' in resp.url.lower():
                    raise PermissionError(
                        "NASA GES DISC requires application authorization. "
                        "Go to https://urs.earthdata.nasa.gov → Applications → "
                        "Authorized Apps → Approve More Applications → "
                        "search for 'NASA GESDISC DATA ARCHIVE' and approve it."
                    )
            resp.raise_for_status()
            return resp

        response = self.execute_with_retry(
            _fetch,
            max_retries=3,
            base_delay=30.0,
            backoff_factor=2.0,
            retry_condition=self.is_retryable_http_error,
        )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.nc4', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)
            ds = xr.open_dataset(tmp_path, engine='netcdf4')
            ds = ds.load()
            ds.close()
            return ds
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def download(self, output_dir: Path) -> Path:
        """Download NLDAS-2 forcing data for the configured domain and period."""
        forcing_dir = self._attribute_dir("forcing")
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        out_path = forcing_dir / f"domain_{self.domain_name}_nldas2_{start_str}_{end_str}.nc"

        if self._skip_if_exists(out_path):
            return out_path

        self._validate_coverage()
        self.logger.info(f"Acquiring NLDAS-2 forcing for bbox: {self.bbox}")

        session = self._get_authenticated_session()

        variables = self._get_config_value(
            lambda: None, default=list(_DEFAULT_VARIABLES),
            dict_key='NLDAS_VARIABLES')
        temporal_step = int(self._get_config_value(
            lambda: None, default=1,
            dict_key='NLDAS_TEMPORAL_STEP_HOURS'))
        hours = list(range(0, 24, temporal_step))

        lat_indices, lon_indices = _compute_grid_indices(self.bbox)
        self.logger.info(
            f"NLDAS-2 grid subset: lat[{lat_indices[0]}:{lat_indices[1]}], "
            f"lon[{lon_indices[0]}:{lon_indices[1]}], "
            f"temporal step: {temporal_step}h ({len(hours)} files/day)"
        )

        chunks = self.generate_temporal_chunks(self.start_date, self.end_date, freq='MS')
        self.logger.info(f"Processing {len(chunks)} monthly chunks")

        chunk_files = []
        for chunk in chunks:
            result = self._process_month(session, chunk, variables, hours,
                                         lat_indices, lon_indices, forcing_dir)
            if result:
                chunk_files.append(result)

        if not chunk_files:
            raise RuntimeError("No NLDAS-2 data could be downloaded for the specified period")

        final_path = self.merge_netcdf_chunks(
            chunk_files, out_path,
            time_slice=(self.start_date, self.end_date),
            cleanup=True,
        )

        self.logger.info(f"NLDAS-2 acquisition complete: {final_path}")
        return final_path

    def _process_month(self, session, chunk, variables, hours,
                       lat_indices, lon_indices, forcing_dir):
        """Download and merge one month of NLDAS-2 data."""
        chunk_start, chunk_end = chunk
        month_str = chunk_start.strftime('%Y%m')
        chunk_path = forcing_dir / f"nldas2_chunk_{month_str}.nc"

        if chunk_path.exists():
            self.logger.info(f"Using cached chunk: {month_str}")
            return chunk_path

        dates = pd.date_range(chunk_start, chunk_end, freq='D')
        daily_datasets = []

        for date in dates:
            day_ds = self._download_day(session, date, variables, hours,
                                        lat_indices, lon_indices)
            if day_ds is not None:
                daily_datasets.append(day_ds)

        if not daily_datasets:
            self.logger.warning(f"No data retrieved for month {month_str}")
            return None

        month_ds = xr.concat(daily_datasets, dim='time')
        encoding = self.get_netcdf_encoding(month_ds, compression=True, complevel=1)
        month_ds.to_netcdf(chunk_path, encoding=encoding)
        self.logger.info(f"Saved NLDAS-2 chunk: {month_str}")
        return chunk_path

    def _download_day(self, session, date, variables, hours,
                      lat_indices, lon_indices):
        """Download all hourly files for a single day and merge."""
        hourly_datasets = []

        for hour in hours:
            base_url = _build_opendap_url(date, hour)
            subset_url = _build_subset_url(
                base_url, variables, lat_indices, lon_indices
            )
            try:
                ds = self._download_subset_file(session, subset_url)
                if ds is not None:
                    hourly_datasets.append(ds)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"Failed to download NLDAS-2 {date.strftime('%Y-%m-%d')} {hour:02d}:00: {e}"
                )

        if not hourly_datasets:
            self.logger.warning(f"No data for {date.strftime('%Y-%m-%d')}")
            return None

        return xr.concat(hourly_datasets, dim='time')
