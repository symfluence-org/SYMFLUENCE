# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MERRA-2 Reanalysis Acquisition Handler

Cloud-based acquisition of NASA MERRA-2 (Modern-Era Retrospective analysis for
Research and Applications, Version 2) reanalysis data via OPeNDAP.

MERRA-2 Overview:
    Data Type: Global atmospheric reanalysis
    Resolution: 0.5deg x 0.625deg (~50km), hourly
    Coverage: Global, 1980-present
    Variables: Temperature, pressure, humidity, wind, radiation, precipitation
    Source: NASA GMAO / GES DISC

Collections:
    M2T1NXSLV: Single-level diagnostics (T2M, PS, QV2M, winds)
    M2T1NXRAD: Radiation diagnostics (SWGDN, LWGAB)
    M2T1NXFLX: Surface flux diagnostics (PRECTOT, PRECSNO)

Data Access:
    OPeNDAP: https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/
    Requires NASA Earthdata authentication (via earthaccess or ~/.netrc)

Stream Mapping:
    100: 1980-1991
    200: 1992-2000
    300: 2001-2010
    400: 2011-present

References:
    Gelaro, R., et al. (2017). The Modern-Era Retrospective Analysis for Research
    and Applications, Version 2 (MERRA-2). J. Climate, 30, 5419-5454.
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

# OPeNDAP base URL for MERRA-2 at GES DISC
_OPENDAP_BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2"

# Collection metadata: (collection_id, short_name, variables)
_COLLECTIONS = {
    'slv': {
        'collection': 'M2T1NXSLV.5.12.4',
        'short': 'tavg1_2d_slv_Nx',
        'variables': ['T2M', 'PS', 'QV2M', 'U2M', 'V2M', 'U10M', 'V10M'],
    },
    'rad': {
        'collection': 'M2T1NXRAD.5.12.4',
        'short': 'tavg1_2d_rad_Nx',
        'variables': ['SWGDN', 'LWGAB'],
    },
    'flx': {
        'collection': 'M2T1NXFLX.5.12.4',
        'short': 'tavg1_2d_flx_Nx',
        'variables': ['PRECTOT', 'PRECTOTCORR', 'PRECSNO'],
    },
}

# Stream number by year range
_STREAM_MAP = [
    (1980, 1991, 100),
    (1992, 2000, 200),
    (2001, 2010, 300),
    (2011, 9999, 400),
]

# MERRA-2 grid parameters
_LAT_MIN, _LAT_STEP, _NLAT = -90.0, 0.5, 361      # -90 to 90 by 0.5
_LON_MIN, _LON_STEP, _NLON = -180.0, 0.625, 576    # -180 to 179.375 by 0.625
_NTIME = 24  # hourly data


def _get_stream(year: int) -> int:
    """Return MERRA-2 stream number for a given year."""
    for y_start, y_end, stream in _STREAM_MAP:
        if y_start <= year <= y_end:
            return stream
    return 400


def _build_opendap_url(collection: str, short_name: str, date: pd.Timestamp) -> str:
    """Build OPeNDAP URL for a single MERRA-2 daily file."""
    stream = _get_stream(date.year)
    ymd = date.strftime('%Y%m%d')
    ym = date.strftime('%Y/%m')
    return (
        f"{_OPENDAP_BASE}/{collection}/{ym}/"
        f"MERRA2_{stream}.{short_name}.{ymd}.nc4"
    )


def _compute_grid_indices(bbox):
    """Compute MERRA-2 grid indices for a bounding box.

    Returns (lat_start, lat_end), (lon_start, lon_end) as integer indices
    into the MERRA-2 0.5x0.625 degree grid.
    """
    lat_min = min(bbox['lat_min'], bbox['lat_max'])
    lat_max = max(bbox['lat_min'], bbox['lat_max'])
    lon_min = min(bbox['lon_min'], bbox['lon_max'])
    lon_max = max(bbox['lon_min'], bbox['lon_max'])

    lat_start = max(0, int(math.floor((lat_min - _LAT_MIN) / _LAT_STEP)))
    lat_end = min(_NLAT - 1, int(math.ceil((lat_max - _LAT_MIN) / _LAT_STEP)))
    lon_start = max(0, int(math.floor((lon_min - _LON_MIN) / _LON_STEP)))
    lon_end = min(_NLON - 1, int(math.ceil((lon_max - _LON_MIN) / _LON_STEP)))

    return (lat_start, lat_end), (lon_start, lon_end)


def _build_subset_url(base_url, variables, lat_indices, lon_indices):
    """Build OPeNDAP subset URL with .nc4 extension for HTTPS download.

    Hyrax (GES DISC's OPeNDAP server) serves subsetted NetCDF4 files when
    the URL has a .nc4 extension appended and DAP2 constraint expressions.
    """
    lat_s, lat_e = lat_indices
    lon_s, lon_e = lon_indices

    # Variable constraints: VAR[time][lat][lon]
    constraints = []
    for var in variables:
        constraints.append(f"{var}[0:{_NTIME - 1}][{lat_s}:{lat_e}][{lon_s}:{lon_e}]")

    # Coordinate variables
    constraints.append(f"time[0:{_NTIME - 1}]")
    constraints.append(f"lat[{lat_s}:{lat_e}]")
    constraints.append(f"lon[{lon_s}:{lon_e}]")

    return f"{base_url}.nc4?{','.join(constraints)}"


@R.acquisition_handlers.add('MERRA2')
class MERRA2Acquirer(
    BaseAcquisitionHandler, RetryMixin, ChunkedDownloadMixin, SpatialSubsetMixin
):
    """MERRA-2 reanalysis acquisition via OPeNDAP with Earthdata authentication.

    Downloads hourly MERRA-2 forcing data from NASA GES DISC using OPeNDAP
    server-side subsetting over authenticated HTTPS. Three collections are
    merged to produce a complete forcing dataset: single-level diagnostics,
    radiation, and surface fluxes.

    Acquisition Strategy:
        1. Authenticate via earthaccess (or ~/.netrc fallback)
        2. Compute MERRA-2 grid indices for the domain bounding box
        3. Generate monthly temporal chunks
        4. For each day, build OPeNDAP subset URLs (server-side spatial crop)
        5. Download subsetted NetCDF4 files via authenticated HTTPS
        6. Merge daily files into monthly chunks, then into final output

    Configuration:
        MERRA2_VARIABLES: List of variables to download (default: all)
        MERRA2_COLLECTIONS: List of collection keys to use (default: all)

    Output:
        NetCDF file: domain_{domain_name}_merra2_{start}_{end}.nc
        Variables: T2M, PS, QV2M, U2M, V2M, U10M, V10M, SWGDN, LWGAB,
                   PRECTOT, PRECTOTCORR, PRECSNO

    References:
        Gelaro et al. (2017). MERRA-2. J. Climate, 30, 5419-5454.
    """

    def _get_authenticated_session(self) -> requests.Session:
        """HTTPS session authenticated for NASA Earthdata GES DISC.

        Bearer token first, then username/password (config, env, .netrc,
        earthaccess login); raises ``EarthdataAuthError`` if neither exists.
        """
        return self._earthdata_auth().session(
            origin_auth=False, required=True, purpose="GES DISC (MERRA-2)",
        )

    def _download_subset_file(self, session, subset_url):
        """Download a subsetted OPeNDAP NetCDF4 file via HTTPS.

        Returns an xarray Dataset loaded into memory, or None on failure.
        """
        def _fetch():
            resp = session.get(subset_url, timeout=120)
            # Detect GES DISC authorization redirect (returns HTML instead of data)
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

        # Write to temp file, open with xarray, load into memory
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
        forcing_dir = self._attribute_dir("forcing")
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        out_path = forcing_dir / f"domain_{self.domain_name}_merra2_{start_str}_{end_str}.nc"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Acquiring MERRA-2 reanalysis for bbox: {self.bbox}")

        # Authenticate
        session = self._get_authenticated_session()

        # Determine which collections and variables to download
        collection_keys = self._get_config_value(lambda: None, default=list(_COLLECTIONS.keys()), dict_key='MERRA2_COLLECTIONS')
        custom_vars = self._get_config_value(lambda: None, default=None, dict_key='MERRA2_VARIABLES')

        active_collections = {}
        for key in collection_keys:
            if key not in _COLLECTIONS:
                self.logger.warning(f"Unknown MERRA-2 collection key: {key}")
                continue
            info = _COLLECTIONS[key].copy()
            if custom_vars:
                info['variables'] = [v for v in info['variables'] if v in custom_vars]
                if not info['variables']:
                    continue
            active_collections[key] = info

        if not active_collections:
            raise ValueError("No valid MERRA-2 collections/variables configured")

        # Compute grid indices for server-side spatial subsetting
        lat_indices, lon_indices = _compute_grid_indices(self.bbox)
        self.logger.info(
            f"MERRA-2 grid subset: lat[{lat_indices[0]}:{lat_indices[1]}], "
            f"lon[{lon_indices[0]}:{lon_indices[1]}]"
        )

        # Generate monthly chunks
        chunks = self.generate_temporal_chunks(self.start_date, self.end_date, freq='MS')
        self.logger.info(f"Processing {len(chunks)} monthly chunks across {len(active_collections)} collections")

        def process_month(chunk):
            chunk_start, chunk_end = chunk
            month_str = chunk_start.strftime('%Y%m')
            chunk_path = forcing_dir / f"merra2_chunk_{month_str}.nc"

            if chunk_path.exists():
                self.logger.info(f"Using cached chunk: {month_str}")
                return chunk_path

            dates = pd.date_range(chunk_start, chunk_end, freq='D')
            monthly_datasets = []

            for date in dates:
                day_datasets = []
                for key, info in active_collections.items():
                    base_url = _build_opendap_url(info['collection'], info['short'], date)
                    subset_url = _build_subset_url(
                        base_url, info['variables'], lat_indices, lon_indices
                    )
                    try:
                        ds = self._download_subset_file(session, subset_url)
                        if ds is not None:
                            day_datasets.append(ds)
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.warning(
                            f"Failed to download MERRA-2 {key} for {date.strftime('%Y-%m-%d')}: {e}"
                        , exc_info=True)
                        continue

                if day_datasets:
                    merged_day = xr.merge(day_datasets)
                    monthly_datasets.append(merged_day)

            if not monthly_datasets:
                self.logger.warning(f"No data retrieved for month {month_str}")
                return None

            month_ds = xr.concat(monthly_datasets, dim='time')
            encoding = self.get_netcdf_encoding(month_ds, compression=True, complevel=1)
            month_ds.to_netcdf(chunk_path, encoding=encoding)
            self.logger.info(f"Saved MERRA-2 chunk: {month_str}")
            return chunk_path

        # Process chunks (serial to respect Earthdata rate limits)
        chunk_files = []
        for chunk in chunks:
            result = process_month(chunk)
            if result:
                chunk_files.append(result)

        if not chunk_files:
            raise RuntimeError("No MERRA-2 data could be downloaded for the specified period")

        # Merge all monthly chunks
        final_path = self.merge_netcdf_chunks(
            chunk_files, out_path,
            time_slice=(self.start_date, self.end_date),
            cleanup=True,
        )

        self.logger.info(f"MERRA-2 acquisition complete: {final_path}")
        return final_path
