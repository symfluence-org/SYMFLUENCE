# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Acquisition handler for RDRS/CaSR (Canadian Surface Reanalysis) datasets.

Provides cloud-based acquisition for CaSR v3.2 (successor to RDRS v3.1) and
RDRS v2.1 via PAVICS THREDDS OPeNDAP for efficient remote subsetting, with
HTTP fallback to ECCC's GPSC-C server.

Data sources:
  - Primary:  PAVICS THREDDS OPeNDAP (CaSR v3.2, 1980-2024, hourly)
  - Fallback: ECCC GPSC-C tiled per-variable NetCDF files (CaSR v3.2,
              35x35-cell spatial tiles, 4-year time blocks) — only the tiles
              and variables covering the domain are fetched
  - Last resort: ECCC GPSC-C full-domain daily NetCDF files (CaSR v3.2,
              ~700 MB per day; all variables, all of North America)
  - Legacy:   PAVICS THREDDS OPeNDAP (RDRS v2.1, 1980-2018, hourly)
"""
from __future__ import annotations

import concurrent.futures
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..utils import create_robust_session

# PAVICS THREDDS OPeNDAP endpoints
PAVICS_OPENDAP_CASR_V32 = (
    "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/"
    "datasets/reanalyses/1hr_NAM_GovCan_CaSR_v32_198001-202412.ncml"
)
PAVICS_OPENDAP_RDRS_V21 = (
    "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/dodsC/"
    "datasets/reanalyses/1hr_RDRSv2.1_NAM.ncml"
)

# ECCC GPSC-C HTTP endpoints for CaSR v3.2
GPSCC_CASR_V32_BASE_URL = (
    "https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/data/CaSRv3.2/netcdf"
)
GPSCC_CASR_V32_TILE_BASE_URL = (
    "https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/data/CaSRv3.2/netcdf_tile"
)

# Guard rails for the last-resort full-domain daily fallback (_download_http_daily).
# Each daily file covers the whole CaSR domain (measured ~700-730 MB for v3.2), so
# the download volume scales with the date range and NOT with catchment size.
DAILY_FULL_DOMAIN_FILE_BYTES = 700 * 1024 ** 2
# ~2 months. Enough for a short diagnostic fetch; anything longer is a multi-year
# run that must use a server-side-subsetting endpoint instead.
MAX_UNGUARDED_DAILY_FILES = 60

# CaSR v3.2 rotated-pole grid geometry (fixed property of the dataset; read
# from the dataset files themselves). Used only to decide which 35x35-cell
# tiles to download — final cell selection always uses the 2-D lat/lon
# coordinate arrays shipped inside each tile file, so a small drift here is
# absorbed by the tile-selection buffer.
CASR_V32_GRID = {
    'pole_lat': 31.758312454493154,   # rotated_pole:grid_north_pole_latitude
    'pole_lon': 87.59703130293302,    # rotated_pole:grid_north_pole_longitude
    'rlat0': -44.100002288818359,     # rlat of global index 0
    'rlon0': -35.397216796875,        # rlon of global index 0
    'step': 0.09,                     # grid spacing in rotated degrees
    'nlat': 778,
    'nlon': 706,
    'tile': 35,                       # tile edge length in cells
}

# Forcing variables required by SYMFLUENCE (CaSR v3.2 raw archive names,
# without the 'CaSR_v3.2_' prefix). A_* = analysis, P_* = forecast.
DEFAULT_CASR_V32_VARIABLES = (
    'A_PR0_SFC',    # precipitation accumulation over the hour (m)
    'P_TT_1.5m',    # air temperature at 1.5 m (deg C)
    'P_HU_1.5m',    # specific humidity at 1.5 m (kg/kg)
    'P_UVC_10m',    # wind modulus at 10 m (knots)
    'P_P0_SFC',     # surface pressure (mb)
    'P_FB_SFC',     # downward solar flux (W m-2)
    'P_FI_SFC',     # surface incoming infrared flux (W m-2)
)


@R.acquisition_handlers.add('RDRS')
@R.acquisition_handlers.add('RDRS_v3.1')
class RDRSAcquirer(BaseAcquisitionHandler):
    """
    Acquisition handler for CaSR/RDRS reanalysis data.

    RDRS has been renamed to CaSR (Canadian Surface Reanalysis) by ECCC.
    This handler acquires CaSR v3.2 by default (successor to v3.1), falling
    back to RDRS v2.1 if configured. The primary pathway uses PAVICS THREDDS
    OPeNDAP for efficient server-side spatial and temporal subsetting; when
    PAVICS is unreachable it falls back to ECCC's GPSC-C tiled archive,
    downloading only the tiles/variables/period covering the domain.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process RDRS/CaSR data for the configured time period."""
        output_dir.mkdir(parents=True, exist_ok=True)
        final_file = output_dir / f"domain_{self.domain_name}_RDRS_{self.start_date.year}_{self.end_date.year}.nc"

        if final_file.exists() and not self._get_config_value(
            lambda: self.config.data.force_download,
            dict_key='FORCE_DOWNLOAD',
        ):
            return final_file

        # Acquisition pathway. The default ("auto") tries OPeNDAP and falls back
        # to the tiled HTTP archive on any error — resilient, but the choice is
        # then decided by runtime network conditions, so two machines can end up
        # on different endpoints. OPeNDAP and the tiled archive do not always
        # return identical data (a large OPeNDAP request can silently truncate;
        # see _validate_acquired_variables), so a run that lands on a different
        # endpoint gets slightly different forcing and a different result. Pin
        # the method to make acquisition reproducible across machines.
        method = str(self._get_config_value(
            lambda: None, default='auto', dict_key='RDRS_ACQUISITION_METHOD')).lower()

        if method in ('opendap', 'pavics'):
            return self._download_opendap(final_file)
        if method in ('tiled', 'http', 'gpsc'):
            return self._download_http(output_dir, final_file)

        # auto: OPeNDAP first, tiled archive on failure
        try:
            return self._download_opendap(final_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"OPeNDAP pathway failed: {e}. Falling back to HTTP.", exc_info=True)
            return self._download_http(output_dir, final_file)

    # ------------------------------------------------------------------
    # Shared spatial subsetting
    # ------------------------------------------------------------------

    def _subset_bbox(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Spatially subset a CaSR/RDRS dataset (rotated-pole grid with 2-D
        lat/lon auxiliary coordinates) to the configured bounding box.

        Longitudes are normalized to [-180, 180) before masking — the GPSC-C
        archive stores them in [0, 360), which would otherwise never match a
        western-hemisphere bbox. Falls back to the nearest grid cell for
        bounding boxes smaller than the grid resolution (point-scale domains).
        """
        if not self.bbox:
            return ds

        lat_vals = np.asarray(ds.lat.values)
        lon_vals = ((np.asarray(ds.lon.values) + 180.0) % 360.0) - 180.0

        mask = (
            (lat_vals >= self.bbox['lat_min']) & (lat_vals <= self.bbox['lat_max']) &
            (lon_vals >= self.bbox['lon_min']) & (lon_vals <= self.bbox['lon_max'])
        )
        y_indices, x_indices = np.where(mask)

        if len(y_indices) > 0 and len(x_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
        else:
            # Bounding box smaller than grid resolution (e.g. point-scale domain).
            # Fall back to nearest grid cell.
            center_lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
            center_lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
            dist = np.sqrt((lat_vals - center_lat)**2 + (lon_vals - center_lon)**2)
            y_near, x_near = np.unravel_index(dist.argmin(), dist.shape)
            self.logger.info(
                f"Bounding box smaller than grid resolution, using nearest grid cell "
                f"at ({lat_vals[y_near, x_near]:.4f}, {lon_vals[y_near, x_near]:.4f})"
            )
            y_min = y_max = y_near
            x_min = x_max = x_near

        # Add buffer for safety
        y_min = max(0, y_min - 2)
        y_max = min(ds.sizes['rlat'] - 1, y_max + 2)
        x_min = max(0, x_min - 2)
        x_max = min(ds.sizes['rlon'] - 1, x_max + 2)

        ds = ds.isel(rlat=slice(y_min, y_max + 1), rlon=slice(x_min, x_max + 1))

        # Persist the normalized longitudes so downstream consumers (grid
        # shapefile creation, remapping) see standard [-180, 180) coordinates.
        ds = ds.assign_coords(lon=(ds.lon.dims, ((ds.lon.values + 180.0) % 360.0) - 180.0, ds.lon.attrs))

        self.logger.info(f"Spatially subsetted to {ds.sizes['rlat']}x{ds.sizes['rlon']} grid")
        return ds

    # ------------------------------------------------------------------
    # OPeNDAP pathway (PAVICS)
    # ------------------------------------------------------------------

    def _download_opendap(self, final_file: Path) -> Path:
        """Download CaSR/RDRS data via PAVICS THREDDS OPeNDAP."""
        version = self._get_config_value(lambda: None, default='v3.2', dict_key='RDRS_VERSION')

        if version == 'v2.1':
            opendap_url = PAVICS_OPENDAP_RDRS_V21
            self.logger.info("Accessing RDRS v2.1 via PAVICS THREDDS OPeNDAP")
        else:
            opendap_url = PAVICS_OPENDAP_CASR_V32
            self.logger.info("Accessing CaSR v3.2 (successor to RDRS v3.1) via PAVICS THREDDS OPeNDAP")

        # Allow user override of OPeNDAP URL
        opendap_url = self._get_config_value(lambda: None, default=opendap_url, dict_key='RDRS_OPENDAP_URL')

        self.logger.info(f"Opening OPeNDAP dataset: {opendap_url}")
        ds = xr.open_dataset(opendap_url)

        # Spatial subsetting using lat/lon coordinates on rotated pole grid
        if self.bbox:
            self.logger.info("Computing spatial mask from OPeNDAP coordinates...")
            ds = self._subset_bbox(ds)

        # Temporal subsetting
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        if ds.time.size == 0:
            raise ValueError(f"No data found for time range {self.start_date} to {self.end_date}")

        time_steps = ds.time.size
        self.logger.info(f"Downloading {time_steps} time steps via OPeNDAP...")

        # Load data from remote server and save locally
        ds_loaded = ds.load()
        self._validate_acquired_variables(ds_loaded, source='PAVICS OPeNDAP endpoint')

        # Use chunked encoding for better memory management
        encoding = {}
        for var in ds_loaded.data_vars:
            shape = ds_loaded[var].shape
            if len(shape) == 3:  # time, rlat, rlon
                encoding[var] = {'chunksizes': (min(168, shape[0]), shape[1], shape[2])}

        ds_loaded.to_netcdf(final_file, encoding=encoding)
        self.logger.info(f"Successfully saved data to {final_file.name}")
        return final_file

    # ------------------------------------------------------------------
    # HTTP fallback (ECCC GPSC-C)
    # ------------------------------------------------------------------

    def _download_http(self, output_dir: Path, final_file: Path) -> Path:
        """
        HTTP fallback from ECCC GPSC-C.

        Prefers the tiled per-variable archive (only the tiles/variables
        covering the domain are fetched); falls back to full-domain daily
        files (~700 MB/day) only if the tiled pathway fails.
        """
        try:
            return self._download_http_tiled(output_dir, final_file)
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(
                f"Tiled CaSR pathway failed: {e}. Falling back to full-domain "
                "daily files (very slow: ~700 MB per simulated day).",
                exc_info=True,
            )
            return self._download_http_daily(output_dir, final_file)

    # -- Tiled per-variable archive ------------------------------------

    @staticmethod
    def _geo_to_rotated(lat: float, lon: float, pole_lat: float, pole_lon: float) -> Tuple[float, float]:
        """Convert geographic coordinates to CF rotated-pole coordinates (degrees)."""
        lat_r = np.radians(lat)
        plat = np.radians(pole_lat)
        dlon = np.radians(lon - pole_lon)
        s = np.sin(lat_r) * np.sin(plat) + np.cos(lat_r) * np.cos(plat) * np.cos(dlon)
        rlat = np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
        y = -np.cos(lat_r) * np.sin(dlon)
        x = -np.cos(lat_r) * np.sin(plat) * np.cos(dlon) + np.sin(lat_r) * np.cos(plat)
        rlon = np.degrees(np.arctan2(y, x))
        return float(rlat), float(rlon)

    def _casr_index_window(self) -> Tuple[int, int, int, int]:
        """
        Global (0-based) rlat/rlon index window covering the bbox, with a
        2-cell buffer. Transforms the bbox corners and edge midpoints since
        the rotated rectangle is not axis-aligned in geographic space.
        """
        g = CASR_V32_GRID
        lat_min, lat_max = self.bbox['lat_min'], self.bbox['lat_max']
        lon_min, lon_max = self.bbox['lon_min'], self.bbox['lon_max']
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2

        points = [
            (lat_min, lon_min), (lat_min, lon_mid), (lat_min, lon_max),
            (lat_mid, lon_min), (lat_mid, lon_max),
            (lat_max, lon_min), (lat_max, lon_mid), (lat_max, lon_max),
        ]
        rlats, rlons = zip(*(self._geo_to_rotated(la, lo, g['pole_lat'], g['pole_lon']) for la, lo in points))

        y_min = int(np.floor((min(rlats) - g['rlat0']) / g['step'])) - 2
        y_max = int(np.ceil((max(rlats) - g['rlat0']) / g['step'])) + 2
        x_min = int(np.floor((min(rlons) - g['rlon0']) / g['step'])) - 2
        x_max = int(np.ceil((max(rlons) - g['rlon0']) / g['step'])) + 2

        y_min = max(0, min(y_min, g['nlat'] - 1))
        y_max = max(0, min(y_max, g['nlat'] - 1))
        x_min = max(0, min(x_min, g['nlon'] - 1))
        x_max = max(0, min(x_max, g['nlon'] - 1))

        if y_max < y_min or x_max < x_min:
            raise ValueError(f"Bounding box {self.bbox} is outside the CaSR v3.2 grid")
        return y_min, y_max, x_min, x_max

    @staticmethod
    def _casr_tile_dirs(y_min: int, y_max: int, x_min: int, x_max: int) -> List[str]:
        """Tile directory names (e.g. 'rlon141-175_rlat386-420') covering an index window."""
        g = CASR_V32_GRID
        t = int(g['tile'])
        n_lat = int(g['nlat'])
        n_lon = int(g['nlon'])
        dirs = []
        for ty in range(y_min // t, y_max // t + 1):
            rlat_lo = ty * t + 1
            rlat_hi = min((ty + 1) * t, n_lat)
            for tx in range(x_min // t, x_max // t + 1):
                rlon_lo = tx * t + 1
                rlon_hi = min((tx + 1) * t, n_lon)
                dirs.append(f"rlon{rlon_lo:03d}-{rlon_hi:03d}_rlat{rlat_lo:03d}-{rlat_hi:03d}")
        return dirs

    def _casr_variables(self) -> List[str]:
        """Full CaSR v3.2 variable names to acquire (config-overridable)."""
        raw = self._get_config_value(lambda: None, default=None, dict_key='RDRS_CASR_VARIABLES')
        if raw:
            names = [v.strip() for v in raw.split(',')] if isinstance(raw, str) else list(raw)
        else:
            names = list(DEFAULT_CASR_V32_VARIABLES)
        return [n if n.startswith('CaSR_') else f"CaSR_v3.2_{n}" for n in names]

    def _download_http_tiled(self, output_dir: Path, final_file: Path) -> Path:
        """
        Download CaSR v3.2 from the GPSC-C tiled per-variable archive.

        The archive splits the domain into 35x35-cell tiles and each variable
        into 4-year files (~60-100 MB each), so a small domain needs only
        `n_tiles x n_variables x n_periods` small files instead of ~700 MB
        per simulated day.
        """
        if not self.bbox:
            raise ValueError("Tiled CaSR download requires a bounding box")

        tile_base = self._get_config_value(
            lambda: None, default=GPSCC_CASR_V32_TILE_BASE_URL, dict_key='RDRS_TILE_BASE_URL')
        var_names = self._casr_variables()

        y_min, y_max, x_min, x_max = self._casr_index_window()
        tile_dirs = self._casr_tile_dirs(y_min, y_max, x_min, x_max)
        self.logger.info(
            f"Tiled CaSR v3.2 download: grid window rlat[{y_min}:{y_max}] rlon[{x_min}:{x_max}] "
            f"-> {len(tile_dirs)} tile(s), {len(var_names)} variable(s)"
        )

        session = create_robust_session(max_retries=3, backoff_factor=1.0)

        # Collect the files to fetch by parsing each tile directory listing
        # (year-block boundaries are read from the server, not assumed).
        # A block labelled Y0-Y1 spans (Y0-1)-12-31T13:00Z .. Y1-12-31T12:00Z.
        tasks: List[Tuple[str, str]] = []  # (tile_dir, file_name)
        for tile_dir in tile_dirs:
            listing = self._list_casr_tile_files(session, tile_base, tile_dir)
            for var in var_names:
                blocks = []
                for fname, (year0, year1) in listing:
                    if not fname.startswith(f"{var}_"):
                        continue
                    block_start = pd.Timestamp(year0 - 1, 12, 31, 13)
                    block_end = pd.Timestamp(year1, 12, 31, 12)
                    if self.start_date <= block_end and self.end_date >= block_start:
                        blocks.append(fname)
                if not blocks:
                    raise RuntimeError(
                        f"No {var} files covering {self.start_date:%Y-%m-%d}..{self.end_date:%Y-%m-%d} "
                        f"in tile {tile_dir} — check {tile_base}/{tile_dir}/"
                    )
                tasks.extend((tile_dir, fname) for fname in blocks)

        total = len(tasks)
        max_workers = int(self._get_config_value(
            lambda: None, default=min(total, 12), dict_key='RDRS_MAX_PARALLEL_DOWNLOADS'))
        max_workers = max(1, min(max_workers, total))
        self.logger.info(f"Downloading {total} CaSR tile files ({max_workers} parallel connections)")

        thread_state = threading.local()

        def get_session() -> requests.Session:
            if not hasattr(thread_state, 'session'):
                thread_state.session = create_robust_session(max_retries=3, backoff_factor=1.0)
            return thread_state.session

        def fetch(task: Tuple[str, str]) -> Path:
            tile_dir, fname = task
            url = f"{tile_base.rstrip('/')}/{tile_dir}/{fname}"
            dest = output_dir / f"temp_casr_tile_{fname}"
            return self._download_casr_tile_file(url, dest, get_session())

        downloaded: List[Path] = []
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                tile_dir, fname = futures[future]
                f = future.result()  # raises on failure -> caller falls back
                downloaded.append(f)
                completed += 1
                self.logger.info(f"Tile download progress: {completed}/{total} ({fname})")

        self.logger.info(f"Assembling {len(downloaded)} CaSR tile files...")
        per_var = []
        for var in var_names:
            var_files = sorted(f for f in downloaded if f.name.startswith(f"temp_casr_tile_{var}_"))
            per_var.append(xr.open_mfdataset(
                var_files, combine='by_coords', chunks={'time': 24 * 31},
                data_vars='minimal', coords='minimal', compat='override'))

        # join='exact' so a coordinate mismatch between per-variable archives
        # raises instead of silently outer-joining and fill-valuing a whole
        # variable (which produced an all-zero downwelling longwave field).
        try:
            ds = xr.merge(per_var, compat='override', join='exact')
        except ValueError as exc:
            raise RuntimeError(
                "CaSR per-variable tile archives do not share identical coordinates, so they "
                "cannot be merged without fabricating fill values. Re-run with the OPeNDAP "
                f"pathway, or report the misaligned tile set upstream. Original error: {exc}"
            ) from exc
        ds = self._subset_bbox(ds)
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        if ds.time.size == 0:
            raise ValueError(f"No data found for time range {self.start_date} to {self.end_date}")

        self.logger.info(f"Saving {ds.time.size} time steps to {final_file.name}...")
        ds_loaded = ds.load()
        self._validate_acquired_variables(ds_loaded, source='tiled CaSR archive')
        for name in ds_loaded.variables:
            # Source chunk sizes (full 35x35 tiles) can exceed the subset dims
            for key in ('chunksizes', 'preferred_chunks', 'original_shape'):
                ds_loaded[name].encoding.pop(key, None)
        ds_loaded.to_netcdf(final_file)
        for src in per_var:
            src.close()

        self.logger.info("Cleaning up temporary tile files...")
        for f in downloaded:
            f.unlink(missing_ok=True)
        return final_file

    def _validate_acquired_variables(self, ds: xr.Dataset, source: str) -> None:
        """
        Fail loudly on degenerate forcing variables.

        A variable that is entirely zero or entirely missing across the whole
        record is never physically meaningful for the fields we acquire, but it
        survives acquisition happily: it has the right name, units and shape.
        Downstream it either trips a model-specific guard much later (SUMMA
        rejects out-of-range longwave) or, for models without one, silently
        corrupts the simulation. Catching it here keeps the failure next to its
        cause.

        Only the variables we actually request are checked. CaSR files ship many
        incidental fields (freezing-rain rate, ice-pellet rate, land masks) that
        are legitimately all-zero for a given domain or season, so validating
        everything would cry wolf.
        """
        try:
            requested = {n.replace('CaSR_v3.2_', '') for n in self._casr_variables()}
        except (AttributeError, KeyError, TypeError, ValueError):
            # Config unavailable/malformed — fall back to the documented set so
            # validation still runs rather than being skipped silently.
            requested = set(DEFAULT_CASR_V32_VARIABLES)

        def is_requested(name: str, da: xr.DataArray) -> bool:
            # Files may carry CF names (rlds) or raw archive names (P_FI_SFC);
            # the CF-renamed variables keep the raw name in `original_variable`.
            return (
                name.replace('CaSR_v3.2_', '') in requested
                or str(da.attrs.get('original_variable', '')) in requested
            )

        degenerate = []
        for name, da in ds.data_vars.items():
            if 'time' not in da.dims or da.size == 0 or not is_requested(name, da):
                continue
            values = np.asarray(da.values, dtype='float64')
            finite = np.isfinite(values)
            if not finite.any():
                degenerate.append(f"{name}: all values missing/NaN")
            elif not np.any(values[finite] != 0.0):
                degenerate.append(f"{name}: all {finite.sum()} finite values are exactly zero")

        if degenerate:
            detail = "\n  - ".join(degenerate)
            raise RuntimeError(
                f"Degenerate forcing variable(s) from the {source} for "
                f"{self.start_date:%Y-%m-%d}..{self.end_date:%Y-%m-%d}:\n  - {detail}\n"
                "The variables were requested and returned with correct metadata but carry no "
                "data, so any simulation using them would be invalid. Retrying often succeeds "
                "on a smaller period; the tiled archive is the more reliable pathway for long "
                "records. Otherwise check the source archive for this period/domain."
            )

    def _list_casr_tile_files(
        self, session: requests.Session, tile_base: str, tile_dir: str
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """List (file_name, (start_year, end_year)) entries in a tile directory."""
        url = f"{tile_base.rstrip('/')}/{tile_dir}/"
        response = session.get(url, timeout=120)
        response.raise_for_status()
        pattern = re.compile(r'href="((?:CaSR|RDRS)[^"]*_(\d{4})-(\d{4})\.nc)"')
        entries = [
            (m.group(1), (int(m.group(2)), int(m.group(3))))
            for m in pattern.finditer(response.text)
        ]
        if not entries:
            raise RuntimeError(f"No CaSR files found in tile listing {url}")
        return entries

    def _download_casr_tile_file(
        self, url: str, dest: Path, session: requests.Session, timeout: int = 900
    ) -> Path:
        """Download one tile file with resume support (Range requests on .part files)."""
        part = dest.with_suffix(dest.suffix + '.part')

        if dest.exists():
            try:
                remote_size = int(session.head(url, timeout=60).headers.get('Content-Length', -1))
            except requests.exceptions.RequestException:
                remote_size = -1
            if remote_size in (-1, dest.stat().st_size):
                return dest
            dest.unlink()

        headers = {}
        mode = 'wb'
        if part.exists() and part.stat().st_size > 0:
            headers['Range'] = f"bytes={part.stat().st_size}-"
            mode = 'ab'

        response = session.get(url, headers=headers, timeout=timeout, stream=True)
        if response.status_code == 200 and mode == 'ab':
            # Server ignored the Range header — restart from scratch
            mode = 'wb'
        elif response.status_code == 416:
            # Range not satisfiable: the .part file is already complete
            part.rename(dest)
            return dest
        response.raise_for_status()

        with open(part, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        part.rename(dest)
        return dest

    # -- Full-domain daily files (last resort) --------------------------

    def _download_http_daily(self, output_dir: Path, final_file: Path) -> Path:
        """Last-resort HTTP download of full-domain daily CaSR NetCDF files."""
        # CaSR v3.2 daily files from GPSC-C (format: YYYYMMDD12.nc)
        base_url = self._get_config_value(lambda: None, default=GPSCC_CASR_V32_BASE_URL, dict_key='RDRS_BASE_URL')
        self.logger.info(f"Downloading CaSR daily files from {base_url}")

        # Generate list of days (files are daily at 12 UTC)
        date_range = pd.date_range(start=self.start_date.normalize(), end=self.end_date.normalize(), freq='D')
        total_files = len(date_range)

        # Volume guard. Unlike OPeNDAP and the tiled archive, this path fetches
        # the FULL CaSR domain for every day (~700 MB/file) and subsets locally,
        # so its cost scales with the date range and is independent of how small
        # the catchment is. A multi-year run reaches petabyte scale: an 8-year
        # calibration is 2922 files ~ 2.1 PB, a 16-year one ~ 4.1 PB. Because
        # 'auto' reaches this path after a *transient* network error on both
        # OPeNDAP and the tiled archive, a momentary blip could otherwise commit
        # a run to an impossible download that reports nothing but a progress
        # line every 5% and silently fills the disk. Refuse instead, and name
        # the two endpoints that do subset server-side.
        est_bytes = total_files * DAILY_FULL_DOMAIN_FILE_BYTES
        if total_files > MAX_UNGUARDED_DAILY_FILES and not self._get_config_value(
            lambda: None, default=False, dict_key='RDRS_ALLOW_FULL_DOMAIN_DAILY'
        ):
            raise RuntimeError(
                f"Refusing the full-domain daily CaSR fallback: {total_files} files "
                f"x ~{DAILY_FULL_DOMAIN_FILE_BYTES / 1024**2:.0f} MB ~ "
                f"{est_bytes / 1024**4:.1f} TB. This path downloads the entire CaSR "
                "domain per day and subsets locally, so it does not shrink with the "
                "catchment. Set RDRS_ACQUISITION_METHOD to 'opendap' or 'tiled' (both "
                "subset server-side) — 'tiled' serves this basin as a handful of ~30 MB "
                "files. It was reached only because both of those failed, which is "
                "usually transient: retrying is normally the right fix. To override "
                "deliberately, set RDRS_ALLOW_FULL_DOMAIN_DAILY: true."
            )

        self.logger.info(
            f"Downloading {total_files} daily CaSR files via HTTP fallback "
            f"(full-domain files, ~{est_bytes / 1024**3:.0f} GB estimated)"
        )

        # Use robust session with connection pooling and retry logic
        session = create_robust_session(max_retries=3, backoff_factor=1.0)

        max_workers = min(total_files, 4)
        downloaded_files = []
        failed_dates = []
        completed = 0
        log_interval = max(1, total_files // 20)  # Log every ~5%

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self._download_daily_file, dt, base_url, output_dir, session): dt
                for dt in date_range
            }
            for future in concurrent.futures.as_completed(future_to_date):
                dt = future_to_date[future]
                completed += 1
                try:
                    f = future.result()
                    if f:
                        downloaded_files.append(f)
                    else:
                        failed_dates.append(dt)
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"HTTP download failed for {dt.strftime('%Y-%m-%d')}: {e}", exc_info=True)
                    failed_dates.append(dt)

                if completed % log_interval == 0 or completed == total_files:
                    self.logger.info(
                        f"HTTP download progress: {completed}/{total_files} "
                        f"({len(downloaded_files)} ok, {len(failed_dates)} failed)"
                    )

        # Retry failed downloads sequentially
        if failed_dates:
            self.logger.info(f"Retrying {len(failed_dates)} failed downloads...")
            retry_session = create_robust_session(max_retries=5, backoff_factor=2.0)
            for dt in failed_dates:
                f = self._download_daily_file(dt, base_url, output_dir, retry_session, timeout=300)
                if f:
                    downloaded_files.append(f)
                else:
                    self.logger.warning(f"Retry also failed for {dt.strftime('%Y-%m-%d')}")

        final_failed = total_files - len(downloaded_files)
        if final_failed > 0:
            self.logger.warning(f"{final_failed}/{total_files} daily files failed to download after retries")

        if not downloaded_files:
            raise RuntimeError(
                "HTTP fallback failed: No CaSR data downloaded. "
                "The GPSC-C server may not yet have data for your time period. "
                "Check https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/download.html "
                "for current data availability."
            )

        downloaded_files.sort()
        self.logger.info(f"Merging {len(downloaded_files)} daily CaSR files...")
        with xr.open_mfdataset(downloaded_files, combine='by_coords', chunks={'time': 24}, data_vars='minimal', coords='minimal', compat='override') as ds:
            if self.bbox:
                self.logger.info("Computing spatial mask for HTTP-downloaded files...")
                ds = self._subset_bbox(ds)

            self.logger.info(f"Saving merged data to {final_file.name}...")
            ds.to_netcdf(final_file)

        self.logger.info("Cleaning up temporary daily files...")
        for f in downloaded_files:
            f.unlink(missing_ok=True)
        return final_file

    def _download_daily_file(
        self, dt: datetime, base_url: str, output_dir: Path,
        session: requests.Session, timeout: int = 120
    ) -> Optional[Path]:
        """Download a single daily CaSR file from GPSC-C (format: YYYYMMDD12.nc)."""
        file_name = dt.strftime("%Y%m%d12.nc")
        url = f"{base_url.rstrip('/')}/{file_name}"
        dest_path = output_dir / f"temp_casr_{file_name}"
        if dest_path.exists():
            return dest_path
        try:
            response = session.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                return dest_path
            else:
                self.logger.debug(f"CaSR download returned status {response.status_code} for {file_name}")
        except requests.exceptions.Timeout:
            self.logger.warning(f"CaSR download timed out for {file_name}")
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"CaSR download failed for {file_name}: {e}")
        return None
