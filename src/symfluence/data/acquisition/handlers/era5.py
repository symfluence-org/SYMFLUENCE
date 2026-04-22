# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ERA5 reanalysis data acquisition handlers.

Provides acquisition from Google Cloud ARCO-ERA5 (Zarr) or ECMWF Climate
Data Store (CDS) with automatic pathway selection and parallel processing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.validation import validate_bounding_box, validate_numeric_range
from symfluence.geospatial.coordinate_utils import BoundingBox, get_bbox_extent

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from .era5_cds import ERA5CDSAcquirer
from .era5_processing import era5_to_summa_schema

# Patterns indicating HDF5/netCDF engine issues on parallel filesystems
_HDF_ERROR_PATTERNS = (
    "HDF error",
    "HDF5 error",
    "Errno -101",
    "unable to lock file",
    "Resource temporarily",
    "unable to synchronously",
)


def _safe_to_netcdf(ds: xr.Dataset, path: Path, encoding: dict = None,
                     logger: logging.Logger = None) -> None:
    """Write dataset to NetCDF with automatic engine fallback for HPC filesystems.

    Loads data into memory first to avoid dask/HDF5 conflicts when writing
    from cloud-backed (Zarr) datasets, clears cloud-store encoding, then
    tries netcdf4 → h5netcdf → uncompressed h5netcdf → local-tempfile as
    fallbacks.

    The final fallback writes to a local temp directory ($TMPDIR or /tmp)
    where POSIX file locking is supported, then moves the file to the target
    path.  This bypasses HDF5 fcntl() locking failures on parallel
    filesystems (Lustre / GPFS / BeeGFS) even when the HDF5 C library was
    compiled with forced locking or loaded before HDF5_USE_FILE_LOCKING was
    set.
    """
    # Belt-and-suspenders: re-enforce locking env var in case the HDF5 C
    # library hasn't been loaded yet (it reads the var at dlopen time).
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    # Materialise into memory so the write is a pure local operation
    ds = ds.load()

    # Clear cloud-store (Zarr) encoding that can confuse NetCDF writers
    for var in ds.data_vars:
        ds[var].encoding.clear()
    for coord in ds.coords:
        ds[coord].encoding.clear()

    # --- Attempt 1: default netcdf4 engine with compression ----------------
    try:
        ds.to_netcdf(path, encoding=encoding, compute=True)
        return
    except (OSError, RuntimeError) as e:
        msg = str(e).lower()
        if not any(pat.lower() in msg for pat in _HDF_ERROR_PATTERNS):
            raise
        if logger:
            logger.warning(
                f"netcdf4 engine failed writing {path.name} "
                f"({e!r}), retrying with h5netcdf engine"
            )

    # --- Attempt 2: h5netcdf engine (different HDF5 binding) ---------------
    try:
        ds.to_netcdf(path, engine="h5netcdf", encoding=encoding, compute=True)
        return
    except (OSError, RuntimeError) as e:
        if logger:
            logger.warning(
                f"h5netcdf with compression also failed ({e!r}), "
                f"retrying without compression"
            )

    # --- Attempt 3: h5netcdf without compression ---------------------------
    try:
        ds.to_netcdf(path, engine="h5netcdf", compute=True)
        return
    except (OSError, RuntimeError) as e:
        if logger:
            logger.warning(
                f"h5netcdf without compression also failed ({e!r}), "
                f"retrying via local temp file to bypass filesystem locking"
            )

    # --- Attempt 4: write to local temp file, then move --------------------
    # On parallel filesystems (Lustre/GPFS/BeeGFS) the HDF5 C library's
    # fcntl() locking can fail even when HDF5_USE_FILE_LOCKING=FALSE, e.g.
    # because the library was compiled with forced locking or was loaded
    # before the env-var was set.  Writing to a local filesystem ($TMPDIR
    # or /tmp) avoids the locking issue; a cross-device move is a plain
    # copy+unlink that does not require HDF5 file locking.
    import shutil
    import tempfile

    tmpdir = os.environ.get('TMPDIR') or tempfile.gettempdir()  # nosec B108 — respects $TMPDIR
    fd, tmp_path = tempfile.mkstemp(suffix='.nc', dir=tmpdir)
    os.close(fd)
    try:
        ds.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding, compute=True)
        shutil.move(tmp_path, str(path))
        if logger:
            logger.info(
                f"Successfully wrote {Path(path).name} via local temp file "
                f"(parallel filesystem locking workaround)"
            )
    except Exception:
        # Clean up temp file on failure before re-raising
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def has_cds_credentials() -> bool:
    """
    Check if CDS API credentials are available.

    Checks for credentials in two locations:
    1. ~/.cdsapirc file (standard CDS API configuration)
    2. CDSAPI_KEY environment variable

    Returns:
        True if credentials are found, False otherwise

    Note:
        The .cdsapirc file should contain:
        url: https://cds.climate.copernicus.eu/api
        key: <YOUR_API_KEY>

        (For users on pre-September-2024 CDS setups: the old
        ``https://cds.climate.copernicus.eu/api/v2`` endpoint and the
        old ``<UID>:<API_KEY>`` key format are no longer supported.
        Regenerate your key at cds.climate.copernicus.eu/profile and
        upgrade ``cdsapi>=0.7.0`` — this is enforced by pyproject.toml.)
    """
    return os.path.exists(os.path.expanduser('~/.cdsapirc')) or 'CDSAPI_KEY' in os.environ


def diagnose_cds_credentials() -> Optional[str]:
    """Inspect ``~/.cdsapirc`` and ``CDSAPI_*`` env vars and return a
    human-readable description of any problem, or ``None`` if the
    setup looks usable against the post-September-2024 CDS endpoint.

    Diagnoses the specific failure modes that bit users during the
    2024 CDS migration and that our own reviewers hit during paper
    reproduction. Designed to be called from error-raise sites so the
    message the user sees points at a concrete thing to fix rather
    than a generic "credentials not found".

    Returns:
        ``None`` when credentials look valid for the new CDS endpoint,
        otherwise a multi-line string suitable for embedding in an
        exception message.
    """
    rc_path = Path(os.path.expanduser('~/.cdsapirc'))
    env_key = os.environ.get('CDSAPI_KEY')
    env_url = os.environ.get('CDSAPI_URL')

    if not rc_path.exists() and not env_key:
        return (
            "No CDS API credentials found. SYMFLUENCE looked for:\n"
            f"  1. {rc_path} (the standard cdsapi config file)\n"
            "  2. CDSAPI_KEY / CDSAPI_URL environment variables\n"
            "and neither was present. To set up CDS access:\n"
            "  1. Register at https://cds.climate.copernicus.eu and "
            "accept the data-use terms for the dataset you need\n"
            "     (for ERA5, visit the ERA5 catalog page and click "
            "'Accept terms' at least once).\n"
            "  2. Copy your API key from https://cds.climate.copernicus.eu/profile\n"
            "  3. Create ~/.cdsapirc with exactly these two lines:\n"
            "       url: https://cds.climate.copernicus.eu/api\n"
            "       key: <YOUR_API_KEY>\n"
            "Alternative: you can skip ~/.cdsapirc and install gcsfs "
            "(``pip install gcsfs``) to use the ARCO-ERA5 path on "
            "Google Cloud, which needs no credentials."
        )

    problems: List[str] = []
    if env_key and ':' in env_key:
        problems.append(
            "CDSAPI_KEY looks like the pre-September-2024 '<UID>:<API_KEY>' "
            "format, which the new CDS rejects. Regenerate your key at "
            "https://cds.climate.copernicus.eu/profile — the new format is a "
            "single token, no colon."
        )
    if env_url and '/api/v2' in env_url:
        problems.append(
            f"CDSAPI_URL is set to {env_url!r}, which is the "
            "pre-September-2024 endpoint. Change it to "
            "'https://cds.climate.copernicus.eu/api' (no /v2)."
        )

    if rc_path.exists():
        try:
            rc_text = rc_path.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            return (
                f"~/.cdsapirc exists at {rc_path} but could not be read: "
                f"{exc}. Check the file permissions."
            )
        rc_lines = [
            ln.strip() for ln in rc_text.splitlines()
            if ln.strip() and not ln.lstrip().startswith('#')
        ]
        rc_map = {}
        for ln in rc_lines:
            if ':' in ln:
                k, _, v = ln.partition(':')
                rc_map[k.strip().lower()] = v.strip()
        rc_url = rc_map.get('url', '')
        rc_key = rc_map.get('key', '')

        if not rc_url:
            problems.append(
                f"~/.cdsapirc at {rc_path} is missing a 'url:' line. "
                "Add: url: https://cds.climate.copernicus.eu/api"
            )
        elif '/api/v2' in rc_url:
            problems.append(
                f"~/.cdsapirc url is '{rc_url}', which is the "
                "pre-September-2024 endpoint. Change to "
                "'https://cds.climate.copernicus.eu/api' (no /v2)."
            )
        if not rc_key:
            problems.append(
                f"~/.cdsapirc at {rc_path} is missing a 'key:' line. "
                "Add: key: <YOUR_API_KEY> (copy from "
                "https://cds.climate.copernicus.eu/profile)."
            )
        elif ':' in rc_key:
            problems.append(
                "~/.cdsapirc key looks like the pre-September-2024 "
                "'<UID>:<API_KEY>' format, which the new CDS rejects. "
                "Regenerate at https://cds.climate.copernicus.eu/profile — "
                "the new format is a single token, no colon."
            )

    if problems:
        return (
            "CDS credential setup has issue(s) that will prevent downloads:\n  - "
            + "\n  - ".join(problems)
        )
    return None


@AcquisitionRegistry.register('ERA5')
class ERA5Acquirer(BaseAcquisitionHandler):
    """
    Dispatcher for ERA5 reanalysis data acquisition.

    Automatically selects between two acquisition pathways:
    1. ARCO-ERA5 (Google Cloud Zarr): Faster, no queue, preferred for speed
    2. CDS (ECMWF Climate Data Store): NetCDF download, uses CDS API queue

    Pathway selection order:
    - If ERA5_USE_CDS config/env is set, use that preference
    - Otherwise auto-detect: ARCO if gcsfs is available, else CDS if credentials exist

    Both pathways produce equivalent SUMMA-compatible forcing files with proper
    unit conversions and variable standardization.
    """
    def download(self, output_dir: Path) -> Path:
        """
        Download ERA5 data using automatically selected pathway.

        Selects between ARCO-ERA5 (Google Cloud) and CDS (ECMWF) based on:
        1. Configuration: ERA5_USE_CDS or environment variable
        2. Auto-detection: Prefer ARCO if gcsfs available, else CDS if credentials exist
        3. Fallback: Try alternate pathway if selected one fails

        Args:
            output_dir: Directory to save downloaded ERA5 files

        Returns:
            Path to downloaded file (single file) or directory (multiple files)

        Raises:
            ImportError: If neither gcsfs nor CDS credentials are available
            Exception: If both pathways fail

        Note:
            ARCO pathway is faster (no queue) but requires gcsfs.
            CDS pathway requires ~/.cdsapirc credentials.
        """
        # Default to ARCO if libraries available, falling back to CDS

        # Get ERA5_USE_CDS from typed config (supports both typed and dict config)
        use_cds = self._get_config_value(lambda: self.config.forcing.era5_use_cds)

        # Also check era5 subsection
        if use_cds is None:
            use_cds = self._get_config_value(lambda: self.config.forcing.era5.use_cds)

        # Check environment variable as fallback
        if use_cds is None:
            env_use_cds = os.environ.get('ERA5_USE_CDS')
            if env_use_cds:
                use_cds = env_use_cds.lower() in ('true', 'yes', '1', 'on')
                self.logger.info(f"Using ERA5_USE_CDS from environment: {use_cds}")

        self.logger.info(f"ERA5_USE_CDS config value: {use_cds} (type: {type(use_cds)})")

        if use_cds is None:
            # Auto-detect preference: ARCO (faster) > CDS
            # Both pathways now have the longwave radiation fix, so prefer ARCO for speed
            from importlib.util import find_spec
            if find_spec("gcsfs") and find_spec("xarray"):
                self.logger.info("Auto-detecting ERA5 pathway: ARCO (Google Cloud) - faster, no queue")
                self.logger.info("  To use CDS instead, set ERA5_USE_CDS=true in config or environment")
                use_cds = False
            else:
                if has_cds_credentials():
                    self.logger.info("gcsfs not available, falling back to CDS pathway")
                    use_cds = True
                else:
                    self.logger.error("Neither gcsfs nor CDS credentials available for ERA5 download")
                    diag = diagnose_cds_credentials() or (
                        "No CDS credentials found at ~/.cdsapirc or in "
                        "CDSAPI_KEY/CDSAPI_URL environment variables."
                    )
                    raise ImportError(
                        "ERA5 download requires either gcsfs (for the ARCO cloud "
                        "path, no credentials needed) or CDS credentials (for the "
                        "CDS API path). Neither is configured.\n"
                        "  To use ARCO: pip install gcsfs\n"
                        "  To use CDS:\n"
                        f"{diag}"
                    )
        else:
            # Handle string values like "true", "True", "yes", etc.
            if isinstance(use_cds, str):
                use_cds = use_cds.lower() in ('true', 'yes', '1', 'on')

        self.logger.info(f"Using CDS pathway: {use_cds}")

        if use_cds:
            self.logger.info("Using CDS pathway for ERA5")
            try:
                return ERA5CDSAcquirer(self.config, self.logger).download(output_dir)
            except (
                ImportError,
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                PermissionError,
                DataAcquisitionError,
            ) as e:
                # Keep the silent-fallback design intent (ARCO is fine for
                # the common case) but give the user enough detail to
                # diagnose a CDS setup problem rather than chasing the
                # ensuing ARCO error as if it were the primary failure.
                # The two most common CDS setup errors post-Sept-2024
                # are: (a) a ~/.cdsapirc pointing at /api/v2 which the
                # new CDS rejects, and (b) cdsapi<0.7.0 not speaking
                # the new API. Both look like generic failures from
                # inside this except block.
                base_msg = (
                    "CDS pathway failed: %s. Falling back to ARCO (Google Cloud).\n"
                    "  If the ARCO fallback also fails, the root cause is usually CDS setup:\n"
                    "    • ~/.cdsapirc url must be https://cds.climate.copernicus.eu/api (no /v2).\n"
                    "    • cdsapi must be >=0.7.0 (new API). Check with: python -c 'import cdsapi; print(cdsapi.__version__)'.\n"
                    "    • The API key is now a single token (not <UID>:<KEY>). Regenerate at "
                    "cds.climate.copernicus.eu/profile."
                )
                diag = diagnose_cds_credentials()
                if diag:
                    self.logger.warning(
                        base_msg + "\n  Detected problem(s) with your local setup:\n%s",
                        e,
                        diag,
                    )
                else:
                    self.logger.warning(base_msg, e)

        self.logger.info("Using ARCO (Google Cloud) pathway for ERA5")
        return ERA5ARCOAcquirer(self.config, self.logger).download(output_dir)

class ERA5ARCOAcquirer(BaseAcquisitionHandler):
    """
    ERA5 acquisition via Google Cloud ARCO-ERA5 (Analysis-Ready Cloud-Optimized).

    Downloads ERA5 reanalysis data from the Google Cloud public bucket in Zarr
    format. This pathway is typically faster than CDS as it doesn't require
    queuing and supports efficient spatial/temporal subsetting.

    Features:
    - No authentication required (public bucket)
    - Efficient cloud-native access via Zarr format
    - Automatic longitude wrapping for domains crossing the antimeridian
    - Parallel monthly chunk processing
    - Automatic conversion to SUMMA-compatible format
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download ERA5 from Google Cloud ARCO-ERA5 Zarr store.

        Process:
        1. Connect to Google Cloud public bucket (anonymous access)
        2. Open ERA5 Zarr dataset with lazy loading
        3. Subset spatially by bounding box (handles antimeridian crossing)
        4. Subset temporally into monthly chunks
        5. Optionally temporally subsample (e.g., every 3 hours)
        6. Process chunks in parallel (if configured)
        7. Convert to SUMMA-compatible format with unit conversions
        8. Save as compressed NetCDF files

        Args:
            output_dir: Directory to save processed ERA5 NetCDF files

        Returns:
            Path to single output file (if one month) or output directory (if multiple months)

        Raises:
            ImportError: If gcsfs or required dependencies not installed
            ValueError: If BOUNDING_BOX_COORDS not provided in configuration
            Exception: If spatial bounding box results in empty selection

        Note:
            - Processes data in monthly chunks to manage memory
            - Supports parallel processing for faster downloads
            - Automatically handles longitude wrapping for domains crossing antimeridian
            - Output files named: domain_{DOMAIN_NAME}_ERA5_merged_{YYYYMM}.nc
        """
        self.logger.info("Downloading ERA5 data from Google Cloud ARCO-ERA5")
        domain_name = self.domain_name

        try:
            import gcsfs
            from pandas.tseries.offsets import MonthEnd
        except ImportError as e:
            raise ImportError("gcsfs and xarray are required for ERA5 cloud access.") from e

        gcs = gcsfs.GCSFileSystem(token="anon")  # nosec B106 - anonymous access to public GCS
        default_store = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        zarr_store = self._get_config_value(
            lambda: self.config.forcing.era5.zarr_path, default=default_store
        )

        mapper = gcs.get_mapper(zarr_store)

        # Reset the CPython thread-pool shutdown flag that may have been set
        # by _python_exit() when the main thread finished (common when Panel
        # runs via pn.serve()).  zarr v3 codecs use asyncio.to_thread() →
        # ThreadPoolExecutor.submit() which raises RuntimeError if this flag
        # is True.
        import concurrent.futures.thread as _cft
        if hasattr(_cft, '_shutdown'):
            _cft._shutdown = False
        if hasattr(_cft, '_global_shutdown'):
            _cft._global_shutdown = False

        ds = xr.open_zarr(mapper, consolidated=True, chunks={})
        ds = ds.assign_coords(longitude=ds.longitude.load(), latitude=ds.latitude.load(), time=ds.time.load())

        if not self.bbox:
            raise ValueError("BOUNDING_BOX_COORDS is required for ERA5 cloud access.")

        bbox_info = _prepare_bbox_for_era5(ds, self.bbox, self.logger)
        lat_min_raw = bbox_info["lat_min"]
        lat_max_raw = bbox_info["lat_max"]
        lon_min = bbox_info["lon_min"]
        lon_max = bbox_info["lon_max"]
        wrap_longitude = bbox_info["wrap_longitude"]
        lat_descending = bbox_info["lat_descending"]
        lon_min_value = bbox_info["lon_min_value"]
        lon_max_value = bbox_info["lon_max_value"]
        lat_res = bbox_info["lat_resolution"]
        lon_res = bbox_info["lon_resolution"]

        step = self._get_config_value(
            lambda: self.config.forcing.era5.time_step_hours, default=1
        )
        step = int(step)

        # Validate time step parameter
        validate_numeric_range(
            step, min_val=1, max_val=24,
            param_name="time_step_hours",
            context="ERA5 acquisition"
        )
        if 24 % step != 0:
            self.logger.warning(
                f"time_step_hours ({step}) does not divide evenly into 24 hours. "
                "This may cause irregular time intervals in the output."
            )

        era5_start = self.start_date - pd.Timedelta(hours=step)
        era5_end = self.end_date

        current_month_start = era5_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        chunks = []
        while current_month_start <= era5_end:
            month_end = (current_month_start + MonthEnd(1)).replace(hour=23, minute=0, second=0, microsecond=0)
            chunk_start, chunk_end = max(era5_start, current_month_start), min(era5_end, month_end)
            if chunk_start <= chunk_end: chunks.append((chunk_start, chunk_end))
            current_month_start = (current_month_start.replace(day=28) + pd.Timedelta(days=4)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        default_vars = ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure", "total_precipitation", "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards"]
        # Critical variables required for hydrological modeling
        critical_vars = ["2m_temperature", "total_precipitation", "surface_pressure"]

        requested_vars = self._get_config_value(
            lambda: self.config.forcing.era5.variables, default=default_vars
        ) or default_vars
        available_vars = [v for v in requested_vars if v in ds.data_vars]

        # Validate that at least critical variables are available
        if not available_vars:
            raise DataAcquisitionError(
                f"No requested ERA5 variables available in dataset. "
                f"Requested: {requested_vars}. Available: {list(ds.data_vars)}"
            )

        missing_critical = [v for v in critical_vars if v not in available_vars]
        if missing_critical:
            self.logger.warning(
                f"ERA5 dataset missing critical forcing variables: {missing_critical}. "
                "Hydrological model may produce incomplete results."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = []

        # Check for skip_existing option (default True to resume interrupted downloads)
        skip_existing = self._get_config_value(
            lambda: self.config.forcing.skip_existing_files, default=True
        )

        # Default to parallel processing if not specified
        n_workers_cfg = self._get_config_value(lambda: self.config.system.num_processes)
        if n_workers_cfg is not None:
            n_workers = int(n_workers_cfg)
        else:
            import os
            # Use available CPUs but cap at 8 to avoid overwhelming I/O
            n_workers = min(8, os.cpu_count() or 1)

        self.logger.info(f"Processing ERA5 with {n_workers} workers")

        if n_workers <= 1:
            for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                file_year, file_month = chunk_start.year, chunk_start.month
                chunk_file = output_dir / f"domain_{domain_name}_ERA5_merged_{file_year}{file_month:02d}.nc"

                # Check if file exists and is valid (skip if already downloaded)
                if skip_existing and chunk_file.exists():
                    try:
                        with xr.open_dataset(chunk_file) as existing:
                            if "time" in existing.dims and existing.sizes["time"] > 0:
                                self.logger.info(f"✓ Skipping ERA5 chunk {i}/{len(chunks)} ({chunk_file.name}) - already exists")
                                chunk_files.append(chunk_file)
                                continue
                    except (OSError, ValueError, KeyError):
                        self.logger.info(f"  Existing file {chunk_file.name} is invalid, re-downloading")

                self.logger.info(f"Processing ERA5 chunk {i}/{len(chunks)}: {chunk_start.strftime('%Y-%m')} to {chunk_end.strftime('%Y-%m')}")
                time_start = chunk_start if i == 1 else chunk_start - pd.Timedelta(hours=step)
                ds_t = ds.sel(time=slice(time_start, chunk_end))
                if "time" not in ds_t.dims or ds_t.sizes["time"] < 2: continue
                ds_ts = _subset_era5_bbox(
                    ds_t,
                    lat_min_raw,
                    lat_max_raw,
                    lon_min,
                    lon_max,
                    wrap_longitude,
                    lat_descending,
                    (lon_min_value, lon_max_value),
                )

                # Check for empty spatial dimensions (bounding box too small for grid resolution)
                if "latitude" not in ds_ts.dims or "longitude" not in ds_ts.dims:
                    self.logger.warning(f"Chunk {i}: Missing spatial dimensions after bounding box selection")
                    continue
                if ds_ts.sizes.get("latitude", 0) == 0 or ds_ts.sizes.get("longitude", 0) == 0:
                    self.logger.warning(
                        f"Chunk {i}: Empty spatial dimensions after bounding box selection. "
                        f"Verify BOUNDING_BOX_COORDS covers at least one ERA5 cell "
                        f"(~{lat_res:.2f}° lat x {lon_res:.2f}° lon)."
                    )
                    continue

                if step > 1 and "time" in ds_ts.dims: ds_ts = ds_ts.isel(time=slice(0, None, step))
                if "time" not in ds_ts.dims or ds_ts.sizes["time"] < 2: continue
                ds_chunk = era5_to_summa_schema(ds_ts[[v for v in available_vars if v in ds_ts.data_vars]], source='arco', logger=self.logger)
                if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 1: continue
                # chunk_file already defined at start of loop
                encoding = {var: {"zlib": True, "complevel": 1, "chunksizes": (min(168, ds_chunk.sizes["time"]), ds_chunk.sizes["latitude"], ds_chunk.sizes["longitude"])} for var in ds_chunk.data_vars}
                _safe_to_netcdf(ds_chunk, chunk_file, encoding=encoding, logger=self.logger)
                self.logger.info(f"✓ Successfully saved ERA5 chunk {i}/{len(chunks)} to {chunk_file.name}")
                chunk_files.append(chunk_file)
        else:
            from concurrent.futures import ThreadPoolExecutor

            # Pre-check which chunks need downloading (skip existing valid files)
            chunks_to_process = []
            for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                file_year, file_month = chunk_start.year, chunk_start.month
                chunk_file = output_dir / f"domain_{domain_name}_ERA5_merged_{file_year}{file_month:02d}.nc"

                if skip_existing and chunk_file.exists():
                    try:
                        with xr.open_dataset(chunk_file) as existing:
                            if "time" in existing.dims and existing.sizes["time"] > 0:
                                self.logger.info(f"✓ Skipping ERA5 chunk {i}/{len(chunks)} ({chunk_file.name}) - already exists")
                                chunk_files.append(chunk_file)
                                continue
                    except (OSError, ValueError, KeyError):
                        self.logger.info(f"  Existing file {chunk_file.name} is invalid, re-downloading")

                chunks_to_process.append((i, chunk_start, chunk_end))

            if not chunks_to_process:
                self.logger.info("All ERA5 chunks already downloaded, skipping acquisition")
            else:
                self.logger.info(f"Downloading {len(chunks_to_process)}/{len(chunks)} ERA5 chunks (skipped {len(chunks) - len(chunks_to_process)} existing)")

                def process_chunk(i, chunk_start, chunk_end):
                    return _process_era5_chunk_threadsafe(
                        i,
                        (chunk_start, chunk_end),
                        ds,
                        available_vars,
                        step,
                        lat_min_raw,
                        lat_max_raw,
                        lon_min,
                        lon_max,
                        wrap_longitude,
                        lat_descending,
                        lon_min_value,
                        lon_max_value,
                        output_dir,
                        domain_name,
                        len(chunks),
                        self.logger,
                    )
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(process_chunk, i, cs, ce) for i, cs, ce in chunks_to_process]
                    for future in futures:
                        _, cf, _ = future.result()
                        if cf: chunk_files.append(cf)

        return output_dir if len(chunk_files) > 1 else (chunk_files[0] if chunk_files else output_dir)


def _process_era5_chunk_threadsafe(
    idx,
    times,
    ds,
    vars,
    step,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    wrap_lon,
    lat_descending,
    lon_min_value,
    lon_max_value,
    out_dir,
    dom,
    total,
    logger=None,
):
    """
    Process a single ERA5 time chunk in a thread-safe manner for parallel execution.

    This function handles one monthly chunk of ERA5 data within a parallel processing
    workflow. It performs temporal and spatial subsetting, optional temporal resampling,
    and conversion to SUMMA-compatible format.

    Args:
        idx: Chunk index number (1-based) for tracking progress
        times: Tuple of (start_time, end_time) for this chunk
        ds: Xarray dataset containing full ERA5 data
        vars: List of ERA5 variables to extract
        step: Temporal step for resampling (e.g., 3 for every 3 hours)
        lat_min: Minimum latitude for spatial subsetting
        lat_max: Maximum latitude for spatial subsetting
        lon_min: Minimum longitude for spatial subsetting (normalized to 0-360)
        lon_max: Maximum longitude for spatial subsetting (normalized to 0-360)
        wrap_lon: Whether longitude wraps across antimeridian
        lat_descending: Whether latitude coordinates are descending
        lon_min_value: Minimum longitude value in dataset
        lon_max_value: Maximum longitude value in dataset
        out_dir: Output directory for processed chunk
        dom: Domain name for output filename
        total: Total number of chunks (for logging)
        logger: Logger instance for progress reporting

    Returns:
        Tuple of (chunk_index, output_file_path, status_message)
        - If successful: (idx, Path, "success")
        - If skipped: (idx, None, "skipped: reason")
        - If error: (idx, None, error_message)

    Note:
        - Adds one extra timestep before chunk_start (except for first chunk) to ensure continuity
        - Skips chunks with insufficient time dimension (<2 timesteps)
        - Checks for empty spatial dimensions after bbox selection
        - Applies zlib compression with complevel=1 for faster writing
        - Thread-safe: Each invocation operates on independent data slices
    """
    start, end = times
    try:
        ts = start if idx == 1 else start - pd.Timedelta(hours=step)
        ds_t = ds.sel(time=slice(ts, end))
        if "time" not in ds_t.dims or ds_t.sizes["time"] < 2: return idx, None, "skipped"
        ds_ts = _subset_era5_bbox(
            ds_t,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            wrap_lon,
            lat_descending,
            (lon_min_value, lon_max_value),
        )

        # Check for empty spatial dimensions
        if "latitude" not in ds_ts.dims or "longitude" not in ds_ts.dims:
            return idx, None, "skipped: missing spatial dimensions"
        if ds_ts.sizes.get("latitude", 0) == 0 or ds_ts.sizes.get("longitude", 0) == 0:
            return idx, None, "skipped: empty spatial dimensions after bbox selection"

        if step > 1 and "time" in ds_ts.dims: ds_ts = ds_ts.isel(time=slice(0, None, step))
        if "time" not in ds_ts.dims or ds_ts.sizes["time"] < 2: return idx, None, "skipped"
        ds_chunk = era5_to_summa_schema(ds_ts[[v for v in vars if v in ds_ts.data_vars]], source='arco', logger=logger)
        cf = out_dir / f"domain_{dom}_ERA5_merged_{start.year}{start.month:02d}.nc"
        encoding = {v: {"zlib": True, "complevel": 1, "chunksizes": (min(168, ds_chunk.sizes["time"]), ds_chunk.sizes["latitude"], ds_chunk.sizes["longitude"])} for v in ds_chunk.data_vars}
        _safe_to_netcdf(ds_chunk, cf, encoding=encoding, logger=logger)
        return idx, cf, "success"
    except KeyError as e:
        return idx, None, f"missing variable or dimension: {e}"
    except ValueError as e:
        return idx, None, f"data value error: {e}"
    except (OSError, IOError, RuntimeError) as e:
        return idx, None, f"file I/O error: {e}"
    except MemoryError as e:
        return idx, None, f"memory error (chunk may be too large): {e}"
    except (TypeError, AttributeError, IndexError) as e:
        return idx, None, f"chunk processing type/index error: {e}"


def _prepare_bbox_for_era5(ds: xr.Dataset, bbox: Dict[str, float], logger: logging.Logger) -> Dict[str, float]:
    """
    Prepare and normalize bounding box for ERA5 data extraction.

    Ensures the bounding box covers at least one ERA5 grid cell by expanding it if
    necessary. Also normalizes longitude coordinates to 0-360 range and determines
    if longitude wrapping across the antimeridian is required.

    Process:
        1. Convert bbox dict to BoundingBox object
        2. Calculate ERA5 grid resolution from dataset coordinates
        3. Compare bbox extent to grid resolution
        4. If bbox smaller than one grid cell, expand it symmetrically
        5. Normalize longitude to 0-360 range
        6. Determine if longitude wrapping is needed (crosses antimeridian)
        7. Check if latitude coordinates are descending

    Args:
        ds: ERA5 dataset with latitude/longitude coordinates
        bbox: Dictionary with keys: lat_min, lat_max, lon_min, lon_max
        logger: Logger for warnings about bbox expansion

    Returns:
        Dictionary containing:
            - lat_min: Minimum latitude (possibly expanded)
            - lat_max: Maximum latitude (possibly expanded)
            - lon_min: Normalized minimum longitude (0-360)
            - lon_max: Normalized maximum longitude (0-360)
            - wrap_longitude: Boolean indicating if lon crosses antimeridian
            - lat_descending: Boolean indicating if lat coordinates descend
            - lon_min_value: Minimum longitude value in dataset
            - lon_max_value: Maximum longitude value in dataset
            - lat_resolution: Median latitude resolution (degrees)
            - lon_resolution: Median longitude resolution (degrees)

    Note:
        - ERA5 native resolution is typically 0.25° (~30km at equator)
        - Bbox expansion ensures at least one grid cell is captured
        - Expansion is symmetric and respects -90/90 latitude bounds
        - Warns user when expansion occurs

    Example:
        >>> # Small bbox (0.1° x 0.1°) with ERA5 0.25° resolution
        >>> bbox = {'lat_min': 50.0, 'lat_max': 50.1, 'lon_min': -115.0, 'lon_max': -114.9}
        >>> result = _prepare_bbox_for_era5(era5_ds, bbox, logger)
        >>> # Bbox expanded to ensure coverage of at least one 0.25° grid cell
        >>> result['lat_min']  # ~49.925 (expanded)
        >>> result['lon_min']  # 245.0 (normalized to 0-360)
    """
    # Validate bounding box before processing
    validate_bounding_box(bbox, context="ERA5 data acquisition", logger=logger)

    bbox_obj = BoundingBox(
        lat_min=float(bbox["lat_min"]),
        lat_max=float(bbox["lat_max"]),
        lon_min=float(bbox["lon_min"]),
        lon_max=float(bbox["lon_max"]),
    )

    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    lat_descending = len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]

    lat_res = float(np.median(np.abs(np.diff(lat_vals)))) if len(lat_vals) > 1 else 0.0
    lon_res = float(np.median(np.abs(np.diff(lon_vals)))) if len(lon_vals) > 1 else 0.0
    lat_extent, lon_extent = get_bbox_extent(bbox_obj.to_dict())

    lat_buffer = max(lat_res - lat_extent, 0.0) / 2.0
    lon_buffer = max(lon_res - lon_extent, 0.0) / 2.0
    if lat_buffer > 0 or lon_buffer > 0:
        expanded_lat_min = max(-90.0, bbox_obj.lat_min - lat_buffer)
        expanded_lat_max = min(90.0, bbox_obj.lat_max + lat_buffer)
        expanded_bbox = BoundingBox(
            lat_min=expanded_lat_min,
            lat_max=expanded_lat_max,
            lon_min=bbox_obj.lon_min - lon_buffer,
            lon_max=bbox_obj.lon_max + lon_buffer,
        )
        logger.warning(
            "Bounding box is smaller than ERA5 grid resolution "
            f"({lat_res:.2f}° lat, {lon_res:.2f}° lon). "
            f"Expanding from {bbox_obj.to_dict()} to {expanded_bbox.to_dict()} to include at least one grid cell."
        )
        bbox_obj = expanded_bbox

    bbox_norm = bbox_obj.normalize_longitude('0-360')
    wrap_longitude = bbox_norm.lon_min > bbox_norm.lon_max

    return {
        "lat_min": bbox_obj.lat_min,
        "lat_max": bbox_obj.lat_max,
        "lon_min": bbox_norm.lon_min,
        "lon_max": bbox_norm.lon_max,
        "wrap_longitude": wrap_longitude,
        "lat_descending": lat_descending,
        "lon_min_value": float(lon_vals.min()),
        "lon_max_value": float(lon_vals.max()),
        "lat_resolution": lat_res,
        "lon_resolution": lon_res,
    }


def _subset_era5_bbox(
    ds: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    wrap_lon: bool,
    lat_descending: bool,
    lon_range: Tuple[float, float],
) -> xr.Dataset:
    """
    Spatially subset ERA5 dataset by bounding box with antimeridian handling.

    Performs spatial subsetting of ERA5 data accounting for:
    - Descending vs. ascending latitude coordinates
    - Longitude wrapping across the antimeridian (180°/-180°)
    - Proper slice ordering based on coordinate direction

    Args:
        ds: ERA5 dataset to subset
        lat_min: Minimum latitude for subsetting
        lat_max: Maximum latitude for subsetting
        lon_min: Minimum longitude (normalized to 0-360)
        lon_max: Maximum longitude (normalized to 0-360)
        wrap_lon: If True, longitude wraps across antimeridian (lon_min > lon_max)
        lat_descending: If True, latitude coordinates are in descending order
        lon_range: Tuple of (min_lon_in_dataset, max_lon_in_dataset)

    Returns:
        Spatially subset xarray Dataset

    Note:
        - Handles antimeridian crossing by selecting two longitude ranges and concatenating
        - For wrap_lon=True: selects [lon_min:360] and [0:lon_max], then concatenates
        - Latitude slicing depends on coordinate order (descending vs ascending)
        - Returns dataset with only the spatial subset, time dimension preserved

    Example:
        >>> # Domain crossing antimeridian (170°E to -170°W = 170° to 190° in 0-360)
        >>> # With wrap_lon=True, selects 170-360 and 0-190
        >>> ds_subset = _subset_era5_bbox(
        ...     ds, lat_min=50, lat_max=60,
        ...     lon_min=170, lon_max=190, wrap_lon=True,
        ...     lat_descending=True, lon_range=(0, 360)
        ... )
    """
    lon_min_value, lon_max_value = lon_range
    lat_slice = slice(lat_max, lat_min) if lat_descending else slice(lat_min, lat_max)
    ds_lat = ds.sel(latitude=lat_slice)

    if wrap_lon and lon_min > lon_max:
        ds_left = ds_lat.sel(longitude=slice(lon_min, lon_max_value))
        ds_right = ds_lat.sel(longitude=slice(lon_min_value, lon_max))
        ds_ts = xr.concat([ds_left, ds_right], dim="longitude")
        ds_ts = ds_ts.sortby("longitude")
    else:
        ds_ts = ds_lat.sel(longitude=slice(lon_min, lon_max))

    return ds_ts
