# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Centralized HDF5/netCDF4 thread safety configuration.

This module provides a single source of truth for HDF5 file locking workarounds
and thread safety settings. Previously, this logic was duplicated across 8+ files
with inconsistent implementations.

The HDF5 library is not thread-safe by default, and concurrent access from
background threads (like tqdm's monitor) can cause segmentation faults during
netCDF/HDF5 file operations.

Usage:
    # At application startup (done automatically in symfluence/__init__.py)
    from symfluence.core.hdf5_safety import configure_hdf5_safety
    configure_hdf5_safety()

    # In worker processes
    from symfluence.core.hdf5_safety import apply_worker_environment
    apply_worker_environment()

    # Get environment dict for subprocess
    from symfluence.core.hdf5_safety import get_worker_environment
    env = get_worker_environment()
    subprocess.run(cmd, env=env)
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Environment Variable Constants
# =============================================================================

HDF5_ENV_VARS: Dict[str, str] = {
    'HDF5_USE_FILE_LOCKING': 'FALSE',
    'HDF5_DISABLE_VERSION_CHECK': '1',
    'NETCDF_DISABLE_LOCKING': '1',
}
"""Environment variables for HDF5/netCDF file locking safety."""


THREAD_ENV_VARS: Dict[str, str] = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',  # Prevent OpenMP conflicts on macOS
}
"""Environment variables to force single-threaded execution in numerical libraries."""

# Module-level flag: set to True when h5py and netCDF4 link against different
# libhdf5 builds.  When True, the h5netcdf engine fallback in
# BaseDatasetHandler.open_dataset() must be skipped because importing h5py
# would corrupt netcdf4's HDF5 state for the rest of the process.
hdf5_library_conflict: bool = False


# =============================================================================
# Configuration Functions
# =============================================================================

def configure_hdf5_safety(disable_tqdm_monitor: bool = True) -> None:
    """
    Configure HDF5/netCDF4 thread safety at application startup.

    This function must be called BEFORE any HDF5/netCDF4/xarray imports occur.
    It is automatically called by symfluence/__init__.py.

    Args:
        disable_tqdm_monitor: If True, disable tqdm's background monitor thread
                              which can cause segfaults with netCDF4/HDF5.
    """
    # Set ALL HDF5 and threading environment variables
    # These must be set BEFORE importing libraries to prevent thread pool creation.
    # HDF5 locking vars are FORCE-SET (not setdefault) because on HPC systems
    # the run session often differs from the install session, and the HDF5 C
    # library reads HDF5_USE_FILE_LOCKING at dlopen time.  If the var is unset
    # (or the module environment sets it to something else), file-locking errors
    # on parallel filesystems (Lustre/GPFS/BeeGFS) are almost guaranteed.
    env_vars = get_worker_environment(include_thread_limits=True)
    for key, value in env_vars.items():
        if key in HDF5_ENV_VARS:
            # Force-set HDF5/netCDF locking vars — these are critical on HPC
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    # Configure xarray to minimize file caching
    try:
        import xarray as xr
        xr.set_options(
            file_cache_maxsize=1,  # Minimal file cache
        )
    except (ImportError, AttributeError):
        pass  # xarray not yet imported or doesn't support this option

    # Check for conflicting HDF5 library builds (e.g. pip h5py + pip netCDF4
    # each bundling their own libhdf5).  Must run BEFORE any h5py/h5netcdf
    # import to avoid poisoning the process.
    _check_hdf5_library_conflict()

    # Disable tqdm monitor thread to prevent segfaults
    if disable_tqdm_monitor:
        _disable_tqdm_monitor()


def get_worker_environment(include_thread_limits: bool = True) -> Dict[str, str]:
    """
    Get environment variables for worker processes.

    Returns a dictionary of environment variables that should be set in
    worker processes to ensure HDF5/netCDF safety and single-threaded
    execution of numerical libraries.

    Args:
        include_thread_limits: If True, include variables that limit threading
                               in numerical libraries (OMP, MKL, etc.)

    Returns:
        Dictionary of environment variables to set
    """
    env_vars = HDF5_ENV_VARS.copy()
    if include_thread_limits:
        env_vars.update(THREAD_ENV_VARS)
    return env_vars


def apply_worker_environment() -> None:
    """
    Apply worker environment variables to the current process.

    Call this at the start of worker processes to ensure HDF5 safety
    and single-threaded execution.
    """
    env_vars = get_worker_environment(include_thread_limits=True)
    os.environ.update(env_vars)


def merge_with_current_env(include_thread_limits: bool = True) -> Dict[str, str]:
    """
    Create a copy of the current environment merged with worker settings.

    Useful for subprocess execution where you need the full environment.

    Args:
        include_thread_limits: If True, include thread limiting variables

    Returns:
        Complete environment dictionary for subprocess execution
    """
    env = os.environ.copy()
    env.update(get_worker_environment(include_thread_limits))
    return env


def clear_xarray_cache() -> None:
    """
    Clear xarray's file manager cache to prevent stale file handles.

    This should be called after intensive file operations, especially
    in worker processes that may have residual file handles from
    previous iterations.
    """
    try:
        import xarray as xr

        # Try different cache clearing approaches for various xarray versions
        # xarray >= 0.19: CachingFileManager uses a module-level cache
        if hasattr(xr.backends, 'file_manager'):
            fm = xr.backends.file_manager
            # Old style cache (xarray < 2022)
            if hasattr(fm, 'FILE_CACHE'):
                fm.FILE_CACHE.clear()
            # Try to find CachingFileManager's cache
            if hasattr(fm, 'CachingFileManager'):
                cfm = fm.CachingFileManager
                cache = getattr(cfm, '_cache', None)
                if cache is not None:
                    try:
                        cache.clear()
                    except (TypeError, AttributeError):
                        pass

        # Also try closing any open file handles via the backends
        for backend_name in ['netcdf4', 'h5netcdf', 'scipy']:
            try:
                backend_mod = getattr(xr.backends, f'{backend_name}_', None)
                if backend_mod and hasattr(backend_mod, '_clear_cache'):
                    backend_mod._clear_cache()
            except (AttributeError, TypeError):
                pass

    except (ImportError, AttributeError):
        pass  # xarray internals may vary by version

    # Also try to clear netCDF4's internal cache if available
    try:
        import netCDF4 as nc4
        # netCDF4 doesn't have a public cache API, but we can trigger cleanup
        if hasattr(nc4, '_clear_cache'):
            nc4._clear_cache()
    except (ImportError, AttributeError):
        pass

    # Force garbage collection to release file handles
    gc.collect()
    gc.collect()  # Second pass for cyclic references


# =============================================================================
# Internal Helpers
# =============================================================================

def _find_bundled_libhdf5(pkg_name: str) -> Optional[Path]:
    """Find a bundled libhdf5 shipped inside a pip wheel for *pkg_name*.

    Pip wheels for h5py and netCDF4 vendor their own private copy of
    libhdf5 into a ``<pkg>.libs/`` (Linux) or ``<pkg>/.dylibs/``
    (macOS) directory next to the package.  Conda-installed packages
    instead link against a single shared ``$CONDA_PREFIX/lib/libhdf5``
    and have no bundled copy.

    Returns the resolved path to the bundled libhdf5, or None if the
    package is not installed or does not bundle its own copy.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or spec.origin is None:
            return None
        pkg_dir = Path(spec.origin).parent

        # Linux pip wheels: <site-packages>/<pkg>.libs/libhdf5-*.so*
        # macOS pip wheels: <site-packages>/<pkg>/.dylibs/libhdf5*.dylib
        search_dirs = [
            pkg_dir.parent / f"{pkg_name}.libs",      # Linux
            pkg_dir / ".dylibs",                       # macOS (inside pkg)
        ]
        # netCDF4 pip wheel uses the distribution name "netcdf4" (lowercase)
        if pkg_name == "netCDF4":
            search_dirs.append(pkg_dir.parent / "netcdf4.libs")

        for d in search_dirs:
            if not d.is_dir():
                continue
            for f in d.iterdir():
                name = f.name.lower()
                if 'libhdf5' in name and 'libhdf5_hl' not in name:
                    return f.resolve()
    except Exception:  # noqa: BLE001
        pass
    return None


def _check_hdf5_library_conflict() -> None:
    """Detect conflicting libhdf5 builds between h5py and netCDF4.

    When both packages are pip-installed, each bundles its own private
    copy of ``libhdf5`` (with different build hashes).  If h5py is
    imported first, its libhdf5 symbols shadow netCDF4's, corrupting
    HDF5 global state and causing ``NetCDF: HDF error`` on every
    subsequent netcdf4 operation — even reads.

    This check runs at startup (before any h5py import) by looking for
    bundled libhdf5 files in pip wheel vendor directories — no Python
    HDF5 modules are imported.

    Sets the module-level ``hdf5_library_conflict`` flag and logs a
    warning with clear fix instructions.
    """
    global hdf5_library_conflict

    h5py_lib = _find_bundled_libhdf5('h5py')
    nc4_lib = _find_bundled_libhdf5('netCDF4')

    if h5py_lib is None or nc4_lib is None:
        # At least one package doesn't bundle its own libhdf5 (likely
        # conda-installed or not present at all).  No conflict.
        return

    if h5py_lib == nc4_lib:
        return  # Same file — no conflict

    hdf5_library_conflict = True
    logger.warning(
        "h5py and netCDF4 bundle DIFFERENT libhdf5 builds:\n"
        "  h5py    → %s\n"
        "  netCDF4 → %s\n"
        "Importing h5py will corrupt netcdf4's HDF5 state, causing "
        "'NetCDF: HDF error' on all subsequent operations.\n"
        "Fix: install both from the same package manager:\n"
        "  conda:  pip uninstall h5py netCDF4 -y && conda install h5py netcdf4\n"
        "  pip:    pip install --force-reinstall --no-binary h5py --no-binary "
        "netCDF4 h5py netCDF4\n"
        "The h5netcdf engine fallback will be DISABLED for this session to "
        "prevent the conflict from being triggered.",
        h5py_lib,
        nc4_lib,
    )


def _disable_tqdm_monitor() -> None:
    """Disable tqdm's background monitor thread."""
    try:
        import tqdm
        tqdm.tqdm.monitor_interval = 0
        if tqdm.tqdm.monitor is not None:
            try:
                tqdm.tqdm.monitor.exit()
            except (AttributeError, RuntimeError):
                pass  # Monitor may already be stopped
            tqdm.tqdm.monitor = None
    except ImportError:
        pass


def ensure_thread_safety() -> None:
    """
    Ensure thread-safe environment for netCDF4/HDF5 operations.

    This is a convenience function that combines environment setup
    with cache clearing. Call before intensive HDF5 operations.

    Note:
        This is equivalent to calling:
        - apply_worker_environment()
        - _disable_tqdm_monitor()
        - clear_xarray_cache()
    """
    apply_worker_environment()
    _disable_tqdm_monitor()
    clear_xarray_cache()


def prepare_for_netcdf_operation() -> None:
    """
    Aggressive preparation before intensive netCDF/HDF5 operations.

    This function should be called before operations that are known to
    cause issues with HDF5 thread safety (e.g., easymore remapping).
    It performs a more thorough cleanup than ensure_thread_safety().

    This function:
    1. Sets all HDF5/netCDF safety environment variables
    2. Disables tqdm's monitor thread
    3. Clears xarray's file cache
    4. Forces garbage collection of netCDF4 Dataset objects
    5. Attempts to close any lingering file handles
    6. Disables xarray's file caching to prevent cache-related segfaults
    """
    # Apply all environment variables
    apply_worker_environment()

    # Disable tqdm monitor
    _disable_tqdm_monitor()

    # Disable xarray's file caching to prevent cache-related segfaults
    try:
        import xarray as xr
        # Set cache size to 0 to disable caching
        if hasattr(xr.backends, 'file_manager'):
            xr.set_options(file_cache_maxsize=1)  # Minimal cache
    except (ImportError, AttributeError):
        pass

    # First GC pass to find unreferenced Dataset objects
    gc.collect()

    # Try to close any open netCDF4 datasets
    try:
        import netCDF4 as nc4
        # netCDF4 tracks open datasets internally in some versions
        if hasattr(nc4, '_active_datasets'):
            for ds in list(nc4._active_datasets):
                try:
                    ds.close()
                except Exception:  # noqa: BLE001 — must-not-raise contract
                    pass
    except (ImportError, AttributeError):
        pass

    # Clear xarray cache
    clear_xarray_cache()

    # Additional GC passes for thorough cleanup
    gc.collect()
    gc.collect()
