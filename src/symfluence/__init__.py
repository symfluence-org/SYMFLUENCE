# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE: SYnergistic Modelling Framework for Linking and Unifying
Earth-system Nexii for Computational Exploration.

A computational environmental modeling platform that streamlines the
hydrological modeling workflow from domain setup to evaluation. Provides
an integrated framework for multi-model comparison, parameter optimization,
and automated workflow management.

Main entry points:
    SYMFLUENCE: Main workflow orchestrator class
    SymfluenceConfig: Configuration management for workflows

Example:
    >>> from symfluence import SYMFLUENCE, SymfluenceConfig
    >>> config = SymfluenceConfig.from_file('config.yaml')
    >>> workflow = SYMFLUENCE(config)
    >>> workflow.run()

For CLI usage:
    $ symfluence workflow run --config config.yaml
    $ symfluence --help
"""
# src/symfluence/__init__.py

# ============================================================================
# CRITICAL: Windows conda DLL path fix
# Must run BEFORE any C-extension imports (netCDF4, GDAL, HDF5, etc.).
# Python 3.8+ on Windows no longer uses PATH for DLL search; we must
# explicitly register conda's Library\bin via os.add_dll_directory().
# Also sets GDAL_DATA if a conda environment is detected.
# ============================================================================
import os as _os
import sys as _sys

# Must be kept alive at module level — the DLL directory is removed if the
# handle returned by os.add_dll_directory() is garbage-collected.
_dll_directory_handles = []

if _sys.platform == "win32":
    _conda_prefix = _os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        _library_bin = _os.path.join(_conda_prefix, "Library", "bin")
        if _os.path.isdir(_library_bin):
            _dll_directory_handles.append(_os.add_dll_directory(_library_bin))

        _gdal_data = _os.path.join(_conda_prefix, "Library", "share", "gdal")
        if _os.path.isdir(_gdal_data):
            _os.environ.setdefault("GDAL_DATA", _gdal_data)

        _proj_data = _os.path.join(_conda_prefix, "Library", "share", "proj")
        if _os.path.isdir(_proj_data):
            _os.environ.setdefault("PROJ_DATA", _proj_data)

    # Prevent OpenMP duplicate-library abort when pip-torch and conda
    # co-exist (each ships its own OpenMP runtime).
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Register torch's own DLL directory so its bundled copies of
    # uv.dll / libiomp5md.dll are found before conda's versions.
    try:
        import importlib.util as _ilu
        _torch_spec = _ilu.find_spec("torch")
        if _torch_spec and _torch_spec.submodule_search_locations:
            _torch_lib = _os.path.join(
                list(_torch_spec.submodule_search_locations)[0], "lib"
            )
            if _os.path.isdir(_torch_lib):
                _dll_directory_handles.append(_os.add_dll_directory(_torch_lib))
    except Exception:  # noqa: BLE001 — top-level fallback
        pass

    # PyTorch must be imported BEFORE conda's HDF5/netCDF4 libraries.
    # Conda's h5py loads HDF5 DLLs that make torch's shm.dll unloadable
    # if torch hasn't already loaded its own DLLs first.
    try:
        import torch as _torch  # noqa: F401
        del _torch
    except (ImportError, OSError):
        pass

del _os, _sys

# ============================================================================
# CRITICAL: HDF5/netCDF4 thread safety fix
# Must be set BEFORE any HDF5/netCDF4/xarray imports occur.
# The netCDF4/HDF5 libraries are not thread-safe by default, and tqdm's
# background monitor thread can cause segmentation faults when running
# concurrently with netCDF file operations (e.g., in easymore remapping).
# ============================================================================
from symfluence.core.hdf5_safety import configure_hdf5_safety

configure_hdf5_safety(disable_tqdm_monitor=True)

import logging
import os
import warnings

try:
    from .symfluence_version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("symfluence")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"

# Expose core components for a cleaner API
from .core import SYMFLUENCE
from .core.config.models import SymfluenceConfig
from .core.exceptions import (
    ConfigurationError,
    ConfigValidationError,
    DataAcquisitionError,
    EvaluationError,
    FileOperationError,
    GeospatialError,
    ModelExecutionError,
    OptimizationError,
    ReportingError,
    SYMFLUENCEError,
    ValidationError,
)

__all__ = [
    "SYMFLUENCE",
    "SymfluenceConfig",
    "SYMFLUENCEError",
    "ConfigurationError",
    "ConfigValidationError",
    "ModelExecutionError",
    "DataAcquisitionError",
    "OptimizationError",
    "GeospatialError",
    "ValidationError",
    "FileOperationError",
    "EvaluationError",
    "ReportingError",
    "__version__",
]

# Suppress overly verbose external logging/warnings
rpy2_logger = logging.getLogger("rpy2.rinterface_lib.embedded")
rpy2_logger.setLevel(logging.WARNING)
rpy2_logger.addHandler(logging.NullHandler())
rpy2_logger.propagate = False

warnings.filterwarnings(
    "ignore",
    message="(?s).*Conversion of an array with ndim > 0 to a scalar is deprecated.*",
    category=DeprecationWarning,
)

os.environ.setdefault(
    "PYTHONWARNINGS",
    r"ignore:Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\.:UserWarning",
)

warnings.filterwarnings(
    "ignore",
    message=r"Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\.",
    category=UserWarning,
)

try:
    import pyproj

    _orig_transform = pyproj.transformer.Transformer.transform

    def _warnless_transform(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="(?s).*Conversion of an array with ndim > 0 to a scalar is deprecated.*",
                category=DeprecationWarning,
            )
            return _orig_transform(self, *args, **kwargs)

    pyproj.transformer.Transformer.transform = _warnless_transform
except ImportError:
    pass

# ============================================================================
# Unified registry bootstrap
# Populate static registrations (delineation aliases, BMI lazy imports, etc.)
# before model modules are imported.
# ============================================================================
from symfluence.core._bootstrap import bootstrap as _bootstrap_registries

_bootstrap_registries()
