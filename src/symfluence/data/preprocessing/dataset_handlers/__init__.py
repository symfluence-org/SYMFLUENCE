# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Dataset Handlers for SYMFLUENCE

This package provides dataset-specific handlers for different forcing datasets.
Each handler encapsulates all dataset-specific logic including variable mappings,
unit conversions, grid specifications, and shapefile creation.

Available Handlers:
    - RDRSHandler: Regional Deterministic Reforecast System
    - CASRHandler: Canadian Arctic System Reanalysis
    - ERA5Handler: ECMWF Reanalysis v5
    - CARRAHandler: Copernicus Arctic Regional Reanalysis
    - CERRAHandler: Copernicus European Regional Reanalysis
    - AORCHandler: NOAA Analysis of Record for Calibration
    - CONUS404Handler: NCAR/USGS CONUS404 WRF reanalysis
    - NEXGDDPCMIP6Handler: NASA NEX-GDDP-CMIP6 downscaled climate data
    - HRRRHandler: High-Resolution Rapid Refresh
    - DaymetHandler: Daymet daily surface weather (ORNL DAAC)

Usage:
    from dataset_handlers import DatasetRegistry

    # Get the appropriate handler for a dataset
    handler = DatasetRegistry.get_handler('era5', config, logger, project_dir)

    # Use the handler
    handler.merge_forcings(...)
    handler.create_shapefile(...)
"""

import importlib as _importlib
import logging as _logging
import warnings as _warnings

from .base_dataset import (
    STANDARD_VARIABLE_ATTRIBUTES,
    BaseDatasetHandler,
    apply_standard_variable_attributes,
)
from .dataset_registry import DatasetRegistry

_logger = _logging.getLogger(__name__)

# Import handlers with fail-safe pattern to allow partial loading
# This helps diagnose which specific handler has import issues

_handler_imports = [
    ('rdrs_utils', 'RDRSHandler'),
    ('casr_utils', 'CASRHandler'),
    ('era5_utils', 'ERA5Handler'),
    ('carra_utils', 'CARRAHandler'),
    ('cerra_utils', 'CERRAHandler'),
    ('aorc_utils', 'AORCHandler'),
    ('conus404_utils', 'CONUS404Handler'),
    ('nex_gddp_utils', 'NEXGDDPCMIP6Handler'),
    ('hrrr_utils', 'HRRRHandler'),
    ('daymet_utils', 'DaymetHandler'),
    ('nwm3_retrospective_utils', 'NWM3RetrospectiveHandler'),
]

# Track successfully imported handlers
_loaded_handlers = []
_failed_handlers = []

for _module_name, _handler_name in _handler_imports:
    try:
        _module = _importlib.import_module(f'.{_module_name}', __name__)
        _handler_class = getattr(_module, _handler_name)
        globals()[_handler_name] = _handler_class
        _loaded_handlers.append(_handler_name)
    except Exception as _e:  # noqa: BLE001 — optional dependency
        _failed_handlers.append((_handler_name, str(_e)))
        globals()[_handler_name] = None
        _logger.warning("Failed to import %s from %s: %s", _handler_name, _module_name, _e)

# Clean up temporary variables
del _handler_imports
try:
    del _module_name, _handler_name, _module, _handler_class
except NameError:
    pass

if _failed_handlers:
    _warnings.warn(
        f"Some dataset handlers failed to import: {[h[0] for h in _failed_handlers]}. "
        f"See stderr for details.",
        ImportWarning
    )

__all__ = [
    "BaseDatasetHandler",
    "STANDARD_VARIABLE_ATTRIBUTES",
    "apply_standard_variable_attributes",
    "DatasetRegistry",
    "RDRSHandler",
    "CASRHandler",
    "ERA5Handler",
    "CARRAHandler",
    "CERRAHandler",
    "AORCHandler",
    "CONUS404Handler",
    "NEXGDDPCMIP6Handler",
    "HRRRHandler",
    "DaymetHandler",
    "NWM3RetrospectiveHandler",
]

__version__ = "1.0.1"
