# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data Preprocessing utilities for SYMFLUENCE.

This module provides:
- ForcingResampler: Orchestrates forcing data remapping
- ShapefileManager: CRS alignment and HRU ID management
- ElevationCalculator: DEM-based elevation statistics
- RemappingWeightGenerator/Applier: EASYMORE weight operations
- GeospatialStatistics: Catchment attribute computation
"""

import logging as _logging
from typing import Any

_logger = _logging.getLogger(__name__)

# Fail-safe imports to allow partial loading
try:
    from .forcing_resampler import ForcingResampler
except ImportError as _e:
    ForcingResampler: Any = None  # type: ignore[no-redef]
    _logger.warning("Failed to import ForcingResampler: %s", _e)

try:
    from .geospatial_statistics import GeospatialStatistics
except ImportError as _e:
    GeospatialStatistics: Any = None  # type: ignore[no-redef]
    _logger.warning("Failed to import GeospatialStatistics: %s", _e)

try:
    from .shapefile_manager import ShapefileManager
except ImportError as _e:
    ShapefileManager: Any = None  # type: ignore[no-redef]
    _logger.warning("Failed to import ShapefileManager: %s", _e)

try:
    from .elevation_calculator import ElevationCalculator, create_elevation_calculator
except ImportError as _e:
    ElevationCalculator: Any = None  # type: ignore[no-redef]
    create_elevation_calculator: Any = None  # type: ignore[no-redef]
    _logger.warning("Failed to import elevation_calculator: %s", _e)

try:
    from .remapping_weights import (
        BatchProcessor,
        RemappingWeightApplier,
        RemappingWeightGenerator,
    )
except ImportError as _e:
    RemappingWeightGenerator: Any = None  # type: ignore[no-redef]
    RemappingWeightApplier: Any = None  # type: ignore[no-redef]
    BatchProcessor: Any = None  # type: ignore[no-redef]
    _logger.warning("Failed to import remapping_weights: %s", _e)

try:
    from .elevation_correction import ElevationCorrectionProcessor
except ImportError as _e:
    ElevationCorrectionProcessor: Any = None  # type: ignore[no-redef]
    _logger.debug("ElevationCorrectionProcessor not available: %s", _e)

__all__ = [
    'ForcingResampler',
    'GeospatialStatistics',
    'ShapefileManager',
    'ElevationCalculator',
    'create_elevation_calculator',
    'RemappingWeightGenerator',
    'RemappingWeightApplier',
    'BatchProcessor',
    'ElevationCorrectionProcessor',
]
