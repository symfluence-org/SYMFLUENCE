# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data Preprocessing utilities for SYMFLUENCE.

This module provides:
- ForcingResampler: Orchestrates forcing data remapping
- RemappingWeightGenerator/Applier: EASYMORE weight operations
- GeospatialStatistics: Catchment attribute computation
"""
from __future__ import annotations

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
    # Canonical home of the EASYMORE weight classes (the old duplicate
    # ``remapping_weights`` module was dead code and has been removed).
    from .resampling import RemappingWeightApplier, RemappingWeightGenerator
except ImportError as _e:
    RemappingWeightGenerator: Any = None  # type: ignore[no-redef]
    RemappingWeightApplier: Any = None  # type: ignore[no-redef]
    _logger.warning("Failed to import resampling weight classes: %s", _e)

try:
    from .elevation_correction import ElevationCorrectionProcessor
except ImportError as _e:
    ElevationCorrectionProcessor: Any = None  # type: ignore[no-redef]
    _logger.debug("ElevationCorrectionProcessor not available: %s", _e)

__all__ = [
    'ForcingResampler',
    'GeospatialStatistics',
    'RemappingWeightGenerator',
    'RemappingWeightApplier',
    'ElevationCorrectionProcessor',
]
