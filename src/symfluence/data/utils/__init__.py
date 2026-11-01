# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data utilities for SYMFLUENCE.

Provides common utilities for data processing:
- Spatial operations (cropping, subsetting, masking)
- Variable standardization and unit conversion
- Archive utilities
"""
from __future__ import annotations

import logging as _logging
from typing import Any

_logger = _logging.getLogger(__name__)

# Fail-safe imports to prevent package loading failures in CI
# If an import fails, we set None and log the error

# Spatial utilities
try:
    from .spatial_utils import (
        BBox,
        SpatialSubsetMixin,
        create_spatial_mask,
        crop_raster_to_bbox,
        find_nearest_grid_point,
        get_bbox_center,
        normalize_longitude,
        read_raster_multiband_window,
        read_raster_window,
        subset_xarray_to_bbox,
        validate_bbox,
    )
except ImportError as e:
    _logger.warning("Failed to import spatial_utils: %s", e)
    crop_raster_to_bbox: Any = None  # type: ignore[no-redef]
    read_raster_window: Any = None  # type: ignore[no-redef]
    read_raster_multiband_window: Any = None  # type: ignore[no-redef]
    create_spatial_mask: Any = None  # type: ignore[no-redef]
    find_nearest_grid_point: Any = None  # type: ignore[no-redef]
    get_bbox_center: Any = None  # type: ignore[no-redef]
    subset_xarray_to_bbox: Any = None  # type: ignore[no-redef]
    normalize_longitude: Any = None  # type: ignore[no-redef]
    validate_bbox: Any = None  # type: ignore[no-redef]
    SpatialSubsetMixin: Any = None  # type: ignore[no-redef]
    BBox: Any = None  # type: ignore[no-redef]

# Variable utilities
try:
    from symfluence.core.modeling.variable_utils import VariableHandler, VariableStandardizer
except ImportError as e:
    _logger.warning("Failed to import variable_utils: %s", e)
    VariableHandler: Any = None  # type: ignore[no-redef]
    VariableStandardizer: Any = None  # type: ignore[no-redef]

__all__ = [
    # Spatial utilities
    'crop_raster_to_bbox',
    'read_raster_window',
    'read_raster_multiband_window',
    'create_spatial_mask',
    'find_nearest_grid_point',
    'get_bbox_center',
    'subset_xarray_to_bbox',
    'normalize_longitude',
    'validate_bbox',
    'SpatialSubsetMixin',
    'BBox',
    # Variable utilities
    'VariableHandler',
    'VariableStandardizer',
]
