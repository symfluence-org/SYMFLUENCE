# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Core reporting utilities.

This module provides shared utilities for the reporting subsystem.
"""
from __future__ import annotations

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.core.decorators import skip_if_not_visualizing
from symfluence.reporting.core.plot_utils import (
    align_timeseries,
    calculate_flow_duration_curve,
    calculate_metrics,
)
from symfluence.reporting.core.shapefile_helper import (
    ShapefileHelper,
    resolve_default_name,
)

__all__ = [
    # Base plotter
    'BasePlotter',
    # Plot utilities
    'calculate_metrics',
    'calculate_flow_duration_curve',
    'align_timeseries',
    # Shapefile helper
    'ShapefileHelper',
    'resolve_default_name',
    # Decorators
    'skip_if_not_visualizing',
]
