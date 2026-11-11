# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data processors for reporting and visualization.

This module provides processors for preparing data for visualization,
calculating metrics, and handling spatial data processing.
"""
from __future__ import annotations

from .data_processor import DataProcessor
from .spatial_processor import SpatialProcessor

__all__ = [
    "DataProcessor",
    "SpatialProcessor",
]
