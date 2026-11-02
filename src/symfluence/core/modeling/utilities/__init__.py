# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared utilities for model preprocessors.

Provides common functionality for time window management, forcing data
processing, data quality handling, and dataset alignment that is used
across multiple model preprocessors (SUMMA, FUSE, NGEN, GR, MESH).

Also includes RoutingDecider for unified routing decision logic across models.
"""
from __future__ import annotations

from .base_forcing_processor import BaseForcingProcessor
from .base_remap_generator import BaseRemapGenerator, RemapData
from .base_topology_generator import BaseTopologyGenerator, TopologyData
from .data_quality_handler import DataQualityHandler
from .dataset_alignment_manager import DatasetAlignmentManager, align_forcing_datasets
from .forcing_data_processor import ForcingDataProcessor
from .periods import parse_period, split_period, to_pydatetime
from .routing_decider import RoutingDecider
from .runoff_loader import (
    MODEL_CONFIGS,
    ModelRunoffConfig,
    detect_runoff_variable,
    fix_time_precision,
    resolve_runoff_file,
)
from .time_window_manager import TimeWindowManager

__all__ = [
    'ForcingDataProcessor',
    'DataQualityHandler',
    'DatasetAlignmentManager',
    'BaseForcingProcessor',
    'BaseRemapGenerator',
    'BaseTopologyGenerator',
    'MODEL_CONFIGS',
    'ModelRunoffConfig',
    'RemapData',
    'RoutingDecider',
    'TopologyData',
    'TimeWindowManager',
    'parse_period',
    'split_period',
    'to_pydatetime',
    'align_forcing_datasets',
    'detect_runoff_variable',
    'fix_time_precision',
    'resolve_runoff_file',
]
