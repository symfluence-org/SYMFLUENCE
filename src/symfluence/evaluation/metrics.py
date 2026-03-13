#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Hydrological performance metrics public API.

This module remains the stable import surface for legacy callers while the
implementation is split into focused submodules:
- metrics_core: core metric calculations and preprocessing helpers
- metrics_hydrograph: hydrograph signature metrics
- metrics_registry: metric metadata/lookup/interpretation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from symfluence.evaluation.metrics_core import (
    _apply_transformation,
    _clean_data,
    bias,
    calculate_all_metrics,
    correlation,
    kge,
    kge_np,
    kge_prime,
    log_nse,
    mae,
    mare,
    nrmse,
    nse,
    pbias,
    r_squared,
    rmse,
    volumetric_efficiency,
)
from symfluence.evaluation.metrics_hydrograph import (
    baseflow_index,
    flow_duration_curve_metrics,
    hydrograph_signatures,
    peak_timing_error,
    recession_constant,
)
from symfluence.evaluation.metrics_registry import (
    METRIC_REGISTRY,
    canonicalize_metric_name,
    get_metric_function,
    get_metric_info,
    interpret_metric,
    list_available_metrics,
)
from symfluence.evaluation.metrics_types import MetricInfo

__all__ = [
    # Core metric functions
    "nse",
    "log_nse",
    "kge",
    "kge_prime",
    "kge_np",
    "rmse",
    "nrmse",
    "mae",
    "mare",
    "bias",
    "pbias",
    "correlation",
    "r_squared",
    "volumetric_efficiency",
    # Hydrograph signature metrics
    "peak_timing_error",
    "recession_constant",
    "baseflow_index",
    "flow_duration_curve_metrics",
    "hydrograph_signatures",
    # Convenience functions
    "calculate_all_metrics",
    "calculate_metrics",
    "canonicalize_metric_name",
    "get_metric_function",
    "get_metric_info",
    "list_available_metrics",
    # Registry and metadata
    "METRIC_REGISTRY",
    "MetricInfo",
    # Helper functions
    "interpret_metric",
    # Semi-private helpers used by tests and legacy callers
    "_clean_data",
    "_apply_transformation",
]


def calculate_metrics(
    observed: Union[np.ndarray, pd.Series],
    simulated: Union[np.ndarray, pd.Series],
    metrics: Optional[List[str]] = None,
    transfo: float = 1.0,
) -> Dict[str, float]:
    """Calculate selected performance metrics.

    If ``metrics`` is omitted, all standard metrics are calculated.
    """
    if metrics is None:
        return calculate_all_metrics(observed, simulated, transfo)

    result: Dict[str, float] = {}
    for metric_name in metrics:
        canonical = canonicalize_metric_name(metric_name)
        func = get_metric_function(canonical)
        if func is not None:
            try:
                if canonical in ("correlation", "R2", "logNSE", "MARE", "VE"):
                    result[canonical] = float(func(observed, simulated))
                else:
                    result[canonical] = float(func(observed, simulated, transfo))
            except TypeError:
                result[canonical] = float(func(observed, simulated))
        else:
            result[canonical] = np.nan

    return result
