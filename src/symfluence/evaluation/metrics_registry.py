# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Metric registry, lookup helpers, and interpretation utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union, cast

import numpy as np

from symfluence.evaluation.metrics_core import (
    bias,
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
from symfluence.evaluation.metrics_types import MetricInfo

METRIC_REGISTRY: Dict[str, Dict[str, Union[Callable, MetricInfo, None]]] = {
    "NSE": {
        "function": nse,
        "info": MetricInfo(
            name="NSE",
            full_name="Nash-Sutcliffe Efficiency",
            range=(float("-inf"), 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="Measures how well simulated values match observed variance",
            reference="Nash & Sutcliffe (1970)",
        ),
    },
    "logNSE": {
        "function": log_nse,
        "info": MetricInfo(
            name="logNSE",
            full_name="Log-transformed Nash-Sutcliffe Efficiency",
            range=(float("-inf"), 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="NSE on log-transformed values, emphasizes low flows",
            reference="Krause et al. (2005)",
        ),
    },
    "KGE": {
        "function": kge,
        "info": MetricInfo(
            name="KGE",
            full_name="Kling-Gupta Efficiency",
            range=(float("-inf"), 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="Decomposes NSE into correlation, variability, and bias",
            reference="Gupta et al. (2009)",
        ),
    },
    "KGEp": {
        "function": kge_prime,
        "info": MetricInfo(
            name="KGEp",
            full_name="Modified Kling-Gupta Efficiency",
            range=(float("-inf"), 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="KGE using coefficient of variation instead of std",
            reference="Kling et al. (2012)",
        ),
    },
    "KGEnp": {
        "function": kge_np,
        "info": MetricInfo(
            name="KGEnp",
            full_name="Non-parametric Kling-Gupta Efficiency",
            range=(float("-inf"), 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="KGE using Spearman correlation and flow duration curves",
            reference="Pool et al. (2018)",
        ),
    },
    "VE": {
        "function": volumetric_efficiency,
        "info": MetricInfo(
            name="VE",
            full_name="Volumetric Efficiency",
            range=(float("-inf"), 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="Fraction of water delivered at the proper time",
            reference="Criss & Winston (2008)",
        ),
    },
    "RMSE": {
        "function": rmse,
        "info": MetricInfo(
            name="RMSE",
            full_name="Root Mean Square Error",
            range=(0.0, float("inf")),
            optimal=0.0,
            direction="minimize",
            units="same as input",
            description="Average magnitude of simulation errors",
            reference="Standard",
        ),
    },
    "NRMSE": {
        "function": nrmse,
        "info": MetricInfo(
            name="NRMSE",
            full_name="Normalized Root Mean Square Error",
            range=(0.0, float("inf")),
            optimal=0.0,
            direction="minimize",
            units="dimensionless",
            description="RMSE normalized by observed standard deviation",
            reference="Standard",
        ),
    },
    "MAE": {
        "function": mae,
        "info": MetricInfo(
            name="MAE",
            full_name="Mean Absolute Error",
            range=(0.0, float("inf")),
            optimal=0.0,
            direction="minimize",
            units="same as input",
            description="Average absolute simulation error",
            reference="Standard",
        ),
    },
    "MARE": {
        "function": mare,
        "info": MetricInfo(
            name="MARE",
            full_name="Mean Absolute Relative Error",
            range=(0.0, float("inf")),
            optimal=0.0,
            direction="minimize",
            units="dimensionless",
            description="Average relative simulation error",
            reference="Standard",
        ),
    },
    "bias": {
        "function": bias,
        "info": MetricInfo(
            name="bias",
            full_name="Mean Error (Bias)",
            range=(float("-inf"), float("inf")),
            optimal=0.0,
            direction="minimize",
            units="same as input",
            description="Mean difference between simulated and observed",
            reference="Standard",
        ),
    },
    "PBIAS": {
        "function": pbias,
        "info": MetricInfo(
            name="PBIAS",
            full_name="Percent Bias",
            range=(float("-inf"), float("inf")),
            optimal=0.0,
            direction="minimize",
            units="percent",
            description="Percentage difference in total volumes",
            reference="Standard",
        ),
    },
    "correlation": {
        "function": correlation,
        "info": MetricInfo(
            name="correlation",
            full_name="Pearson Correlation Coefficient",
            range=(-1.0, 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="Linear correlation between observed and simulated",
            reference="Standard",
        ),
    },
    "R2": {
        "function": r_squared,
        "info": MetricInfo(
            name="R2",
            full_name="Coefficient of Determination",
            range=(0.0, 1.0),
            optimal=1.0,
            direction="maximize",
            units="dimensionless",
            description="Proportion of variance explained by the model",
            reference="Standard",
        ),
    },
}

METRIC_REGISTRY["log_likelihood"] = {
    "function": None,  # Computed in evaluator, not a standalone metric function
    "info": MetricInfo(
        name="log_likelihood",
        full_name="Gaussian Log-Likelihood",
        range=(float("-inf"), 0.0),
        optimal=0.0,
        direction="maximize",
        units="dimensionless",
        description="Gaussian log-likelihood with observation and model error",
        reference="Vrugt (2016); Pastorello et al. (2020)",
    ),
}

# ---------------------------------------------------------------------------
# Alias → canonical name mapping (kept separate from the registry so that
# provenance records and config diffs always use the canonical name).
# ---------------------------------------------------------------------------
_METRIC_ALIASES: Dict[str, str] = {
    "kge": "KGE",
    "kling_gupta": "KGE",
    "nse": "NSE",
    "nash_sutcliffe": "NSE",
    "kge_prime": "KGEp",
    "kgep": "KGEp",
    "kge_np": "KGEnp",
    "kgenp": "KGEnp",
    "r_squared": "R2",
    "r2": "R2",
    "log_nse": "logNSE",
    "lognse": "logNSE",
    "ve": "VE",
    "rmse": "RMSE",
    "nrmse": "NRMSE",
    "mae": "MAE",
    "mare": "MARE",
    "pbias": "PBIAS",
}


def canonicalize_metric_name(name: str) -> str:
    """Resolve a metric name or alias to its canonical registry key.

    Returns the input unchanged if it is already canonical or unrecognised.
    """
    if name in METRIC_REGISTRY:
        return name
    return _METRIC_ALIASES.get(name.lower(), name)


def get_metric_function(name: str) -> Optional[Callable]:
    """Get a metric function by canonical or alias name."""
    canonical = canonicalize_metric_name(name)
    if canonical in METRIC_REGISTRY:
        return cast(Callable, METRIC_REGISTRY[canonical]["function"])
    return None


def get_metric_info(name: str) -> Optional[MetricInfo]:
    """Get metric metadata by canonical or alias name."""
    canonical = canonicalize_metric_name(name)
    if canonical in METRIC_REGISTRY:
        return cast(MetricInfo, METRIC_REGISTRY[canonical]["info"])
    return None


def list_available_metrics() -> List[str]:
    """List all primary metric names (excluding aliases)."""
    return [
        "NSE",
        "logNSE",
        "KGE",
        "KGEp",
        "KGEnp",
        "VE",
        "RMSE",
        "NRMSE",
        "MAE",
        "MARE",
        "bias",
        "PBIAS",
        "correlation",
        "R2",
    ]


def interpret_metric(name: str, value: float) -> str:
    """Provide a human-readable interpretation of a metric value."""
    info = get_metric_info(name)
    if info is None:
        return f"{name} = {value:.3f}: Unknown metric"

    if np.isnan(value):
        return f"{name} = NaN: Could not be calculated (insufficient data or invalid values)"

    if info.direction == "maximize":
        if info.optimal == 1.0:
            if value >= 0.9:
                category = "Excellent"
            elif value >= 0.75:
                category = "Good"
            elif value >= 0.5:
                category = "Satisfactory"
            elif value >= 0.0:
                category = "Poor"
            else:
                category = "Unsatisfactory"
        else:
            category = "N/A"
    else:
        if info.optimal == 0.0:
            if value == 0:
                category = "Perfect"
            elif name == "PBIAS":
                abs_val = abs(value)
                if abs_val < 10:
                    category = "Excellent"
                elif abs_val < 25:
                    category = "Good"
                elif abs_val < 50:
                    category = "Satisfactory"
                else:
                    category = "Poor"
            else:
                category = "See context"
        else:
            category = "N/A"

    return f"{name} = {value:.3f}: {category}"
