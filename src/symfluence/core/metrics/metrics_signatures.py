# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Signature-based calibration objectives.

Each function takes ``(observed, simulated)`` flow arrays (time-ordered) and
returns a non-negative error that is **minimised** (0 = the simulated
signature matches the observed one). They are registered in the metric
registry so they can be used as ``OPTIMIZATION_METRIC`` or as COMPOSITE /
NSGA-II objective components.

* ``FDC_SLOPE_ERROR`` -- |%BiasFMS|: relative error (%) of the slope of the
  mid-segment (20-70% exceedance) of the log flow-duration curve
  (Yilmaz et al., 2008).
* ``PEAK_FLOW_BIAS`` -- |%BiasFHV|: relative volume error (%) of the
  highest 2% of flows (Yilmaz et al., 2008).
* ``BFI_ERROR`` -- absolute difference of the baseflow index from the
  Lyne-Hollick digital filter (alpha = 0.925).

A runoff-ratio error would need precipitation, which a two-series objective
does not have; volume errors are covered by ``PBIAS``.

Reference:
    Yilmaz, K.K., Gupta, H.V. and Wagener, T. (2008). A process-based
    diagnostic approach to model evaluation: Application to the NWS
    distributed hydrologic model. Water Resources Research, 44, W09417.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from symfluence.core.metrics.metrics_core import _clean_data
from symfluence.core.metrics.metrics_hydrograph import baseflow_index

__all__ = ["fdc_slope_error", "peak_flow_bias", "bfi_error"]

ArrayLike = Union[np.ndarray, pd.Series]

_MIN_POINTS = 30


def _fdc_mid_slope(sorted_desc: np.ndarray, lo: float = 0.2, hi: float = 0.7) -> float:
    n = sorted_desc.size
    q_lo = sorted_desc[int(lo * n)]
    q_hi = sorted_desc[int(hi * n)]
    if q_lo <= 0 or q_hi <= 0:
        return float("nan")
    return float(np.log(q_lo) - np.log(q_hi))


def fdc_slope_error(observed: ArrayLike, simulated: ArrayLike) -> float:
    """|%BiasFMS|: percent error of the FDC mid-segment slope (minimise)."""
    obs, sim = _clean_data(observed, simulated)
    if len(obs) < _MIN_POINTS:
        return float("nan")
    s_obs = _fdc_mid_slope(np.sort(obs)[::-1])
    s_sim = _fdc_mid_slope(np.sort(sim)[::-1])
    if not (np.isfinite(s_obs) and np.isfinite(s_sim)) or s_obs == 0:
        return float("nan")
    return float(abs((s_sim - s_obs) / s_obs) * 100.0)


def peak_flow_bias(observed: ArrayLike, simulated: ArrayLike, fraction: float = 0.02) -> float:
    """|%BiasFHV|: percent volume error of the top ``fraction`` of flows (minimise)."""
    obs, sim = _clean_data(observed, simulated)
    if len(obs) < _MIN_POINTS:
        return float("nan")
    k = max(1, int(np.ceil(fraction * len(obs))))
    top_obs = np.sort(obs)[::-1][:k]
    top_sim = np.sort(sim)[::-1][:k]
    total = float(np.sum(top_obs))
    if total == 0:
        return float("nan")
    return float(abs((np.sum(top_sim) - total) / total) * 100.0)


def bfi_error(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Absolute baseflow-index difference |BFI_sim - BFI_obs| (minimise)."""
    obs, sim = _clean_data(observed, simulated)
    if len(obs) < _MIN_POINTS:
        return float("nan")
    b_obs = baseflow_index(obs)
    b_sim = baseflow_index(sim)
    if not (np.isfinite(b_obs) and np.isfinite(b_sim)):
        return float("nan")
    return float(abs(b_sim - b_obs))
