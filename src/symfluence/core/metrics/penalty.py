# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Failure / penalty handling for calibration scores.

Two value spaces exist during calibration:

* **raw metric space** -- what a worker's ``calculate_metrics`` returns
  (KGE 0.8, RMSE 12.3 m3/s, ...), each metric with its own direction;
* **score space** -- the value the optimizer maximises, produced by
  :class:`~symfluence.core.metrics.metric_transformer.MetricTransformer`
  (KGE 0.8 -> 0.8, RMSE 12.3 -> -12.3).

A failed evaluation is signalled in score space by *exactly*
``ModelDefaults.PENALTY_SCORE``. Magnitude thresholds (``score <= -900``) are
wrong there: ``-RMSE`` of a large river is legitimately below -900. Use
:func:`is_penalty` / :func:`sanitize_score` instead of comparing by hand.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from symfluence.core.constants import ModelDefaults
from symfluence.core.metrics.metric_transformer import MetricTransformer

PENALTY_SCORE: float = ModelDefaults.PENALTY_SCORE

# Raw-metric sentinels that workers use to signal a failed run: several model
# workers return ``{'KGE': -999.0}``; StreamflowMetrics returns PENALTY_SCORE
# for maximise metrics and -PENALTY_SCORE for minimise ones (see
# :func:`raw_failure_value`).
FAILURE_SENTINELS = frozenset({-999.0, PENALTY_SCORE, -PENALTY_SCORE})


def is_penalty(score: Any, penalty: float = PENALTY_SCORE) -> Any:
    """True when a score-space value marks a failed evaluation.

    A score is a failure when it is ``None``, non-numeric, non-finite, or
    exactly equal to the penalty constant. Accepts scalars or arrays (an array
    returns an elementwise boolean mask).
    """
    if score is None:
        return True
    if isinstance(score, np.ndarray):
        arr = score.astype(float, copy=False)
        return ~np.isfinite(arr) | (arr == penalty)
    try:
        value = float(score)
    except (TypeError, ValueError):
        return True
    return (not np.isfinite(value)) or value == penalty


def sanitize_score(score: Any, penalty: float = PENALTY_SCORE) -> Any:
    """Map ``None`` / NaN / +-inf (scalar or array) to the penalty score."""
    if isinstance(score, np.ndarray):
        arr = np.array(score, dtype=float)
        arr[~np.isfinite(arr)] = penalty
        return arr
    if score is None:
        return penalty
    try:
        value = float(score)
    except (TypeError, ValueError):
        return penalty
    return value if np.isfinite(value) else penalty


def is_failure_sentinel(raw_value: Any) -> bool:
    """True when a *raw* metric value is a worker failure sentinel.

    ``None``, non-numeric and non-finite values count, as do the literal
    sentinels in :data:`FAILURE_SENTINELS`.
    """
    if raw_value is None:
        return True
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return True
    return (not np.isfinite(value)) or value in FAILURE_SENTINELS


def raw_failure_value(metric_name: str, penalty: float = PENALTY_SCORE) -> float:
    """Raw metric value that transforms to exactly ``penalty`` in score space.

    For a maximise metric (KGE) that is ``penalty`` itself; for a minimise
    metric (RMSE) it is ``-penalty`` (a huge error), so a failed run can never
    out-score a real one after ``transform_for_maximization``.
    """
    if MetricTransformer.is_minimize_metric(metric_name):
        return -penalty
    return penalty
