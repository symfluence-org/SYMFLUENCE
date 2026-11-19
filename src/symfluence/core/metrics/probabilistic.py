# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Ensemble / predictive-distribution utilities and scores.

Used to turn GLUE behavioural runs and DREAM posterior runs into predictive
quantiles and to score those predictions against observations:

* :func:`weighted_quantiles` -- per-time-step quantiles of a weighted ensemble;
* :func:`interval_coverage` -- fraction of observations inside a band;
* :func:`average_interval_width` -- mean band width (sharpness);
* :func:`ensemble_crps` -- continuous ranked probability score of a
  (weighted) ensemble, averaged over time.

References:
    Gneiting, T. and Raftery, A.E. (2007). Strictly proper scoring rules,
    prediction, and estimation. JASA, 102(477), 359-378.

    Hersbach, H. (2000). Decomposition of the continuous ranked probability
    score for ensemble prediction systems. Weather and Forecasting, 15(5),
    559-570.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "normalise_weights",
    "weighted_quantiles",
    "interval_coverage",
    "average_interval_width",
    "ensemble_crps",
    "predictive_scores",
]


def normalise_weights(weights: Optional[np.ndarray], n_members: int) -> np.ndarray:
    """Non-negative weights summing to 1 (uniform when ``weights`` is None)."""
    if weights is None:
        return np.full(n_members, 1.0 / n_members)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n_members:
        raise ValueError(f"Expected {n_members} weights, got {w.size}")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Weights must be finite and non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must not all be zero")
    return w / total


def weighted_quantiles(
    ensemble: np.ndarray,
    quantiles: Sequence[float],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-time-step quantiles of a weighted ensemble.

    Uses the weighted empirical CDF with each member placed at the midpoint
    of its cumulative-weight interval and linear interpolation in between
    (for equal weights this is the Hazen plotting position). This is the
    standard GLUE construction of likelihood-weighted prediction bounds
    (Beven & Binley, 1992).

    Args:
        ensemble: Array ``(n_members, n_times)``; NaN entries and
            zero-weight members are ignored per time step (the remaining
            weights are renormalised).
        quantiles: Probabilities in [0, 1].
        weights: Member weights ``(n_members,)``; None = equal weights.

    Returns:
        Array ``(len(quantiles), n_times)``; NaN where a time step has no
        finite member.
    """
    ens = np.atleast_2d(np.asarray(ensemble, dtype=float))
    m, n_t = ens.shape
    w = normalise_weights(weights, m)
    qs = np.asarray(quantiles, dtype=float)
    if np.any((qs < 0) | (qs > 1)):
        raise ValueError("quantiles must be in [0, 1]")
    out = np.full((qs.size, n_t), np.nan)
    order = np.argsort(ens, axis=0)  # NaN sorts last
    for t in range(n_t):
        idx = order[:, t]
        vals = ens[idx, t]
        ww = w[idx]
        finite = np.isfinite(vals) & (ww > 0)
        if not finite.any():
            continue
        vals = vals[finite]
        ww = ww[finite]
        if ww.sum() <= 0:
            continue
        ww = ww / ww.sum()
        cum = np.cumsum(ww) - 0.5 * ww
        out[:, t] = np.interp(qs, cum, vals)
    return out


def interval_coverage(obs: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of finite observations with ``lower <= obs <= upper``."""
    obs = np.asarray(obs, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return float("nan")
    inside = (obs[valid] >= lower[valid]) & (obs[valid] <= upper[valid])
    return float(np.mean(inside))


def average_interval_width(
    lower: np.ndarray, upper: np.ndarray, obs: Optional[np.ndarray] = None
) -> float:
    """Mean of ``upper - lower`` (over steps with a finite observation when given)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(lower) & np.isfinite(upper)
    if obs is not None:
        valid &= np.isfinite(np.asarray(obs, dtype=float))
    if not valid.any():
        return float("nan")
    return float(np.mean(upper[valid] - lower[valid]))


def ensemble_crps(
    ensemble: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Mean CRPS of a weighted ensemble forecast over time.

    For the ensemble CDF ``F = sum_i w_i H(x - x_i)``::

        CRPS = sum_i w_i |x_i - y| - 1/2 sum_i sum_j w_i w_j |x_i - x_j|

    The double sum is evaluated in ``O(m log m)`` per step from the sorted
    members: ``sum_ij w_i w_j |x_i - x_j| = 2 sum_i w_i x_i (W_<i - W_>i)``.

    Args:
        ensemble: ``(n_members, n_times)``.
        obs: ``(n_times,)``; steps with a NaN observation are skipped.
        weights: Member weights; None = equal.

    Returns:
        Time-mean CRPS (same units as the flows), NaN if no step is scorable.
    """
    ens = np.atleast_2d(np.asarray(ensemble, dtype=float))
    y = np.asarray(obs, dtype=float).ravel()
    m, n_t = ens.shape
    if y.size != n_t:
        raise ValueError(f"obs length {y.size} != ensemble length {n_t}")
    w = normalise_weights(weights, m)
    keep = np.isfinite(y) & np.all(np.isfinite(ens), axis=0)
    if not keep.any():
        return float("nan")
    x = ens[:, keep]
    y = y[keep]
    term1 = np.sum(w[:, None] * np.abs(x - y[None, :]), axis=0)
    order = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, order, axis=0)
    ws = w[order]
    cum = np.cumsum(ws, axis=0)
    below = cum - ws
    above = 1.0 - cum
    term2 = np.sum(ws * xs * (below - above), axis=0)  # = 1/2 * double sum
    crps = term1 - term2
    return float(np.mean(crps))


def predictive_scores(
    obs: np.ndarray,
    ensemble: np.ndarray,
    weights: Optional[np.ndarray] = None,
    band: Sequence[float] = (0.05, 0.95),
) -> Dict[str, float]:
    """Coverage and width of a central band, plus CRPS, for one period.

    Returns keys ``coverage_<P>`` (P = nominal band in %, e.g. 90),
    ``width_<P>``, ``crps`` and ``n_obs``.
    """
    obs = np.asarray(obs, dtype=float)
    lo_q, hi_q = float(band[0]), float(band[1])
    qs = weighted_quantiles(ensemble, [lo_q, hi_q], weights)
    nominal = int(round((hi_q - lo_q) * 100))
    return {
        f"coverage_{nominal}": interval_coverage(obs, qs[0], qs[1]),
        f"width_{nominal}": average_interval_width(qs[0], qs[1], obs),
        "crps": ensemble_crps(ensemble, obs, weights),
        "n_obs": int(np.isfinite(obs).sum()),
    }
