# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data assimilation diagnostics.

Provides verification metrics for ensemble forecasting quality:
- Rank histogram (reliability)
- CRPS (continuous ranked probability score)
- Spread-error ratio
- Innovation consistency
- Open-loop comparison
- Added-value summaries against an open-loop ensemble and persistence, for
  the filter run (:func:`verification_summary`) and by forecast lead time
  (:func:`lead_time_scores`)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from symfluence.evaluation.forecast_verification import scores as forecast_scores

logger = logging.getLogger(__name__)


def rank_histogram(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Compute rank histogram for ensemble reliability assessment.

    A uniform histogram indicates a well-calibrated ensemble.

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members).
        observations: Shape (n_timesteps,).

    Returns:
        Histogram counts of shape (n_members + 1,).
    """
    valid = ~np.isnan(observations)
    preds = ensemble_predictions[valid]
    obs = observations[valid]

    n_members = preds.shape[1]
    ranks = np.zeros(len(obs), dtype=int)

    for i in range(len(obs)):
        ranks[i] = np.searchsorted(np.sort(preds[i]), obs[i])

    histogram = np.bincount(ranks, minlength=n_members + 1)
    return histogram


def crps_series(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """CRPS of an ensemble at every timestep (NaN where there is no observation).

    Uses the "fair" ensemble estimator (Ferro 2014),

        CRPS = mean_i |x_i - y| - sum_{i<j} |x_i - x_j| / (m (m - 1)),

    which is unbiased for the CRPS of the distribution the members are drawn
    from, so ensembles of different sizes can be compared. With one member it
    reduces to the absolute error, so a deterministic forecast (persistence)
    scores its MAE.

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members); a 1-D array is
            treated as a one-member ensemble.
        observations: Shape (n_timesteps,).
    """
    preds = np.asarray(ensemble_predictions, dtype=float)
    if preds.ndim == 1:
        preds = preds[:, None]
    obs = np.asarray(observations, dtype=float)
    out = np.full(obs.shape, np.nan)
    valid = ~np.isnan(obs) & ~np.isnan(preds).any(axis=1)
    if not valid.any():
        return out
    x = np.sort(preds[valid], axis=1)
    y = obs[valid]
    m = x.shape[1]
    abs_term = np.mean(np.abs(x - y[:, None]), axis=1)
    if m > 1:
        # sum_{i<j} (x_(j) - x_(i)) for sorted x = sum_k (2k - m - 1) x_(k), k = 1..m
        weights = 2.0 * np.arange(1, m + 1) - m - 1
        pair_sum = x @ weights
        out[valid] = abs_term - pair_sum / (m * (m - 1))
    else:
        out[valid] = abs_term
    return out


def crps(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Mean Continuous Ranked Probability Score (lower is better).

    See :func:`crps_series` for the estimator. Earlier versions halved the
    ensemble-spread term, which overstated the CRPS of every spread ensemble.

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members).
        observations: Shape (n_timesteps,).

    Returns:
        Mean CRPS over timesteps with an observation.
    """
    values = crps_series(ensemble_predictions, observations)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')


def spread_error_ratio(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Spread-error ratio: ensemble spread over the RMSE of the ensemble mean.

    For a reliable ensemble of m members the expected squared error of the
    mean is ``(m + 1) / m`` times the mean ensemble variance, so the spread
    is taken as ``sqrt(mean(var) * (m + 1) / m)`` (Fortin et al. 2014) and a
    reliable ensemble scores 1:

    - ratio > 1: over-dispersive (too much spread)
    - ratio < 1: under-dispersive (too little spread; the usual EnKF failure)

    Args:
        ensemble_predictions: Shape (n_timesteps, n_members).
        observations: Shape (n_timesteps,).

    Returns:
        Spread-error ratio.
    """
    valid = ~np.isnan(observations)
    preds = ensemble_predictions[valid]
    obs = observations[valid]
    m = preds.shape[1]
    if m < 2 or len(obs) == 0:
        return float('nan')

    spread = np.sqrt(np.mean(np.var(preds, axis=1, ddof=1)) * (m + 1) / m)
    ens_mean = np.mean(preds, axis=1)
    rmse = np.sqrt(np.mean((ens_mean - obs) ** 2))

    if rmse < 1e-12:
        return float('inf')

    return float(spread / rmse)


def innovation_consistency(
    innovations: np.ndarray,
    ensemble_stds: np.ndarray,
    obs_error_std: float,
) -> float:
    """Check consistency of innovations with expected variance.

    Normalized innovations should have unit variance. Returns the
    ratio of actual to expected innovation variance.

    Args:
        innovations: obs - predicted_mean, shape (n_timesteps,).
        ensemble_stds: Ensemble spread, shape (n_timesteps,).
        obs_error_std: Observation error standard deviation.

    Returns:
        Ratio of actual/expected innovation variance (should be ~1.0).
    """
    valid = ~np.isnan(innovations)
    innov = innovations[valid]
    stds = ensemble_stds[valid]

    expected_var = stds ** 2 + obs_error_std ** 2
    actual_var = innov ** 2

    ratio = np.mean(actual_var) / np.mean(expected_var) if np.mean(expected_var) > 1e-12 else float('inf')
    return float(ratio)


def open_loop_comparison(
    da_predictions: np.ndarray,
    open_loop_predictions: np.ndarray,
    observations: np.ndarray,
) -> Dict[str, float]:
    """Compare DA performance against open-loop (no assimilation) baseline.

    Args:
        da_predictions: DA ensemble mean, shape (n_timesteps,).
        open_loop_predictions: Open-loop predictions, shape (n_timesteps,).
        observations: Observations, shape (n_timesteps,).

    Returns:
        Dictionary with RMSE and correlation for both DA and open-loop.
    """
    valid = ~np.isnan(observations) & ~np.isnan(da_predictions) & ~np.isnan(open_loop_predictions)

    da = da_predictions[valid]
    ol = open_loop_predictions[valid]
    obs = observations[valid]

    if len(obs) == 0:
        return {'da_rmse': np.nan, 'ol_rmse': np.nan, 'da_corr': np.nan, 'ol_corr': np.nan}

    da_rmse = float(np.sqrt(np.mean((da - obs) ** 2)))
    ol_rmse = float(np.sqrt(np.mean((ol - obs) ** 2)))

    da_corr = float(np.corrcoef(da, obs)[0, 1]) if len(obs) > 1 else np.nan
    ol_corr = float(np.corrcoef(ol, obs)[0, 1]) if len(obs) > 1 else np.nan

    return {
        'da_rmse': da_rmse,
        'ol_rmse': ol_rmse,
        'da_corr': da_corr,
        'ol_corr': ol_corr,
        'rmse_improvement': float((ol_rmse - da_rmse) / ol_rmse * 100) if ol_rmse > 1e-12 else 0.0,
    }


# ----------------------------------------------------------------------
# Deterministic scores of the ensemble mean
# ----------------------------------------------------------------------

def nse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency on pairs where both are finite."""
    sim, obs = np.asarray(simulated, float), np.asarray(observed, float)
    ok = np.isfinite(sim) & np.isfinite(obs)
    if ok.sum() < 2:
        return float('nan')
    var = np.sum((obs[ok] - obs[ok].mean()) ** 2)
    return float(1.0 - np.sum((sim[ok] - obs[ok]) ** 2) / var) if var > 0 else float('nan')


def kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Kling-Gupta efficiency (Gupta et al. 2009) on pairs where both are finite."""
    sim, obs = np.asarray(simulated, float), np.asarray(observed, float)
    ok = np.isfinite(sim) & np.isfinite(obs)
    if ok.sum() < 2:
        return float('nan')
    s, o = sim[ok], obs[ok]
    if o.std() == 0 or o.mean() == 0 or s.std() == 0:
        return float('nan')
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def point_scores(forecast: pd.Series, observed: pd.Series,
                 references: Optional[Mapping[str, pd.Series]] = None) -> Dict[str, float]:
    """Bias, MAE, RMSE, NSE and MSE skill vs each reference, on pairs where all are present.

    ``skill_vs_<name> = 1 - MSE(forecast) / MSE(reference)``.  Delegates to
    :func:`symfluence.evaluation.forecast_verification.scores`, so DA lead-time
    scores and deterministic forecast verification share one implementation.
    """
    return forecast_scores(forecast, observed, references)


def skill_score(score: float, reference: float) -> float:
    """``1 - score / reference`` for a negatively oriented score (CRPS, MSE).

    1 is perfect, 0 no better than the reference, negative worse.
    """
    if not np.isfinite(score) or not np.isfinite(reference) or reference <= 0:
        return float('nan')
    return float(1.0 - score / reference)


# ----------------------------------------------------------------------
# Baselines and added-value summaries
# ----------------------------------------------------------------------

def persistence_forecast(observations: np.ndarray, lag: int = 1) -> np.ndarray:
    """Persistence: the observation ``lag`` steps earlier (NaN if missing).

    The filter's recorded predictions are priors: the prediction for step t
    is made before the observation at t is assimilated, i.e. it is a one-step
    forecast issued after step t-1. Its persistence benchmark is therefore
    the observation at t-1 (``lag=1``).
    """
    obs = np.asarray(observations, dtype=float)
    out = np.full(obs.shape, np.nan)
    if lag < len(obs):
        out[lag:] = obs[:len(obs) - lag]
    return out


def _ensemble_scores(ens: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
    """Scores of one ensemble on an already-common sample."""
    mean = ens.mean(axis=1)
    return {
        'n': int(len(obs)),
        'crps': crps(ens, obs) if len(obs) else float('nan'),
        'spread_error_ratio': spread_error_ratio(ens, obs) if ens.shape[1] > 1 else float('nan'),
        'rank_histogram': rank_histogram(ens, obs).tolist() if ens.shape[1] > 1 else [],
        'rmse': float(np.sqrt(np.mean((mean - obs) ** 2))) if len(obs) else float('nan'),
        'nse': nse(mean, obs),
        'kge': kge(mean, obs),
    }


def verification_summary(
    ensemble_predictions: np.ndarray,
    observations: np.ndarray,
    open_loop_predictions: Optional[np.ndarray] = None,
    persistence: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Scores of the filter, the open loop and persistence on one common sample.

    All runs are scored on the same timesteps (observation, and every
    baseline given, present), so the skill scores compare like with like.

    Args:
        ensemble_predictions: Filter prior predictions (n_timesteps, n_members).
        observations: (n_timesteps,), NaN where missing.
        open_loop_predictions: Open-loop ensemble (n_timesteps, n_members).
        persistence: Deterministic persistence forecast (n_timesteps,); see
            :func:`persistence_forecast`.
        mask: Optional boolean (n_timesteps,) restricting the sample (e.g.
            excluding warm-up).

    Returns:
        ``{'enkf': {...}, 'open_loop': {...}, 'persistence': {...},
        'crpss_vs_open_loop': x, 'crpss_vs_persistence': y, 'n': n}``. Each
        ensemble entry has CRPS, spread-error ratio, rank histogram, RMSE,
        NSE and KGE of the ensemble mean; persistence (one member) has CRPS
        (= MAE), RMSE, NSE and KGE. CRPSS = 1 - CRPS / CRPS_reference.
    """
    ens = np.asarray(ensemble_predictions, dtype=float)
    obs = np.asarray(observations, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(ens).all(axis=1)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    ol = None if open_loop_predictions is None else np.asarray(open_loop_predictions, float)
    if ol is not None:
        ok &= np.isfinite(ol).all(axis=1)
    per = None if persistence is None else np.asarray(persistence, float)
    if per is not None:
        ok &= np.isfinite(per)

    out: Dict[str, Any] = {'n': int(ok.sum()), 'enkf': _ensemble_scores(ens[ok], obs[ok])}
    if ol is not None:
        out['open_loop'] = _ensemble_scores(ol[ok], obs[ok])
        out['crpss_vs_open_loop'] = skill_score(out['enkf']['crps'], out['open_loop']['crps'])
    if per is not None:
        p = per[ok]
        out['persistence'] = {
            'n': int(ok.sum()),
            'crps': float(np.mean(np.abs(p - obs[ok]))) if ok.any() else float('nan'),
            'rmse': float(np.sqrt(np.mean((p - obs[ok]) ** 2))) if ok.any() else float('nan'),
            'nse': nse(p, obs[ok]),
            'kge': kge(p, obs[ok]),
        }
        out['crpss_vs_persistence'] = skill_score(
            out['enkf']['crps'], out['persistence']['crps'])
    return out


def lead_time_scores(
    forecasts: np.ndarray,
    issue_steps: Sequence[int],
    observations: np.ndarray,
    open_loop_predictions: Optional[np.ndarray] = None,
    timestep_hours: float = 24.0,
) -> pd.DataFrame:
    """Score ensemble forecasts by lead time against open loop and persistence.

    The forecast issued at step ``t`` (after the analysis at t) for lead
    ``k`` covers step ``t + k``. Benchmarks for the same target step:

    - persistence: the observation at the issue step t, held for every lead;
    - open loop: the never-updated ensemble at step t + k (it does not
      depend on the issue time, so its score varies with lead only through
      the sample).

    Every lead is scored on the target steps where the observation, the
    persistence value and (if given) the open loop are all present.

    Args:
        forecasts: (n_issues, n_leads, n_members).
        issue_steps: Step index of each issue, length n_issues.
        observations: (n_timesteps,).
        open_loop_predictions: (n_timesteps, n_members).
        timestep_hours: Step length, for the ``lead_hours`` column.

    Returns:
        One row per lead: ``lead_steps``, ``lead_hours``, ``n``, ``crps``,
        ``crps_open_loop``, ``mae_persistence``, ``crpss_vs_open_loop``,
        ``crpss_vs_persistence``, ``spread_error_ratio``, ``nse``, ``kge``,
        and the ensemble-mean point scores of :func:`point_scores`
        (``bias``, ``mae``, ``rmse``, MSE ``skill_vs_persistence`` /
        ``skill_vs_open_loop``).
    """
    fc = np.asarray(forecasts, dtype=float)
    obs = np.asarray(observations, dtype=float)
    issues = np.asarray(issue_steps, dtype=int)
    n_t = len(obs)
    ol = None if open_loop_predictions is None else np.asarray(open_loop_predictions, float)
    rows = []
    if fc.ndim != 3 or len(issues) != fc.shape[0]:
        raise ValueError("forecasts must be (n_issues, n_leads, n_members) matching issue_steps")
    for k in range(1, fc.shape[1] + 1):
        target = issues + k
        inside = target < n_t
        tgt = target[inside]
        ens = fc[inside, k - 1, :]
        y = obs[tgt]
        per = obs[issues[inside]]
        ok = np.isfinite(y) & np.isfinite(per) & np.isfinite(ens).all(axis=1)
        ol_ens = ol[tgt] if ol is not None else None
        if ol_ens is not None:
            ok &= np.isfinite(ol_ens).all(axis=1)
        ens, y, per = ens[ok], y[ok], per[ok]
        row: Dict[str, Any] = {'lead_steps': k, 'lead_hours': k * float(timestep_hours),
                               'n': int(ok.sum())}
        row['crps'] = crps(ens, y) if len(y) else float('nan')
        row['mae_persistence'] = float(np.mean(np.abs(per - y))) if len(y) else float('nan')
        row['crpss_vs_persistence'] = skill_score(row['crps'], row['mae_persistence'])
        refs = {'persistence': pd.Series(per)}
        if ol_ens is not None:
            ol_ok = ol_ens[ok]
            row['crps_open_loop'] = crps(ol_ok, y) if len(y) else float('nan')
            row['crpss_vs_open_loop'] = skill_score(row['crps'], row['crps_open_loop'])
            refs['open_loop'] = pd.Series(ol_ok.mean(axis=1) if len(y) else [], dtype=float)
        row['spread_error_ratio'] = (spread_error_ratio(ens, y)
                                     if len(y) and ens.shape[1] > 1 else float('nan'))
        mean = ens.mean(axis=1) if len(y) else np.array([])
        row['kge'] = kge(mean, y)
        point = point_scores(pd.Series(mean, dtype=float), pd.Series(y, dtype=float), refs)
        point.pop('n', None)
        row.update(point)
        rows.append(row)
    return pd.DataFrame(rows)
