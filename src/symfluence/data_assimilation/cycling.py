# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forecast-analysis cycling with baselines.

One pass over the period advances, on the same model object (states are
swapped in and out through the :class:`~.enkf.ensemble_manager.DAModel`
interface):

- the **filter** ensemble: forecast, then an EnKF analysis wherever an
  observation exists (after warm-up, on assimilation steps);
- the **open loop**: the *same* members with the *same* forcing
  perturbations (and the same precipitation-bias draws), never updated. Any
  difference between the two is therefore due to the analyses alone;
- optional **lead-time forecasts**: from the analysis state at each issue
  step, the ensemble is run ``forecast_steps`` ahead without further
  updates.

Recorded filter predictions are *priors* (before the analysis at that step),
so verifying them against the observation at the same step is a fair,
out-of-sample one-step forecast.

Forcing perturbations are drawn once for the whole period (AR(1) in time,
see :meth:`GaussianPerturbation.forcing_noise`) and are shared by the filter,
the open loop and the forecasts. Forecasts therefore use the observed
forcing with the ensemble's perturbations: a hindcast that isolates the
value of the updated initial state. Operational skill with forecast forcing
(NWP) will be lower and must be verified separately.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .enkf.enkf_algorithm import EnKFAlgorithm
from .enkf.observation_error import StreamflowObservationError
from .enkf.state_vector import StateVariableSpec, StateVector

logger = logging.getLogger(__name__)


@dataclass
class CycleSettings:
    """Settings of one forecast-analysis pass (see :func:`run_cycle`)."""

    assimilation_interval: int = 1
    warmup_steps: int = 0
    filter_variant: str = 'stochastic'
    augment_state: bool = True
    enforce_nonnegative: bool = True
    open_loop: bool = True
    forecast_steps: int = 0
    forecast_issue_interval: int = 1
    precip_bias: bool = False
    precip_bias_std: float = 0.2
    precip_bias_walk_std: float = 0.01


@dataclass
class CycleResult:
    """Arrays produced by one pass of the forecast-analysis loop.

    Prediction arrays carry a trailing gauge axis only when there is more
    than one gauge.
    """

    ensemble_predictions: np.ndarray  # (n_t, n_members[, n_gauges]) priors
    observed_values: np.ndarray       # (n_t[, n_gauges]), NaN = none
    analysis_times: List[int] = field(default_factory=list)
    open_loop_predictions: Optional[np.ndarray] = None
    forecasts: Optional[np.ndarray] = None   # (n_issues, n_leads, n_members[, n_gauges])
    forecast_issue_steps: List[int] = field(default_factory=list)
    precip_bias: Optional[np.ndarray] = None  # (n_t, n_members) multiplier after analysis

    @property
    def ensemble_means(self) -> np.ndarray:
        return self.ensemble_predictions.mean(axis=1)

    @property
    def ensemble_stds(self) -> np.ndarray:
        return self.ensemble_predictions.std(axis=1)


def _squeeze_gauges(a: np.ndarray) -> np.ndarray:
    return a[..., 0] if a.shape[-1] == 1 else a


def run_cycle(
    model,
    enkf: EnKFAlgorithm,
    obs_model_units: np.ndarray,
    obs_error: StreamflowObservationError,
    settings: CycleSettings,
    forcing_noise: Optional[Mapping[str, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    state_specs: Optional[List[StateVariableSpec]] = None,
) -> CycleResult:
    """Run filter, open loop and forecasts over the model period.

    Args:
        model: A :class:`DAModel`, already initialised.
        enkf: Analysis algorithm.
        obs_model_units: Observations aligned to the model steps, (n_t,) or
            (n_t, n_gauges), in the units of ``model.step``'s predictions.
        obs_error: Observation-error model.
        settings: Cycle settings.
        forcing_noise: ``{name: (n_members, n_t)}`` multipliers or offsets
            for the variables in ``model.perturbable_forcing``; missing
            names are not perturbed.
        rng: Generator for the precipitation-bias draws.
        state_specs: State layout; defaults to ``model.state_specs``.

    Returns:
        :class:`CycleResult`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    specs = state_specs if state_specs is not None else model.state_specs
    state_vector = StateVector(specs)
    n_state = state_vector.n_state
    n_members = int(model.n_members)

    obs = np.asarray(obs_model_units, dtype=float)
    if obs.ndim == 1:
        obs = obs[:, None]
    n_t = len(np.asarray(model.forcing['precip']))
    if obs.shape[0] != n_t:
        raise ValueError(
            f"Observations must be aligned to the model time axis: expected "
            f"{n_t} values, got shape {np.shape(obs_model_units)}.")
    n_gauges = obs.shape[1]
    if n_gauges > 1 and not settings.augment_state:
        raise ValueError("Several gauges require ENKF_AUGMENT_STATE (predictions in the state).")

    # numeric per-step forcing series; anything else (time axis, metadata)
    # is not passed to the step
    base = {}
    for name, series in model.forcing.items():
        arr = np.asarray(series)
        if name == 'time' or arr.ndim == 0 or arr.shape[0] != n_t \
                or not np.issubdtype(arr.dtype, np.number):
            continue
        base[name] = arr.astype(float).reshape(n_t, -1)[:, 0]
    noise = dict(forcing_noise or {})
    kinds = dict(getattr(model, 'perturbable_forcing', {}) or {})

    use_bias = bool(settings.precip_bias) and 'precip' in base
    if use_bias:
        s0 = float(settings.precip_bias_std)
        # log-bias with mean -s0^2/2: the multiplier exp(beta) has mean 1
        beta0 = rng.normal(-0.5 * s0 ** 2, s0, n_members)
        walk = rng.normal(0.0, float(settings.precip_bias_walk_std), (n_members, n_t))
    else:
        beta0 = np.zeros(n_members)
        walk = None

    def forcing_at(t: int, beta: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, series in base.items():
            value = np.full(n_members, series[t])
            if name in noise and name in kinds:
                if kinds[name] == 'multiplicative':
                    value = value * noise[name][:, t]
                else:
                    value = value + noise[name][:, t]
            if name == 'precip' and use_bias:
                value = value * np.exp(beta)
            out[name] = value
        return out

    def advance(X: np.ndarray, t: int, beta: np.ndarray):
        model.set_state(X)
        pred = np.asarray(model.step(t, forcing_at(t, beta)), dtype=float)
        return model.get_state(), pred.reshape(n_members, -1)

    X = np.asarray(model.get_state(), dtype=float)
    beta = beta0.copy()
    X_ol = X.copy() if settings.open_loop else None
    beta_ol = beta0.copy()

    n_bias = 1 if use_bias else 0
    nonneg = np.zeros(n_state + n_bias + n_gauges, dtype=bool)
    nonneg[:n_state] = True
    nonneg[n_state + n_bias:] = True

    preds = np.empty((n_t, n_members, n_gauges))
    ol_preds = np.empty((n_t, n_members, n_gauges)) if settings.open_loop else None
    bias_trace = np.empty((n_t, n_members)) if use_bias else None
    forecasts: List[np.ndarray] = []
    issue_steps: List[int] = []
    analysis_times: List[int] = []
    interval = max(int(settings.assimilation_interval), 1)

    for t in range(n_t):
        if walk is not None:
            beta = beta + walk[:, t]
            beta_ol = beta_ol + walk[:, t]

        if X_ol is not None and ol_preds is not None:
            X_ol, ol_preds[t] = advance(X_ol, t, beta_ol)

        X, p = advance(X, t, beta)
        preds[t] = p

        y_t = obs[t]
        valid = np.flatnonzero(np.isfinite(y_t))
        if t >= settings.warmup_steps and t % interval == 0 and valid.size:
            y = y_t[valid]
            R = obs_error.covariance(y, valid)
            parts = [X]
            if use_bias:
                parts.append(beta[:, None])
            if settings.augment_state:
                parts.append(p[:, valid])
                Z = np.hstack(parts)
                H = np.zeros((valid.size, Z.shape[1]))
                H[np.arange(valid.size), Z.shape[1] - valid.size + np.arange(valid.size)] = 1.0
                mask = np.concatenate([nonneg[:n_state + n_bias], np.ones(valid.size, bool)])
            else:
                # Without augmentation the observation is the last state
                # element (legacy behaviour for a one-gauge model).
                Z = np.hstack(parts)
                H = np.zeros((1, Z.shape[1]))
                H[0, n_state - 1] = 1.0
                mask = nonneg[:n_state + n_bias]
            Z_a = enkf.analyze(Z, y, H, R, settings.filter_variant,
                               nonnegative=mask if settings.enforce_nonnegative else None)
            X = state_vector.enforce_bounds(Z_a[:, :n_state])
            if use_bias:
                beta = Z_a[:, n_state]
            analysis_times.append(t)

        if bias_trace is not None:
            bias_trace[t] = np.exp(beta)

        if (settings.forecast_steps > 0 and t >= settings.warmup_steps
                and (t - settings.warmup_steps) % settings.forecast_issue_interval == 0
                and t + 1 < n_t):
            fc = np.full((settings.forecast_steps, n_members, n_gauges), np.nan)
            Xf = X.copy()
            for k in range(1, settings.forecast_steps + 1):
                s = t + k
                if s >= n_t:
                    break
                Xf, fc[k - 1] = advance(Xf, s, beta)
            forecasts.append(fc)
            issue_steps.append(t)

    # leave the model holding the filter state
    model.set_state(X)

    return CycleResult(
        ensemble_predictions=_squeeze_gauges(preds),
        observed_values=_squeeze_gauges(obs),
        analysis_times=analysis_times,
        open_loop_predictions=None if ol_preds is None else _squeeze_gauges(ol_preds),
        forecasts=(_squeeze_gauges(np.stack(forecasts)) if forecasts else None),
        forecast_issue_steps=issue_steps,
        precip_bias=bias_trace,
    )


def draw_forcing_noise(
    perturbation: Any,
    model,
    n_t: int,
    correlation_hours: float,
    enabled: bool = True,
) -> Dict[str, np.ndarray]:
    """Draw the period's forcing perturbations for every perturbable variable."""
    if not enabled or not hasattr(perturbation, 'forcing_noise'):
        return {}
    tau_steps = float(correlation_hours) / float(getattr(model, 'timestep_hours', 24) or 24)
    out = {}
    for name in getattr(model, 'perturbable_forcing', {}) or {}:
        out[name] = perturbation.forcing_noise(name, int(model.n_members), n_t, tau_steps)
    return out
