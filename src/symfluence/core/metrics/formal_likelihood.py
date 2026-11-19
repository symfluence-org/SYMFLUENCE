# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Formal (statistical) likelihood for streamflow: Box-Cox + AR(1) residuals.

Informal objectives such as KGE are not likelihoods: plugging them into a
Metropolis acceptance ratio (``exp(KGE / T)``) produces a "posterior" whose
width is set by the arbitrary temperature ``T``, not by the data. This module
provides a residual error model whose density is a proper likelihood
``p(Q_obs | theta)``, so MCMC (DREAM) samples a genuine Bayesian posterior.

Error model (Bates & Campbell, 2001; Schoups & Vrugt, 2010, simplified):

* **Heteroscedasticity** -- flows are Box-Cox transformed,
  ``z = ((q + eps)^lambda - 1) / lambda`` (``log(q + eps)`` for lambda = 0),
  so a residual of fixed size in z-space corresponds to an error that grows
  with flow magnitude in flow space. ``eps`` is a small offset that keeps zero
  flows finite.
* **Autocorrelation** -- residuals ``e_t = z(obs_t) - z(sim_t)`` follow a
  stationary AR(1) process ``e_t = phi * e_{t-1} + a_t`` with Gaussian
  innovations ``a_t ~ N(0, sigma^2)``.

The density is evaluated in observed-flow space (the Box-Cox Jacobian
``(lambda - 1) * sum(log(q_obs + eps))`` is included) so values are
comparable across error-model settings.

Error-model parameters:

* ``sigma`` -- fixed when given; otherwise **integrated out analytically**
  under the Jeffreys prior ``p(sigma) ~ 1/sigma`` (Bates & Campbell, 2001),
  which gives ``log L = lgamma(n/2) - log 2 - (n/2) log(pi * S)`` with ``S``
  the sum of squared innovations. No extra sampled dimension is needed.
* ``phi`` -- fixed when given; otherwise set to its conditional
  least-squares estimate for the current residuals (a profile likelihood in
  ``phi``).

Simplifications relative to the full Schoups & Vrugt (2010) generalised
likelihood: Gaussian (not skew exponential power) innovations, no residual
bias term, and an error variance that is constant in Box-Cox space (the
heteroscedasticity comes from the transform). Gaps in the observed record
are treated as contiguous by the AR(1) recursion.

References:
    Bates, B.C. and Campbell, E.P. (2001). A Markov chain Monte Carlo scheme
    for parameter estimation and inference in conceptual rainfall-runoff
    modeling. Water Resources Research, 37(4), 937-947.

    Schoups, G. and Vrugt, J.A. (2010). A formal likelihood function for
    parameter and predictive inference of hydrologic models with correlated,
    heteroscedastic, and non-Gaussian errors. Water Resources Research,
    46, W10531.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "BoxCoxAR1Settings",
    "box_cox",
    "inverse_box_cox",
    "default_epsilon",
    "estimate_ar1_phi",
    "boxcox_ar1_residuals",
    "boxcox_ar1_error_parameters",
    "boxcox_ar1_log_likelihood",
    "sample_boxcox_ar1_predictive",
]

#: |phi| is clipped below 1 so the stationary AR(1) variance stays finite.
PHI_LIMIT = 0.995


@dataclass(frozen=True)
class BoxCoxAR1Settings:
    """Settings of the Box-Cox AR(1) residual error model.

    Attributes:
        lam: Box-Cox lambda (0 = log transform). 0.3 is a common choice for
            daily streamflow (e.g. Engeland et al., 2010; McInerney et al., 2017).
        phi: Fixed AR(1) coefficient, or None to use the conditional
            least-squares estimate for each simulation.
        sigma: Fixed innovation standard deviation (Box-Cox space), or None to
            integrate it out under the Jeffreys prior.
        epsilon: Offset added to flows before transforming, or None for 1% of
            the mean observed flow (min 1e-6).
    """

    lam: float = 0.3
    phi: Optional[float] = None
    sigma: Optional[float] = None
    epsilon: Optional[float] = None


def box_cox(q: np.ndarray, lam: float) -> np.ndarray:
    """Box-Cox transform of strictly positive values."""
    q = np.asarray(q, dtype=float)
    if lam == 0:
        return np.log(q)
    return (np.power(q, lam) - 1.0) / lam


def inverse_box_cox(z: np.ndarray, lam: float) -> np.ndarray:
    """Inverse Box-Cox transform.

    For ``lam > 0`` values with ``lam * z + 1 <= 0`` lie outside the range
    of the transform; they map to 0 (the lower bound of the flow space).
    """
    z = np.asarray(z, dtype=float)
    if lam == 0:
        return np.exp(z)
    base = lam * z + 1.0
    out = np.zeros_like(base)
    pos = base > 0
    out[pos] = np.power(base[pos], 1.0 / lam)
    return out


def default_epsilon(obs: np.ndarray) -> float:
    """Zero-flow offset: 1% of the mean observed flow (min 1e-6)."""
    obs = np.asarray(obs, dtype=float)
    finite = obs[np.isfinite(obs)]
    mean = float(np.mean(finite)) if finite.size else 0.0
    return max(0.01 * mean, 1e-6) if mean > 0 else 1e-6


def estimate_ar1_phi(residuals: np.ndarray) -> float:
    """Conditional least-squares AR(1) coefficient, clipped to |phi| < 1."""
    e = np.asarray(residuals, dtype=float)
    if e.size < 3:
        return 0.0
    denom = float(np.dot(e[:-1], e[:-1]))
    if denom <= 0:
        return 0.0
    phi = float(np.dot(e[1:], e[:-1]) / denom)
    return float(np.clip(phi, -PHI_LIMIT, PHI_LIMIT))


def _clean(obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    if obs.shape != sim.shape:
        raise ValueError(f"obs and sim must have the same shape, got {obs.shape} and {sim.shape}")
    valid = np.isfinite(obs) & np.isfinite(sim)
    return obs[valid], sim[valid]


def boxcox_ar1_residuals(
    obs: np.ndarray,
    sim: np.ndarray,
    lam: float = 0.3,
    epsilon: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Box-Cox residuals ``z(obs) - z(sim)``.

    Negative flows are clipped to 0 before transforming (a simulated negative
    discharge is outside the support of the transform).

    Returns:
        ``(residuals, log_jacobian, epsilon)`` where ``log_jacobian`` is
        ``(lam - 1) * sum(log(obs + eps))``.
    """
    obs, sim = _clean(obs, sim)
    eps = default_epsilon(obs) if epsilon is None else float(epsilon)
    if eps <= 0:
        raise ValueError(f"epsilon must be > 0, got {eps}")
    q_obs = np.maximum(obs, 0.0) + eps
    q_sim = np.maximum(sim, 0.0) + eps
    e = box_cox(q_obs, lam) - box_cox(q_sim, lam)
    log_jac = (lam - 1.0) * float(np.sum(np.log(q_obs)))
    return e, log_jac, eps


def _innovations(e: np.ndarray, phi: float) -> np.ndarray:
    """Innovations of a stationary AR(1); the first is scaled by sqrt(1 - phi^2)."""
    a = np.empty_like(e)
    a[0] = math.sqrt(1.0 - phi * phi) * e[0]
    a[1:] = e[1:] - phi * e[:-1]
    return a


def boxcox_ar1_error_parameters(
    obs: np.ndarray,
    sim: np.ndarray,
    settings: BoxCoxAR1Settings = BoxCoxAR1Settings(),
) -> Tuple[float, float, float]:
    """Error-model parameters ``(phi, sigma, epsilon)`` for one simulation.

    Fixed values from ``settings`` are returned as given; free ones are the
    conditional estimates used by :func:`boxcox_ar1_log_likelihood`
    (``sigma`` is the maximum-likelihood innovation standard deviation).
    """
    e, _, eps = boxcox_ar1_residuals(obs, sim, settings.lam, settings.epsilon)
    if e.size < 2:
        raise ValueError("At least two valid obs/sim pairs are needed")
    phi = estimate_ar1_phi(e) if settings.phi is None else float(settings.phi)
    if settings.sigma is not None:
        sigma = float(settings.sigma)
    else:
        a = _innovations(e, phi)
        sigma = float(np.sqrt(np.mean(a * a)))
    return phi, sigma, eps


def boxcox_ar1_log_likelihood(
    obs: np.ndarray,
    sim: np.ndarray,
    lam: float = 0.3,
    phi: Optional[float] = None,
    sigma: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> float:
    """Log-likelihood of observed flows under the Box-Cox AR(1) error model.

    Args:
        obs: Observed flows (time-ordered; NaN pairs are dropped).
        sim: Simulated flows, same length as ``obs``.
        lam: Box-Cox lambda (0 = log).
        phi: Fixed AR(1) coefficient (|phi| < 1); None -> conditional estimate.
        sigma: Fixed innovation standard deviation in Box-Cox space;
            None -> integrated out under the Jeffreys prior ``1/sigma``.
        epsilon: Zero-flow offset; None -> :func:`default_epsilon`.

    Returns:
        ``log p(obs | sim)`` (a finite float), or ``-inf`` when fewer than two
        valid pairs exist or the residuals are degenerate.
    """
    e, log_jac, _ = boxcox_ar1_residuals(obs, sim, lam, epsilon)
    n = e.size
    if n < 2:
        return float("-inf")
    phi_v = estimate_ar1_phi(e) if phi is None else float(phi)
    if not abs(phi_v) < 1:
        raise ValueError(f"phi must satisfy |phi| < 1, got {phi_v}")
    a = _innovations(e, phi_v)
    s = float(np.dot(a, a))
    half_log_stationary = 0.5 * math.log(1.0 - phi_v * phi_v)

    if sigma is not None:
        sig = float(sigma)
        if sig <= 0:
            raise ValueError(f"sigma must be > 0, got {sig}")
        ll = (-0.5 * n * math.log(2.0 * math.pi) - n * math.log(sig)
              + half_log_stationary - s / (2.0 * sig * sig))
    else:
        if s <= 0:
            # A perfect fit has unbounded marginal density; cap it via a
            # tiny floor so the value stays finite and comparable.
            s = 1e-300
        ll = (math.lgamma(0.5 * n) - math.log(2.0)
              - 0.5 * n * math.log(math.pi * s) + half_log_stationary)
    return float(ll + log_jac)


def sample_boxcox_ar1_predictive(
    sim: np.ndarray,
    lam: float,
    phi: float,
    sigma: float,
    epsilon: float,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw flow series from the error model around one simulation.

    ``q = z^-1(z(sim + eps) + e) - eps`` with ``e`` a stationary AR(1)
    realisation; values are clipped at 0.

    Returns:
        Array of shape ``(n_draws, len(sim))``.
    """
    sim = np.asarray(sim, dtype=float)
    n = sim.size
    out = np.full((n_draws, n), np.nan)
    if n == 0 or n_draws <= 0:
        return out[:, :0] if n == 0 else out
    eta = rng.standard_normal((n_draws, n)) * sigma
    e = np.empty_like(eta)
    e[:, 0] = eta[:, 0] / math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        e[:, t] = phi * e[:, t - 1] + eta[:, t]
    finite = np.isfinite(sim)
    z = box_cox(np.maximum(np.where(finite, sim, 0.0), 0.0) + epsilon, lam)
    q = inverse_box_cox(z[None, :] + e, lam) - epsilon
    q = np.maximum(q, 0.0)
    q[:, ~finite] = np.nan
    return q
