# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Box-Cox AR(1) formal likelihood against analytic densities."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from symfluence.core.metrics.formal_likelihood import (
    BoxCoxAR1Settings,
    box_cox,
    boxcox_ar1_error_parameters,
    boxcox_ar1_log_likelihood,
    estimate_ar1_phi,
    inverse_box_cox,
    sample_boxcox_ar1_predictive,
)
from symfluence.core.metrics.metrics_registry import get_metric_info
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

pytestmark = pytest.mark.unit


@pytest.fixture
def flows():
    rng = np.random.default_rng(0)
    sim = 5.0 + 3.0 * np.sin(np.linspace(0, 6 * np.pi, 60)) ** 2
    obs = sim * np.exp(rng.normal(0, 0.1, sim.size))
    return obs, sim


def test_lambda_one_iid_matches_gaussian(flows):
    """lambda=1, phi=0, fixed sigma: plain i.i.d. Gaussian on raw residuals."""
    obs, sim = flows
    ll = boxcox_ar1_log_likelihood(obs, sim, lam=1.0, phi=0.0, sigma=0.4, epsilon=1e-6)
    expected = stats.norm.logpdf(obs - sim, scale=0.4).sum()
    assert ll == pytest.approx(expected, rel=1e-10)


def test_ar1_matches_multivariate_normal(flows):
    """Fixed phi/sigma: equals the exact stationary AR(1) Gaussian density."""
    obs, sim = flows
    phi, sigma, n = 0.6, 0.3, obs.size
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    cov = sigma ** 2 / (1 - phi ** 2) * phi ** lags
    expected = stats.multivariate_normal(mean=np.zeros(n), cov=cov).logpdf(obs - sim)
    ll = boxcox_ar1_log_likelihood(obs, sim, lam=1.0, phi=phi, sigma=sigma, epsilon=1e-6)
    assert ll == pytest.approx(expected, rel=1e-8)


def test_box_cox_jacobian_change_of_variables(flows):
    """lambda != 1: density in flow space = density in z space * |dz/dq|."""
    obs, sim = flows
    lam, eps, phi, sigma = 0.3, 0.05, 0.4, 0.2
    e = box_cox(obs + eps, lam) - box_cox(sim + eps, lam)
    n = obs.size
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    cov = sigma ** 2 / (1 - phi ** 2) * phi ** lags
    log_jac = np.sum((lam - 1) * np.log(obs + eps))
    expected = stats.multivariate_normal(np.zeros(n), cov).logpdf(e) + log_jac
    ll = boxcox_ar1_log_likelihood(obs, sim, lam=lam, phi=phi, sigma=sigma, epsilon=eps)
    assert ll == pytest.approx(expected, rel=1e-8)


def test_sigma_marginalised_under_jeffreys_prior(flows):
    """sigma=None integrates sigma out with p(sigma) = 1/sigma."""
    obs, sim = flows
    obs, sim = obs[:12], sim[:12]
    lam, eps, phi = 0.3, 0.05, 0.3

    def integrand(sig):
        return math.exp(boxcox_ar1_log_likelihood(
            obs, sim, lam=lam, phi=phi, sigma=sig, epsilon=eps)) / sig

    numeric, _ = integrate.quad(integrand, 1e-4, 50, limit=200, points=[0.01, 0.1, 1.0])
    ll = boxcox_ar1_log_likelihood(obs, sim, lam=lam, phi=phi, sigma=None, epsilon=eps)
    assert ll == pytest.approx(math.log(numeric), rel=1e-6)


def test_likelihood_prefers_better_fit(flows):
    obs, sim = flows
    assert boxcox_ar1_log_likelihood(obs, sim) > boxcox_ar1_log_likelihood(obs, sim * 1.5)


def test_phi_estimate_recovers_ar1_process():
    rng = np.random.default_rng(1)
    e = np.zeros(20000)
    for t in range(1, e.size):
        e[t] = 0.7 * e[t - 1] + rng.normal()
    assert estimate_ar1_phi(e) == pytest.approx(0.7, abs=0.02)


def test_zero_flows_are_finite():
    obs = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 3.0])
    sim = np.array([0.1, 0.0, 0.8, 2.5, 0.0, 2.0])
    assert np.isfinite(boxcox_ar1_log_likelihood(obs, sim))


def test_inverse_box_cox_roundtrip():
    q = np.array([0.01, 1.0, 10.0, 500.0])
    for lam in (0.0, 0.3, 1.0):
        assert np.allclose(inverse_box_cox(box_cox(q, lam), lam), q)


def test_predictive_draws_have_requested_residual_structure():
    rng = np.random.default_rng(2)
    sim = np.full(5000, 10.0)
    draws = sample_boxcox_ar1_predictive(sim, 1.0, 0.5, 0.2, 1e-6, 4, rng)
    e = draws - sim
    assert draws.shape == (4, 5000)
    assert estimate_ar1_phi(e[0]) == pytest.approx(0.5, abs=0.05)
    assert np.std(e) == pytest.approx(0.2 / math.sqrt(1 - 0.25), rel=0.05)


def test_error_parameters_fixed_values_pass_through(flows):
    obs, sim = flows
    phi, sigma, eps = boxcox_ar1_error_parameters(
        obs, sim, BoxCoxAR1Settings(lam=0.3, phi=0.2, sigma=0.1, epsilon=0.5))
    assert (phi, sigma, eps) == (0.2, 0.1, 0.5)


def test_registered_as_maximised_metric_and_computed_by_streamflow_metrics(flows):
    obs, sim = flows
    assert get_metric_info('loglik_bc_ar1').direction == 'maximize'
    calc = StreamflowMetrics(likelihood=BoxCoxAR1Settings(lam=0.5, phi=0.0))
    out = calc.calculate_metrics(obs, sim, metrics=['loglik_bc_ar1'])
    assert out['loglik_bc_ar1'] == pytest.approx(
        boxcox_ar1_log_likelihood(obs, sim, lam=0.5, phi=0.0))


def test_constant_simulation_is_not_scored_minus_one(flows):
    obs, _ = flows
    sim = np.full(obs.size, 5.0)
    out = StreamflowMetrics().calculate_metrics(obs, sim, metrics=['loglik_bc_ar1', 'kge'])
    assert out['kge'] == -1.0
    assert out['loglik_bc_ar1'] == pytest.approx(boxcox_ar1_log_likelihood(obs, sim))


def test_settings_from_flat_config():
    calc = StreamflowMetrics.from_config({
        'LIKELIHOOD_BOX_COX_LAMBDA': 0.1, 'LIKELIHOOD_AR1_PHI': 0.5,
        'BOX_COX_LAMBDA': 0.4,
    })
    assert calc.box_cox_lambda == 0.4
    assert calc.likelihood == BoxCoxAR1Settings(lam=0.1, phi=0.5, sigma=None, epsilon=None)
