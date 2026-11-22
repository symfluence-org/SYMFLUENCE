# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Weighted predictive quantiles, coverage, width and CRPS."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from symfluence.core.metrics.probabilistic import (
    average_interval_width,
    ensemble_crps,
    interval_coverage,
    predictive_scores,
    weighted_quantiles,
)

pytestmark = pytest.mark.unit


def test_equal_weights_match_hazen_quantiles():
    rng = np.random.default_rng(0)
    ens = rng.normal(size=(200, 3))
    qs = [0.05, 0.5, 0.95]
    got = weighted_quantiles(ens, qs)
    expected = np.quantile(ens, qs, axis=0, method='hazen')
    assert np.allclose(got, expected)


def test_weights_shift_quantiles_towards_heavy_members():
    ens = np.array([[0.0], [10.0]])
    assert weighted_quantiles(ens, [0.5], [1, 1])[0, 0] == pytest.approx(5.0)
    assert weighted_quantiles(ens, [0.5], [9, 1])[0, 0] < 5.0
    # All weight on one member: every quantile is that member
    assert np.allclose(weighted_quantiles(ens, [0.05, 0.95], [0, 1]), 10.0)


def test_weighted_quantiles_equal_replicated_ensemble():
    """Integer weights behave like replicating members."""
    ens = np.array([[1.0], [2.0], [3.0]])
    w = [1, 2, 1]
    rep = np.array([[1.0], [2.0], [2.0], [3.0]])
    for q in (0.2, 0.5, 0.8):
        assert weighted_quantiles(ens, [q], w)[0, 0] == pytest.approx(
            np.quantile(rep, q, method='hazen'), abs=0.26)


def test_coverage_and_width():
    obs = np.array([1.0, 2.0, 3.0, np.nan])
    lo = np.array([0.5, 2.5, 2.0, 0.0])
    hi = np.array([1.5, 3.0, 4.0, 1.0])
    assert interval_coverage(obs, lo, hi) == pytest.approx(2 / 3)
    assert average_interval_width(lo, hi, obs) == pytest.approx((1.0 + 0.5 + 2.0) / 3)


def test_crps_matches_brute_force():
    rng = np.random.default_rng(1)
    ens = rng.gamma(2.0, size=(15, 40))
    obs = rng.gamma(2.0, size=40)
    w = rng.random(15)
    wn = w / w.sum()
    brute = np.mean([
        np.sum(wn * np.abs(ens[:, t] - obs[t]))
        - 0.5 * np.sum(np.outer(wn, wn) * np.abs(np.subtract.outer(ens[:, t], ens[:, t])))
        for t in range(40)
    ])
    assert ensemble_crps(ens, obs, w) == pytest.approx(brute)


def test_crps_of_large_gaussian_ensemble_matches_closed_form():
    rng = np.random.default_rng(2)
    ens = rng.normal(0.0, 1.0, size=(20000, 1))
    y = 0.7
    z = y
    closed = z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi)
    assert ensemble_crps(ens, np.array([y])) == pytest.approx(closed, rel=0.02)


def test_crps_of_point_forecast_is_absolute_error():
    ens = np.array([[2.0, 5.0]])
    assert ensemble_crps(ens, np.array([3.0, 1.0])) == pytest.approx(2.5)


def test_predictive_scores_keys():
    rng = np.random.default_rng(3)
    ens = rng.normal(size=(100, 50))
    obs = rng.normal(size=50)
    sc = predictive_scores(obs, ens)
    assert set(sc) == {'coverage_90', 'width_90', 'crps', 'n_obs'}
    assert 0.7 < sc['coverage_90'] <= 1.0
