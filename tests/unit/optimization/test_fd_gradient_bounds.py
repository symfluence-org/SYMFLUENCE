# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Finite-difference gradients must stay correct on the box boundary.

Perturbations are clamped to the normalized [0, 1] box. Dividing by the
*requested* step rather than the one actually taken silently corrupts the
gradient exactly where a parameter is pinned — halving it (central) or
zeroing it (forward), which freezes the parameter on the bound.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from symfluence.core.calibration.mixins.gradient_optimization import GradientOptimizationMixin


class _Stub(GradientOptimizationMixin):
    def __init__(self):
        self.logger = logging.getLogger("test_fd")
        self._get_config_value = lambda accessor, default=None, dict_key=None: default


@pytest.fixture
def mixin():
    return _Stub()


def _quadratic(peak=0.7):
    """Concave parabola; analytic slope is -2 * (x - peak)."""
    return lambda x: -float((x[0] - peak) ** 2)


@pytest.mark.parametrize("x0,expected", [(1.0, -0.6), (0.0, 1.4), (0.5, 0.4)])
def test_central_gradient_matches_analytic(mixin, x0, expected):
    f = _quadratic()
    _, grad = mixin.compute_fd_gradients(np.array([x0]), f, epsilon=1e-3)
    assert grad[0] == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("x0,expected", [(1.0, -0.6), (0.0, 1.4), (0.5, 0.4)])
def test_forward_gradient_matches_analytic(mixin, x0, expected):
    f = _quadratic()
    x = np.array([x0])
    grad = mixin.compute_fd_gradients_forward(x, f(x), f, epsilon=1e-3)
    assert grad[0] == pytest.approx(expected, abs=2e-3)


def test_forward_gradient_at_upper_bound_is_not_zero(mixin):
    """Regression: a pinned parameter used to report exactly zero slope."""
    f = _quadratic()
    x = np.array([1.0])
    grad = mixin.compute_fd_gradients_forward(x, f(x), f, epsilon=1e-3)
    assert grad[0] != 0.0
    assert grad[0] < 0  # points back into the box


def test_degenerate_box_yields_zero_without_error(mixin):
    """A zero-width box has no slope to measure; must not divide by zero."""
    f = lambda x: 1.0
    grad = mixin.compute_fd_gradients_forward(np.array([0.0]), 1.0, f, epsilon=0.0)
    assert grad[0] == 0.0


def test_line_search_alert_fires_once_when_degenerate(mixin, caplog):
    with caplog.at_level(logging.ERROR):
        for i in range(1, 31):
            mixin._warn_if_line_search_degenerate(i, i, epsilon=1e-4)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "steepest descent" in errors[0].getMessage()
    assert "gradient_epsilon" in errors[0].getMessage()


def test_line_search_alert_silent_when_healthy(mixin, caplog):
    with caplog.at_level(logging.ERROR):
        for i in range(1, 31):
            mixin._warn_if_line_search_degenerate(2, i, epsilon=1e-4)
    assert not [r for r in caplog.records if r.levelno == logging.ERROR]
