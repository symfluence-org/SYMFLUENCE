# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Signature-based calibration objectives are registered and minimised."""
from __future__ import annotations

import numpy as np
import pytest

from symfluence.core.config.models.optimization import OptimizationConfig
from symfluence.core.metrics.metric_transformer import MetricTransformer
from symfluence.core.metrics.metrics_registry import get_metric_function
from symfluence.core.metrics.metrics_signatures import bfi_error, fdc_slope_error, peak_flow_bias
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

pytestmark = pytest.mark.unit

SIGNATURES = ('FDC_SLOPE_ERROR', 'PEAK_FLOW_BIAS', 'BFI_ERROR')


@pytest.fixture
def obs():
    rng = np.random.default_rng(0)
    q = np.zeros(730)
    for t in range(1, q.size):
        q[t] = 0.9 * q[t - 1] + rng.gamma(0.3, 4.0)
    return q + 0.5


@pytest.mark.parametrize('fn', [fdc_slope_error, peak_flow_bias, bfi_error])
def test_perfect_simulation_scores_zero(fn, obs):
    assert fn(obs, obs) == pytest.approx(0.0, abs=1e-12)


def test_errors_grow_with_distortion(obs):
    flattened = obs.mean() + 0.5 * (obs - obs.mean())
    assert fdc_slope_error(obs, flattened) > 10
    assert peak_flow_bias(obs, flattened) > 10
    assert peak_flow_bias(obs, obs * 1.2) == pytest.approx(20.0)
    smoothed = np.convolve(obs, np.ones(15) / 15, mode='same')
    assert bfi_error(obs, smoothed) > bfi_error(obs, obs)


def test_short_series_is_nan():
    assert np.isnan(fdc_slope_error(np.ones(5), np.ones(5)))


@pytest.mark.parametrize('name', SIGNATURES)
def test_registered_minimise_objectives(name, obs):
    assert OptimizationConfig(OPTIMIZATION_METRIC=name).metric == name
    assert MetricTransformer.get_direction(name) == 'minimize'
    assert MetricTransformer.transform_for_maximization(name, 5.0) == -5.0
    assert get_metric_function(name.lower()) is not None
    assert StreamflowMetrics.supports(name.lower())
    out = StreamflowMetrics().calculate_metrics(obs, obs * 1.1, metrics=[name.lower()])
    assert np.isfinite(out[name.lower()]) and out[name.lower()] >= 0
