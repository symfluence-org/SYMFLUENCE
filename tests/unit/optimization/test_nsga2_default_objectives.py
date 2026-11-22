# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NSGA-II defaults to a non-degenerate KGE / KGE_LOG front, and workers compute it."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.config.models.optimization import NSGA2Config, OptimizationConfig
from symfluence.core.metrics.streamflow_metrics import configured_objective_metrics

pytestmark = pytest.mark.unit


def test_default_secondary_metric_is_kge_log():
    cfg = NSGA2Config()
    assert (cfg.primary_metric, cfg.secondary_metric) == ('KGE', 'KGE_LOG')
    assert OptimizationConfig().nsga2.secondary_metric == 'KGE_LOG'


def test_objective_metrics_follow_active_algorithm():
    flat = {'ITERATIVE_OPTIMIZATION_ALGORITHM': 'NSGA-II', 'OPTIMIZATION_METRIC': 'KGE'}
    assert configured_objective_metrics(flat) == ['KGE', 'KGE_LOG']
    flat['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DDS'
    assert configured_objective_metrics(flat) == ['KGE']
    flat['COMPOSITE_METRIC'] = {'NSE': 0.5, 'BFI_ERROR': 0.5}
    flat['OPTIMIZATION_METRIC'] = 'COMPOSITE'
    assert configured_objective_metrics(flat) == ['NSE', 'BFI_ERROR']


def test_worker_requests_the_secondary_objective():
    class _W(BaseWorker):
        def apply_parameters(self, *a, **k):
            return True

        def run_model(self, *a, **k):
            return True

        def calculate_metrics(self, *a, **k):
            return {}

    worker = _W({'ITERATIVE_OPTIMIZATION_ALGORITHM': 'NSGA-II', 'OPTIMIZATION_METRIC': 'KGE'})
    assert worker._requested_metrics() == ['kge', 'kge_log', 'nse']


def test_evaluator_adds_configured_objectives():
    """Evaluator-scored workers (e.g. SUMMA) now produce KGE_LOG / signatures."""
    from symfluence.core.modeling.evaluators.base import ModelEvaluator

    ev = SimpleNamespace(config={'ITERATIVE_OPTIMIZATION_ALGORITHM': 'NSGA-II',
                                 'OPTIMIZATION_METRIC': 'BFI_ERROR'})
    idx = pd.date_range('2000-01-01', periods=100)
    obs = pd.Series(np.linspace(1, 5, 100) + np.sin(np.arange(100)), index=idx)
    out = ModelEvaluator._calculate_objective_metrics(ev, obs, obs * 1.1, {'KGE': 0.9})
    assert set(out) == {'BFI_ERROR', 'KGE_LOG'}
    assert out['BFI_ERROR'] == pytest.approx(0.0, abs=1e-9)
