# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Workers that score their own obs/sim produce LOGLIK_BC_AR1; others make DREAM fall back."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.metrics.formal_likelihood import boxcox_ar1_log_likelihood
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

pytestmark = pytest.mark.unit

INDEX = pd.date_range('2001-01-01', periods=400, freq='D')
OBS = pd.Series(5 + 3 * np.sin(np.arange(400) / 15.0) ** 2, index=INDEX)
SIM = OBS * 1.1


def _config(tmp_path, **extra):
    return {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'DOMAIN_NAME': 'd',
            'OPTIMIZATION_METRIC': 'LOGLIK_BC_AR1',
            'CALIBRATION_PERIOD': '2001-01-01, 2002-02-04', **extra}


@pytest.fixture
def observations(monkeypatch):
    monkeypatch.setattr(StreamflowMetrics, 'load_observations',
                        lambda self, *a, **k: (OBS.to_numpy(), OBS.index))


def _expected():
    return boxcox_ar1_log_likelihood(OBS.to_numpy(), SIM.to_numpy())


def test_raven_worker_returns_formal_likelihood(tmp_path, monkeypatch, observations):
    from symfluence.models.raven import adapter

    monkeypatch.setattr(adapter, 'discharge', lambda output_dir: SIM)
    worker = adapter.RavenWorker(_config(tmp_path), logging.getLogger('t'))
    metrics = worker.calculate_metrics(tmp_path, _config(tmp_path))
    assert metrics['loglik_bc_ar1'] == pytest.approx(_expected())
    assert worker._extract_primary_score(metrics, _config(tmp_path)) == pytest.approx(_expected())


def test_ngen_direct_metrics_return_formal_likelihood(tmp_path, observations):
    from symfluence.models.ngen.calibration.worker import NgenWorker

    out = tmp_path / 'out'
    out.mkdir()
    pd.DataFrame({'q_cms': SIM}).to_csv(out / 'nex.csv')
    worker = NgenWorker(_config(tmp_path), logging.getLogger('t'))
    metrics = worker._calculate_metrics_direct(out, _config(tmp_path))
    assert metrics['loglik_bc_ar1'] == pytest.approx(_expected())


def test_objective_extras_only_adds_missing_metrics():
    class _W(BaseWorker):
        def apply_parameters(self, *a, **k):
            return True

        def run_model(self, *a, **k):
            return True

        def calculate_metrics(self, *a, **k):
            return {}

    w = _W({'OPTIMIZATION_METRIC': 'KGE_LOG'})
    extras = w._objective_extras(OBS, SIM, {'kge': 0.9, 'nse': 0.8})
    assert set(extras) == {'kge_log'}


class TestDreamFallsBackWhenWorkerCannotScore:
    """Pre-flight: an incapable worker turns formal DREAM into informal, with a warning."""

    @staticmethod
    def _stub(worker_supports: bool, config):
        created = []

        class _Worker:
            def supports_formal_likelihood(self, cfg=None):
                return worker_supports

        stub = SimpleNamespace(
            logger=logging.getLogger('t'), FORMAL_LIKELIHOOD_METRIC='LOGLIK_BC_AR1',
            _STREAMFLOW_TARGETS=BaseModelOptimizer._STREAMFLOW_TARGETS,
            _with_informal_dream=BaseModelOptimizer._with_informal_dream,
            _create_worker=lambda: created.append(1) or _Worker(),
        )
        from symfluence.core.mixins.config import resolve_config_value
        stub._get_config_value = lambda acc, default=None, dict_key=None: resolve_config_value(
            stub._config, acc, dict_key=dict_key)
        stub._config_before_likelihood = config
        stub._config = config
        stub._config = BaseModelOptimizer._apply_dream_likelihood(stub, config)
        stub.worker = _Worker()
        BaseModelOptimizer._check_formal_likelihood_support(stub)
        return stub, created

    def test_incapable_worker_falls_back_to_informal(self, caplog):
        caplog.set_level(logging.WARNING)
        cfg = {'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DREAM', 'OPTIMIZATION_METRIC': 'KGE'}
        stub, created = self._stub(False, cfg)
        assert stub._config['OPTIMIZATION_METRIC'] == 'KGE'
        assert stub._config['DREAM_LIKELIHOOD'] == 'informal'
        assert created == [1]  # worker rebuilt on the informal config
        assert 'falls back to DREAM_LIKELIHOOD: informal' in caplog.text

    def test_capable_worker_keeps_formal(self):
        cfg = {'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DREAM', 'OPTIMIZATION_METRIC': 'KGE'}
        stub, created = self._stub(True, cfg)
        assert stub._config['OPTIMIZATION_METRIC'] == 'LOGLIK_BC_AR1'
        assert created == []

    def test_multi_gauge_is_not_a_single_series_likelihood(self):
        class _W(BaseWorker):
            def apply_parameters(self, *a, **k):
                return True

            def run_model(self, *a, **k):
                return True

            def calculate_metrics(self, *a, **k):
                return {}

        assert _W({}).supports_formal_likelihood() is True
        assert _W({'MULTI_GAUGE_CALIBRATION': True}).supports_formal_likelihood() is False

    def test_ignacio_declares_no_support(self):
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker
        assert IGNACIOWorker.SUPPORTS_FORMAL_LIKELIHOOD is False
