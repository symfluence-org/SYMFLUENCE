# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""DREAM samples a genuine posterior when given a log-likelihood (T = 1)."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from symfluence.core.calibration.optimizers.algorithms import get_algorithm
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.constants import ModelDefaults

pytestmark = pytest.mark.unit

LOG = logging.getLogger('test_dream_formal')
MU, SD = 0.4, 0.05


def _callbacks():
    return {
        'evaluate_solution': Mock(),
        'denormalize_params': lambda x: {f'p{i}': float(v) for i, v in enumerate(x)},
        'record_iteration': Mock(),
        'update_best': Mock(),
        'log_progress': Mock(),
    }


def _run_dream(evaluate_population, iterations=600, seed=11, **cfg):
    config = {'NUMBER_OF_ITERATIONS': iterations, 'POPULATION_SIZE': 8,
              'RANDOM_SEED': seed, **cfg}
    algo = get_algorithm('DREAM', config, LOG)
    return algo.optimize(n_params=1, evaluate_population=evaluate_population, **_callbacks())


def test_gaussian_log_likelihood_gives_matching_posterior():
    def loglik(pop, it=0):
        return -0.5 * ((pop[:, 0] - MU) / SD) ** 2

    res = _run_dream(loglik)
    post = res['posterior_samples'][:, 0]
    assert res['likelihood'] == 'formal'
    assert post.mean() == pytest.approx(MU, abs=0.01)
    assert post.std() == pytest.approx(SD, rel=0.15)
    assert res['posterior_log_likelihood'].shape == post.shape
    assert res['posterior_samples'].shape == (res['n_posterior_samples'], 1)


def test_failed_evaluations_are_never_accepted():
    """A penalty score is a failed run (zero likelihood), even when valid fits
    score below the penalty value -- as large-sample log-likelihoods do."""
    def loglik(pop, it=0):
        ll = -20000.0 - 0.5 * ((pop[:, 0] - MU) / SD) ** 2
        ll[pop[:, 0] > 0.6] = ModelDefaults.PENALTY_SCORE
        return ll

    res = _run_dream(loglik, iterations=200)
    assert res['posterior_samples'][:, 0].max() <= 0.6
    assert res['best_score'] != ModelDefaults.PENALTY_SCORE


def test_informal_mode_warns(caplog):
    caplog.set_level(logging.WARNING)
    _run_dream(lambda pop, it=0: -np.abs(pop[:, 0] - MU), iterations=10,
               DREAM_LIKELIHOOD='informal')
    assert 'not a Bayesian posterior' in caplog.text


class TestOptimizerSwapsObjective:
    """BaseModelOptimizer makes DREAM calibrate on LOGLIK_BC_AR1 by default."""

    @staticmethod
    def _apply(config):
        stub = SimpleNamespace(logger=LOG, _STREAMFLOW_TARGETS=BaseModelOptimizer._STREAMFLOW_TARGETS,
                               FORMAL_LIKELIHOOD_METRIC=BaseModelOptimizer.FORMAL_LIKELIHOOD_METRIC)
        from symfluence.core.mixins.config import resolve_config_value
        stub._get_config_value = lambda acc, default=None, dict_key=None: resolve_config_value(
            config, acc, dict_key=dict_key)
        return BaseModelOptimizer._apply_dream_likelihood(stub, config)

    def test_formal_default_replaces_metric_and_composite(self, caplog):
        caplog.set_level(logging.WARNING)
        out = self._apply({'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DREAM',
                           'OPTIMIZATION_METRIC': 'KGE', 'COMPOSITE_METRIC': {'KGE': 1.0}})
        assert out['OPTIMIZATION_METRIC'] == 'LOGLIK_BC_AR1'
        assert out['COMPOSITE_METRIC'] is None
        assert 'LOGLIK_BC_AR1' in caplog.text

    def test_informal_keeps_metric(self):
        cfg = {'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DREAM', 'OPTIMIZATION_METRIC': 'KGE',
               'DREAM_LIKELIHOOD': 'informal'}
        assert self._apply(cfg)['OPTIMIZATION_METRIC'] == 'KGE'

    def test_other_algorithms_untouched(self):
        cfg = {'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS', 'OPTIMIZATION_METRIC': 'KGE'}
        assert self._apply(cfg) is cfg

    def test_non_streamflow_target_keeps_metric(self):
        cfg = {'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DREAM', 'OPTIMIZATION_METRIC': 'KGE',
               'OPTIMIZATION_TARGET': 'swe'}
        assert self._apply(cfg)['OPTIMIZATION_METRIC'] == 'KGE'

    def test_typed_config_copy(self, tmp_path):
        from symfluence.core.config.models import SymfluenceConfig
        cfg = SymfluenceConfig.from_minimal(
            domain_name='d', model='GR', SYMFLUENCE_DATA_DIR=str(tmp_path),
            SYMFLUENCE_CODE_DIR=str(tmp_path), EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00', POUR_POINT_COORDS='64/-19',
            ITERATIVE_OPTIMIZATION_ALGORITHM='DREAM', OPTIMIZATION_METRIC='NSE')
        out = self._apply(cfg)
        assert out.optimization.metric == 'LOGLIK_BC_AR1'
        assert cfg.optimization.metric == 'NSE'
        from symfluence.core.config.flattening import flatten_nested_config
        assert flatten_nested_config(out)['OPTIMIZATION_METRIC'] == 'LOGLIK_BC_AR1'


def test_worker_never_substitutes_kge_for_the_likelihood():
    from symfluence.core.calibration.workers.base_worker import BaseWorker

    class _W(BaseWorker):
        def apply_parameters(self, *a, **k):
            return True

        def run_model(self, *a, **k):
            return True

        def calculate_metrics(self, *a, **k):
            return {}

    worker = _W({'OPTIMIZATION_METRIC': 'LOGLIK_BC_AR1'})
    score = worker._extract_primary_score({'kge': 0.8, 'nse': 0.7},
                                          {'OPTIMIZATION_METRIC': 'LOGLIK_BC_AR1'})
    assert score == worker.penalty_score
