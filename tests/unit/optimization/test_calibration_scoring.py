# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for calibration score handling (direction, NaN, penalty).

Each test pins one audit finding:

* a minimise metric (RMSE) falling back to KGE was negated, so a crashed
  ``{'kge': -999}`` run scored +999 and won the calibration;
* zero-variance / failed raw metrics were negated into winning scores;
* NaN leaked through the sequential evaluation path and froze DDS / DE;
* multi-objective metric-dict objectives were never direction-transformed;
* NSGA-II cleaned NaN only for offspring, not the initial population;
* magnitude thresholds (``<= -900``) rejected legitimate -RMSE scores;
* the best-row lookup used ``idxmin`` on an already-maximised score column;
* INITIAL_GUESS was ignored by most algorithms;
* small numerical issues (FD step at bounds, Adam NaN stall, log chain rule).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from symfluence.core.calibration.mixins.results_tracking import ResultsTrackingMixin
from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm
from symfluence.core.calibration.optimizers.algorithms.basin_hopping import BasinHoppingAlgorithm
from symfluence.core.calibration.optimizers.algorithms.cmaes import CMAESAlgorithm
from symfluence.core.calibration.optimizers.algorithms.dds import DDSAlgorithm
from symfluence.core.calibration.optimizers.algorithms.de import DEAlgorithm
from symfluence.core.calibration.optimizers.algorithms.ga import GAAlgorithm
from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm
from symfluence.core.calibration.optimizers.algorithms.nelder_mead import NelderMeadAlgorithm
from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm
from symfluence.core.calibration.optimizers.algorithms.pso import PSOAlgorithm
from symfluence.core.calibration.optimizers.algorithms.sce_ua import SCEUAAlgorithm
from symfluence.core.calibration.optimizers.algorithms.simulated_annealing import (
    SimulatedAnnealingAlgorithm,
)
from symfluence.core.calibration.optimizers.evaluators.population_evaluator import (
    PopulationEvaluator,
)
from symfluence.core.calibration.parameters.base_parameter_manager import (
    denormalize_derivative,
    denormalize_value,
)
from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerResult
from symfluence.core.constants import ModelDefaults
from symfluence.core.metrics.penalty import is_penalty, sanitize_score
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics
from symfluence.optimization.optimization_manager import OptimizationManager

PENALTY = ModelDefaults.PENALTY_SCORE
LOG = logging.getLogger("test.calibration_scoring")


class _Worker(BaseWorker):
    def apply_parameters(self, *a, **k):
        return True

    def run_model(self, *a, **k):
        return True

    def calculate_metrics(self, *a, **k):
        return {}


# ---------------------------------------------------------------------------
# 1. Direction-aware primary score / fallback / degenerate metrics
# ---------------------------------------------------------------------------

class TestPrimaryScoreDirection:
    RMSE = {"OPTIMIZATION_METRIC": "RMSE"}

    def test_rmse_is_negated(self):
        assert _Worker({})._extract_primary_score({"RMSE": 12.0}, self.RMSE) == -12.0

    def test_kge_fallback_refused_for_minimize_metric(self):
        # Used to return -0.8 (KGE negated under RMSE's direction).
        assert _Worker({})._extract_primary_score({"kge": 0.8}, self.RMSE) == PENALTY

    def test_crashed_kge_sentinel_never_wins_under_rmse(self):
        # Used to return +999, beating every real run.
        assert _Worker({})._extract_primary_score({"kge": -999.0}, self.RMSE) == PENALTY

    def test_same_direction_fallback_still_allowed(self):
        cfg = {"OPTIMIZATION_METRIC": "NSE"}
        assert _Worker({})._extract_primary_score({"kge": 0.7}, cfg) == pytest.approx(0.7)

    @pytest.mark.parametrize("raw", [-999.0, PENALTY, -PENALTY, float("nan"), float("inf")])
    def test_failure_sentinels_map_to_penalty(self, raw):
        assert _Worker({})._extract_primary_score({"RMSE": raw}, self.RMSE) == PENALTY

    def test_composite_failed_rmse_component_is_penalised(self):
        cfg = {"OPTIMIZATION_METRIC": "COMPOSITE", "COMPOSITE_METRIC": {"KGE": 0.5, "RMSE": 0.5}}
        score = _Worker({})._extract_primary_score({"KGE": 0.8, "RMSE": -PENALTY}, cfg)
        assert score < -1000  # failed component dominates, never rewards

    def test_zero_variance_sim_rmse_is_real_value(self):
        obs, sim = np.arange(10.0), np.ones(10)
        m = StreamflowMetrics().calculate_metrics(obs, sim, ["kge", "rmse"])
        assert m["kge"] == -1.0
        # Used to be -1.0, i.e. +1.0 after the RMSE transform (a "perfect" score).
        assert m["rmse"] == pytest.approx(np.sqrt(np.mean((obs - sim) ** 2)))

    def test_failed_rmse_transforms_to_exact_penalty(self):
        # Log transform of non-positive flows leaves no data -> metric fails.
        # The raw failure value used to be -9999 for RMSE too (+9999 after the
        # transform); it is now direction-aware.
        m = StreamflowMetrics().calculate_metrics(
            np.array([-1.0, -2.0, -3.0]), np.array([-1.0, -2.0, -4.0]), ["rmse_log", "kge_log"])
        assert m == {"rmse_log": -PENALTY, "kge_log": PENALTY}
        cfg = {"OPTIMIZATION_METRIC": "RMSE_LOG"}
        assert _Worker({})._extract_primary_score(m, cfg) == PENALTY


# ---------------------------------------------------------------------------
# 2. NaN on the sequential evaluation path
# ---------------------------------------------------------------------------

def _sequential_evaluator(score):
    ev = PopulationEvaluator.__new__(PopulationEvaluator)
    ev.logger = LOG
    ev.metrics_tracker = None
    ev._task_error_counts = {}
    ev.use_parallel = False
    ev.num_processes = 1
    ev.task_builder = MagicMock()
    ev.task_builder.param_manager.denormalize_parameters = lambda x: {"p": float(x[0])}
    ev.task_builder.build_task = lambda **k: {"individual_id": 0, "params": k["params"]}
    ev.worker = MagicMock()
    ev.worker.evaluate.return_value = WorkerResult(individual_id=0, params={}, score=score)
    return ev


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_sequential_path_sanitises_non_finite(bad):
    ev = _sequential_evaluator(bad)
    assert ev.evaluate_solution(np.array([0.5])) == PENALTY
    np.testing.assert_array_equal(ev.evaluate_population(np.full((3, 1), 0.5)), [PENALTY] * 3)


def test_parallel_path_sanitises_inf():
    ev = _sequential_evaluator(0.0)
    fit = ev._extract_scores([{"individual_id": 0, "score": float("inf")},
                              {"individual_id": 1, "score": 0.4}], 2)
    np.testing.assert_array_equal(fit, [PENALTY, 0.4])


# ---------------------------------------------------------------------------
# 3. Multi-objective objectives are direction-transformed; NSGA-II init NaN
# ---------------------------------------------------------------------------

def test_metrics_dict_objectives_are_transformed():
    ev = _sequential_evaluator(0.0)
    results = [
        {"individual_id": 0, "metrics": {"KGE": 0.8, "RMSE": 10.0}},
        {"individual_id": 1, "metrics": {"kge": 0.5, "rmse": float("nan")}},
        {"individual_id": 2, "metrics": {"KGE": -999.0}},
    ]
    obj = ev._extract_objectives(results, 3, 2, objective_names=["KGE", "RMSE"])
    np.testing.assert_array_equal(obj[0], [0.8, -10.0])
    np.testing.assert_array_equal(obj[1], [0.5, PENALTY])
    np.testing.assert_array_equal(obj[2], [PENALTY, PENALTY])


def test_explicit_objectives_nan_sanitised():
    ev = _sequential_evaluator(0.0)
    obj = ev._extract_objectives([{"individual_id": 0, "objectives": [0.5, float("nan")]}], 1, 2)
    np.testing.assert_array_equal(obj[0], [0.5, PENALTY])


def _cb(record):
    def evaluate_population(pop, it):
        record.extend(np.array(pop, dtype=float))
        f = -np.sum((np.asarray(pop) - 0.3) ** 2, axis=1)
        f[0] = np.nan if it == 0 else f[0]
        return f

    def evaluate_solution(x, it):
        record.append(np.array(x, dtype=float))
        return float(-np.sum((np.asarray(x) - 0.3) ** 2))

    return dict(
        evaluate_solution=evaluate_solution,
        evaluate_population=evaluate_population,
        denormalize_params=lambda x: {f"p{i}": float(v) for i, v in enumerate(np.ravel(x))},
        record_iteration=lambda *a, **k: None,
        update_best=lambda *a, **k: None,
        log_progress=lambda *a, **k: None,
    )


def test_nsga2_initial_population_nan_is_cleaned():
    algo = NSGA2Algorithm({"NUMBER_OF_ITERATIONS": 1, "POPULATION_SIZE": 6,
                           "RANDOM_SEED": 1}, LOG)
    out = algo.optimize(n_params=2, **_cb([]))
    assert np.isfinite(out["best_score"])


# ---------------------------------------------------------------------------
# 4. Explicit failure signal instead of magnitude thresholds
# ---------------------------------------------------------------------------

def test_is_penalty_helper():
    assert is_penalty(PENALTY) and is_penalty(None) and is_penalty(float("nan"))
    assert not is_penalty(-1500.0)  # -RMSE of a large river
    np.testing.assert_array_equal(is_penalty(np.array([PENALTY, -1500.0, np.nan])),
                                  [True, False, True])
    assert sanitize_score(float("inf")) == PENALTY


def test_valid_score_accepts_large_negative_rmse():
    assert WorkerResult(0, {}, score=-1500.0).valid_score
    assert not WorkerResult(0, {}, score=PENALTY).valid_score


def test_update_best_accepts_large_negative_rmse():
    class _T(ResultsTrackingMixin):
        def __init__(self):
            self.logger = LOG
            self.__dict__["_config"] = {}
            self._best_score = float("-inf")
            self._best_params = None
            self._best_iteration = None

    t = _T()
    assert t.update_best(-1500.0, {"p": 1.0}, 3)
    assert t.best_score == -1500.0
    assert not t.update_best(PENALTY, {"p": 2.0}, 4)


def test_dds_counts_large_negative_scores_as_progress_not_crash():
    algo = DDSAlgorithm({"NUMBER_OF_ITERATIONS": 5, "RANDOM_SEED": 1}, LOG)
    cb = _cb([])
    cb["evaluate_solution"] = lambda x, it: -1500.0 - float(np.sum(x))
    out = algo.optimize(n_params=2, **cb)
    assert out["best_score"] < -1500.0 and out["best_score"] != PENALTY


# ---------------------------------------------------------------------------
# 5. Best-row selection
# ---------------------------------------------------------------------------

def test_best_row_uses_idxmax_on_maximised_score():
    df = pd.DataFrame({"iteration": [0, 1, 2], "score": [-50.0, -10.0, -80.0], "k": [1, 2, 3]})
    assert OptimizationManager._best_row_index(df) == 1


# ---------------------------------------------------------------------------
# 6. INITIAL_GUESS seeding
# ---------------------------------------------------------------------------

GUESS = np.array([0.123, 0.876])


@pytest.mark.parametrize("cls", [
    PSOAlgorithm, DEAlgorithm, SCEUAAlgorithm, GAAlgorithm, NSGA2Algorithm,
    SimulatedAnnealingAlgorithm, BasinHoppingAlgorithm, LBFGSAlgorithm,
    NelderMeadAlgorithm, DDSAlgorithm,
])
def test_initial_guess_is_evaluated(cls):
    np.random.seed(0)
    cfg = {"NUMBER_OF_ITERATIONS": 1, "POPULATION_SIZE": 6, "RANDOM_SEED": 0,
           "NUMBER_OF_COMPLEXES": 2, "LBFGS_STEPS": 1, "LR": 0.01}
    seen: list = []
    algo = cls(cfg, LOG)
    try:
        algo.optimize(n_params=2, initial_guess=GUESS, **_cb(seen))
    except Exception:  # noqa: BLE001 - only the first evaluations matter here
        pass
    assert any(np.allclose(x, GUESS) for x in seen), f"{cls.__name__} ignored initial_guess"


def test_cmaes_mean_starts_at_initial_guess(monkeypatch):
    class _ZeroNormal:
        def standard_normal(self, n):
            return np.zeros(n)

    # CMA-ES draws from its per-run Generator, not the global np.random.
    monkeypatch.setattr(CMAESAlgorithm, "_new_rng", lambda self: _ZeroNormal())
    seen: list = []
    CMAESAlgorithm({"NUMBER_OF_ITERATIONS": 1, "POPULATION_SIZE": 4}, LOG).optimize(
        n_params=2, initial_guess=GUESS, **_cb(seen))
    assert np.allclose(seen[0], GUESS)


def test_initial_guess_wrong_length_ignored():
    algo = PSOAlgorithm({}, LOG)
    assert algo._initial_guess(3, {"initial_guess": GUESS}) is None


# ---------------------------------------------------------------------------
# 7. Small numerical fixes
# ---------------------------------------------------------------------------

def test_fd_gradient_at_upper_bound_uses_actual_step():
    algo = PSOAlgorithm({}, LOG)
    _, grad = algo._compute_fd_gradients(np.array([1.0]), lambda x, s: float(2.0 * x[0]), 1e-3)
    assert grad[0] == pytest.approx(2.0)  # used to be 1.0 (divided by 2*eps)


def test_adam_moves_off_failed_point_and_stops():
    calls = []

    def grad(x):
        calls.append(x.copy())
        return float("nan"), np.full_like(x, np.nan)

    algo = AdamAlgorithm({"ADAM_STEPS": 50, "RANDOM_SEED": 0}, LOG)
    cb = _cb([])
    out = algo.optimize(n_params=2, compute_gradient=grad, gradient_mode="native", **cb)
    assert len(calls) == AdamAlgorithm.MAX_CONSECUTIVE_FAILURES  # stopped early
    assert not np.allclose(calls[0], calls[1])  # moved instead of stalling
    assert out["best_score"] == PENALTY


def test_log_bounds_chain_rule_matches_finite_difference():
    b = {"min": 1e-4, "max": 10.0, "transform": "log"}
    n, h = 0.4, 1e-6
    fd = (denormalize_value(b, n + h) - denormalize_value(b, n - h)) / (2 * h)
    assert denormalize_derivative(b, n) == pytest.approx(fd, rel=1e-6)
    assert denormalize_derivative({"min": 0.0, "max": 5.0}, n) == 5.0


def test_native_gradient_callback_uses_log_chain_rule():
    from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer

    class _Opt(BaseModelOptimizer):
        def _get_model_name(self):
            return "X"

        def _get_final_file_manager_path(self, *a, **k):
            return None

        def _run_model_for_final_evaluation(self, *a, **k):
            return None

    bounds = {"k": {"min": 1e-4, "max": 10.0, "transform": "log"}}
    opt = _Opt.__new__(_Opt)
    opt.logger = LOG
    opt.target_metric = "KGE"
    opt.param_manager = Mock()
    opt.param_manager.all_param_names = ["k"]
    opt.param_manager.get_parameter_bounds.return_value = bounds
    opt.param_manager.denormalize_parameters = lambda x: {"k": denormalize_value(bounds["k"], x[0])}
    opt.worker = Mock()
    opt.worker.supports_native_gradients.return_value = True
    opt.worker.evaluate_with_gradient.return_value = (0.1, {"k": 1.0})
    _, g = opt._create_gradient_callback()(np.array([0.4]))
    assert g[0] == pytest.approx(denormalize_derivative(bounds["k"], 0.4))


# ---------------------------------------------------------------------------
# Follow-up to 1: workers request the configured metric, not just KGE/NSE
# ---------------------------------------------------------------------------

class _StreamflowWorker(_Worker):
    """Mirrors the PRMS/VIC/CLM/... calculate_metrics call pattern."""

    def calculate_metrics(self, output_dir, config, **kwargs):
        obs, sim = kwargs["obs"], kwargs["sim"]
        return StreamflowMetrics().calculate_metrics(
            obs, sim, metrics=self._requested_metrics(config))


@pytest.mark.parametrize("metric,expected", [
    ("RMSE", lambda o, s: -np.sqrt(np.mean((o - s) ** 2))),
    ("MAE", lambda o, s: -np.mean(np.abs(o - s))),
    ("PBIAS", lambda o, s: -abs(100 * np.sum(s - o) / np.sum(o))),
    ("KGE", None),
])
def test_worker_scores_configured_minimise_metric(metric, expected):
    obs = np.linspace(1.0, 20.0, 50)
    good, bad = obs * 1.05, obs * 1.5
    cfg = {"OPTIMIZATION_METRIC": metric}
    w = _StreamflowWorker(cfg)
    s_good = w._extract_primary_score(w.calculate_metrics(None, cfg, obs=obs, sim=good), cfg)
    s_bad = w._extract_primary_score(w.calculate_metrics(None, cfg, obs=obs, sim=bad), cfg)
    assert np.isfinite(s_good) and s_good != PENALTY
    assert s_good > s_bad  # correctly oriented: the better sim scores higher
    if expected is not None:
        assert s_good == pytest.approx(expected(obs, good), rel=1e-6)


def test_requested_metrics_helper():
    w = _Worker({})
    assert w._requested_metrics({"OPTIMIZATION_METRIC": "RMSE"}) == ["rmse", "kge", "nse"]
    assert w._requested_metrics({"OPTIMIZATION_METRIC": "KGE"}) == ["kge", "nse"]
    comp = {"OPTIMIZATION_METRIC": "COMPOSITE",
            "COMPOSITE_METRIC": {"KGE_LOG": 0.5, "PBIAS": 0.3, "SNOW_SCORE": 0.2}}
    # SNOW_SCORE is not a streamflow metric: the worker adds it itself.
    assert w._requested_metrics(comp) == ["kge_log", "pbias", "kge", "nse"]
