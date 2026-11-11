# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Regression tests for the round-5 optimizer audit.

* every algorithm draws from its per-run Generator, never global ``np.random``;
* worker task seeds differ between generations;
* L-BFGS line search succeeds, never accepts a worse point, records its
  final iterate;
* a failed task is charged to the individual that failed, not individual 0;
* NSGA-II does not invent a second objective;
* Bayesian optimization keeps penalty scores out of the GP;
* crash statistics on the sequential population path;
* partial INITIAL_GUESS overlays model defaults;
* one canonical file slug per algorithm, previous results rotated;
* CMA-ES sigma stop is reachable; AsyncDDS budget / stagnation accounting;
* lazily-built parallel helpers are cached.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from symfluence.core.calibration.mixins.parallel.execution_strategies.sequential import (
    SequentialExecutionStrategy,
)
from symfluence.core.calibration.mixins.parallel_execution import ParallelExecutionMixin
from symfluence.core.calibration.mixins.results_tracking import ResultsTrackingMixin
from symfluence.core.calibration.optimizers.algorithms import (
    algorithm_slug,
    get_algorithm,
    list_algorithms,
)
from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.optimizers.evaluators.population_evaluator import (
    PopulationEvaluator,
)
from symfluence.core.calibration.optimizers.evaluators.task_builder import TaskBuilder
from symfluence.core.calibration.optimizers.metrics_tracker import EvaluationMetricsTracker
from symfluence.core.constants import ModelDefaults
from symfluence.optimization.optimization_manager import OptimizationManager

PENALTY = ModelDefaults.PENALTY_SCORE
LOG = logging.getLogger("test.optimizer_round5")


def sphere(x: np.ndarray) -> float:
    return -float(np.sum((np.asarray(x) - 0.3) ** 2))


class Run:
    """Minimal stand-in for the optimizer's callbacks around one algorithm run."""

    def __init__(self, objective):
        self.objective = objective
        self.evaluated: list = []
        self.batches: list = []
        self.records: list = []
        self.updates: list = []

    def evaluate_solution(self, x, proc_id=0):
        x = np.array(x, dtype=float)
        self.evaluated.append(x)
        return float(self.objective(x))

    def evaluate_population(self, population, iteration=0):
        self.batches.append(len(population))
        return np.array([self.evaluate_solution(x) for x in population])

    def record(self, iteration, score, params, additional_metrics=None):
        self.records.append((iteration, score, dict(params), additional_metrics))

    def update_best(self, score, params, iteration):
        self.updates.append((iteration, score, dict(params)))

    def go(self, name, config, n_params, logger=LOG, **kwargs):
        algorithm = get_algorithm(name, config, logger)
        return algorithm.optimize(
            n_params=n_params,
            evaluate_solution=self.evaluate_solution,
            evaluate_population=self.evaluate_population,
            denormalize_params=lambda x: {f"p{i}": float(v) for i, v in enumerate(x)},
            record_iteration=self.record,
            update_best=self.update_best,
            log_progress=lambda *a, **k: None,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 1. Per-run RNG
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list_algorithms())
def test_same_seed_same_trajectory_despite_global_rng_use(name):
    """Two runs with one RANDOM_SEED evaluate identical points, even when the
    objective itself consumes the process-global ``np.random`` stream (and the
    global stream starts in a different state each time)."""
    config = {"NUMBER_OF_ITERATIONS": 6, "POPULATION_SIZE": 8, "RANDOM_SEED": 11}

    def run(global_seed):
        np.random.seed(global_seed)

        def noisy_consumer(x):
            np.random.random(3)  # something else in the process using global RNG
            return sphere(x)

        r = Run(noisy_consumer)
        result = r.go(name, dict(config), 4, num_processes=1)
        return np.array(r.evaluated), result["best_score"]

    first, best_first = run(1)
    second, best_second = run(2)

    assert first.shape == second.shape
    np.testing.assert_array_equal(first, second)
    assert best_first == best_second


def test_different_seeds_give_different_trajectories():
    def run(seed):
        r = Run(sphere)
        r.go("dds", {"NUMBER_OF_ITERATIONS": 10, "RANDOM_SEED": seed}, 4)
        return np.array(r.evaluated)

    assert not np.array_equal(run(1), run(2))


def test_algorithm_immune_to_in_process_worker_reseeding():
    """A worker running in-process that reseeds the global RNG with the same
    seed every generation (SUMMA's pool fallback) must not rewind the search."""
    def run(reseed):
        class R(Run):
            def evaluate_population(self, population, iteration=0):
                out = []
                for i, x in enumerate(population):
                    if reseed:
                        np.random.seed(42 + i + 1000)
                    out.append(self.evaluate_solution(x))
                return np.array(out)

        r = R(sphere)
        r.go("pso", {"NUMBER_OF_ITERATIONS": 8, "POPULATION_SIZE": 6, "RANDOM_SEED": 42}, 4)
        return np.array(r.evaluated)

    np.testing.assert_array_equal(run(False), run(True))


# ---------------------------------------------------------------------------
# 2. Worker task seeds
# ---------------------------------------------------------------------------

def _task_builder(tmp_path, config=None, num_processes=2):
    param_manager = MagicMock()
    param_manager.denormalize_parameters.side_effect = lambda x: {"a": float(x[0])}
    param_manager.original_depths = None
    return TaskBuilder(
        config=config or {},
        project_dir=tmp_path,
        domain_name="d",
        optimization_settings_dir=tmp_path / "settings",
        default_sim_dir=tmp_path / "sim",
        parallel_dirs={},
        num_processes=num_processes,
        target_metric="KGE",
        param_manager=param_manager,
        logger=LOG,
    )


def test_task_seeds_do_not_repeat_between_generations(tmp_path):
    builder = _task_builder(tmp_path)
    population = np.full((4, 1), 0.5)

    gen0 = [t["random_seed"] for t in builder.build_population_tasks(population, 0, base_random_seed=42)]
    gen1 = [t["random_seed"] for t in builder.build_population_tasks(population, 1, base_random_seed=42)]

    assert gen0 == [1042, 1043, 1044, 1045]  # first batch keeps the historical values
    assert len(set(gen0 + gen1)) == 8        # used to be the same four seeds again

    # reproducible for a given base seed
    again = _task_builder(tmp_path)
    assert [t["random_seed"] for t in again.build_population_tasks(population, 0, base_random_seed=42)] == gen0


def test_task_seed_absent_without_base_seed(tmp_path):
    builder = _task_builder(tmp_path)
    for base in (None, "None"):
        tasks = builder.build_population_tasks(np.full((2, 1), 0.5), 0, base_random_seed=base)
        assert all("random_seed" not in t for t in tasks)


# ---------------------------------------------------------------------------
# 3 + 4. L-BFGS
# ---------------------------------------------------------------------------

def test_lbfgs_line_search_succeeds_on_sphere(caplog):
    r = Run(sphere)
    with caplog.at_level(logging.DEBUG, logger=LOG.name):
        result = r.go("lbfgs", {"NUMBER_OF_ITERATIONS": 20}, 6)

    failures = [rec for rec in caplog.records if "line search failed" in rec.getMessage()]
    assert not failures  # used to fail on 19 of 20 steps
    start_score = sphere(np.full(6, 0.5))
    assert result["best_score"] > start_score
    assert result["best_score"] > -1e-8
    # 13 evaluations per finite-difference gradient; the old search burned
    # ~20 gradients per failed step (several thousand evaluations in total).
    assert len(r.evaluated) <= 400


def test_lbfgs_beats_start_on_anisotropic_quadratic():
    def quad(x):
        return -float(np.sum(np.arange(1, len(x) + 1) * (x - 0.3) ** 2))

    r = Run(quad)
    result = r.go("lbfgs", {"NUMBER_OF_ITERATIONS": 30}, 6)
    assert result["best_score"] > -1e-6
    assert len(r.evaluated) <= 1000


def test_lbfgs_never_moves_to_a_worse_point():
    """Rugged objective: the trajectory of accepted iterates never decreases."""
    def rugged(x):
        z = (np.asarray(x) - 0.3) * 10
        return -float(np.sum(z * z - 10 * np.cos(2 * np.pi * z) + 10)) / len(x)

    r = Run(rugged)
    result = r.go("lbfgs", {"NUMBER_OF_ITERATIONS": 15}, 4)
    scores = [score for _, score, _, _ in r.records]
    assert all(b >= a for a, b in zip(scores, scores[1:]))
    assert result["best_score"] >= rugged(np.full(4, 0.5))


def test_lbfgs_line_search_rejects_non_improving_steps():
    algo = get_algorithm("lbfgs", {}, LOG)

    def eval_min(x):  # minimisation objective that increases along +x
        return float(np.sum(x)), np.ones_like(x)

    x = np.full(3, 0.5)
    g, grad = eval_min(x)
    # +grad is an ascent direction for g: no step may be returned
    step, g_new, _ = algo._line_search_with_gradient_func(x, grad, g, grad, eval_min, 1.0, 1e-4, 0.9)
    assert step is None and g_new == g
    # -grad is a descent direction: accepted, and strictly better
    step, g_new, _ = algo._line_search_with_gradient_func(x, -grad, g, grad, eval_min, 1.0, 1e-4, 0.9)
    assert step is not None and g_new < g


def test_lbfgs_records_final_iterate():
    r = Run(sphere)
    result = r.go("lbfgs", {"NUMBER_OF_ITERATIONS": 3}, 4)

    assert r.records[-1][1] == result["best_score"]
    assert r.records[-1][2] == result["best_params"]
    assert r.updates[-1][1] == result["best_score"]
    assert r.updates[-1][2] == result["best_params"]


# ---------------------------------------------------------------------------
# 5 + 8. Failed-task ids and crash statistics
# ---------------------------------------------------------------------------

def _evaluator(tracker=None):
    return PopulationEvaluator(None, None, None, False, 1, "X", LOG, metrics_tracker=tracker)


def _failing_worker(task):
    if task["individual_id"] == 2:
        raise RuntimeError("model blew up")
    return {"individual_id": task["individual_id"], "score": 0.5 + 0.1 * task["individual_id"]}


def test_sequential_failure_is_charged_to_the_failed_individual():
    tasks = [{"individual_id": i, "proc_id": 0} for i in range(4)]
    results = SequentialExecutionStrategy(LOG).execute(tasks, _failing_worker, 1)

    assert results[2]["individual_id"] == 2
    assert results[2]["score"] is None
    assert "model blew up" in results[2]["error"]

    scores = _evaluator()._extract_scores(results, 4)
    np.testing.assert_allclose(scores, [0.5, 0.6, PENALTY, 0.8])  # used to be [PENALTY, 0.6, PENALTY, 0.8]


def test_sequential_failure_objectives_charged_to_the_failed_individual():
    def worker(task):
        if task["individual_id"] == 1:
            raise ValueError("bad")
        return {"individual_id": task["individual_id"], "objectives": [0.7, 0.4]}

    results = SequentialExecutionStrategy(LOG).execute(
        [{"individual_id": i} for i in range(3)], worker, 1)
    objectives = _evaluator()._extract_objectives(results, 3, 2, objective_names=["KGE", "NSE"])

    np.testing.assert_allclose(objectives[0], [0.7, 0.4])
    np.testing.assert_allclose(objectives[1], [PENALTY, PENALTY])
    np.testing.assert_allclose(objectives[2], [0.7, 0.4])


@pytest.mark.parametrize("bad_id", [None, 7, -1, "0"])
def test_results_without_a_usable_id_are_skipped_not_assigned_to_zero(bad_id, caplog):
    results = [{"individual_id": 0, "score": 0.9}, {"individual_id": 1, "score": 0.8}]
    orphan = {"score": None, "error": "boom"}
    if bad_id is not None:
        orphan["individual_id"] = bad_id
    results.append(orphan)

    with caplog.at_level(logging.WARNING, logger=LOG.name):
        scores = _evaluator()._extract_scores(results, 3)

    np.testing.assert_allclose(scores, [0.9, 0.8, PENALTY])
    assert any("unusable individual_id" in rec.getMessage() for rec in caplog.records)

    objectives = _evaluator()._extract_objectives(
        [{"individual_id": 0, "objectives": [0.9, 0.1]}, orphan], 2, 2)
    np.testing.assert_allclose(objectives[0], [0.9, 0.1])


def test_sequential_population_path_tracks_crashes():
    tracker = EvaluationMetricsTracker(max_iterations=10, logger=LOG, elapsed_time_fn=lambda: "0s")
    evaluator = _evaluator(tracker)
    scores = iter([0.8, PENALTY, 0.7, PENALTY])
    evaluator.evaluate_solution = lambda params, proc_id=0: next(scores)

    evaluator.evaluate_population(np.zeros((4, 2)))

    stats = tracker.get_crash_stats()
    assert stats["total_evaluations"] == 4  # used to be 0 ("Crashes: 0/0")
    assert stats["crash_count"] == 2


# ---------------------------------------------------------------------------
# 6. NSGA-II objectives
# ---------------------------------------------------------------------------

NSGA_CFG = {"NUMBER_OF_ITERATIONS": 3, "POPULATION_SIZE": 8, "RANDOM_SEED": 1}


def test_nsga2_single_objective_writes_no_obj2_and_no_pareto(caplog):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("objectives evaluator used in single-objective mode")

    r = Run(sphere)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        result = r.go(
            "nsga2", dict(NSGA_CFG), 3,
            objective_names=["KGE", "NSE"], multiobjective=False,
            evaluate_population_objectives=must_not_be_called,
        )

    assert any("SINGLE-objective" in rec.getMessage() for rec in caplog.records)
    for _, _, _, extra in r.records:
        assert not any(key.startswith("obj2_") for key in (extra or {}))
    assert result["pareto_front"] is None
    assert result["pareto_objectives"] is None
    assert result["objective_names"] == ["KGE"]
    assert result["best_score"] == max(sphere(x) for x in r.evaluated)


def test_nsga2_two_objectives_are_evaluated_and_reported():
    calls = []

    def objectives(population, names, iteration):
        calls.append(list(names))
        pop = np.asarray(population)
        return np.column_stack([-np.sum((pop - 0.2) ** 2, axis=1), -np.sum((pop - 0.8) ** 2, axis=1)])

    r = Run(sphere)
    result = r.go(
        "nsga2", dict(NSGA_CFG), 3,
        objective_names=["KGE", "NSE"], multiobjective=True,
        evaluate_population_objectives=objectives,
    )

    assert calls and all(names == ["KGE", "NSE"] for names in calls)
    assert not r.evaluated  # the single-score evaluator is not used
    assert all("obj2_NSE" in extra for _, _, _, extra in r.records)
    front = result["pareto_objectives"]
    assert front.shape[1] == 2
    assert not np.allclose(front[:, 0], front[:, 1])  # a real trade-off, not a copy


def test_nsga2_falls_back_to_single_objective_for_whole_run():
    """All-penalty objectives -> single-objective for the rest of the run (it
    used to keep requesting the penalty objectives every generation)."""
    calls = []

    def dead_objectives(population, names, iteration):
        calls.append(iteration)
        return np.full((len(population), len(names)), PENALTY)

    r = Run(sphere)
    result = r.go(
        "nsga2", dict(NSGA_CFG), 3,
        objective_names=["KGE", "NSE"], multiobjective=True,
        evaluate_population_objectives=dead_objectives,
    )

    assert calls == [0]
    assert result["pareto_front"] is None
    assert result["best_score"] > PENALTY
    assert not any("obj2_NSE" in (extra or {}) for _, _, _, extra in r.records)


@pytest.mark.parametrize("multi_target,primary,secondary,target,expected", [
    (False, "KGE", "KGE", "NSE", (["NSE"], False)),        # default: one objective
    (False, "KGE", "NSE", "KGE", (["KGE", "NSE"], True)),  # two metrics, one target
    (True, "KGE", "KGE", "KGE", (["KGE", "KGE"], True)),   # two targets
])
def test_nsga2_objective_setup(multi_target, primary, secondary, target, expected):
    fake = SimpleNamespace(
        target_metric=target,
        _get_config_value=lambda accessor, default=None, dict_key=None: multi_target,
        _get_nsga2_objective_names=lambda: [primary, secondary],
    )
    assert BaseModelOptimizer._nsga2_objective_setup(fake) == expected


@pytest.mark.parametrize("config,expected", [
    ({"ITERATIVE_OPTIMIZATION_ALGORITHM": "NSGA-II"}, False),
    ({"ITERATIVE_OPTIMIZATION_ALGORITHM": "NSGA-II", "NSGA2_MULTI_TARGET": True}, True),
    ({"ITERATIVE_OPTIMIZATION_ALGORITHM": "MOEAD", "MOEAD_MULTI_TARGET": True}, True),
    ({"ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS"}, True),  # other callers: unchanged
])
def test_multi_target_mode_follows_the_multi_target_flag(tmp_path, config, expected):
    builder = _task_builder(tmp_path, config=config)
    task = builder.build_task(0, {"a": 1.0}, 0, "e", multiobjective=True, objective_names=["KGE", "NSE"])
    assert task["multiobjective"] is True
    assert task["multi_target_mode"] is expected


# ---------------------------------------------------------------------------
# 7. Bayesian optimization and penalty scores
# ---------------------------------------------------------------------------

def test_bo_surrogate_observations_revalue_penalties():
    algo = get_algorithm("bayesian_opt", {}, LOG)
    X = np.arange(8, dtype=float).reshape(4, 2)

    y = np.array([0.2, PENALTY, 0.6, float("nan")])
    X_fit, y_fit = algo._surrogate_observations(X, y)
    assert X_fit.shape == X.shape
    np.testing.assert_allclose(y_fit, [0.2, -0.2, 0.6, -0.2])  # worst valid - spread
    assert y[1] == PENALTY  # input untouched

    # one valid observation: still strictly below it
    _, y_fit = algo._surrogate_observations(X[:2], np.array([0.5, PENALTY]))
    assert y_fit[1] < 0.5 and y_fit[1] > PENALTY / 100

    # no valid observation: dropped
    X_fit, y_fit = algo._surrogate_observations(X[:2], np.array([PENALTY, PENALTY]))
    assert len(X_fit) == 0 and len(y_fit) == 0

    # nothing failed: returned as is
    X_fit, y_fit = algo._surrogate_observations(X[:2], np.array([0.1, 0.3]))
    np.testing.assert_array_equal(y_fit, [0.1, 0.3])


def test_bo_gp_never_sees_the_penalty_constant(monkeypatch):
    def crash_region(x):
        return PENALTY if x[0] > 0.6 else sphere(x)

    algo_cls = type(get_algorithm("bayesian_opt", {}, LOG))
    seen = []
    original = algo_cls._fit_gp

    def spy(self, X, y):
        seen.append(np.array(y))
        return original(self, X, y)

    monkeypatch.setattr(algo_cls, "_fit_gp", spy)
    r = Run(crash_region)
    result = r.go("bayesian_opt", {"NUMBER_OF_ITERATIONS": 6, "RANDOM_SEED": 3}, 3)

    assert seen
    assert any(crash_region(x) == PENALTY for x in r.evaluated)  # crashes did occur
    for y in seen:
        assert y.min() > -100  # not -9999
    assert result["best_score"] > PENALTY


def test_bo_runs_when_every_initial_sample_failed():
    class InitialSamplesFail(Run):
        def evaluate_population(self, population, iteration=0):
            self.batches.append(len(population))
            return np.full(len(population), PENALTY)

    r = InitialSamplesFail(sphere)
    result = r.go("bayesian_opt", {"NUMBER_OF_ITERATIONS": 6, "RANDOM_SEED": 3}, 3)
    assert len(r.evaluated) == 6  # random sampling until a run succeeds, then the GP
    assert result["best_score"] > PENALTY


# ---------------------------------------------------------------------------
# 9. Partial INITIAL_GUESS
# ---------------------------------------------------------------------------

def _guess_host(defaults, names=("a", "b", "c")):
    param_manager = MagicMock()
    param_manager.all_param_names = list(names)
    param_manager.get_initial_parameters.return_value = defaults
    return SimpleNamespace(param_manager=param_manager, logger=LOG)


def test_partial_initial_guess_overlays_model_defaults(caplog):
    host = _guess_host({"a": 1.0, "b": 2.0})
    with caplog.at_level(logging.INFO, logger=LOG.name):
        merged = BaseModelOptimizer._complete_initial_guess(host, {"a": 9.0})

    assert merged == {"a": 9.0, "b": 2.0}  # b from defaults, c has none
    messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO]
    assert any("model defaults for: b" in m for m in messages)
    assert any("no model default for: c" in m for m in messages)


def test_complete_initial_guess_is_left_alone():
    host = _guess_host({"a": 1.0, "b": 2.0, "c": 3.0})
    guess = {"a": 7.0, "b": 8.0, "c": 9.0}
    assert BaseModelOptimizer._complete_initial_guess(host, guess) == guess
    host.param_manager.get_initial_parameters.assert_not_called()


def test_initial_guess_survives_unreadable_defaults():
    host = _guess_host(None)
    host.param_manager.get_initial_parameters.side_effect = ValueError("no trial params")
    assert BaseModelOptimizer._complete_initial_guess(host, {"a": 9.0}) == {"a": 9.0}


# ---------------------------------------------------------------------------
# 10 + 11. Result file names and run-directory safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling,slug", [
    ("NSGA-II", "nsga2"), ("nsga2", "nsga2"), ("CMA-ES", "cmaes"), ("cmaes", "cmaes"),
    ("ASYNC-DDS", "async_dds"), ("async_dds", "async_dds"), ("MOEA-D", "moead"),
    ("moead", "moead"), ("DDS", "dds"), ("SCE-UA", "sce-ua"), ("L-BFGS", "lbfgs"),
    ("not-a-registered/algo", "not_a_registered_algo"),
])
def test_algorithm_slug_is_one_name_per_algorithm(spelling, slug):
    assert algorithm_slug(spelling) == slug


def test_every_registered_algorithm_has_a_filesystem_safe_slug():
    for name in list_algorithms():
        slug = algorithm_slug(name)
        assert slug == name
        assert "/" not in slug and "\\" not in slug and " " not in slug


def test_best_params_lookup_prefers_slug_and_still_finds_old_names(tmp_path):
    manager = SimpleNamespace(experiment_id="exp")
    find = OptimizationManager._find_best_params_json

    assert find(manager, tmp_path, "NSGA-II") is None

    legacy = tmp_path / "exp_nsga-ii_best_params.json"  # written by older releases
    legacy.write_text("{}", encoding="utf-8")
    assert find(manager, tmp_path, "nsga2") == legacy
    assert find(manager, tmp_path, "NSGA-II") == legacy

    current = tmp_path / "exp_nsga2_best_params.json"
    current.write_text("{}", encoding="utf-8")
    assert find(manager, tmp_path, "NSGA-II") == current


def test_progress_plots_go_to_the_run_directory(tmp_path):
    reporting = MagicMock()
    host = SimpleNamespace(
        reporting_manager=reporting,
        results_dir=tmp_path / "SUMMA" / "async_dds_exp",
        experiment_id="exp",
        target_metric="KGE",
        _iteration_history=[],
        _get_config_value=lambda accessor, default=None, dict_key=None: default,
    )
    BaseModelOptimizer._visualize_progress(host, "async_dds")

    args = reporting.visualize_optimization_progress.call_args[0]
    assert args[1] == host.results_dir  # used to be .../SUMMA/asyncdds_exp


class _Tracker(ResultsTrackingMixin):
    def __init__(self, results_dir: Path):
        self._config = {"EXPERIMENT_ID": "exp"}
        self.logger = LOG
        self.results_dir = results_dir
        self.__init_results_tracking__()

    def _get_config_value(self, accessor, default=None, dict_key=None):
        return self._config.get(dict_key, default)


def test_previous_results_are_rotated_not_overwritten(tmp_path, caplog):
    old_csv = tmp_path / "exp_parallel_iteration_results.csv"
    old_csv.write_text("iteration,score\n0,0.1\n1,0.9\n", encoding="utf-8")
    old_json = tmp_path / "exp_nsga-ii_best_params.json"
    old_json.write_text(json.dumps({"best_score": 0.9}), encoding="utf-8")
    other = tmp_path / "other_parallel_iteration_results.csv"  # another experiment
    other.write_text("x", encoding="utf-8")
    (tmp_path / "final_evaluation").mkdir()

    tracker = _Tracker(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        backups = tracker.rotate_previous_results()

    assert len(backups) == 2
    assert not old_csv.exists() and not old_json.exists()
    assert other.exists() and (tmp_path / "final_evaluation").is_dir()
    by_suffix = {b.suffix: b for b in backups}
    assert "0,0.1" in by_suffix[".csv"].read_text(encoding="utf-8")
    assert json.loads(by_suffix[".json"].read_text(encoding="utf-8")) == {"best_score": 0.9}
    assert any("previous run" in rec.getMessage() for rec in caplog.records)

    # the backups are invisible to the result-file globs used by readers
    assert not list(tmp_path.glob("exp_parallel_iteration_results.csv"))
    assert not list(tmp_path.glob("exp_*_best_params.json"))
    assert not [p for p in tmp_path.iterdir() if p.name.endswith("_best_params.json")]

    # the new run's first checkpoint no longer destroys anything
    tracker.record_iteration(0, 0.2, {"a": 1.0})
    tracker.save_results("dds", standard_filename=True)
    assert old_csv.exists()
    assert "0,0.1" in by_suffix[".csv"].read_text(encoding="utf-8")

    # rotating again keeps both generations of backups
    second = tracker.rotate_previous_results()
    assert len(second) == 1 and second[0] not in backups
    assert by_suffix[".csv"].exists()


def test_rotation_is_a_no_op_in_a_fresh_directory(tmp_path):
    assert _Tracker(tmp_path).rotate_previous_results() == []
    assert _Tracker(tmp_path / "missing").rotate_previous_results() == []


# ---------------------------------------------------------------------------
# 12. CMA-ES convergence stop
# ---------------------------------------------------------------------------

def test_cmaes_convergence_threshold_is_reachable():
    # sigma is clamped to >= SIGMA_MIN, so a smaller threshold can never fire
    assert CMAESDefaults.CONVERGENCE_THRESHOLD >= CMAESDefaults.SIGMA_MIN


def test_cmaes_stops_when_sigma_reaches_its_floor(monkeypatch, caplog):
    # Raise the floor so a short run reaches it.
    monkeypatch.setattr(CMAESDefaults, "SIGMA_MIN", 0.05)
    monkeypatch.setattr(CMAESDefaults, "CONVERGENCE_THRESHOLD", 0.05)
    r = Run(sphere)
    with caplog.at_level(logging.INFO, logger=LOG.name):
        result = r.go("cmaes", {"NUMBER_OF_ITERATIONS": 400, "POPULATION_SIZE": 8, "RANDOM_SEED": 5}, 3)

    assert any("CMA-ES converged" in rec.getMessage() for rec in caplog.records)
    assert r.records[-1][0] < 400
    assert result["final_sigma"] <= 0.05


def test_cmaes_healthy_run_is_not_stopped_early():
    r = Run(sphere)
    result = r.go("cmaes", {"NUMBER_OF_ITERATIONS": 40, "POPULATION_SIZE": 8, "RANDOM_SEED": 5}, 3)
    assert r.records[-1][0] == 40
    assert result["final_sigma"] > CMAESDefaults.SIGMA_MIN


# ---------------------------------------------------------------------------
# 13. DREAM R-hat
# ---------------------------------------------------------------------------

def test_dream_reports_rhat_from_chain_histories(caplog):
    r = Run(sphere)
    with caplog.at_level(logging.INFO, logger=LOG.name):
        result = r.go("dream", {"NUMBER_OF_ITERATIONS": 60, "POPULATION_SIZE": 8, "RANDOM_SEED": 2}, 2)

    assert np.isfinite(result["r_hat"]) and result["r_hat"] > 0.9
    assert any("max R-hat" in rec.getMessage() for rec in caplog.records)
    assert r.records[-1][0] == 60  # diagnostic only: no early stop


# ---------------------------------------------------------------------------
# 14. AsyncDDS accounting
# ---------------------------------------------------------------------------

ASYNC_CFG = {
    "NUMBER_OF_ITERATIONS": 30, "ASYNC_DDS_POOL_SIZE": 2, "ASYNC_DDS_BATCH_SIZE": 2,
    "MAX_STAGNATION_BATCHES": 3, "RANDOM_SEED": 4,
}


def test_async_dds_all_crash_batches_do_not_count_as_stagnation(caplog):
    state = {"n": 0}

    def crashes_after_pool(x):
        state["n"] += 1
        return sphere(x) if state["n"] <= 2 else PENALTY

    r = Run(crashes_after_pool)
    with caplog.at_level(logging.INFO, logger=LOG.name):
        r.go("async_dds", dict(ASYNC_CFG), 4, num_processes=2)

    # Budget = 30 batches x 2 = 60 runs, the initial pool of 2 included, and
    # crashed runs count: 29 batches. It used to stop after 3 "stagnant"
    # all-crash batches.
    assert len(r.evaluated) == 60
    assert len(r.batches) - 1 == 29
    assert not any("stagnation" in rec.getMessage().lower() for rec in caplog.records
                   if rec.levelno >= logging.INFO)
    assert any("Total evaluation budget: 60" in rec.getMessage() for rec in caplog.records)


def test_async_dds_counts_crashed_runs_against_the_budget():
    state = {"n": 0}

    def crashes_every_other(x):
        state["n"] += 1
        return PENALTY if state["n"] % 2 == 0 else sphere(x)

    r = Run(crashes_every_other)
    r.go("async_dds", dict(ASYNC_CFG, MAX_STAGNATION_BATCHES=1000), 4, num_processes=2)
    assert len(r.evaluated) == 60


def test_async_dds_still_stops_on_real_stagnation():
    state = {"n": 0}

    def never_improves(x):
        state["n"] += 1
        return 1.0 if state["n"] <= 2 else 0.0  # valid, but never better than the pool

    r = Run(never_improves)
    r.go("async_dds", dict(ASYNC_CFG), 4, num_processes=2)
    assert len(r.batches) - 1 == 3


# ---------------------------------------------------------------------------
# 16. Lazily-built parallel helpers
# ---------------------------------------------------------------------------

def test_parallel_helpers_are_built_once():
    class Host(ParallelExecutionMixin):
        def __init__(self):
            self._config = {"NUM_PROCESSES": 1}
            self.logger = LOG

    host = Host()
    assert host._directory_manager is host._directory_manager
    assert host._config_updater is host._config_updater
    assert host._worker_env_config is host._worker_env_config
