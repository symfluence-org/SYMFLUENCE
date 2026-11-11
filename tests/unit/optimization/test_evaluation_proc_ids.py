# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Every evaluation must carry a valid process id.

Nelder-Mead, simulated annealing, Bayesian optimisation and single-objective
MOEA/D call ``evaluate_solution(x, iteration)``. The optimizer's callback used
to forward that second argument as ``proc_id``, so on the sequential path those
evaluations were built against another process's directories (or, once the
counter passed NUM_PROCESSES, against the non-parallel default directories).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from symfluence.core.calibration.optimizers.algorithms import ALGORITHM_REGISTRY
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.optimizers.evaluators.population_evaluator import (
    PopulationEvaluator,
)
from symfluence.core.calibration.optimizers.evaluators.task_builder import TaskBuilder

pytestmark = [pytest.mark.unit, pytest.mark.optimization]

LOG = logging.getLogger("test.evaluation_proc_ids")
N_PARAMS = 3
CONFIG = {
    "OPTIMIZATION_METRIC": "KGE",
    "NUMBER_OF_ITERATIONS": 6,
    "POPULATION_SIZE": 6,
    "RANDOM_SEED": 7,
    "NUM_PROCESSES": 1,
}


def _score(params) -> float:
    return 1.0 - float(sum((v - 0.3) ** 2 for v in params.values()))


class _Harness:
    """Real callbacks -> real evaluator -> real task builder, stub worker."""

    def __init__(self, tmp_path, num_processes=1):
        self.proc_ids: list = []
        self.settings_dirs: list = []
        param_manager = MagicMock()
        param_manager.denormalize_parameters.side_effect = (
            lambda x: {f"p{i}": float(v) for i, v in enumerate(np.atleast_1d(x))}
        )
        param_manager.original_depths = None
        parallel_dirs = {
            pid: {
                "settings_dir": tmp_path / f"process_{pid}" / "settings",
                "sim_dir": tmp_path / f"process_{pid}" / "sim",
                "root": tmp_path / f"process_{pid}",
            }
            for pid in range(num_processes)
        }
        builder = TaskBuilder(
            config=dict(CONFIG),
            project_dir=tmp_path,
            domain_name="d",
            optimization_settings_dir=tmp_path / "default_settings",
            default_sim_dir=tmp_path / "default_sim",
            parallel_dirs=parallel_dirs,
            num_processes=num_processes,
            target_metric="KGE",
            param_manager=param_manager,
            logger=LOG,
        )
        harness = self

        class _Worker:
            def evaluate(self, task):
                harness._seen(task.proc_id, task.settings_dir)
                return SimpleNamespace(score=_score(task.params))

        def execute_batch(tasks, worker_func):
            return [worker_func(t) for t in tasks]

        def batch_worker(task):
            harness._seen(task["proc_id"], task.get("proc_settings_dir"))
            score = _score(task["params"])
            return {
                "individual_id": task["individual_id"],
                "score": score,
                "objectives": [score, score - 0.1],
            }

        self.evaluator = PopulationEvaluator(
            builder, _Worker(), execute_batch, False, num_processes, "X", LOG,
        )
        self.evaluator._resolve_worker_function = lambda: batch_worker
        self.builder = builder

        fake = SimpleNamespace(
            _get_config_value=lambda *a, **k: 10,
            get_crash_stats=lambda: {"crash_count": 0, "crash_rate": 0},
            record_iteration=lambda *a, **k: None,
            update_best=lambda *a, **k: None,
            save_results=lambda *a, **k: None,
            save_best_params=lambda *a, **k: None,
            log_iteration_progress=lambda *a, **k: None,
            log_initial_population=lambda *a, **k: None,
            logger=LOG,
            results_dir=tmp_path,
            experiment_id="test",
            num_processes=num_processes,
            population_evaluator=self.evaluator,
            _metrics_tracker=SimpleNamespace(track_evaluation=lambda s: None),
            param_manager=param_manager,
        )
        fake._evaluate_solution = (
            lambda x, **kw: BaseModelOptimizer._evaluate_solution(fake, x, **kw)
        )
        fake._evaluate_population = (
            lambda pop, iteration=0: self.evaluator.evaluate_population(pop, iteration)
        )
        self.callbacks, self.kwargs = BaseModelOptimizer._build_algorithm_callbacks(fake, "X")

    def _seen(self, proc_id, settings_dir):
        self.proc_ids.append(proc_id)
        self.settings_dirs.append(settings_dir)


def _fd_gradient(evaluate):
    def compute(x):
        return evaluate(x, 0), np.zeros_like(x)
    return compute


@pytest.mark.parametrize("name", sorted(ALGORITHM_REGISTRY))
@pytest.mark.parametrize("multiobjective", [False, True])
def test_registered_algorithm_only_uses_valid_proc_ids(tmp_path, name, multiobjective):
    harness = _Harness(tmp_path)
    algorithm = ALGORITHM_REGISTRY[name](dict(CONFIG), LOG)

    extra = {}
    if multiobjective:
        extra = {
            "evaluate_population_objectives": (
                lambda pop, names, iteration=0:
                harness.evaluator.evaluate_population_objectives(pop, names, iteration)
            ),
            "objective_names": ["KGE", "NSE"],
            "multiobjective": True,
        }

    algorithm.optimize(n_params=N_PARAMS, **harness.callbacks, **harness.kwargs, **extra)

    assert harness.proc_ids, "the algorithm evaluated nothing"
    assert set(harness.proc_ids) == {0}, (
        f"{name} evaluated with process ids {sorted(set(harness.proc_ids))}; "
        "with one process the only valid id is 0"
    )


def test_step_counter_is_not_a_process_id(tmp_path):
    """evaluate_solution(x, 5) on a 2-process run still uses process 0."""
    harness = _Harness(tmp_path, num_processes=2)
    harness.callbacks["evaluate_solution"](np.full(N_PARAMS, 0.5), 5)
    harness.callbacks["evaluate_solution"](np.full(N_PARAMS, 0.5), 1)
    assert harness.proc_ids == [0, 0]
    assert all("process_0" in str(d) for d in harness.settings_dirs)


def test_proc_id_cannot_be_passed_positionally(tmp_path):
    harness = _Harness(tmp_path)
    with pytest.raises(TypeError):
        harness.evaluator.evaluate_solution(np.full(N_PARAMS, 0.5), 3)


@pytest.mark.parametrize("bad", [-1, 1, 7, 1.0, None, True])
def test_task_builder_rejects_out_of_range_proc_id(tmp_path, bad):
    harness = _Harness(tmp_path)
    with pytest.raises(ValueError, match="Invalid proc_id"):
        harness.builder.build_task(0, {"p0": 0.5}, proc_id=bad, evaluation_id="x")
