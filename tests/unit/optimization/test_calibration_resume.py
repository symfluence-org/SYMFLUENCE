# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Calibration checkpoint/resume (CALIBRATION_RESUME / ``--resume``).

A resumed run replays the logged evaluations through the algorithm and then
continues; a seeded resumed run must end exactly where an uninterrupted one
does, for every registered algorithm.
"""
from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from symfluence.core.calibration.optimizers.algorithms import get_algorithm, list_algorithms
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.optimizers.metrics_tracker import EvaluationMetricsTracker
from symfluence.core.calibration.optimizers.resume import (
    CalibrationResumeError,
    EvaluationLog,
    load_resume_state,
    make_fingerprint,
    resume_state_path,
    write_json_atomic,
)

LOG = logging.getLogger("test.calibration_resume")


class Interrupted(Exception):
    """Stands in for the process dying mid-calibration."""


def sphere(x: np.ndarray) -> float:
    return -float(np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))


def two_objectives(population, names, iteration=0):
    pop = np.asarray(population, dtype=float)
    return np.column_stack([
        -np.sum((pop - 0.3) ** 2, axis=1),
        -np.sum((pop - 0.6) ** 2, axis=1),
    ])


# ---------------------------------------------------------------------------
# Algorithm level: every registered algorithm replays exactly
# ---------------------------------------------------------------------------

class Harness:
    """The optimizer's callbacks around one algorithm run, with an evaluation
    budget after which the 'process dies'."""

    def __init__(self, log: EvaluationLog, max_requests: Optional[int] = None):
        self.log = log
        self.max_requests = max_requests
        self.requests = 0
        self.model_runs = 0
        self.records: List = []

    def _charge(self, n: int = 1) -> None:
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise Interrupted()
        self.requests += 1
        self.model_runs += n

    def evaluate_solution(self, x, step_id=0):
        self._charge()
        return sphere(x)

    def evaluate_population(self, population, iteration=0):
        self._charge(len(population))
        return np.array([sphere(x) for x in population])

    def evaluate_objectives(self, population, names, iteration=0):
        self._charge(len(population))
        return two_objectives(population, names, iteration)

    def go(self, name: str, config: Dict[str, Any], n_params: int, algorithm=None):
        algorithm = algorithm or get_algorithm(name, dict(config), LOG)
        kwargs: Dict[str, Any] = {'num_processes': 1}
        if name == 'nsga2':
            kwargs['evaluate_population_objectives'] = self.log.wrap_objectives(
                self.evaluate_objectives)
            kwargs['objective_names'] = ['KGE', 'NSE']
            kwargs['multiobjective'] = True
        result = algorithm.optimize(
            n_params=n_params,
            evaluate_solution=self.log.wrap_solution(self.evaluate_solution),
            evaluate_population=self.log.wrap_population(self.evaluate_population),
            denormalize_params=lambda x: {f"p{i}": float(v) for i, v in enumerate(x)},
            record_iteration=lambda it, score, params, additional_metrics=None: (
                self.records.append((it, score, dict(params)))),
            update_best=lambda score, params, it: None,
            log_progress=lambda *a, **k: None,
            **kwargs,
        )
        return algorithm, result


ALGO_CONFIG = {"NUMBER_OF_ITERATIONS": 6, "POPULATION_SIZE": 8, "RANDOM_SEED": 7}


@pytest.mark.parametrize("name", list_algorithms())
def test_every_algorithm_resumes_to_the_uninterrupted_result(name):
    full = Harness(EvaluationLog())
    _, expected = full.go(name, ALGO_CONFIG, 4)
    total = full.model_runs
    assert full.requests >= 2

    # Die half-way, keeping the requests that completed.
    dying = Harness(EvaluationLog(), max_requests=max(1, full.requests // 2))
    with pytest.raises(Interrupted):
        dying.go(name, ALGO_CONFIG, 4)
    completed = dying.log.entries
    assert completed, "nothing was evaluated before the interruption"

    resumed = Harness(EvaluationLog(completed))
    _, got = resumed.go(name, ALGO_CONFIG, 4)

    # Only the evaluations after the checkpoint ran the 'model' again ...
    assert resumed.model_runs == total - dying.model_runs
    # ... and the run ended exactly where the uninterrupted one did.
    assert got['best_score'] == expected['best_score']
    np.testing.assert_array_equal(got['best_solution'], expected['best_solution'])
    assert resumed.records == full.records
    assert len(resumed.log.entries) == len(full.log.entries)


@pytest.mark.parametrize("name", ["dds", "de"])
def test_unseeded_run_resumes_with_its_original_entropy(name):
    config = {k: v for k, v in ALGO_CONFIG.items() if k != "RANDOM_SEED"}
    full = Harness(EvaluationLog())
    algorithm, expected = full.go(name, config, 3)
    entropy = algorithm._run_entropy

    dying = Harness(EvaluationLog(), max_requests=max(1, full.requests // 2))
    first = get_algorithm(name, dict(config), LOG)
    first._resume_entropy = entropy  # what the checkpointer does on resume
    with pytest.raises(Interrupted):
        dying.go(name, config, 3, algorithm=first)

    second = get_algorithm(name, dict(config), LOG)
    second._resume_entropy = entropy
    resumed = Harness(EvaluationLog(dying.log.entries))
    _, got = resumed.go(name, config, 3, algorithm=second)
    assert got['best_score'] == expected['best_score']


def test_replay_mismatch_is_an_error_not_a_silent_restart():
    log = EvaluationLog([{'kind': 'solution', 'x': [0.1, 0.2], 'result': -1.0}])
    evaluate = log.wrap_solution(lambda x, s=0: pytest.fail("must not run the model"))
    with pytest.raises(CalibrationResumeError, match="diverged"):
        evaluate(np.array([0.1, 0.25]))


def test_native_gradient_requests_are_logged_and_replayed():
    calls = []

    def grad(x):
        calls.append(np.array(x))
        return float(np.sum(x ** 2)), 2 * np.asarray(x)

    log = EvaluationLog()
    wrapped = log.wrap_gradient(grad)
    loss, g = wrapped(np.array([0.5, 0.25]))
    replay = EvaluationLog(log.entries).wrap_gradient(lambda x: pytest.fail("ran the model"))
    loss2, g2 = replay(np.array([0.5, 0.25]))
    assert loss2 == loss and np.array_equal(g2, g) and len(calls) == 1


# ---------------------------------------------------------------------------
# Optimizer level: checkpoint files, CSV append, fingerprint
# ---------------------------------------------------------------------------

class _ParamManager:
    def __init__(self, n: int, upper: float = 10.0):
        self.all_param_names = [f"p{i}" for i in range(n)]
        self.bounds = {name: {'min': 0.0, 'max': upper} for name in self.all_param_names}

    def get_parameter_bounds(self):
        return dict(self.bounds)

    def denormalize_parameters(self, x):
        return {name: float(v) * self.bounds[name]['max'] for name, v in zip(self.all_param_names, x)}

    def normalize_parameters(self, params):
        return np.array([params[n] / self.bounds[n]['max'] for n in self.all_param_names])

    def get_initial_parameters(self):
        return None


class _Optimizer(BaseModelOptimizer):
    """BaseModelOptimizer around a synthetic objective (no model, no workers)."""

    budget: Optional[int] = None

    def __init__(self, config: Dict[str, Any], results_dir: Path, n_params: int = 3,
                 upper: float = 10.0):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = LOG
        self.results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        self.__init_results_tracking__()
        self.param_manager = _ParamManager(n_params, upper)
        self.max_iterations = config['NUMBER_OF_ITERATIONS']
        self.population_size = config.get('POPULATION_SIZE', 8)
        self.target_metric = 'KGE'
        # As BaseModelOptimizer.__init__: an unset RANDOM_SEED is drawn and recorded.
        from symfluence.core.provenance import resolve_seed
        self.random_seed, self.random_seed_source = resolve_seed(
            config.get('RANDOM_SEED'), LOG, context="calibration")
        self._runtime_overrides = {}
        self._metrics_tracker = EvaluationMetricsTracker(self.max_iterations, LOG, lambda: "0s")
        self._resume_checkpointer = None
        self.model_runs = 0

    # -- BaseModelOptimizer abstract hooks / heavy collaborators ----------
    def _get_model_name(self):
        return "SYNTH"

    def _get_final_file_manager_path(self):
        return self.results_dir

    def _run_model_for_final_evaluation(self, output_dir):
        return True

    def _log_calibration_alignment(self):
        pass

    def _visualize_progress(self, algorithm):
        pass

    def run_final_evaluation(self, best_params):
        return None

    def _charge(self, n: int) -> None:
        if self.budget is not None and self.model_runs + n > self.budget:
            raise Interrupted()
        self.model_runs += n

    def _evaluate_solution(self, normalized_params, *, proc_id=0):
        self._charge(1)
        score = sphere(normalized_params)
        self._metrics_tracker.track_evaluation(score)
        return score

    def _evaluate_population(self, population, iteration=0):
        self._charge(len(population))
        return np.array([sphere(x) for x in population])


def _config(tmp_path: Path, algorithm: str, **extra) -> Dict[str, Any]:
    cfg = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'synth',
        'EXPERIMENT_ID': 'resume_test',
        'EXPERIMENT_TIME_START': '2000-01-01 00:00',
        'EXPERIMENT_TIME_END': '2000-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'HYDROLOGICAL_MODEL': 'GR',
        'FORCING_DATASET': 'ERA5',
        'ITERATIVE_OPTIMIZATION_ALGORITHM': algorithm.upper(),
        'NUMBER_OF_ITERATIONS': 40 if algorithm == 'dds' else 6,
        'POPULATION_SIZE': 8,
        'RANDOM_SEED': 42,
        'CHECKPOINT_INTERVAL': 2,
        'SKIP_WARM_START': True,
    }
    cfg.update(extra)
    return cfg


def _run(tmp_path, algorithm, results_dir, budget=None, **extra):
    opt = _Optimizer(_config(tmp_path, algorithm, **extra), results_dir)
    opt.budget = budget
    if budget is not None:
        with pytest.raises(Interrupted):
            opt._run_optimization_impl(algorithm)
    else:
        opt._run_optimization_impl(algorithm)
    return opt


def _results_csv(results_dir: Path) -> pd.DataFrame:
    return pd.read_csv(results_dir / 'resume_test_parallel_iteration_results.csv')


@pytest.mark.parametrize("algorithm", ["dds", "de"])
def test_resumed_seeded_run_matches_uninterrupted_run(tmp_path, algorithm, caplog):
    reference = _run(tmp_path, algorithm, tmp_path / 'reference')
    total = reference.model_runs

    run_dir = tmp_path / 'run'
    dead = _run(tmp_path, algorithm, run_dir, budget=total // 2)
    state_file = resume_state_path(run_dir, 'resume_test', algorithm)
    state = load_resume_state(state_file)
    assert not state['complete']
    assert 0 < state['n_evaluations'] <= dead.model_runs
    assert state['rng']['state'] is not None
    rows_before = _results_csv(run_dir)

    with caplog.at_level(logging.INFO, logger=LOG.name):
        resumed = _run(tmp_path, algorithm, run_dir, CALIBRATION_RESUME=True)
    assert "replay verified against the checkpoint" in caplog.text

    # Only the evaluations after the last checkpoint were run again.
    assert resumed.model_runs == total - state['n_evaluations']
    assert resumed.best_score == reference.best_score
    assert resumed.best_params == reference.best_params
    assert resumed.best_iteration == reference.best_iteration

    got, expected = _results_csv(run_dir), _results_csv(tmp_path / 'reference')
    cols = [c for c in expected.columns if c != 'timestamp']
    pd.testing.assert_frame_equal(got[cols], expected[cols])
    # Appended, not rewritten: the checkpointed rows keep their timestamps.
    n_kept = state['n_history']
    assert list(got['timestamp'][:n_kept]) == list(rows_before['timestamp'][:n_kept])
    # Nothing was rotated aside.
    assert not list(run_dir.glob('*.prev-*'))
    assert load_resume_state(state_file)['complete']

    best = json.loads((run_dir / f'resume_test_{algorithm}_best_params.json').read_text())
    assert best['best_score'] == reference.best_score


def test_unseeded_optimizer_run_resumes_with_its_drawn_seed(tmp_path, caplog):
    """A drawn seed is restored on resume, so the run replays and ends as a seeded one would."""
    run_dir = tmp_path / 'run'
    dead = _run(tmp_path, 'dds', run_dir, budget=20, RANDOM_SEED=None)
    assert dead.random_seed_source == 'drawn'
    drawn = dead.random_seed
    state = load_resume_state(resume_state_path(run_dir, 'resume_test', 'dds'))
    assert state['fingerprint']['fields']['random_seed'] is None
    assert state['rng']['entropy'] == drawn

    with caplog.at_level(logging.INFO, logger=LOG.name):
        resumed = _run(tmp_path, 'dds', run_dir, CALIBRATION_RESUME=True, RANDOM_SEED=None)
    assert "replay verified against the checkpoint" in caplog.text
    assert resumed.random_seed == drawn

    reference = _run(tmp_path, 'dds', tmp_path / 'reference', RANDOM_SEED=drawn)
    assert resumed.best_score == reference.best_score
    assert resumed.best_params == reference.best_params
    conditions = json.loads(
        (run_dir / 'resume_test_dds_calibration_conditions.json').read_text())
    assert conditions['random_seed'] == drawn
    assert conditions['random_seed_source'] == 'drawn'


def test_resume_refuses_changed_bounds(tmp_path):
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=15)
    opt = _Optimizer(_config(tmp_path, 'dds', CALIBRATION_RESUME=True), run_dir, upper=20.0)
    with pytest.raises(CalibrationResumeError, match="bounds"):
        opt._run_optimization_impl('dds')
    assert opt.model_runs == 0


def test_resume_refuses_changed_seed(tmp_path):
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=15)
    with pytest.raises(CalibrationResumeError, match="random_seed"):
        _run(tmp_path, 'dds', run_dir, CALIBRATION_RESUME=True, RANDOM_SEED=43)


def test_resume_refuses_changed_budget_mode(tmp_path):
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=15)
    with pytest.raises(CalibrationResumeError, match="budget"):
        _run(tmp_path, 'dds', run_dir, CALIBRATION_RESUME=True,
             CALIBRATION_BUDGET_MODE='per_parameter', EVALUATIONS_PER_PARAMETER=20)


def test_resume_refuses_changed_forcing_correction(tmp_path, monkeypatch):
    from symfluence.core.calibration.forcing_correction import ForcingCorrectionPlan

    def plan(multiplier):
        return ForcingCorrectionPlan(mode='fixed', model='SYNTH', fixed={'precip': multiplier})

    run_dir = tmp_path / 'run'
    monkeypatch.setattr(_ParamManager, 'forcing_correction', plan(1.1), raising=False)
    _run(tmp_path, 'dds', run_dir, budget=15)
    monkeypatch.setattr(_ParamManager, 'forcing_correction', plan(1.3), raising=False)
    with pytest.raises(CalibrationResumeError, match="forcing_correction"):
        _run(tmp_path, 'dds', run_dir, CALIBRATION_RESUME=True)


def test_resume_with_changed_iterations_diverges_loudly(tmp_path):
    """DDS's perturbation schedule depends on NUMBER_OF_ITERATIONS, so the
    replay cannot match; that must stop the run, not continue elsewhere."""
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=25)
    opt = _Optimizer(
        _config(tmp_path, 'dds', CALIBRATION_RESUME=True, NUMBER_OF_ITERATIONS=80), run_dir)
    with pytest.raises(CalibrationResumeError, match="diverged"):
        opt._run_optimization_impl('dds')


def test_resume_without_checkpoint_starts_fresh_with_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        opt = _run(tmp_path, 'dds', tmp_path / 'run', CALIBRATION_RESUME=True)
    assert "no resume state" in caplog.text
    assert opt.model_runs == 41


def test_fresh_run_rotates_previous_resume_state(tmp_path):
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=15)
    _run(tmp_path, 'dds', run_dir)
    assert list(run_dir.glob('resume_test_dds_resume_state.prev-*.json'))
    assert load_resume_state(resume_state_path(run_dir, 'resume_test', 'dds'))['complete']


def test_algorithm_without_resume_support_raises(tmp_path, monkeypatch):
    from symfluence.core.calibration.optimizers.algorithms.dds import DDSAlgorithm
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=15)
    monkeypatch.setattr(DDSAlgorithm, 'supports_resume', False)
    with pytest.raises(CalibrationResumeError, match="does not support calibration resume"):
        _run(tmp_path, 'dds', run_dir, CALIBRATION_RESUME=True)


def test_completed_run_resumes_without_model_runs(tmp_path):
    run_dir = tmp_path / 'run'
    first = _run(tmp_path, 'de', run_dir)
    again = _run(tmp_path, 'de', run_dir, CALIBRATION_RESUME=True)
    assert again.model_runs == 0
    assert again.best_score == first.best_score


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    target = tmp_path / 'state.json'
    write_json_atomic(target, {'a': 1})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr('symfluence.core.calibration.optimizers.resume.os.replace', boom)
    with pytest.raises(OSError):
        write_json_atomic(target, {'a': 2})
    assert json.loads(target.read_text()) == {'a': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['state.json']


def test_fingerprint_is_order_independent():
    a = make_fingerprint({'bounds': {'x': [0, 1], 'y': [1, 2]}, 'metric': 'KGE'})
    b = make_fingerprint({'metric': 'KGE', 'bounds': {'y': [1, 2], 'x': [0, 1]}})
    assert a['sha256'] == b['sha256']


@pytest.mark.parametrize("flags,expected", [
    ({}, {}),
    ({'resume_calibration': True}, {'CALIBRATION_RESUME': True}),
    ({'resume_calibration': True, 'force_rerun': True},
     {'CALIBRATION_RESUME': True, 'FORCE_RUN_ALL_STEPS': True}),
])
def test_resume_flag_sets_config_override(flags, expected):
    from symfluence.cli.commands.workflow_commands import _force_rerun_overrides
    assert _force_rerun_overrides(Namespace(**flags)) == expected


@pytest.mark.parametrize("argv", [
    ['workflow', 'run', '--resume'],
    ['workflow', 'step', 'calibrate_model', '--resume'],
    ['workflow', 'steps', 'calibrate_model', '--resume'],
])
def test_cli_accepts_resume(argv):
    from symfluence.cli.argument_parser import CLIParser
    args = CLIParser().parse_args(argv)
    assert args.resume_calibration is True


def test_resume_switch_does_not_invalidate_calibration_marker():
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.core.stage_marker import compute_stage_hash

    base = dict(_config(Path('/tmp'), 'dds'))
    plain = SymfluenceConfig(**base)
    resumed = SymfluenceConfig(**{**base, 'CALIBRATION_RESUME': True})
    assert compute_stage_hash(plain, 'calibrate_model') == compute_stage_hash(resumed, 'calibrate_model')


def test_rng_state_mismatch_after_replay_is_an_error(tmp_path):
    run_dir = tmp_path / 'run'
    _run(tmp_path, 'dds', run_dir, budget=15)
    path = resume_state_path(run_dir, 'resume_test', 'dds')
    state = json.loads(path.read_text())
    state['rng']['state']['state']['state'] += 1  # PCG64 internal state
    write_json_atomic(path, state)
    with pytest.raises(CalibrationResumeError, match="random-number generator"):
        _run(tmp_path, 'dds', run_dir, CALIBRATION_RESUME=True)
