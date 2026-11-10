# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Population Evaluator for Optimization

Handles batch evaluation of parameter populations with parallel execution.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from symfluence.core.constants import ModelDefaults
from symfluence.core.logging_utils import log_once
from symfluence.core.metrics.metric_transformer import MetricTransformer
from symfluence.core.metrics.penalty import is_failure_sentinel, is_penalty, sanitize_score

from .task_builder import TaskBuilder


class PopulationEvaluator:
    """
    Evaluates populations of solutions for optimization algorithms.

    Handles:
    - Single solution evaluation
    - Population batch evaluation (single objective)
    - Population batch evaluation (multi-objective)
    - Parallel execution coordination
    - Result extraction and error handling
    """

    DEFAULT_PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    def __init__(
        self,
        task_builder: TaskBuilder,
        worker: Any,
        execute_batch: Callable,
        use_parallel: bool,
        num_processes: int,
        model_name: str,
        logger: Optional[logging.Logger] = None,
        metrics_tracker: Optional[Any] = None
    ):
        """
        Initialize population evaluator.

        Args:
            task_builder: Task builder instance
            worker: Worker instance for evaluations
            execute_batch: Batch execution function
            use_parallel: Whether to use parallel execution
            num_processes: Number of parallel processes
            model_name: Model name (e.g., 'SUMMA', 'FUSE')
            logger: Optional logger instance
            metrics_tracker: Optional EvaluationMetricsTracker. Batch
                evaluations report each individual's outcome to it, so
                population-based algorithms count crashes like the
                single-solution path already does. Without it the progress
                line reports ``Crashes: 0/0`` no matter how many individuals
                crashed.
        """
        self.task_builder = task_builder
        self.worker = worker
        self.execute_batch = execute_batch
        self.use_parallel = use_parallel
        self.num_processes = num_processes
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_tracker = metrics_tracker
        # Occurrence counters for identical task-error messages, so a broken
        # setup that fails every evaluation logs one ERROR instead of thousands
        # of identical WARNINGs.
        self._task_error_counts: Dict[str, int] = {}

    def _track_batch(self, scores: np.ndarray) -> None:
        """Report one batch of evaluation outcomes to the metrics tracker.

        ``scores`` holds the primary score per individual, already defaulted to
        the penalty score for anything that crashed or returned nothing, which
        is exactly what the tracker treats as a crash.
        """
        if self.metrics_tracker is None:
            return
        for score in scores:
            self.metrics_tracker.track_evaluation(float(score))

    def _log_task_error(self, idx: Any, error: Any, result: Dict) -> None:
        """
        Log a worker task error, de-duplicated by exception message.

        The first occurrence of a given message is logged at ERROR (with the
        worker traceback when the result carries one); identical repeats are
        logged at DEBUG with an occurrence counter.
        """
        error_str = str(error)
        count = self._task_error_counts.get(error_str, 0) + 1
        self._task_error_counts[error_str] = count

        truncated = error_str[:500] if len(error_str) > 500 else error_str
        if count == 1:
            traceback_str = result.get('traceback')
            message = f"Task {idx} worker error: {truncated}"
            if traceback_str:
                message += f"\n{traceback_str}"
            self.logger.error(message)
        else:
            self.logger.debug(
                f"Task {idx} worker error (repeat #{count} of identical message): {truncated}"
            )

    def _log_all_penalty_batch(self, n_results: int, kind: str) -> None:
        """
        Emit one actionable ERROR when an entire batch returned only penalty
        values (first occurrence only; repeats at DEBUG via log_once).
        """
        log_once(
            self.logger,
            logging.ERROR,
            key=f'population-all-penalty-{kind}',
            message=(
                f"All {n_results} individuals in this population returned penalty "
                f"{kind}. This indicates a broken model or forcing setup for this "
                f"run rather than a poor parameter region - check the first task "
                f"error logged above, and verify the model runs outside calibration "
                f"before continuing."
            ),
        )

    def _resolve_worker_function(self) -> Callable:
        """
        Resolve module-level worker function for MPI compatibility.

        Returns:
            Worker function callable
        """
        worker_func = None

        try:
            worker_module_name = self.worker.__class__.__module__
            worker_module = sys.modules.get(worker_module_name)

            func_name = f"_evaluate_{self.model_name.lower()}_parameters_worker"
            if worker_module and hasattr(worker_module, func_name):
                worker_func = getattr(worker_module, func_name)
                self.logger.debug(
                    f"Resolved MPI worker function from loaded module: "
                    f"{worker_module_name}.{func_name}"
                )
        except (ValueError, RuntimeError) as e:
            self.logger.debug(f"Dynamic worker resolution failed: {e}")

        if worker_func is None:
            worker_func = self.worker.evaluate_worker_function

        return worker_func

    def _extract_scores(
        self,
        results: List[Dict],
        n_individuals: int
    ) -> np.ndarray:
        """
        Extract fitness scores from batch results.

        Args:
            results: List of result dictionaries
            n_individuals: Number of individuals in population

        Returns:
            Array of fitness scores
        """
        fitness = np.full(n_individuals, self.DEFAULT_PENALTY_SCORE)
        valid_count = 0

        for result in results:
            idx = result.get('individual_id', 0)
            score = result.get('score')
            error = result.get('error')

            if error:
                self._log_task_error(idx, error, result)

            fitness[idx] = self._sanitize(score)
            if not is_penalty(fitness[idx], self.DEFAULT_PENALTY_SCORE):
                valid_count += 1
            else:
                self.logger.debug(f"Task {idx} returned score={score}")

        self.logger.debug(f"Batch results: {len(results)} returned, {valid_count} valid scores")
        if results and valid_count == 0:
            self._log_all_penalty_batch(len(results), 'scores')
        self._track_batch(fitness)
        return fitness

    def _warn_on_dead_objectives(
        self,
        objectives: np.ndarray,
        objective_names: Optional[List[str]] = None,
    ) -> None:
        """Warn when a secondary objective carries no selection pressure.

        Validity above is judged on the *primary* objective only, so a
        secondary objective that is identical across the whole population —
        typically every individual falling back to the penalty score because
        its observations are missing — passes silently and quietly demotes a
        multi-objective calibration to a single-objective one. Observed with a
        multivariate streamflow+TWS experiment whose GRACE download had been
        interrupted: the TWS axis sat at the penalty value for every
        generation, and the run completed with no indication that half the
        objective had been inert.
        """
        if objectives.ndim != 2 or objectives.shape[1] < 2 or not len(objectives):
            return
        for j in range(1, objectives.shape[1]):
            col = objectives[:, j]
            if not np.all(col == col[0]):
                continue
            name = (objective_names[j] if objective_names
                    and j < len(objective_names) else f"#{j + 1}")
            is_penalty = col[0] == self.DEFAULT_PENALTY_SCORE
            log_once(
                self.logger,
                logging.ERROR if is_penalty else logging.WARNING,
                key=f'dead-objective-{j}',
                message=(
                    f"Objective {name} is identical ({col[0]:g}) across all "
                    f"{len(col)} individuals"
                    + (" and equals the penalty score, which usually means its "
                       "observations are missing or unreadable"
                       if is_penalty else "")
                    + ". It exerts no selection pressure, so this run is "
                      "effectively optimizing the remaining objective(s) only."
                ),
            )

    def _extract_objectives(
        self,
        results: List[Dict],
        n_individuals: int,
        n_objectives: int,
        objective_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract objective values from batch results.

        Supports two result formats:
        1. Explicit 'objectives' list (SUMMA workers)
        2. Metrics dict fallback — extracts named metrics from 'metrics' or
           top-level keys (JAX/in-memory workers)

        Args:
            results: List of result dictionaries
            n_individuals: Number of individuals in population
            n_objectives: Number of objectives
            objective_names: Ordered metric names (e.g., ['KGE', 'NSE'])

        Returns:
            Array of objective values (n_individuals x n_objectives)
        """
        objectives = np.full((n_individuals, n_objectives), self.DEFAULT_PENALTY_SCORE)
        valid_count = 0

        for result in results:
            idx = result.get('individual_id', 0)
            obj = result.get('objectives')
            error = result.get('error')

            if error:
                self._log_task_error(idx, error, result)

            if obj and len(obj) == n_objectives:
                # Explicit objectives list (SUMMA workers) -- already in
                # maximization convention.
                objectives[idx] = self._sanitize(np.array(obj, dtype=float))
            elif objective_names:
                # Fall back to the raw metrics dict (JAX/in-memory/other
                # workers): look each objective up and transform it to the
                # maximization convention, as SUMMA's worker does.
                metrics = result.get('metrics') or {}
                objectives[idx] = [
                    self._objective_from_metrics(name, metrics, result)
                    for name in objective_names
                ]
            else:
                self.logger.debug(f"Task {idx} returned objectives={obj}")

            if not is_penalty(objectives[idx][0], self.DEFAULT_PENALTY_SCORE):
                valid_count += 1

        self.logger.debug(
            f"Batch objectives: {len(results)} returned, {valid_count} valid objective sets"
        )
        if results and valid_count == 0:
            self._log_all_penalty_batch(len(results), 'objectives')
        self._warn_on_dead_objectives(objectives, objective_names)
        # Track on the primary objective — an individual whose first objective
        # is the penalty score is one whose model run failed.
        self._track_batch(objectives[:, 0])
        return objectives

    def _sanitize(self, score):
        """None / NaN / +-inf (scalar or array) -> penalty score.

        The single choke point for both the parallel and the sequential
        evaluation paths: a NaN fitness otherwise freezes DDS (``f_new >
        f_best`` is never true) and makes DE's ``argmax`` return the NaN slot.
        """
        return sanitize_score(score, self.DEFAULT_PENALTY_SCORE)

    def _objective_from_metrics(
        self,
        name: str,
        metrics: Dict[str, Any],
        result: Dict[str, Any],
    ) -> float:
        """Raw metric ``name`` from a result, transformed for maximization."""
        raw = None
        wanted = name.lower()
        for source in (metrics, result):
            if not isinstance(source, dict):
                continue
            for key in (name, f"Calib_{name}"):
                if key in source:
                    raw = source[key]
                    break
            else:
                for key, value in source.items():
                    if key.lower() in (wanted, f"calib_{wanted}"):
                        raw = value
                        break
            if raw is not None:
                break
        if is_failure_sentinel(raw):
            return self.DEFAULT_PENALTY_SCORE
        return self._sanitize(MetricTransformer.transform_for_maximization(name, float(raw)))

    def evaluate_solution(
        self,
        normalized_params: np.ndarray,
        proc_id: int = 0
    ) -> float:
        """
        Evaluate a single normalized parameter set.

        Args:
            normalized_params: Normalized parameters [0, 1]
            proc_id: Process ID for parallel execution

        Returns:
            Fitness score
        """
        from symfluence.core.calibration.workers.base_worker import WorkerTask

        params = self.task_builder.param_manager.denormalize_parameters(normalized_params)

        # Use TaskBuilder to ensure all model-specific paths (like mizuroute_settings_dir)
        # are correctly included in the task.
        task_data = self.task_builder.build_task(
            individual_id=0,
            params=params,
            proc_id=proc_id,
            evaluation_id="single_eval"
        )

        task = WorkerTask.from_legacy_dict(task_data)

        from symfluence.core.calibration.eval_watchdog import get_watchdog
        get_watchdog().beat()
        result = self.worker.evaluate(task)
        return self._sanitize(result.score)

    def evaluate_population(
        self,
        population: np.ndarray,
        iteration: int = 0,
        base_random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate a population of solutions (single objective).

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            Array of fitness scores
        """
        n_individuals = len(population)
        fitness = np.full(n_individuals, self.DEFAULT_PENALTY_SCORE)

        if self.use_parallel and n_individuals > 1:
            # Parallel evaluation
            tasks = self.task_builder.build_population_tasks(
                population,
                iteration=iteration,
                base_random_seed=base_random_seed
            )

            worker_func = self._resolve_worker_function()
            results = self.execute_batch(tasks, worker_func)
            fitness = self._extract_scores(results, n_individuals)
        else:
            # Sequential evaluation
            for i, params_normalized in enumerate(population):
                fitness[i] = self.evaluate_solution(params_normalized, proc_id=0)

        return fitness

    def evaluate_population_objectives(
        self,
        population: np.ndarray,
        objective_names: List[str],
        iteration: int = 0,
        base_random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate a population for multiple objectives.

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            objective_names: Ordered list of objective metric names
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            Array of objective values (n_individuals x n_objectives)
        """
        n_individuals = len(population)
        n_objectives = len(objective_names)
        objectives = np.full((n_individuals, n_objectives), self.DEFAULT_PENALTY_SCORE)

        tasks = self.task_builder.build_population_tasks(
            population,
            iteration=iteration,
            multiobjective=True,
            objective_names=objective_names,
            base_random_seed=base_random_seed
        )

        worker_func = self._resolve_worker_function()
        results = self.execute_batch(tasks, worker_func)
        objectives = self._extract_objectives(
            results, n_individuals, n_objectives, objective_names=objective_names
        )

        return objectives

    def evaluate_trials(
        self,
        trials: List[np.ndarray],
        trial_indices: List[int],
        iteration: int,
        base_random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Evaluate trial solutions (e.g., for DE algorithm).

        Args:
            trials: List of trial parameter arrays (normalized)
            trial_indices: Corresponding indices in population
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            Array of fitness scores indexed by trial_indices
        """
        pop_size = len(trials)
        trial_fitness = np.full(pop_size, self.DEFAULT_PENALTY_SCORE)

        if self.use_parallel and len(trials) > 1:
            tasks = self.task_builder.build_trial_tasks(
                trials,
                trial_indices,
                iteration,
                base_random_seed=base_random_seed
            )

            worker_func = self._resolve_worker_function()
            results = self.execute_batch(tasks, worker_func)

            for result in results:
                idx = result.get('individual_id', 0)
                score = result.get('score')
                error = result.get('error')

                if error:
                    self._log_task_error(idx, error, result)

                if idx in trial_indices:
                    trial_idx = trial_indices.index(idx)
                    trial_fitness[trial_idx] = self._sanitize(score)
        else:
            # Sequential evaluation
            for i, trial in enumerate(trials):
                proc_id = trial_indices[i] % self.num_processes
                trial_fitness[i] = self.evaluate_solution(trial, proc_id=proc_id)

        return trial_fitness
