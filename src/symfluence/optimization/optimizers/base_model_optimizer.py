# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Base Model Optimizer

Abstract base class providing unified optimization infrastructure for all hydrological
models. Implements template method pattern to delegate model-specific operations while
centralizing algorithm execution, parallel processing, results tracking, and final
evaluation workflows.

Mixin Components:
    ConfigurableMixin, ParallelExecutionMixin, ResultsTrackingMixin,
    RetryExecutionMixin, GradientOptimizationMixin

Abstract Methods (Must Implement in Subclass):
    _get_model_name() -> str
    _run_model_for_final_evaluation(output_dir) -> bool
    _get_final_file_manager_path() -> Path

Optional Overrides:
    _create_parameter_manager(), _create_calibration_target(), _create_worker(),
    _apply_best_parameters_for_final(), _get_settings_directory()
"""

import logging
import random
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from symfluence.core import ConfigurableMixin
from symfluence.core.constants import ModelDefaults
from symfluence.core.exceptions import OptimizationError, require_not_none

from ..mixins import GradientOptimizationMixin, ParallelExecutionMixin, ResultsTrackingMixin, RetryExecutionMixin
from ..workers.base_worker import BaseWorker
from .algorithms import ALGORITHM_REGISTRY, get_algorithm
from .component_factory import OptimizerComponentFactory
from .evaluators import PopulationEvaluator, TaskBuilder
from .final_evaluation import FinalEvaluationOrchestrator, FinalResultsSaver
from .metrics_tracker import EvaluationMetricsTracker

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseModelOptimizer(
    ConfigurableMixin,
    ParallelExecutionMixin,
    ResultsTrackingMixin,
    RetryExecutionMixin,
    GradientOptimizationMixin,
    ABC
):
    """Abstract base class for model-specific optimizers.

    Implements template method pattern providing unified optimization across all
    hydrological models. Uses mixins for parallel execution, results tracking,
    retry logic, and gradient-based optimization.

    Subclasses must implement: _get_model_name(), _run_model_for_final_evaluation(),
    _get_final_file_manager_path(). Components (param_manager, worker,
    calibration_target) are created via overridable factory methods with
    registry-based defaults.
    """

    # Default algorithm parameters
    DEFAULT_ITERATIONS = 100
    DEFAULT_POPULATION_SIZE = 30
    DEFAULT_PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize the model optimizer.

        Args:
            config: Configuration (typed SymfluenceConfig or legacy dict)
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Setup paths using typed config accessors
        # Note: dict_key enables fallback for legacy dict-based configs
        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir, default='.',
            dict_key='SYMFLUENCE_DATA_DIR'
        ))
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name, default='default',
            dict_key='DOMAIN_NAME'
        )
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        # Note: experiment_id is provided by ConfigMixin property

        # Optimization settings directory
        if optimization_settings_dir is not None:
            self.optimization_settings_dir = Path(optimization_settings_dir)
        else:
            model_name = self._get_model_name()
            self.optimization_settings_dir = (
                self.project_dir / 'settings' / model_name
            )

        # Results directory - now includes model name to avoid overwrites between models
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm, default='optimization',
            dict_key='OPTIMIZATION_ALGORITHM'
        ).lower()
        model_name = self._get_model_name()
        self.results_dir = (
            self.project_dir / 'optimization' / model_name /
            f"{algorithm}_{self.experiment_id}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results tracking
        self.__init_results_tracking__()

        # Composed helpers (must be created before model-specific components)
        self._factory = OptimizerComponentFactory(self.config, self.logger, self.project_dir)

        # Create model-specific components
        self.param_manager = self._create_parameter_manager()
        self.calibration_target = self._create_calibration_target()
        self.worker = self._create_worker()

        # Algorithm parameters (using typed config with dict_key fallback)
        self.max_iterations = self._get_config_value(
            lambda: self.config.optimization.iterations, default=self.DEFAULT_ITERATIONS,
            dict_key='OPTIMIZATION_MAX_ITERATIONS'
        )
        self.population_size = self._get_config_value(
            lambda: self.config.optimization.population_size, default=self.DEFAULT_POPULATION_SIZE,
            dict_key='OPTIMIZATION_POPULATION_SIZE'
        )
        self.target_metric = self._get_config_value(
            lambda: self.config.optimization.metric, default='KGE',
            dict_key='OPTIMIZATION_METRIC'
        )

        # Random seed
        self.random_seed = self._get_config_value(
            lambda: self.config.system.random_seed, dict_key='RANDOM_SEED'
        )
        if self.random_seed is not None and self.random_seed != 'None':
            self._set_random_seeds(int(self.random_seed))

        # Parallel processing state
        self.parallel_dirs: Dict[int, Dict[str, Any]] = {}
        self.default_sim_dir = self.results_dir  # Initialize with results_dir as fallback
        # Setup directories if NUM_PROCESSES is set, regardless of count (for isolation)
        num_processes = self._get_config_value(
            lambda: self.config.system.num_processes, default=1,
            dict_key='NUM_PROCESSES'
        )
        if num_processes >= 1:
            self._setup_parallel_dirs()

        # Runtime config overrides (for algorithm-specific settings like Adam/LBFGS)
        self._runtime_overrides: Dict[str, Any] = {}

        # Algorithm registry
        self._registry = ALGORITHM_REGISTRY

        # Lazy-initialized components
        self._task_builder: Optional[TaskBuilder] = None
        self._population_evaluator: Optional[PopulationEvaluator] = None
        self._final_evaluation_runner: Optional[Any] = None
        self._results_saver: Optional[FinalResultsSaver] = None

        # Composed helpers (factory created earlier, before component creation)
        self._metrics_tracker = EvaluationMetricsTracker(
            self.max_iterations, self.logger, self.format_elapsed_time
        )
        self._final_orchestrator: Optional[FinalEvaluationOrchestrator] = None

    # =========================================================================
    # Lazy-initialized component properties
    # =========================================================================

    @property
    def task_builder(self) -> TaskBuilder:
        """Lazy-initialized task builder."""
        if self._task_builder is None:
            self._task_builder = TaskBuilder(
                config=self.config,
                project_dir=self.project_dir,
                domain_name=self.domain_name,
                optimization_settings_dir=self.optimization_settings_dir,
                default_sim_dir=self.default_sim_dir,
                parallel_dirs=self.parallel_dirs,
                num_processes=self.num_processes,
                target_metric=self.target_metric,
                param_manager=self.param_manager,
                logger=self.logger
            )
            if hasattr(self, 'summa_exe_path'):
                self._task_builder.set_summa_exe_path(self.summa_exe_path)
        return require_not_none(self._task_builder, "task_builder", OptimizationError)

    @property
    def population_evaluator(self) -> PopulationEvaluator:
        """Lazy-initialized population evaluator."""
        if self._population_evaluator is None:
            self._population_evaluator = PopulationEvaluator(
                task_builder=self.task_builder,
                worker=self.worker,
                execute_batch=self.execute_batch,
                use_parallel=self.use_parallel,
                num_processes=self.num_processes,
                model_name=self._get_model_name(),
                logger=self.logger
            )
        return require_not_none(self._population_evaluator, "population_evaluator", OptimizationError)

    @property
    def results_saver(self) -> FinalResultsSaver:
        """Lazy-initialized results saver."""
        if self._results_saver is None:
            self._results_saver = FinalResultsSaver(
                results_dir=self.results_dir,
                experiment_id=self.experiment_id,
                domain_name=self.domain_name,
                logger=self.logger
            )
        return require_not_none(self._results_saver, "results_saver", OptimizationError)

    @property
    def final_orchestrator(self) -> FinalEvaluationOrchestrator:
        """Lazy-initialized final evaluation orchestrator."""
        if self._final_orchestrator is None:
            self._final_orchestrator = FinalEvaluationOrchestrator(
                config=self.config,
                logger=self.logger,
                optimization_settings_dir=self.optimization_settings_dir,
                results_saver=self.results_saver,
            )
        return self._final_orchestrator

    def _visualize_progress(self, algorithm: str) -> None:
        """Helper to visualize optimization progress if reporting manager available."""
        if self.reporting_manager:
            calibration_variable = self._get_config_value(
                lambda: self.config.optimization.calibration_variable, default='streamflow'
            )
            self.reporting_manager.visualize_optimization_progress(
                self._iteration_history,
                self.results_dir.parent / f"{algorithm.lower()}_{self.experiment_id}", # Matches results_dir logic or pass results_dir
                calibration_variable,
                self.target_metric
            )

            calibrate_depth = self._get_config_value(
                lambda: self.config.model.summa.calibrate_depth, default=False
            )
            if calibrate_depth:
                self.reporting_manager.visualize_optimization_depth_parameters(
                    self._iteration_history,
                    self.results_dir.parent / f"{algorithm.lower()}_{self.experiment_id}"
                )

    # =========================================================================
    # Default factory methods using registry-based discovery
    # =========================================================================

    def _create_parameter_manager_default(self):
        """Default factory for parameter managers using registry discovery.

        Delegates to OptimizerComponentFactory for registry lookup and
        instantiation.  Override _create_parameter_manager() if a
        non-standard constructor is needed.
        """
        return self._factory.create_parameter_manager(self._get_model_name())

    def _create_worker_default(self) -> BaseWorker:
        """Default factory for workers using registry discovery.

        Delegates to OptimizerComponentFactory.
        """
        return self._factory.create_worker(self._get_model_name())

    def _create_calibration_target_default(self):
        """Default factory for calibration targets using centralized factory.

        Delegates to OptimizerComponentFactory.
        """
        model_name = self._get_model_name()
        target_type = str(self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow', dict_key='OPTIMIZATION_TARGET'
        )).lower()
        return self._factory.create_calibration_target(model_name, target_type)

    def _get_settings_directory(self) -> Path:
        """Get model settings directory. Delegates to OptimizerComponentFactory."""
        return self._factory.get_settings_directory(self._get_model_name())

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the name of the model being optimized."""
        pass

    @abstractmethod
    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run the model for final evaluation (model-specific implementation)."""
        pass

    @abstractmethod
    def _get_final_file_manager_path(self) -> Path:
        """Get path to the file manager used for final evaluation."""
        pass


    def _load_best_previous_params(self) -> Optional[Dict[str, float]]:
        """Load best parameters from previous optimization runs (warm-start).

        Scans sibling run directories for best_params.json files and returns
        the parameters from the run with the highest score.  Prefers runs
        that used the same optimization metric to avoid comparing scores
        across incompatible metrics (e.g. KGE vs COMPOSITE).  Accepts partial
        matches when at least 50% of expected parameters are present —
        missing parameters are filled with log-aware midpoints from the
        parameter manager.
        """
        import json
        parent_dir = self.results_dir.parent
        if not parent_dir.exists():
            return None

        # Get current model's expected parameter names for validation
        expected_params = set(self.param_manager.all_param_names)
        min_overlap = max(1, len(expected_params) // 2)  # 50% threshold

        # Determine current metric for same-metric preference
        current_metric = self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE',
            dict_key='OPTIMIZATION_METRIC'
        )

        # Collect all valid candidates grouped by metric match
        same_metric_best: Dict[str, Any] = {'score': -float('inf'), 'params': None, 'run': None, 'missing': set()}
        other_metric_best: Dict[str, Any] = {'score': -float('inf'), 'params': None, 'run': None, 'missing': set()}

        for run_dir in parent_dir.iterdir():
            if not run_dir.is_dir() or run_dir == self.results_dir:
                continue
            for f in run_dir.iterdir():
                if f.name.endswith('_best_params.json'):
                    try:
                        data = json.loads(f.read_text(encoding='utf-8'))
                        score = data.get('best_score', -float('inf'))
                        params = data.get('best_params')
                        run_metric = data.get('metric')  # may be None for legacy runs
                        if not params:
                            continue
                        loaded_keys = set(params.keys())
                        overlap = expected_params & loaded_keys
                        missing = expected_params - loaded_keys
                        if len(overlap) < min_overlap:
                            self.logger.debug(
                                f"Skipping {run_dir.name}: insufficient "
                                f"overlap ({len(overlap)}/{len(expected_params)}, "
                                f"need {min_overlap})"
                            )
                            continue

                        # Categorise by metric match
                        metrics_match = (
                            run_metric is not None
                            and run_metric.upper() == current_metric.upper()
                        )
                        target = same_metric_best if metrics_match else other_metric_best
                        if score > target['score']:
                            target['score'] = score
                            target['params'] = params
                            target['run'] = run_dir.name
                            target['missing'] = missing
                    except (json.JSONDecodeError, OSError):
                        continue

        # Prefer same-metric runs; fall back to cross-metric only if none
        if same_metric_best['params'] is not None:
            best_score = same_metric_best['score']
            best_params = same_metric_best['params']
            best_run = same_metric_best['run']
            best_missing = same_metric_best['missing']
            if other_metric_best['params'] is not None and other_metric_best['score'] > best_score:
                self.logger.info(
                    f"Ignoring higher-scoring {other_metric_best['run']} "
                    f"(score={other_metric_best['score']:.4f}, different metric) "
                    f"in favour of same-metric run {best_run} "
                    f"(score={best_score:.4f}, metric={current_metric})"
                )
        elif other_metric_best['params'] is not None:
            best_score = other_metric_best['score']
            best_params = other_metric_best['params']
            best_run = other_metric_best['run']
            best_missing = other_metric_best['missing']
            self.logger.warning(
                f"No same-metric ({current_metric}) runs found; "
                f"warm-starting from {best_run} which used a different metric "
                f"(score={best_score:.4f})"
            )
        else:
            best_params = None
            best_run = None
            best_missing = set()

        if best_params is not None:
            # Fill missing parameters with raw file values (not midpoints).
            # These params weren't calibrated in the previous run, so their
            # file values represent what the model was actually running with
            # when it achieved its best score.
            if best_missing:
                try:
                    defaults = self.param_manager.get_initial_parameters(
                        skip_boundary_check=True
                    ) or {}
                except TypeError:
                    # Model's param manager doesn't support the kwarg
                    defaults = self.param_manager.get_initial_parameters() or {}
                filled: Dict[str, float] = {}
                for p in sorted(best_missing):
                    if p in defaults:
                        val = defaults[p]
                        try:
                            val = float(val)
                        except (TypeError, ValueError):
                            import numpy as np
                            val = float(np.asarray(val).flat[0])
                        best_params[p] = val
                        filled[p] = val
                    else:
                        # Fallback: midpoint from bounds
                        bounds = self.param_manager.param_bounds.get(p, {'min': 0.1, 'max': 10.0})
                        transform = bounds.get('transform', 'linear')
                        if transform == 'log' and bounds['min'] > 0:
                            import numpy as np
                            mid = float(np.exp((np.log(bounds['min']) + np.log(bounds['max'])) / 2))
                        else:
                            mid = float((bounds['min'] + bounds['max']) / 2)
                        best_params[p] = mid
                        filled[p] = mid
                self.logger.info(
                    f"Warm-starting from {best_run} ({current_metric}={best_score:.4f}); "
                    f"filled {len(filled)} missing params: "
                    + ", ".join(f"{k}={v:.4g}" for k, v in filled.items())
                )
            else:
                self.logger.info(
                    f"Warm-starting from {best_run} ({current_metric}={best_score:.4f})"
                )
            # Only return expected params (drop any extra from previous run)
            return {k: v for k, v in best_params.items() if k in expected_params}
        return None

    def _create_parameter_manager(self):
        """
        Create the model-specific parameter manager.

        Default implementation uses registry-based discovery via
        _create_parameter_manager_default(). Override if:
        - Non-standard constructor signature needed
        - Pre-initialization logic required
        - Custom path resolution needed

        Examples of when to override:
        - FUSE: Needs fuse_sim_dir computed before super().__init__()
        - SUMMA: Uses summa_settings_dir instead of generic settings_dir
        - GR: Uses gr_setup_dir instead of generic settings_dir

        Returns:
            Parameter manager instance
        """
        return self._create_parameter_manager_default()

    def _create_calibration_target(self):
        """
        Create the model-specific calibration target.

        Default implementation uses centralized factory via
        _create_calibration_target_default(). Override rarely needed as
        the factory handles registry lookup and fallbacks.

        Returns:
            Calibration target instance
        """
        return self._create_calibration_target_default()

    def _create_worker(self) -> BaseWorker:
        """
        Create the model-specific worker.

        Default implementation uses registry-based discovery via
        _create_worker_default(). Override rarely needed as all workers
        use the same constructor signature.

        Returns:
            Worker instance
        """
        return self._create_worker_default()

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _get_nsga2_objective_names(self) -> List[str]:
        """Resolve NSGA-II objective metric names in priority order."""
        primary_metric = self._get_config_value(
            lambda: self.config.optimization.nsga2.primary_metric, default=self.target_metric
        )
        secondary_metric = self._get_config_value(
            lambda: self.config.optimization.nsga2.secondary_metric, default=self.target_metric
        )
        return [str(primary_metric).upper(), str(secondary_metric).upper()]

    def _get_moead_objective_names(self) -> List[str]:
        """Resolve MOEA/D objective metric names in priority order."""
        primary_metric = self._get_config_value(
            lambda: self.config.optimization.moead_primary_metric, default=self.target_metric,
            dict_key='MOEAD_PRIMARY_METRIC'
        )
        secondary_metric = self._get_config_value(
            lambda: self.config.optimization.moead_secondary_metric, default=self.target_metric,
            dict_key='MOEAD_SECONDARY_METRIC'
        )
        return [str(primary_metric).upper(), str(secondary_metric).upper()]

    def _supports_multi_objective(self) -> bool:
        """Check if this model supports multi-objective optimization (NSGA-II).

        All models support multi-objective optimization. SUMMA workers return
        explicit 'objectives' lists; JAX/in-memory workers return 'metrics'
        dicts from which objectives are extracted by name (e.g., KGE, NSE).

        Returns:
            True if multi-objective is supported, False otherwise.
        """
        return True

    def _log_calibration_alignment(self) -> None:
        """Log basic calibration alignment info before optimization starts."""
        try:
            # Check if this is a multivariate target
            if hasattr(self.calibration_target, 'variables') and self.calibration_target.variables:
                # Multivariate calibration: check each variable's observed data
                from symfluence.evaluation.registry import EvaluationRegistry

                all_found = True
                for var in self.calibration_target.variables:
                    evaluator = EvaluationRegistry.get_evaluator(
                        var, self.config, self.logger, self.project_dir, target=var
                    )
                    if evaluator and hasattr(evaluator, '_load_observed_data'):
                        obs = evaluator._load_observed_data()
                        if obs is None or obs.empty:
                            self.logger.warning(f"Calibration check: no observed data found for {var}")
                            all_found = False
                        else:
                            self.logger.info(f"Calibration check: {var} has {len(obs)} observed data points")

                if not all_found:
                    self.logger.warning("Some variables in multivariate calibration lack observed data")
                return

            # Single-target calibration
            if not hasattr(self.calibration_target, '_load_observed_data'):
                return

            obs = self.calibration_target._load_observed_data()
            if obs is None or obs.empty:
                self.logger.warning("Calibration check: no observed data found")
                return

            if not isinstance(obs.index, pd.DatetimeIndex):
                obs.index = pd.to_datetime(obs.index)

            calib_period = self.calibration_target._parse_date_range(
                self._get_config_value(lambda: self.config.domain.calibration_period, default='')
            )
            obs_period = obs.copy()
            if calib_period[0] and calib_period[1]:
                obs_period = obs_period[(obs_period.index >= calib_period[0]) & (obs_period.index <= calib_period[1])]

            eval_timestep = str(self._get_config_value(
                lambda: self.config.optimization.calibration_timestep, default='native'
            )).lower()
            if eval_timestep != 'native' and hasattr(self.calibration_target, '_resample_to_timestep'):
                obs_period = self.calibration_target._resample_to_timestep(obs_period, eval_timestep)

            sim_start = self._get_config_value(lambda: self.config.domain.time_start)
            sim_end = self._get_config_value(lambda: self.config.domain.time_end)
            overlap = obs_period
            if sim_start and sim_end:
                sim_start = pd.Timestamp(sim_start)
                sim_end = pd.Timestamp(sim_end)
                overlap = obs_period[(obs_period.index >= sim_start) & (obs_period.index <= sim_end)]

            self.logger.debug(
                "Calibration data check | timestep=%s | obs=%d | calib_window=%d | overlap_with_sim=%d",
                eval_timestep,
                len(obs),
                len(obs_period),
                len(overlap)
            )
        except (KeyError, IndexError, TypeError, ValueError) as e:
            self.logger.debug(f"Calibration alignment check failed: {e}")

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def _adjust_end_time_for_forcing(self, end_time_str: str) -> str:
        """Adjust end time to align with forcing data timestep."""
        try:
            forcing_timestep_seconds = self._get_config_value(
                lambda: self.config.forcing.time_step_size, default=3600
            )

            if forcing_timestep_seconds >= 3600:  # Hourly or coarser
                # Parse the end time
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')

                # Calculate the last valid hour based on timestep
                forcing_timestep_hours = forcing_timestep_seconds / 3600
                last_hour = int(24 - (24 % forcing_timestep_hours)) - forcing_timestep_hours
                if last_hour < 0:
                    last_hour = 0

                # Adjust if needed
                if end_time.hour > last_hour or (end_time.hour == 23 and last_hour < 23):
                    end_time = end_time.replace(hour=int(last_hour), minute=0)
                    adjusted_str = end_time.strftime('%Y-%m-%d %H:%M')
                    self.logger.info(f"Adjusted end time from {end_time_str} to {adjusted_str} for {forcing_timestep_hours}h forcing")
                    return adjusted_str

            return end_time_str

        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not adjust end time: {e}")
            return end_time_str

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel processing directories."""
        # Determine algorithm for directory naming
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm, default='optimization'
        ).lower()

        # Use algorithm-specific directory
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        # If the primary simulations directory is not writable (common on macOS
        # sandboxed mounts or read-only network drives), fall back to a local
        # scratch directory so calibration can proceed.
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback = Path(tempfile.gettempdir()) / "symfluence" / self.domain_name / f'run_{algorithm}'
            fallback.mkdir(parents=True, exist_ok=True)
            self.logger.warning(
                f"Simulations directory not writable: {base_dir}. "
                f"Falling back to scratch: {fallback}"
            )
            base_dir = fallback

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            self._get_model_name(),
            self.experiment_id
        )

        # For non-parallel runs, set a default output directory for fallback
        # This ensures SUMMA outputs go to the simulation directory, not the optimization results directory
        if not self.use_parallel and self.parallel_dirs:
            # Use process_0 directories as the default
            self.default_sim_dir = self.parallel_dirs[0].get('sim_dir', self.results_dir)
        else:
            self.default_sim_dir = self.results_dir

    # =========================================================================
    # Evaluation methods
    # =========================================================================

    def log_iteration_progress(
        self,
        algorithm_name: str,
        iteration: int,
        best_score: float,
        secondary_score: Optional[float] = None,
        secondary_label: Optional[str] = None,
        n_improved: Optional[int] = None,
        population_size: Optional[int] = None,
        crash_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log optimization progress. Delegates to EvaluationMetricsTracker."""
        self._metrics_tracker.log_iteration_progress(
            algorithm_name, iteration, best_score,
            secondary_score=secondary_score, secondary_label=secondary_label,
            n_improved=n_improved, population_size=population_size,
            crash_stats=crash_stats
        )

    def log_initial_population(
        self,
        algorithm_name: str,
        population_size: int,
        best_score: float
    ) -> None:
        """Log initial population completion. Delegates to EvaluationMetricsTracker."""
        self._metrics_tracker.log_initial_population(algorithm_name, population_size, best_score)

    def _evaluate_solution(
        self,
        normalized_params: np.ndarray,
        proc_id: int = 0
    ) -> float:
        """
        Evaluate a normalized parameter set.

        Args:
            normalized_params: Normalized parameters [0, 1]
            proc_id: Process ID for parallel execution

        Returns:
            Fitness score
        """
        score = self.population_evaluator.evaluate_solution(normalized_params, proc_id)
        self._metrics_tracker.track_evaluation(score)
        return score

    def get_crash_stats(self) -> Dict[str, Any]:
        """Return crash rate statistics. Delegates to EvaluationMetricsTracker."""
        return self._metrics_tracker.get_crash_stats()

    def _create_gradient_callback(self) -> Optional[Callable]:
        """Create native gradient callback if worker supports autodiff.

        Returns:
            Callable (x_normalized -> (loss, gradient)) or None if unsupported.
        """
        # Check if worker supports native gradients
        if not hasattr(self, 'worker') or self.worker is None:
            return None

        if not hasattr(self.worker, 'supports_native_gradients'):
            return None

        if not self.worker.supports_native_gradients():
            return None

        # Get optimization metric from config
        # Uses self.target_metric which is set in __init__ from config.optimization.metric
        # This matches _extract_primary_score in base_worker.py to ensure FD and native
        # gradient paths optimize the same objective
        metric = self.target_metric.lower()

        # Get parameter names and bounds for gradient transformation
        param_names = self.param_manager.all_param_names
        bounds = self.param_manager.get_parameter_bounds()

        # Compute scale factors for gradient chain rule
        # d(loss)/d(x_norm) = d(loss)/d(x_phys) * d(x_phys)/d(x_norm)
        # where d(x_phys)/d(x_norm) = (upper - lower) for linear scaling
        scale_factors = np.array([
            bounds[name]['max'] - bounds[name]['min']
            for name in param_names
        ])

        def gradient_callback(x_normalized: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Compute loss and gradient for normalized parameters.

            Args:
                x_normalized: Parameters in [0,1] normalized space

            Returns:
                Tuple of (loss, gradient_normalized) where:
                - loss: Scalar loss value (negative of metric, for minimization)
                - gradient_normalized: Gradient in normalized [0,1] space
            """
            # Denormalize to physical parameters
            params_dict = self.param_manager.denormalize_parameters(x_normalized)

            # Call worker's evaluate_with_gradient
            loss, grad_dict = self.worker.evaluate_with_gradient(params_dict, metric)

            if grad_dict is None:
                raise RuntimeError(
                    f"Worker returned None gradient despite supporting native gradients. "
                    f"Check {self.worker.__class__.__name__}.evaluate_with_gradient() implementation."
                )

            # Convert gradient dict to array (same order as param_names)
            grad_physical = np.array([grad_dict[name] for name in param_names])

            # Transform gradient from physical to normalized space via chain rule
            grad_normalized = grad_physical * scale_factors

            return loss, grad_normalized

        self.logger.info(
            f"Native gradient callback created for {self._get_model_name()} "
            f"({len(param_names)} parameters)"
        )
        return gradient_callback

    def _get_gradient_mode(self) -> str:
        """
        Get gradient computation mode from configuration.

        Returns:
            One of: 'auto', 'native', 'finite_difference'
            - 'auto': Use native gradients if available, else finite differences
            - 'native': Require native gradients (error if unavailable)
            - 'finite_difference': Always use FD (useful for comparison/debugging)
        """
        return self._get_config_value(
            lambda: self.config.optimization.gradient_mode,
            default='auto',
            dict_key='GRADIENT_MODE'
        )

    def _evaluate_population(
        self,
        population: np.ndarray,
        iteration: int = 0
    ) -> np.ndarray:
        """Evaluate a population of normalized parameter sets in parallel.

        Args:
            population: Array shape (n_individuals, n_parameters) in [0, 1]
            iteration: Current generation number

        Returns:
            Array shape (n_individuals,) with fitness scores
        """
        base_seed = self.random_seed if hasattr(self, 'random_seed') else None
        return self.population_evaluator.evaluate_population(
            population, iteration, base_random_seed=base_seed
        )

    def _evaluate_population_objectives(
        self,
        population: np.ndarray,
        objective_names: List[str],
        iteration: int = 0
    ) -> np.ndarray:
        """Evaluate population for multiple objectives (NSGA-II/MOEA-D).

        Args:
            population: Array shape (n_individuals, n_parameters) in [0, 1]
            objective_names: List of metric names to extract (e.g., ['KGE', 'NSE'])
            iteration: Current generation number

        Returns:
            Array shape (n_individuals, n_objectives)
        """
        base_seed = self.random_seed if hasattr(self, 'random_seed') else None
        return self.population_evaluator.evaluate_population_objectives(
            population, objective_names, iteration, base_random_seed=base_seed
        )

    # =========================================================================
    # Algorithm implementations
    # =========================================================================

    def _save_pareto_front(self, result: Dict[str, Any], algorithm_name: str) -> Optional[Path]:
        """Save Pareto front from multi-objective optimization results.

        Args:
            result: Algorithm result dict containing 'pareto_front' and 'pareto_objectives'
            algorithm_name: Name of the algorithm

        Returns:
            Path to saved Pareto front CSV, or None if saving failed
        """
        import pandas as pd

        pareto_solutions = result.get('pareto_front')
        pareto_objectives = result.get('pareto_objectives')

        if pareto_solutions is None or pareto_objectives is None:
            return None

        if len(pareto_solutions) == 0:
            self.logger.warning("Pareto front is empty, nothing to save")
            return None

        # Build records with objectives and parameters
        records = []
        param_names = self.param_manager.all_param_names

        for i, (solution, objectives) in enumerate(zip(pareto_solutions, pareto_objectives)):
            record = {'solution_id': i}

            # Add objectives
            for j, obj in enumerate(objectives):
                record[f'obj_{j}'] = obj

            # Add denormalized parameters
            params_dict = self.param_manager.denormalize_parameters(solution)
            for param_name in param_names:
                val = params_dict.get(param_name)
                if isinstance(val, (list, np.ndarray)):
                    val = val[0] if len(val) > 0 else None
                record[param_name] = val

            records.append(record)

        df = pd.DataFrame(records)

        # Generate filename (sanitize algorithm name for filesystem)
        safe_algorithm = algorithm_name.lower().replace('/', '_')
        filename = f"{self.experiment_id}_{safe_algorithm}_pareto_front.csv"
        pareto_path = self.results_dir / filename

        df.to_csv(pareto_path, index=False)
        self.logger.info(f"Saved Pareto front ({len(records)} solutions) to {pareto_path}")

        return pareto_path

    def _run_default_only(self, algorithm_name: str) -> Path:
        """
        Run a single default evaluation when no parameters are configured.
        """
        self.start_timing()
        self.logger.info(
            f"No parameters configured for {self._get_model_name()} - running default evaluation only"
        )

        score = self.DEFAULT_PENALTY_SCORE
        final_result = self.run_final_evaluation({})
        if final_result and isinstance(final_result, dict):
            metrics = final_result.get('final_metrics', {})
            score = metrics.get(self.target_metric, self.DEFAULT_PENALTY_SCORE)

            self.record_iteration(0, score, {})
            self.update_best(score, {}, 0)
            self.save_best_params(algorithm_name)
        # Save results
        results_path = self.save_results(algorithm_name, standard_filename=True)
        if results_path is None:
            raise OptimizationError(f"Failed to save results for {algorithm_name} default evaluation")
        return results_path

    def _build_algorithm_callbacks(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build callback functions and kwargs dict for algorithm.optimize().

        Returns:
            Tuple of (callbacks_dict, kwargs_dict) where callbacks_dict contains
            the core function bindings and kwargs_dict contains additional settings.
        """
        def evaluate_solution(normalized_params, proc_id=0):
            return self._evaluate_solution(normalized_params, proc_id)

        def evaluate_population(population, iteration=0):
            return self._evaluate_population(population, iteration)

        def denormalize_params(normalized):
            return self.param_manager.denormalize_parameters(normalized)

        def record_iteration(iteration, score, params, additional_metrics=None):
            crash_stats = self.get_crash_stats()
            merged = dict(additional_metrics or {}, **{
                'crash_count': crash_stats['crash_count'],
                'crash_rate': crash_stats['crash_rate'],
            })
            self.record_iteration(iteration, score, params, additional_metrics=merged)

        def update_best(score, params, iteration):
            self.update_best(score, params, iteration)

        def log_progress(alg_name, iteration, best_score, n_improved=None, pop_size=None, secondary_score=None, secondary_label=None):
            self.log_iteration_progress(
                alg_name, iteration, best_score,
                secondary_score=secondary_score, secondary_label=secondary_label,
                n_improved=n_improved, population_size=pop_size,
                crash_stats=self.get_crash_stats()
            )

        def save_checkpoint(algorithm_name: str, iteration: int):
            try:
                self.save_results(algorithm_name, standard_filename=True)
                self.save_best_params(algorithm_name)
                self.logger.debug(f"Checkpoint saved at iteration {iteration}")
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"Checkpoint save failed at iteration {iteration}: {e}")

        callbacks = {
            'evaluate_solution': evaluate_solution,
            'evaluate_population': evaluate_population,
            'denormalize_params': denormalize_params,
            'record_iteration': record_iteration,
            'update_best': update_best,
            'log_progress': log_progress,
        }

        kwargs = {
            'log_initial_population': self.log_initial_population,
            'num_processes': self.num_processes if hasattr(self, 'num_processes') else 1,
            'save_checkpoint': save_checkpoint,
        }

        return callbacks, kwargs

    def run_optimization(self, algorithm_name: str) -> Path:
        """Run optimization using a specified algorithm from the registry.

        Args:
            algorithm_name: Algorithm name (case-insensitive)

        Returns:
            Path to results JSON file
        """
        self.start_timing()
        self.logger.info(f"Starting {algorithm_name.upper()} optimization for {self._get_model_name()}")
        self._log_calibration_alignment()

        n_params = len(self.param_manager.all_param_names)
        if n_params == 0:
            return self._run_default_only(algorithm_name)

        # Get algorithm instance from registry
        algorithm = get_algorithm(algorithm_name, self.config, self.logger)

        # Build callbacks and kwargs for the algorithm
        callbacks, kwargs = self._build_algorithm_callbacks()

        # Seed optimization with best previous result (warm-start) or def file defaults
        skip_warm_start = self._get_config_value(lambda: None, default=False, dict_key='SKIP_WARM_START')
        try:
            if skip_warm_start:
                self.logger.info(
                    "SKIP_WARM_START is set — skipping warm-start from previous runs. "
                    "Optimization will start from def file defaults or config-specified initial parameters."
                )
                initial_params_dict = self.param_manager.get_initial_parameters()
            else:
                initial_params_dict = self._load_best_previous_params()
                if initial_params_dict is None:
                    initial_params_dict = self.param_manager.get_initial_parameters()
                    if initial_params_dict:
                        self.logger.debug("Using initial parameter guess for optimization seeding")
            if initial_params_dict:
                initial_guess = self.param_manager.normalize_parameters(initial_params_dict)
                kwargs['initial_guess'] = initial_guess
                self.logger.info(
                    f"Initial guess prepared: {len(initial_guess)} params, "
                    f"mean={initial_guess.mean():.4f}, "
                    f"sample: {list(initial_params_dict.items())[:3]}"
                )
            else:
                self.logger.warning("No initial parameter guess available (dict was None/empty)")
        except (KeyError, AttributeError, ValueError) as e:
            self.logger.warning(f"Failed to prepare initial parameter guess: {e}")

        # Guard multi-objective algorithms: only SUMMA supports multi-objective
        if algorithm_name.lower() in ['nsga2', 'nsga-ii', 'moead', 'moea-d', 'moea_d']:
            if not self._supports_multi_objective():
                raise OptimizationError(
                    f"{algorithm_name.upper()} multi-objective optimization is not supported for {self._get_model_name()}. "
                    f"Only SUMMA models have the worker infrastructure to return multiple objectives. "
                    f"Use single-objective algorithms like DDS, PSO, DE, or SCE-UA instead."
                )

        # For NSGA-II, add multi-objective support
        if algorithm_name.lower() in ['nsga2', 'nsga-ii']:
            kwargs['evaluate_population_objectives'] = self._evaluate_population_objectives
            kwargs['objective_names'] = self._get_nsga2_objective_names()
            kwargs['multiobjective'] = bool(self._get_config_value(
                lambda: self.config.optimization.nsga2.multi_target, default=False
            ))

        # For MOEA/D, add multi-objective support only when explicitly enabled
        if algorithm_name.lower() in ['moead', 'moea-d', 'moea_d']:
            moead_multi = bool(self._get_config_value(
                lambda: self.config.optimization.moead_multi_target, default=False,
                dict_key='MOEAD_MULTI_TARGET'
            ))
            kwargs['multiobjective'] = moead_multi
            if moead_multi:
                kwargs['evaluate_population_objectives'] = self._evaluate_population_objectives
                kwargs['objective_names'] = self._get_moead_objective_names()

        # For gradient-based algorithms (Adam, L-BFGS), add native gradient support
        if algorithm_name.lower() in ['adam', 'lbfgs']:
            gradient_callback = self._create_gradient_callback()
            gradient_mode = self._get_gradient_mode()

            if gradient_callback is not None:
                kwargs['compute_gradient'] = gradient_callback
                self.logger.info(
                    f"Native gradient support enabled for {algorithm_name.upper()} "
                    f"(mode: {gradient_mode})"
                )
            else:
                self.logger.info(
                    f"Using finite-difference gradients for {algorithm_name.upper()} "
                    f"(native gradients not available for {self._get_model_name()})"
                )

            kwargs['gradient_mode'] = gradient_mode

        # Run the algorithm
        result = algorithm.optimize(
            n_params=n_params,
            **callbacks,
            **kwargs
        )

        # Save results
        results_path = self.save_results(algorithm.name, standard_filename=True)
        self.save_best_params(algorithm.name)
        self._visualize_progress(algorithm.name)

        # Save Pareto front for multi-objective algorithms
        if result.get('pareto_objectives') is not None and result.get('pareto_front') is not None:
            self._save_pareto_front(result, algorithm.name)

        self.logger.info(f"{algorithm.name} completed in {self.format_elapsed_time()}")

        # Run final evaluation on full period
        if result.get('best_params'):
            final_result = self.run_final_evaluation(result['best_params'])
            if final_result:
                self._save_final_evaluation_results(final_result, algorithm.name)

        return results_path  # type: ignore[return-value]

    # =========================================================================
    # Algorithm convenience methods - delegate to run_optimization()
    # =========================================================================

    def run_dds(self) -> Path:
        """Run Dynamically Dimensioned Search (DDS) optimization."""
        return self.run_optimization('dds')

    def run_pso(self) -> Path:
        """Run Particle Swarm Optimization (PSO)."""
        return self.run_optimization('pso')

    def run_de(self) -> Path:
        """Run Differential Evolution (DE) optimization."""
        return self.run_optimization('de')

    def run_sce(self) -> Path:
        """Run Shuffled Complex Evolution (SCE-UA) optimization."""
        return self.run_optimization('sce-ua')

    def run_async_dds(self) -> Path:
        """Run Asynchronous Parallel DDS optimization."""
        return self.run_optimization('async_dds')

    def run_nsga2(self) -> Path:
        """Run NSGA-II multi-objective optimization."""
        return self.run_optimization('nsga2')

    def run_cmaes(self) -> Path:
        """Run CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization."""
        return self.run_optimization('cmaes')

    def run_dream(self) -> Path:
        """Run DREAM (DiffeRential Evolution Adaptive Metropolis) optimization."""
        return self.run_optimization('dream')

    def run_glue(self) -> Path:
        """Run GLUE (Generalized Likelihood Uncertainty Estimation) analysis."""
        return self.run_optimization('glue')

    def run_basin_hopping(self) -> Path:
        """Run Basin Hopping global optimization."""
        return self.run_optimization('basin_hopping')

    def run_nelder_mead(self) -> Path:
        """Run Nelder-Mead simplex optimization."""
        return self.run_optimization('nelder_mead')

    def run_ga(self) -> Path:
        """Run Genetic Algorithm (GA) optimization."""
        return self.run_optimization('ga')

    def run_bayesian_opt(self) -> Path:
        """Run Bayesian Optimization with Gaussian Process surrogate."""
        return self.run_optimization('bayesian_opt')

    def run_moead(self) -> Path:
        """Run MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)."""
        return self.run_optimization('moead')

    def run_simulated_annealing(self) -> Path:
        """Run Simulated Annealing optimization."""
        return self.run_optimization('simulated_annealing')

    def run_abc(self) -> Path:
        """Run Approximate Bayesian Computation (ABC-SMC) for likelihood-free inference."""
        return self.run_optimization('abc')

    def run_adam(self, steps: int = 100, lr: float = 0.01) -> Path:
        """
        Run Adam gradient-based optimization.

        Args:
            steps: Number of optimization steps (passed via config ADAM_STEPS)
            lr: Learning rate (passed via config ADAM_LR)

        Returns:
            Path to results file
        """
        # Store parameters in runtime overrides for the algorithm to use
        self._runtime_overrides['ADAM_STEPS'] = steps
        self._runtime_overrides['ADAM_LR'] = lr
        return self.run_optimization('adam')

    def run_lbfgs(self, steps: int = 50, lr: float = 0.1) -> Path:
        """
        Run L-BFGS gradient-based optimization.

        Args:
            steps: Maximum number of steps (passed via config LBFGS_STEPS)
            lr: Initial step size (passed via config LBFGS_LR)

        Returns:
            Path to results file
        """
        # Store parameters in runtime overrides for the algorithm to use
        self._runtime_overrides['LBFGS_STEPS'] = steps
        self._runtime_overrides['LBFGS_LR'] = lr
        return self.run_optimization('lbfgs')

    # =========================================================================
    # Final Evaluation
    # =========================================================================

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with best parameters over full experiment period.

        Evaluates the model on both calibration and evaluation windows, then restores
        settings for reproducibility. Subclasses may override for custom behavior.

        Args:
            best_params: Best parameters from optimization (physical units)

        Returns:
            Dict with 'final_metrics', 'calibration_metrics', 'evaluation_metrics',
            'success', 'best_params', or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Running model with best parameters over full simulation period...")

        try:
            # Update file manager for full period
            self._update_file_manager_for_final_run()

            # Apply best parameters directly
            if not self._apply_best_parameters_for_final(best_params):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            # Setup output directory
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # Update file manager output path
            self._update_file_manager_output_path(final_output_dir)

            # Run model directly using specific hook
            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error(f"{self._get_model_name()} run failed during final evaluation")
                return None

            # Calculate metrics for both periods (calibration_only=False)
            metrics = self.calibration_target.calculate_metrics(
                final_output_dir,
                calibration_only=False
            )

            if not metrics:
                self.logger.error("Failed to calculate final evaluation metrics")
                return None

            # Extract period-specific metrics
            calib_metrics = self._extract_period_metrics(metrics, 'Calib')
            eval_metrics = self._extract_period_metrics(metrics, 'Eval')

            # Log detailed results
            self._log_final_evaluation_results(calib_metrics, eval_metrics)

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            return final_result

        except (IOError, OSError, ValueError, RuntimeError) as e:
            self.logger.error(f"Error in final evaluation: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Restore optimization settings
            self._restore_model_decisions_for_optimization()
            self._restore_file_manager_for_optimization()

    def _extract_period_metrics(self, all_metrics: Dict, period_prefix: str) -> Dict:
        """Extract metrics for a specific period. Delegates to FinalEvaluationOrchestrator."""
        return self.final_orchestrator.extract_period_metrics(all_metrics, period_prefix)

    def _log_final_evaluation_results(self, calib_metrics: Dict, eval_metrics: Dict) -> None:
        """Log final evaluation results. Delegates to FinalEvaluationOrchestrator."""
        self.final_orchestrator.log_results(calib_metrics, eval_metrics)

    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager for full experiment period. Delegates to FinalEvaluationOrchestrator."""
        self.final_orchestrator.update_file_manager_for_final_run(
            self._get_final_file_manager_path()
        )


    def _restore_model_decisions_for_optimization(self) -> None:
        """Restore model decisions. Delegates to FinalEvaluationOrchestrator."""
        self.final_orchestrator.restore_model_decisions()

    def _restore_file_manager_for_optimization(self) -> None:
        """Restore file manager to calibration period. Delegates to FinalEvaluationOrchestrator."""
        self.final_orchestrator.restore_file_manager(self._get_final_file_manager_path())

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """Apply best parameters for final evaluation."""
        try:
            return self.worker.apply_parameters(
                best_params,
                self.optimization_settings_dir,
                config=self.config
            )
        except (ValueError, IOError, RuntimeError) as e:
            self.logger.error(f"Error applying parameters for final evaluation: {e}")
            return False

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """Update file manager output path. Delegates to FinalEvaluationOrchestrator."""
        self.final_orchestrator.update_file_manager_output_path(
            self._get_final_file_manager_path(), output_dir
        )

    def _save_final_evaluation_results(
        self,
        final_result: Dict[str, Any],
        algorithm: str
    ) -> None:
        """
        Save final evaluation results to JSON file.

        Args:
            final_result: Final evaluation results dictionary
            algorithm: Algorithm name (e.g., 'PSO', 'DDS')
        """
        self.final_orchestrator.save_results(final_result, algorithm)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Cleanup parallel processing directories and temporary files."""
        self._shutdown_mpi_strategy()
        if self.parallel_dirs:
            self.cleanup_parallel_processing(self.parallel_dirs)
