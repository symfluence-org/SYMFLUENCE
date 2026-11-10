# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Base worker for model evaluation during optimization.

Provides ``WorkerTask``, ``WorkerResult`` dataclasses and the ``BaseWorker``
abstract class implementing Template Method: apply_parameters → run_model →
calculate_metrics, with exponential-backoff retry for transient failures.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from symfluence.core.constants import ModelDefaults
from symfluence.core.exceptions import RetryExhaustedError
from symfluence.core.metrics.metric_transformer import MetricTransformer
from symfluence.core.metrics.penalty import is_failure_sentinel, is_penalty, sanitize_score
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.mixins.logging import _class_logger_name

logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    """Single model evaluation task: parameters + paths + config."""
    individual_id: int
    params: Dict[str, float]
    proc_id: int
    config: Dict[str, Any]
    settings_dir: Path
    output_dir: Path
    sim_dir: Optional[Path] = None
    iteration: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.settings_dir, str):
            self.settings_dir = Path(self.settings_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.sim_dir, str):
            self.sim_dir = Path(self.sim_dir)

    @classmethod
    def from_legacy_dict(cls, task_data: Dict[str, Any]) -> 'WorkerTask':
        """
        Create a WorkerTask from a legacy dictionary format.

        This maintains backward compatibility with existing worker functions.

        Args:
            task_data: Dictionary in legacy format

        Returns:
            WorkerTask instance
        """
        # Handle various key naming conventions
        individual_id = task_data.get('individual_id', task_data.get('task_id', 0))
        params = task_data.get('params', task_data.get('parameters', {}))
        proc_id = task_data.get('proc_id', task_data.get('process_id', 0))

        # Settings directory - various key names used
        settings_dir = (
            task_data.get('proc_settings_dir') or
            task_data.get('settings_dir') or
            task_data.get('optimization_settings_dir') or
            Path('.')
        )

        # Output directory
        output_dir = (
            task_data.get('proc_output_dir') or
            task_data.get('output_dir') or
            Path('.')
        )

        # Simulation directory
        sim_dir = (
            task_data.get('proc_sim_dir') or
            task_data.get('sim_dir') or
            None
        )

        # Config - could be nested or at top level
        config = task_data.get('config', {})

        # Convert Pydantic model to dict if needed
        if hasattr(config, 'model_dump'):
            config = config.model_dump()
        elif hasattr(config, 'dict'):
            config = config.dict()

        if not config:
            # Extract config keys from task_data itself
            config_keys = [
                'EXPERIMENT_ID', 'DOMAIN_NAME', 'ROOT_PATH', 'HYDROLOGICAL_MODEL',
                'ROUTING_MODEL', 'CALIBRATION_METRIC', 'CALIBRATION_PERIOD',
                'EVALUATION_PERIOD', 'NUM_PROCESSES', 'DOMAIN_DEFINITION_METHOD',
                'SYMFLUENCE_DATA_DIR',
            ]
            config = {k: task_data[k] for k in config_keys if k in task_data}

        # Additional data
        additional_keys = set(task_data.keys()) - {
            'individual_id', 'task_id', 'params', 'parameters', 'proc_id',
            'process_id', 'proc_settings_dir', 'settings_dir',
            'optimization_settings_dir', 'proc_output_dir', 'output_dir',
            'proc_sim_dir', 'sim_dir', 'config', 'iteration'
        } - set(config.keys())
        additional_data = {k: task_data[k] for k in additional_keys}

        return cls(
            individual_id=individual_id,
            params=params,
            proc_id=proc_id,
            config=config,
            settings_dir=settings_dir,
            output_dir=output_dir,
            sim_dir=sim_dir,
            iteration=task_data.get('iteration'),
            additional_data=additional_data
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format.

        Returns:
            Dictionary in legacy format
        """
        result = {
            'individual_id': self.individual_id,
            'params': self.params,
            'proc_id': self.proc_id,
            'proc_settings_dir': str(self.settings_dir),
            'proc_output_dir': str(self.output_dir),
            'config': self.config,
        }
        if self.sim_dir:
            result['proc_sim_dir'] = str(self.sim_dir)
        if self.iteration is not None:
            result['iteration'] = self.iteration
        result.update(self.additional_data)
        return result


@dataclass
class WorkerResult:
    """Result of a worker evaluation: score + metrics + error info."""
    individual_id: int
    params: Dict[str, float]
    score: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    runtime: Optional[float] = None
    iteration: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the evaluation was successful."""
        return self.error is None and self.score is not None

    @property
    def valid_score(self) -> bool:
        """Check if the score is valid (not NaN, not penalty value)."""
        return not is_penalty(self.score)

    @classmethod
    def failure(
        cls,
        individual_id: int,
        params: Dict[str, float],
        error: str,
        penalty_score: float = ModelDefaults.PENALTY_SCORE
    ) -> 'WorkerResult':
        """
        Create a failure result.

        Args:
            individual_id: Task identifier
            params: Parameters that were attempted
            error: Error message
            penalty_score: Penalty score to assign

        Returns:
            WorkerResult indicating failure
        """
        return cls(
            individual_id=individual_id,
            params=params,
            score=penalty_score,
            error=error
        )

    @classmethod
    def from_legacy_dict(cls, result_data: Dict[str, Any]) -> 'WorkerResult':
        """
        Create a WorkerResult from a legacy dictionary format.

        Args:
            result_data: Dictionary in legacy format

        Returns:
            WorkerResult instance
        """
        individual_id = result_data.get('individual_id', result_data.get('task_id', 0))
        params = result_data.get('params', result_data.get('parameters', {}))

        # Score may be under various keys
        score = (
            result_data.get('score') or
            result_data.get('fitness') or
            result_data.get('objective') or
            result_data.get('kge') or
            result_data.get('nse')
        )

        # Handle metrics
        metrics = result_data.get('metrics', {})
        if not metrics:
            # Look for common metric keys
            metric_keys = ['kge', 'nse', 'rmse', 'mae', 'bias', 'correlation']
            metrics = {k: result_data[k] for k in metric_keys if k in result_data}

        return cls(
            individual_id=individual_id,
            params=params,
            score=score,
            metrics=metrics,
            error=result_data.get('error'),
            runtime=result_data.get('runtime'),
            iteration=result_data.get('iteration'),
            additional_data={
                k: v for k, v in result_data.items()
                if k not in {'individual_id', 'task_id', 'params', 'parameters',
                            'score', 'fitness', 'objective', 'metrics', 'error',
                            'runtime', 'iteration', 'kge', 'nse', 'rmse', 'mae',
                            'bias', 'correlation'}
            }
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format.

        Returns:
            Dictionary in legacy format
        """
        result = {
            'individual_id': self.individual_id,
            'params': self.params,
            'score': self.score,
            'metrics': self.metrics,
        }
        if self.error:
            result['error'] = self.error
        if self.runtime is not None:
            result['runtime'] = self.runtime
        if self.iteration is not None:
            result['iteration'] = self.iteration
        # Flatten common metrics for backward compatibility
        for key in ['kge', 'nse', 'rmse', 'mae', 'bias']:
            if key in self.metrics:
                result[key] = self.metrics[key]
        result.update(self.additional_data)
        return result


class BaseWorker(ABC):
    """Abstract base for model evaluation workers (Template Method pattern).

    Subclasses implement three abstract methods:

    - ``apply_parameters(params, settings_dir, **kw) → bool``
    - ``run_model(config, settings_dir, output_dir, **kw) → bool``
    - ``calculate_metrics(output_dir, config, **kw) → Dict[str, float]``

    The ``evaluate(task)`` template method orchestrates these three steps
    with exponential-backoff retry for transient failures (stale file handle,
    permission denied, etc.) and returns a ``WorkerResult``.

    Config keys: ``CALIBRATION_METRIC`` (default 'KGE'),
    ``PENALTY_SCORE`` (-999.0), ``WORKER_MAX_RETRIES`` (3),
    ``WORKER_BASE_DELAY`` (0.5s).
    """

    # Default retry settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 0.5
    DEFAULT_PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    # Transient errors that warrant retry
    TRANSIENT_ERRORS = (
        'stale file handle',
        'resource temporarily unavailable',
        'no such file or directory',
        'permission denied',
        'connection refused',
        'broken pipe',
    )

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(
            _class_logger_name(self.__class__))

        # Shadow the subclass-level shared StreamflowMetrics instance when a
        # non-default Box-Cox lambda is configured (used by *_BOX_COX metrics)
        box_cox_lambda = self._cfg('BOX_COX_LAMBDA')
        if box_cox_lambda is not None:
            self._streamflow_metrics = StreamflowMetrics(
                box_cox_lambda=float(box_cox_lambda))

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Get config value from the worker's config dict.

        Workers receive flat config dicts from the optimization loop.

        Args:
            key: Flat config key (e.g., 'DOMAIN_NAME')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        return default

    def _get_config_value(
        self,
        typed_accessor: Any = None,
        default: Any = None,
        dict_key: Optional[str] = None
    ) -> Any:
        """Get config value with typed accessor + dict fallback.

        Mirrors :class:`~symfluence.core.mixins.config.ConfigMixin` so that
        workers can use the same typed-config access pattern as optimizers.

        Args:
            typed_accessor: Callable accessing typed config,
                e.g. ``lambda: self.config.domain.name``
            default: Fallback value
            dict_key: Legacy flat dict key (e.g. 'DOMAIN_NAME')

        Returns:
            Configuration value or *default*.
        """
        return resolve_config_value(self.config, typed_accessor, default, dict_key)

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self._cfg('WORKER_MAX_RETRIES', self.DEFAULT_MAX_RETRIES)

    @property
    def base_delay(self) -> float:
        """Base delay for exponential backoff."""
        return self._cfg('WORKER_BASE_DELAY', self.DEFAULT_BASE_DELAY)

    @property
    def penalty_score(self) -> float:
        """Penalty score for failed evaluations."""
        return self._cfg('PENALTY_SCORE', self.DEFAULT_PENALTY_SCORE)

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to model configuration files.

        Args:
            params: Dictionary of parameter names to values
            settings_dir: Path to settings directory
            **kwargs: Additional model-specific arguments

        Returns:
            True if parameters were applied successfully, False otherwise
        """
        pass

    @abstractmethod
    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run the model simulation.

        Args:
            config: Configuration dictionary
            settings_dir: Path to settings directory
            output_dir: Path for outputs
            **kwargs: Additional model-specific arguments

        Returns:
            True if model ran successfully, False otherwise
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate objective metrics from model outputs.

        Args:
            output_dir: Path to model outputs
            config: Configuration dictionary
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary of metric names to values
        """
        pass

    # =========================================================================
    # Template method - main evaluation logic
    # =========================================================================

    def evaluate(self, task: WorkerTask) -> WorkerResult:
        """
        Evaluate a parameter set by running the full workflow.

        This is the template method that orchestrates:
        1. Parameter application
        2. Model execution
        3. Metric calculation

        Includes retry logic for transient failures.

        Args:
            task: WorkerTask containing parameters and paths

        Returns:
            WorkerResult with score and metrics
        """
        start_time = time.time()

        try:
            result = self._evaluate_with_retry(task)
            result.runtime = time.time() - start_time
            return result

        except (ValueError, RuntimeError, IOError) as e:
            self.logger.error(
                f"Evaluation failed for individual {task.individual_id}: {e}"
            )
            return WorkerResult.failure(
                individual_id=task.individual_id,
                params=task.params,
                error=str(e),
                penalty_score=self.penalty_score
            )

    def _evaluate_with_retry(self, task: WorkerTask) -> WorkerResult:
        """
        Execute evaluation with retry logic for transient failures.

        Args:
            task: WorkerTask to evaluate

        Returns:
            WorkerResult from evaluation
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._evaluate_once(task)

            except (ValueError, RuntimeError, IOError, TimeoutError) as e:
                last_error = e

                if attempt >= self.max_retries:
                    raise

                if not self._is_transient_error(e):
                    raise

                delay = self.base_delay * (2 ** attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RetryExhaustedError("Evaluation failed: retry loop completed without success or captured error")

    def _evaluate_once(self, task: WorkerTask) -> WorkerResult:
        """
        Execute a single evaluation attempt.

        Args:
            task: WorkerTask to evaluate

        Returns:
            WorkerResult from evaluation
        """
        # Step 1: Apply parameters
        # Note: proc_output_dir, sim_dir, proc_id must be passed explicitly because
        # they're excluded from additional_data (since they're primary fields in WorkerTask)
        if not self.apply_parameters(
            task.params,
            task.settings_dir,
            config=task.config,
            proc_output_dir=task.output_dir,
            output_dir=task.output_dir,
            sim_dir=task.sim_dir,
            proc_id=task.proc_id,
            **task.additional_data
        ):
            return WorkerResult.failure(
                individual_id=task.individual_id,
                params=task.params,
                error="Failed to apply parameters",
                penalty_score=self.penalty_score
            )

        # Step 2: Run model
        if not self.run_model(
            task.config,
            task.settings_dir,
            task.output_dir,
            sim_dir=task.sim_dir,
            proc_id=task.proc_id,
            params=task.params,
            **task.additional_data
        ):
            return WorkerResult.failure(
                individual_id=task.individual_id,
                params=task.params,
                error="Model execution failed",
                penalty_score=self.penalty_score
            )

        # Step 3: Calculate metrics
        metrics = self.calculate_metrics(
            task.output_dir,
            task.config,
            sim_dir=task.sim_dir,
            settings_dir=str(task.settings_dir),
            proc_id=task.proc_id,
            **task.additional_data
        )

        # Determine primary score
        score = self._extract_primary_score(metrics, task.config)

        return WorkerResult(
            individual_id=task.individual_id,
            params=task.params,
            score=score,
            metrics=metrics,
            iteration=task.iteration
        )

    def _extract_primary_score(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> float:
        """
        Extract the primary optimization score from metrics.

        Supports two modes:
        1. Single metric: Extracts one metric (e.g., CALIBRATION_METRIC: KGE)
        2. Composite metric: Weighted combination of multiple metrics
           (e.g., CALIBRATION_METRIC: COMPOSITE with COMPOSITE_METRIC: {KGE: 0.5, KGE_LOG: 0.5})

        Automatically transforms scores to maximization convention using
        MetricTransformer, so optimization algorithms can always maximize.

        Args:
            metrics: Dictionary of calculated metrics
            config: Configuration dictionary

        Returns:
            Primary score for optimization (transformed for maximization)
        """
        # Get configured metric name (config is a task config dict, not self.config)
        if isinstance(config, dict):
            metric_name = config.get(
                'OPTIMIZATION_METRIC',
                config.get('CALIBRATION_METRIC', 'KGE')
            )
            composite_config = config.get('COMPOSITE_METRIC')
        else:
            # Pydantic SymfluenceConfig
            try:
                metric_name = config.optimization.metric
            except AttributeError:
                metric_name = 'KGE'
            try:
                composite_config = config.optimization.composite_metric
            except AttributeError:
                composite_config = None

        # Check for composite objective function
        if (metric_name.upper() == 'COMPOSITE' or composite_config) and isinstance(composite_config, dict):
            return self._extract_composite_score(metrics, composite_config)

        # Debug log to trace metric configuration
        self.logger.debug(
            f"Primary metric extraction: OPTIMIZATION_METRIC={config.get('OPTIMIZATION_METRIC')}, "
            f"CALIBRATION_METRIC={config.get('CALIBRATION_METRIC')}, using metric_name={metric_name}"
        )

        source_metric = metric_name
        raw_value = self._find_metric_value(metrics, metric_name)
        if raw_value is None:
            source_metric, raw_value = self._find_fallback_metric(metrics, metric_name)

        # Return penalty if no metric found
        if raw_value is None:
            self.logger.warning(
                f"Could not find metric '{metric_name}' in results. "
                f"Available keys: {list(metrics.keys())}"
            )
            return self.penalty_score

        # Worker failure sentinels (-999 / -9999 / NaN) are failures, whatever
        # the metric's direction -- never let them be negated into a win.
        if is_failure_sentinel(raw_value):
            return self.penalty_score

        # Transform to maximization convention under the direction of the
        # metric the value actually came from.
        transformed_score = MetricTransformer.transform_for_maximization(
            source_metric, raw_value
        )

        self.logger.debug(
            f"Score extraction: metric={source_metric}, raw={raw_value:.4f}, "
            f"direction={MetricTransformer.get_direction(source_metric)}, "
            f"transformed={transformed_score}"
        )

        return sanitize_score(transformed_score, self.penalty_score)

    def _extract_composite_score(
        self,
        metrics: Dict[str, float],
        composite_config: Dict[str, float]
    ) -> float:
        """
        Compute a weighted composite score from multiple metrics.

        Allows single-objective optimizers (DDS, SCE) to optimize multi-criteria
        objectives by combining metrics into a single scalar. Each component is
        transformed to maximization convention before weighting, ensuring that
        minimize-metrics (RMSE, MAE) and maximize-metrics (KGE, NSE) can be
        combined correctly.

        Config example:
            COMPOSITE_METRIC:
              KGE: 0.5       # 50% weight on standard KGE
              KGE_LOG: 0.3   # 30% weight on KGE of log-transformed flows
              KGE_INV: 0.2   # 20% weight on KGE of inverse flows

        Args:
            metrics: Dictionary of calculated metrics
            composite_config: Dict mapping metric names to weights

        Returns:
            Weighted composite score (maximization convention)
        """
        # Normalize weights to sum to 1.0
        total_weight = sum(composite_config.values())
        if total_weight <= 0:
            self.logger.error("COMPOSITE_METRIC weights sum to zero or negative")
            return self.penalty_score

        composite_score = 0.0
        components_found = 0

        for metric_name, weight in composite_config.items():
            normalized_weight = weight / total_weight

            raw_value = self._find_metric_value(metrics, metric_name)

            if is_failure_sentinel(raw_value):
                self.logger.warning(
                    f"Composite component '{metric_name}' missing or failed in metrics. "
                    f"Available: {list(metrics.keys())}"
                )
                # Use penalty for missing component
                composite_score += normalized_weight * self.penalty_score
                continue

            # Transform to maximization convention
            transformed = MetricTransformer.transform_for_maximization(
                metric_name, raw_value
            )
            transformed = sanitize_score(transformed, self.penalty_score)

            composite_score += normalized_weight * transformed
            components_found += 1

            self.logger.debug(
                f"  Composite component: {metric_name} = {raw_value:.4f} "
                f"(transformed={transformed:.4f}, weight={normalized_weight:.2f})"
            )

        if components_found == 0:
            self.logger.error("No composite metric components found")
            return self.penalty_score

        self.logger.debug(
            f"Composite score: {composite_score:.4f} "
            f"({components_found}/{len(composite_config)} components)"
        )

        return composite_score

    def _find_metric_value(
        self,
        metrics: Dict[str, float],
        metric_name: str,
    ) -> Optional[float]:
        """
        Find a metric value in the metrics dict with case-insensitive fallback.

        Args:
            metrics: Dictionary of calculated metrics
            metric_name: Name of the metric to find

        Returns:
            The metric value, or None if not found
        """
        # Exact match
        if metric_name in metrics:
            return metrics[metric_name]

        # Calib_ prefix
        if f"Calib_{metric_name}" in metrics:
            return metrics[f"Calib_{metric_name}"]

        # Case-insensitive search
        metric_lower = metric_name.lower()
        for k, v in metrics.items():
            if k.lower() == metric_lower or k.lower() == f"calib_{metric_lower}":
                return v

        return None

    # Generic stand-ins tried when the configured primary metric is missing.
    FALLBACK_METRICS = ('kge', 'nse', 'score', 'fitness', 'objective')

    def _find_fallback_metric(
        self,
        metrics: Dict[str, float],
        metric_name: str,
    ) -> Tuple[str, Optional[float]]:
        """Find a stand-in for a missing primary metric.

        Only a stand-in with the same optimization direction as the configured
        metric is accepted: substituting KGE for a missing RMSE would mix
        score scales (and, before this check, negated KGE so a crashed
        ``{'kge': -999}`` run scored +999). A direction mismatch is refused
        with a warning and the caller falls back to the penalty.

        Returns:
            ``(source_metric_name, value)`` or ``(metric_name, None)``.
        """
        wanted = MetricTransformer.get_direction(metric_name)
        for alt in self.FALLBACK_METRICS:
            for k, v in metrics.items():
                if k.lower() != alt:
                    continue
                if MetricTransformer.get_direction(k) != wanted:
                    self.logger.warning(
                        f"Configured metric '{metric_name}' ({wanted}) missing; "
                        f"refusing fallback to '{k}' "
                        f"({MetricTransformer.get_direction(k)}) -- "
                        f"scoring this run as a failure."
                    )
                    return metric_name, None
                return k, v
        return metric_name, None

    def _is_transient_error(self, error: Exception) -> bool:
        """
        Check if an error is likely transient and worth retrying.

        Args:
            error: The exception to check

        Returns:
            True if the error is likely transient
        """
        error_str = str(error).lower()
        return any(te in error_str for te in self.TRANSIENT_ERRORS)

    # =========================================================================
    # Hooks for subclass customization
    # =========================================================================

    def pre_evaluation(self, task: WorkerTask) -> None:
        """
        Hook called before evaluation begins.

        Subclasses can override to perform setup tasks.

        Args:
            task: WorkerTask about to be evaluated
        """
        pass

    def post_evaluation(self, task: WorkerTask, result: WorkerResult) -> None:
        """
        Hook called after evaluation completes.

        Subclasses can override to perform cleanup or logging.

        Args:
            task: WorkerTask that was evaluated
            result: WorkerResult from evaluation
        """
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    def setup_worker_isolation(self) -> None:
        """
        Setup process isolation for worker.

        Sets environment variables to prevent thread contention
        and file locking issues in parallel execution.
        """
        import os

        env_vars = {
            **WORKER_THREAD_ENV,
            'VECLIB_MAXIMUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            **WORKER_FILE_LOCKING_ENV,
            'HDF5_DISABLE_VERSION_CHECK': '1',
        }

        for key, value in env_vars.items():
            os.environ[key] = value

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static method for use with process pools.

        This is a template that subclasses should override to provide
        a picklable worker function.

        Args:
            task_data: Dictionary with task parameters

        Returns:
            Dictionary with results
        """
        raise NotImplementedError(
            "Subclasses must implement evaluate_worker_function"
        )

    # =========================================================================
    # Native Gradient Support (Optional)
    # =========================================================================

    def supports_native_gradients(self) -> bool:
        """
        Check if this worker supports native gradient computation.

        Native gradients (e.g., via JAX autodiff) can be significantly more
        efficient than finite-difference gradients for gradient-based optimization.
        When supported, gradient computation requires only ~2 model evaluations
        (forward + backward pass) instead of 2N+1 evaluations for N parameters.

        Override this method in subclasses that implement autodiff-capable models.

        Returns:
            True if compute_gradient() and evaluate_with_gradient() are available
            and functional. Default: False (use finite differences).

        Example:
            >>> class HBVWorker(BaseWorker):
            ...     def supports_native_gradients(self) -> bool:
            ...         return HAS_JAX  # True if JAX is installed
        """
        return False

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss with respect to parameters using native method.

        This method should be overridden by workers that support autodiff
        (e.g., JAX, PyTorch). The gradient is computed for the loss function
        (negative of the objective metric), so for maximizing KGE, gradients
        point in the direction of decreasing KGE (increasing loss).

        Args:
            params: Dictionary mapping parameter names to current values
            metric: Objective metric to compute gradient for ('kge', 'nse', etc.)

        Returns:
            Dictionary mapping parameter names to gradient values (d(loss)/d(param)),
            or None if native gradients are not supported.

        Note:
            - Gradients are for the LOSS (negative metric), not the metric itself
            - For maximization problems, negate the gradient for gradient ascent
            - Returns None by default; override in autodiff-capable workers

        Example:
            >>> worker = HBVWorker(config, logger)
            >>> if worker.supports_native_gradients():
            ...     grads = worker.compute_gradient({'fc': 250.0, 'k1': 0.1}, 'kge')
            ...     print(grads)  # {'fc': -0.001, 'k1': 0.05, ...}
        """
        return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Evaluate loss and compute gradient in a single pass.

        This is more efficient than calling evaluate + compute_gradient separately
        when using autodiff, as the forward pass computation can be shared.
        Uses jax.value_and_grad or torch.autograd for efficient computation.

        Args:
            params: Dictionary mapping parameter names to current values
            metric: Objective metric ('kge', 'nse', etc.)

        Returns:
            Tuple of (loss_value, gradient_dict):
            - loss_value: Scalar loss (negative of metric, for minimization)
            - gradient_dict: Dictionary mapping parameter names to gradients,
              or None if native gradients not supported

        Note:
            - Returns (loss, None) by default; override in autodiff-capable workers
            - loss is NEGATIVE of metric (e.g., -KGE) for minimization
            - Subclasses should use value_and_grad for efficiency

        Example:
            >>> worker = HBVWorker(config, logger)
            >>> loss, grads = worker.evaluate_with_gradient({'fc': 250.0}, 'kge')
            >>> print(f"Loss: {loss}, Gradients: {grads}")
        """
        # Default implementation: not supported
        # Subclasses override with actual autodiff implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native gradients. "
            "Override evaluate_with_gradient() or use finite differences."
        )


# =============================================================================
# Shared module-level worker entry point
# =============================================================================

#: One thread per process: parallelism is across the pool, not inside a run.
WORKER_THREAD_ENV: Dict[str, str] = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
}

#: Disable HDF5/NetCDF file locking so concurrent workers can read shared
#: forcing/attribute files (same values as ``BaseWorker.setup_worker_isolation``).
WORKER_FILE_LOCKING_ENV: Dict[str, str] = {
    'NETCDF_DISABLE_LOCKING': '1',
    'HDF5_USE_FILE_LOCKING': 'FALSE',
}

#: Suppress macOS malloc stack-logging noise in child processes.
MACOS_MALLOC_QUIET_ENV: Dict[str, str] = {'MallocStackLogging': '0'}

#: Default random start-up jitter (seconds) to stagger file access across workers.
DEFAULT_WORKER_START_JITTER: Tuple[float, float] = (0.1, 0.5)


def _exit_on_signal(signum, frame) -> None:  # pragma: no cover - exercised via signals
    """Signal handler installed in worker processes: exit promptly on SIGTERM/SIGINT."""
    import sys
    sys.exit(1)


def _install_worker_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers in the current (worker) process.

    ``signal.signal`` raises ``ValueError`` outside the main thread (e.g. when a
    pool runs tasks in threads); handlers are then simply not installed.
    """
    import signal

    try:
        signal.signal(signal.SIGTERM, _exit_on_signal)
        signal.signal(signal.SIGINT, _exit_on_signal)
    except ValueError:
        pass


def run_worker_task(
    worker_cls: type,
    task_data: Dict[str, Any],
    *,
    model_name: str = '',
    isolate: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
    start_jitter: Optional[Tuple[float, float]] = DEFAULT_WORKER_START_JITTER,
    catch_exceptions: bool = True,
    error_prefix: Optional[str] = None,
    pass_config: bool = True,
) -> Dict[str, Any]:
    """Evaluate one legacy task dict with ``worker_cls`` inside a worker process.

    Shared body of the per-model ``_evaluate_<model>_parameters_worker``
    functions. Those must stay thin module-level ``def`` wrappers (with their
    original names) because the MPI strategies import them by
    ``__module__``/``__name__`` and process pools pickle them by reference.

    Args:
        worker_cls: ``BaseWorker`` subclass to instantiate.
        task_data: Legacy task dictionary (see ``WorkerTask.from_legacy_dict``).
        model_name: Label used in the default error prefix.
        isolate: When True, install SIGTERM/SIGINT handlers in this process,
            set single-thread and HDF5/NetCDF no-locking env vars (plus
            ``extra_env``), and sleep a random ``start_jitter`` interval.
        extra_env: Additional env vars set when ``isolate`` is True.
        start_jitter: ``(low, high)`` seconds for the random start-up delay,
            or None for no delay. Only used when ``isolate`` is True.
        catch_exceptions: When True, any exception becomes a penalty result
            dict instead of propagating to the pool/MPI runner.
        error_prefix: Prefix for the ``error`` message of the penalty dict;
            defaults to ``"<model_name> worker exception"``.
        pass_config: Construct the worker as ``worker_cls(config=...)`` from
            ``task_data['config']``; when False, ``worker_cls()``.

    Returns:
        Legacy result dictionary.
    """
    if isolate:
        import os
        import random

        _install_worker_signal_handlers()
        os.environ.update(WORKER_THREAD_ENV)
        os.environ.update(WORKER_FILE_LOCKING_ENV)
        if extra_env:
            os.environ.update(extra_env)
        if start_jitter is not None:
            time.sleep(random.uniform(*start_jitter))

    if not catch_exceptions:
        return _evaluate_task_dict(worker_cls, task_data, pass_config)

    try:
        return _evaluate_task_dict(worker_cls, task_data, pass_config)
    except Exception as e:  # noqa: BLE001 — calibration resilience
        import traceback

        prefix = error_prefix or f'{model_name} worker exception'
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'{prefix}: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1),
        }


def _evaluate_task_dict(
    worker_cls: type, task_data: Dict[str, Any], pass_config: bool
) -> Dict[str, Any]:
    """Instantiate ``worker_cls``, evaluate the task and return the legacy dict."""
    worker = worker_cls(config=task_data.get('config')) if pass_config else worker_cls()
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
