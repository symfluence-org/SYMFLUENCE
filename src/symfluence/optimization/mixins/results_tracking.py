# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Results Tracking Mixin

Provides results persistence and tracking for optimization runs.
Handles iteration history, best solution tracking, and results file I/O.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from symfluence.core.mixins import ConfigMixin

logger = logging.getLogger(__name__)


def _scalar_for_csv(value: Any) -> Any:
    """Reduce a parameter value to a plain scalar for CSV serialisation.

    SUMMA and other array-first parameter managers return
    ``np.array([x])`` or per-HRU ``np.array([x, x, x])`` from
    ``_format_parameter_value``. pandas will happily put those objects
    into a DataFrame, but ``to_csv`` then writes their ``str(array)``
    repr ("[0.1]", "[273.16 273.16 ...]") which downstream consumers
    (SALib's sobol.analyze, VISCOUS) can't read back as numeric.

    We reduce:
      * length-1 arrays / lists / tuples to their single element;
      * multi-element homogeneous arrays to their mean (per-HRU
        parameters are calibrated jointly today, so the mean is a
        lossless summary — if that changes, the right thing is to
        log a separate per-HRU column rather than let the array
        string leak into the CSV);
      * numpy scalar types to their Python equivalent;
      * anything else is returned unchanged.
    """
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        if flat.size == 0:
            return float("nan")
        if flat.size == 1:
            return flat.item()
        # Homogeneous per-HRU broadcast — unique value; preserve it.
        if np.unique(flat).size == 1:
            return flat[0].item()
        # Truly varying per-HRU — mean is a better summary than the
        # array repr. Callers that need per-HRU can log separately.
        return float(flat.mean())
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _scalar_for_csv(value[0])
        try:
            return _scalar_for_csv(np.asarray(value))
        except (TypeError, ValueError):
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


class ResultsTrackingMixin(ConfigMixin):
    """
    Mixin class providing results tracking and persistence for optimizers.

    Requires the following attributes on the class using this mixin:
    - self.config: Dict[str, Any]
    - self.logger: logging.Logger
    - self.results_dir: Path

    Provides:
    - Iteration history recording
    - Best parameter tracking
    - Results file I/O (CSV, JSON)
    - Pareto front storage for multi-objective optimization
    """

    def __init_results_tracking__(self):
        """Initialize results tracking state. Call in optimizer __init__."""
        self._iteration_history: List[Dict[str, Any]] = []
        self._best_score: float = float('-inf')
        self._best_params: Optional[Dict[str, float]] = None
        self._best_iteration: int = -1
        self._pareto_front: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None

    # =========================================================================
    # Iteration tracking
    # =========================================================================

    def record_iteration(
        self,
        iteration: int,
        score: float,
        params: Dict[str, float],
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record results from an optimization iteration.

        Args:
            iteration: Iteration number
            score: Fitness/objective score
            params: Parameter values used
            additional_metrics: Optional additional metrics (NSE, RMSE, etc.)
        """
        # Coerce array-valued parameters down to plain scalars before
        # recording. Model-specific parameter managers (notably SUMMA's)
        # return 1-element numpy arrays or per-HRU arrays from
        # _format_parameter_value so the downstream model wrappers get
        # the array type they expect. Feeding those arrays straight
        # into pandas.DataFrame writes the string representation
        # ("[0.1]", "[273.16]") to CSV — co-author PW reported that
        # the sensitivity-analysis pipeline then read those columns
        # as strings and SALib rejected them with "Bounds are not
        # legal". Convert at the recording boundary so the CSV only
        # carries numeric scalars without touching the shape passed
        # to the model itself.
        scalar_params = {k: _scalar_for_csv(v) for k, v in params.items()}

        record = {
            'iteration': iteration,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            **scalar_params,
        }

        if additional_metrics:
            record.update(additional_metrics)

        self._iteration_history.append(record)

        self.logger.debug(
            f"Iteration {iteration}: score={score:.4f}, params={params}"
        )

    def update_best(
        self,
        score: float,
        params: Dict[str, float],
        iteration: int
    ) -> bool:
        """
        Update best solution if the new score is better.

        Args:
            score: New fitness score
            params: Parameter values
            iteration: Iteration number

        Returns:
            True if the best was updated, False otherwise
        """
        # Handle invalid scores
        if score is None or np.isnan(score) or score <= -900:
            return False

        # Check if better (for maximization problems like KGE)
        if score > self._best_score:
            self._best_score = score
            # Same scalar coercion as record_iteration so the
            # best-params JSON dump doesn't re-leak array reprs.
            self._best_params = {k: _scalar_for_csv(v) for k, v in params.items()}
            self._best_iteration = iteration

            self.logger.debug(
                f"New best at iteration {iteration}: score={score:.4f}"
            )
            return True

        return False

    @property
    def best_score(self) -> float:
        """Get the best score found so far."""
        return self._best_score

    @property
    def best_params(self) -> Optional[Dict[str, float]]:
        """Get the best parameters found so far."""
        return self._best_params

    @property
    def best_iteration(self) -> int:
        """Get the iteration where best was found."""
        return self._best_iteration

    def get_best_result(self) -> Dict[str, Any]:
        """
        Get the best result found so far.

        Returns:
            Dictionary with best score, params, and iteration
        """
        return {
            'score': self._best_score,
            'params': self._best_params,
            'iteration': self._best_iteration,
        }

    def get_iteration_history(self) -> pd.DataFrame:
        """
        Get iteration history as a DataFrame.

        Returns:
            DataFrame with all recorded iterations
        """
        if not self._iteration_history:
            return pd.DataFrame()

        return pd.DataFrame(self._iteration_history)

    # =========================================================================
    # Results persistence
    # =========================================================================

    def save_results(
        self,
        algorithm: str,
        metric_name: str = 'KGE',
        experiment_id: Optional[str] = None,
        standard_filename: bool = False
    ) -> Optional[Path]:
        """
        Save optimization results to a CSV file.

        Args:
            algorithm: Algorithm name (e.g., 'PSO', 'DDS')
            metric_name: Name of the optimization metric
            experiment_id: Optional experiment identifier
            standard_filename: If True, uses the standard SYMFLUENCE naming convention
                              ({experiment_id}_parallel_iteration_results.csv)

        Returns:
            Path to the saved results file
        """
        if experiment_id is None:
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='optimization', dict_key='EXPERIMENT_ID')

        # Create results dataframe
        df = self.get_iteration_history()

        if df.empty:
            self.logger.warning("No results to save")
            return None

        # Generate filename
        if standard_filename:
            filename = f"{experiment_id}_parallel_iteration_results.csv"
        else:
            filename = f"{experiment_id}_{algorithm.lower()}_results.csv"

        results_path = self.results_dir / filename

        # Save to CSV
        df.to_csv(results_path, index=False)

        self.logger.info(f"Saved optimization results to {results_path}")

        return results_path

    def save_best_params(
        self,
        algorithm: str,
        experiment_id: Optional[str] = None
    ) -> Optional[Path]:
        """
        Save best parameters to a JSON file.

        Args:
            algorithm: Algorithm name
            experiment_id: Optional experiment identifier

        Returns:
            Path to the saved parameters file
        """
        if experiment_id is None:
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='optimization', dict_key='EXPERIMENT_ID')

        if self._best_params is None:
            self.logger.warning("No best parameters to save")
            return None

        # Convert numpy types to JSON-serializable types
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types."""
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(val) for key, val in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # Determine the optimization metric used
        metric_name = self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE',
            dict_key='OPTIMIZATION_METRIC'
        )

        # Prepare output
        output = {
            'algorithm': algorithm,
            'experiment_id': experiment_id,
            'metric': metric_name,
            'best_score': float(self._best_score) if self._best_score is not None else None,
            'best_iteration': int(self._best_iteration) if self._best_iteration is not None else None,
            'best_params': convert_to_serializable(self._best_params),
            'timestamp': datetime.now().isoformat(),
        }

        # Generate filename (sanitize algorithm name for filesystem)
        safe_algorithm = algorithm.lower().replace('/', '_')
        filename = f"{experiment_id}_{safe_algorithm}_best_params.json"
        params_path = self.results_dir / filename

        # Save to JSON
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Saved best parameters to {params_path}")

        return params_path

    def load_results(self, results_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load results from a CSV file.

        Args:
            results_path: Path to results file

        Returns:
            DataFrame with loaded results
        """
        results_path = Path(results_path)

        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        return pd.read_csv(results_path)

    def load_best_params(self, params_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load best parameters from a JSON file.

        Args:
            params_path: Path to parameters file

        Returns:
            Dictionary with best parameters
        """
        params_path = Path(params_path)

        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

        with open(params_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # =========================================================================
    # Multi-objective support
    # =========================================================================

    def record_pareto_solution(
        self,
        objectives: List[float],
        params: Dict[str, float],
        dominated: bool = False
    ) -> None:
        """
        Record a solution on the Pareto front.

        Args:
            objectives: List of objective values
            params: Parameter values
            dominated: Whether this solution is dominated
        """
        record = {
            'objectives': objectives,
            'params': params,
            'dominated': dominated,
            'timestamp': datetime.now().isoformat(),
        }

        self._pareto_front.append(record)

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Get non-dominated solutions from the Pareto front.

        Returns:
            List of non-dominated solutions
        """
        return [s for s in self._pareto_front if not s.get('dominated', False)]

    def save_pareto_front(
        self,
        algorithm: str,
        experiment_id: Optional[str] = None
    ) -> Optional[Path]:
        """
        Save Pareto front to a CSV file.

        Args:
            algorithm: Algorithm name
            experiment_id: Optional experiment identifier

        Returns:
            Path to the saved Pareto front file
        """
        if experiment_id is None:
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='optimization', dict_key='EXPERIMENT_ID')

        pareto_solutions = self.get_pareto_front()

        if not pareto_solutions:
            self.logger.warning("No Pareto front to save")
            return None

        # Convert to DataFrame
        records = []
        for sol in pareto_solutions:
            record = {
                f'obj_{i}': obj for i, obj in enumerate(sol['objectives'])
            }
            record.update(sol['params'])
            records.append(record)

        df = pd.DataFrame(records)

        # Generate filename
        filename = f"{experiment_id}_{algorithm.lower()}_pareto_front.csv"
        pareto_path = self.results_dir / filename

        # Save
        df.to_csv(pareto_path, index=False)

        self.logger.info(f"Saved Pareto front to {pareto_path}")

        return pareto_path

    # =========================================================================
    # Timing
    # =========================================================================

    def start_timing(self) -> None:
        """Start timing the optimization run."""
        self._start_time = datetime.now()

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since start.

        Returns:
            Elapsed time in seconds
        """
        if self._start_time is None:
            return 0.0

        return (datetime.now() - self._start_time).total_seconds()

    def format_elapsed_time(self) -> str:
        """
        Get formatted elapsed time string.

        Returns:
            Formatted time string (e.g., "1h 23m 45s")
        """
        elapsed = self.get_elapsed_time()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
