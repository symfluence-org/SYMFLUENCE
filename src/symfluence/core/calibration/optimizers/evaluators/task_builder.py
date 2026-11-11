# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Task Builder for Optimization Evaluations

Constructs task dictionaries for parallel worker execution.
Eliminates duplicated task dict construction across evaluation methods.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    pass


def _ensure_flat_config(config: Any) -> Dict[str, Any]:
    """
    Ensure config is a flat dictionary for worker compatibility.

    Workers expect flat dictionaries with uppercase keys and .get() method.
    SymfluenceConfig objects need to be flattened for multiprocessing serialization.

    Args:
        config: Either a SymfluenceConfig instance or a flat dict

    Returns:
        Flat dictionary with uppercase keys
    """
    # Check if it's a SymfluenceConfig by checking for model_dump (Pydantic method)
    if hasattr(config, 'model_dump'):
        from symfluence.core.config.flattening import flatten_nested_config
        return flatten_nested_config(config)
    # Already a dict
    return config


class TaskBuilder(ConfigMixin):
    """
    Builds task dictionaries for parallel model evaluations.

    Centralizes task dict construction that was previously duplicated in:
    - _evaluate_population
    - _evaluate_population_objectives
    - run_de (inline)
    """

    DEFAULT_PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        optimization_settings_dir: Path,
        default_sim_dir: Path,
        parallel_dirs: Dict[int, Dict[str, Any]],
        num_processes: int,
        target_metric: str,
        param_manager: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize task builder.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name
            optimization_settings_dir: Settings directory path
            default_sim_dir: Default simulation directory
            parallel_dirs: Parallel processing directories by process ID
            num_processes: Number of parallel processes
            target_metric: Target optimization metric
            param_manager: Parameter manager instance
            logger: Optional logger instance
        """
        # Store typed config for _get_config_value typed accessors
        self._typed_config = config if hasattr(config, 'model_dump') else None
        # Flatten config if it's a SymfluenceConfig to ensure workers get a plain dict
        self.config = _ensure_flat_config(config)
        self.project_dir = project_dir
        self.domain_name = domain_name
        self.optimization_settings_dir = optimization_settings_dir
        self.default_sim_dir = default_sim_dir
        self.parallel_dirs = parallel_dirs
        self.num_processes = num_processes
        self.target_metric = target_metric
        self.param_manager = param_manager
        self.logger = logger or logging.getLogger(__name__)

        # Cache paths that are frequently used
        self._summa_exe_path = None

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Get config value using typed accessor with dict fallback.

        Args:
            key: Flat config key (e.g., 'ITERATIVE_OPTIMIZATION_ALGORITHM')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    def set_summa_exe_path(self, path: Path) -> None:
        """Set SUMMA executable path for task construction."""
        self._summa_exe_path = path

    def build_task(
        self,
        individual_id: int,
        params: Dict[str, float],
        proc_id: int,
        evaluation_id: str,
        multiobjective: bool = False,
        objective_names: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build a single task dictionary.

        Args:
            individual_id: Individual index in population
            params: Denormalized parameters
            proc_id: Process ID for parallel execution
            evaluation_id: Unique evaluation identifier
            multiobjective: Whether this is multi-objective evaluation
            objective_names: List of objective metric names
            random_seed: Optional random seed

        Returns:
            Task dictionary for worker execution
        """
        # A process id outside the configured range silently selected the
        # non-parallel default directories (or, inside the range, another
        # process's). Algorithms used to reach this with an iteration counter.
        n_proc = max(1, int(self.num_processes))
        is_int = isinstance(proc_id, (int, np.integer)) and not isinstance(proc_id, bool)
        if not is_int or not 0 <= proc_id < n_proc:
            raise ValueError(
                f"Invalid proc_id {proc_id!r} for an evaluation task: expected an "
                f"integer in [0, {n_proc}) (NUM_PROCESSES={self.num_processes})."
            )
        dirs = self.parallel_dirs.get(proc_id, {})
        settings_dir = Path(dirs.get('settings_dir', self.optimization_settings_dir))
        sim_dir = dirs.get('sim_dir', self.default_sim_dir)
        root_dir = dirs.get('root', self.project_dir)

        # All models use sim_dir for output (e.g., process_0/simulations/run_1/HYPE)
        # HYPE's resultdir in info.txt is set by the worker to match this path
        output_dir = sim_dir

        # Build mizuRoute paths
        if dirs and sim_dir:
            mizuroute_dir = str(Path(sim_dir).parent / 'mizuRoute')
        else:
            mizuroute_dir = str(Path(self.default_sim_dir).parent / 'mizuRoute')

        mizuroute_settings_dir = str(root_dir / 'settings' / 'mizuRoute') if dirs else ''

        # Get file manager path
        file_manager_name = self._cfg('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        file_manager = str(settings_dir / file_manager_name)

        # Get original depths if available
        original_depths = None
        if hasattr(self.param_manager, 'original_depths') and self.param_manager.original_depths is not None:
            original_depths = self.param_manager.original_depths.tolist()

        # Determine target metric
        target_metric = self.target_metric
        if multiobjective:
            assert objective_names is not None
            target_metric = objective_names[0]

        task_dict = {
            'individual_id': individual_id,
            'params': params,
            'proc_id': proc_id,
            'evaluation_id': evaluation_id,
            'config': self.config,
            'target_metric': target_metric,
            'calibration_variable': self._cfg('OPTIMIZATION_TARGET', 'streamflow'),
            'domain_name': self.domain_name,
            'project_dir': str(self.project_dir),
            'proc_settings_dir': str(settings_dir),
            'proc_output_dir': str(output_dir),
            'proc_sim_dir': str(sim_dir),
            'proc_forcing_dir': str(dirs.get('forcing_dir', '')),
            'summa_settings_dir': str(settings_dir),
            'mizuroute_settings_dir': mizuroute_settings_dir,
            'summa_dir': str(sim_dir),
            'mizuroute_dir': mizuroute_dir,
            'file_manager': file_manager,
            'summa_exe': str(self._summa_exe_path) if self._summa_exe_path else '',
            'original_depths': original_depths,
        }

        # Add multi-objective fields if needed
        if multiobjective:
            task_dict['multiobjective'] = True
            task_dict['objective_names'] = objective_names

            # Add target types and metrics for multi-target optimization
            # Determine which algorithm is active to prioritize its keys.
            # The flattened config includes Pydantic defaults for ALL algorithms
            # (e.g., NSGA2_SECONDARY_TARGET='gw_depth' even in a MOEA/D run),
            # so we must check the active algorithm's keys first.
            algorithm = str(self._cfg('ITERATIVE_OPTIMIZATION_ALGORITHM', '')).upper()
            is_moead = algorithm in ('MOEAD', 'MOEA-D', 'MOEA_D')
            is_nsga2 = algorithm in ('NSGA2', 'NSGA-II', 'NSGA_II')

            # Fallback target/metric from general config
            fallback_target = self._cfg('OPTIMIZATION_TARGET', 'streamflow')
            fallback_metric = self._cfg('OPTIMIZATION_METRIC', 'KGE')

            # multi_target_mode = two calibration *targets* (the active
            # algorithm's *_MULTI_TARGET flag). Without it the worker scores
            # the objective metrics on the single OPTIMIZATION_TARGET
            # (multi-metric). Callers that are neither NSGA-II nor MOEA/D keep
            # the historical multi-target behaviour.
            if is_nsga2:
                task_dict['multi_target_mode'] = bool(self._cfg('NSGA2_MULTI_TARGET', False))
            elif is_moead:
                task_dict['multi_target_mode'] = bool(self._cfg('MOEAD_MULTI_TARGET', False))
            else:
                task_dict['multi_target_mode'] = True

            if is_moead:
                task_dict['primary_target_type'] = self._cfg(
                    'MOEAD_PRIMARY_TARGET', fallback_target)
                task_dict['secondary_target_type'] = self._cfg(
                    'MOEAD_SECONDARY_TARGET',
                    self._cfg('OPTIMIZATION_TARGET2', 'tws'))
                task_dict['primary_metric'] = self._cfg(
                    'MOEAD_PRIMARY_METRIC', fallback_metric)
                task_dict['secondary_metric'] = self._cfg(
                    'MOEAD_SECONDARY_METRIC',
                    self._cfg('OPTIMIZATION_METRIC2', 'KGE'))
            elif is_nsga2:
                task_dict['primary_target_type'] = self._cfg(
                    'NSGA2_PRIMARY_TARGET', fallback_target)
                task_dict['secondary_target_type'] = self._cfg(
                    'NSGA2_SECONDARY_TARGET',
                    self._cfg('OPTIMIZATION_TARGET2', 'tws'))
                task_dict['primary_metric'] = self._cfg(
                    'NSGA2_PRIMARY_METRIC', fallback_metric)
                task_dict['secondary_metric'] = self._cfg(
                    'NSGA2_SECONDARY_METRIC',
                    self._cfg('OPTIMIZATION_METRIC2', 'KGE'))
            else:
                # Generic fallback: try MOEAD then NSGA2 then legacy keys
                task_dict['primary_target_type'] = self._cfg(
                    'MOEAD_PRIMARY_TARGET',
                    self._cfg('NSGA2_PRIMARY_TARGET', fallback_target))
                task_dict['secondary_target_type'] = self._cfg(
                    'MOEAD_SECONDARY_TARGET',
                    self._cfg('NSGA2_SECONDARY_TARGET',
                        self._cfg('OPTIMIZATION_TARGET2', 'tws')))
                task_dict['primary_metric'] = self._cfg(
                    'MOEAD_PRIMARY_METRIC',
                    self._cfg('NSGA2_PRIMARY_METRIC', fallback_metric))
                task_dict['secondary_metric'] = self._cfg(
                    'MOEAD_SECONDARY_METRIC',
                    self._cfg('NSGA2_SECONDARY_METRIC',
                        self._cfg('OPTIMIZATION_METRIC2', 'KGE')))

        # Add random seed if provided
        if random_seed is not None:
            task_dict['random_seed'] = random_seed

        return task_dict

    def _next_task_seed(self, base_random_seed: Optional[Any]) -> Optional[int]:
        """Worker seed for the next task of this run, or None when unseeded.

        Workers (SUMMA) reseed their process-global RNGs from this value. It
        used to be ``base + individual_index + 1000``, identical in every
        generation, so each worker replayed the same random stream for the
        whole run -- and a worker running in-process (pool fallback) reset the
        optimizer process's global stream to the same state each generation.
        A running per-run task counter makes every evaluation's seed distinct
        while staying reproducible for a given RANDOM_SEED; the first batch
        keeps the old values.
        """
        if base_random_seed in (None, 'None', ''):
            return None
        counter = getattr(self, '_task_seed_counter', 0)
        self._task_seed_counter = counter + 1
        return int(base_random_seed) + 1000 + counter

    def build_population_tasks(
        self,
        population: np.ndarray,
        iteration: int = 0,
        multiobjective: bool = False,
        objective_names: Optional[List[str]] = None,
        base_random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build task dictionaries for a population of solutions.

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            iteration: Current iteration number
            multiobjective: Whether this is multi-objective evaluation
            objective_names: List of objective metric names
            base_random_seed: Base random seed (individual seeds derived from this)

        Returns:
            List of task dictionaries
        """
        tasks = []

        for i, params_normalized in enumerate(population):
            params = self.param_manager.denormalize_parameters(params_normalized)
            proc_id = i % self.num_processes

            # Per-evaluation worker seed (distinct across generations)
            random_seed = self._next_task_seed(base_random_seed)

            task = self.build_task(
                individual_id=i,
                params=params,
                proc_id=proc_id,
                evaluation_id=f"pop_eval_{i:03d}",
                multiobjective=multiobjective,
                objective_names=objective_names,
                random_seed=random_seed
            )
            tasks.append(task)

        return tasks

    def build_trial_tasks(
        self,
        trials: List[np.ndarray],
        trial_indices: List[int],
        iteration: int,
        base_random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build task dictionaries for trial solutions (e.g., in DE).

        Args:
            trials: List of trial parameter arrays (normalized)
            trial_indices: Corresponding indices in population
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            List of task dictionaries
        """
        tasks = []

        for idx, trial in zip(trial_indices, trials):
            params = self.param_manager.denormalize_parameters(trial)
            proc_id = idx % self.num_processes

            random_seed = self._next_task_seed(base_random_seed)

            task = self.build_task(
                individual_id=idx,
                params=params,
                proc_id=proc_id,
                evaluation_id=f"trial_{iteration:03d}_{idx:03d}",
                random_seed=random_seed
            )
            tasks.append(task)

        return tasks
