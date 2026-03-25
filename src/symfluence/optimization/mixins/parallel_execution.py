# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Parallel execution infrastructure mixin for distributed model evaluation during optimization.

Provides extensible parallel processing framework supporting three execution paradigms: sequential
(single process), multiprocessing (ProcessPool on shared-memory systems), and distributed MPI
(HPC clusters). Coordinates process-specific directory management, configuration file updates,
task distribution, and environment variable setup.

Architecture:
    The ParallelExecutionMixin implements the Mixin Pattern combined with Facade Pattern,
    delegating to specialized helper classes while providing unified interface to optimizers:

    1. Execution Strategies (Strategy Pattern):
       - SequentialExecutionStrategy: Single-process execution (num_processes=1)
       - ProcessPoolExecutionStrategy: Python multiprocessing (shared-memory, num_processes>1 on single machine)
       - MPIExecutionStrategy: MPI-based distributed (HPC clusters with Slurm/PBS/LSF)
       - Automatic strategy selection based on MPI environment detection and config

    2. Helper Classes (Facade):
       - DirectoryManager: Creates/manages process-specific directories to avoid file conflicts
       - ConfigurationUpdater: Updates model config files (fileManager, mizRoute control) for each process
       - TaskDistributor: Assigns tasks to processes and updates task dictionaries with directory info
       - WorkerEnvironmentConfig: Sets environment variables (GDAL, Python paths) for worker processes

    3. Parallel Directory Scheme:
       Base structure: {base_dir}/parallel/{process_id}/
       Prevents concurrent writes to shared files by isolated directories per process
       Example for SUMMA model with 4 processes:
           {base_dir}/parallel/0/settings/ (Process 0)
           {base_dir}/parallel/1/settings/ (Process 1)
           {base_dir}/parallel/2/settings/ (Process 2)
           {base_dir}/parallel/3/settings/ (Process 3)

    4. Configuration Update Workflow:
       For each process:
           1. Copy base settings to process directory
           2. Update fileManager paths: settingsPath → process-specific dir
           3. Update outputPath, outFilePrefix for process isolation
           4. Update mizRoute control: <input_dir>, <output_dir> → process dirs
           5. Update simulation times to calibration period

    5. Task Distribution:
       Tasks distributed round-robin across processes:
           Task 0 → Process 0, Task 4 → Process 0 (if 4 processes)
           Task 1 → Process 1, Task 5 → Process 1
           etc.
       Each task updated with parallel_dirs[process_id] for worker access

Execution Modes:

    Sequential (num_processes=1 or single task):
        - Single Python process executes tasks serially
        - No parallelization overhead
        - Used for debugging, single-task runs
        - Execution: SequentialExecutionStrategy

    ProcessPool (num_processes>1, single machine):
        - Python multiprocessing.Pool for shared-memory parallelism
        - Efficient for multi-core systems (typical workstations/small servers)
        - Spawn worker processes for each task
        - Execution: ProcessPoolExecutionStrategy
        - Configuration: NUM_PROCESSES controls pool size

    MPI Distributed (num_processes>1 + HPC environment):
        - MPI-based execution across multiple compute nodes (HPC clusters)
        - Job submission via Slurm/PBS/LSF (handled upstream by job scheduler)
        - Detects MPI via environment variables: OMPI_COMM_WORLD_RANK, PMI_RANK
        - Master rank (0) distributes tasks, other ranks execute workers
        - Execution: MPIExecutionStrategy
        - Fallback to ProcessPool if MPI fails

Workflow Integration:

    1. Optimizer Setup Phase:
       optimizer.setup_parallel_processing(base_dir, model, exp_id)
       → Creates parallel/{0,1,2,...,N} directories

    2. Configuration Preparation:
       optimizer.copy_base_settings(settings_source, parallel_dirs, model)
       → Copies settings files to all process directories

    3. File Updates:
       optimizer.update_file_managers(parallel_dirs, model, exp_id)
       optimizer.update_mizuroute_controls(parallel_dirs, model, exp_id)
       → Updates paths in fileManager.txt, mizRoute.control for process isolation

    4. Task Distribution:
       optimizer.distribute_tasks(task_list, parallel_dirs)
       → Assigns tasks to processes, adds directory info to each task

    5. Batch Execution:
       optimizer.execute_batch(tasks, worker_func, max_workers)
       → Selects strategy (Sequential/ProcessPool/MPI) and executes tasks

    6. Cleanup:
       optimizer.cleanup_parallel_processing(parallel_dirs)
       → Optional cleanup of parallel directories after run

Key Features:

    - Automatic Strategy Selection: Detects MPI environment and selects optimal execution strategy
    - Graceful Fallback: MPI → ProcessPool → Sequential if execution fails
    - Process Isolation: Process-specific directories prevent concurrent file access conflicts
    - Configuration Management: Automatic path updates for model configs across processes
    - Error Handling: Comprehensive error reporting from all parallel processes
    - Environment Setup: Coordinates worker environment variables (GDAL, Python, etc.)
    - Lazy Initialization: Helper classes created on-demand (properties)
    - Backward Compatibility: Legacy _create_mpi_worker_script() method preserved

Configuration Parameters:
    NUM_PROCESSES: int (default 1)
        - Number of parallel processes to use
        - > 1 triggers ProcessPool or MPI execution
        - Typical values: 1 (sequential), 4, 8, 16, 32 (depends on machine cores)

Properties:
    num_processes: int - Configured number of processes (from NUM_PROCESSES)
    use_parallel: bool - True if num_processes > 1
    max_workers: int - min(num_processes, cpu_count()) - effective worker count
    is_mpi_run: bool - True if running under MPI environment

Required Mixin Attributes:
    self.config: Dict[str, Any] - Configuration object with NUM_PROCESSES, paths
    self.logger: logging.Logger - Logger instance
    self.project_dir: Path - Project directory (used by MPI strategy)

Example Workflows:

    # Sequential Execution (debugging)
    >>> config.NUM_PROCESSES = 1
    >>> result = optimizer.execute_batch(tasks, worker_func)
    # Single process executes tasks serially

    # Multiprocessing (local multi-core)
    >>> config.NUM_PROCESSES = 8
    >>> parallel_dirs = optimizer.setup_parallel_processing(base, 'SUMMA', 'exp1')
    >>> optimizer.copy_base_settings(settings_src, parallel_dirs, 'SUMMA')
    >>> optimizer.update_file_managers(parallel_dirs, 'SUMMA', 'exp1')
    >>> tasks = optimizer.distribute_tasks(all_tasks, parallel_dirs)
    >>> results = optimizer.execute_batch(tasks, worker_func)
    # ProcessPool creates 8 worker processes on local machine

    # MPI Execution (HPC cluster)
    >>> # Slurm job: srun -n 4 python calibrate.py (sets NUM_PROCESSES=4)
    >>> config.NUM_PROCESSES = 4
    >>> if optimizer.is_mpi_run:
    ...     # MPI environment detected
    ...     parallel_dirs = optimizer.setup_parallel_processing(base, 'SUMMA', 'exp1')
    ...     optimizer.copy_base_settings(settings_src, parallel_dirs, 'SUMMA')
    ...     optimizer.update_file_managers(parallel_dirs, 'SUMMA', 'exp1')
    ...     tasks = optimizer.distribute_tasks(all_tasks, parallel_dirs)
    ...     results = optimizer.execute_batch(tasks, worker_func)
    # MPI master distributes tasks to worker ranks

Error Handling:

    - MPI failure → Automatic fallback to ProcessPool
    - ProcessPool failure → Fallback to Sequential
    - Failed tasks return {'individual_id': ..., 'score': None, 'error': 'message'}
    - Comprehensive error logging with traceback

References:
    - Multiprocessing: https://docs.python.org/3/library/multiprocessing.html
    - MPI for Python: https://mpi4py.readthedocs.io/
    - Process Pool Pattern: https://en.wikipedia.org/wiki/Thread_pool
    - Mixin Pattern: Gang of Four design patterns
    - Strategy Pattern: Gang of Four design patterns

See Also:
    - DirectoryManager: Process directory creation and management
    - ConfigurationUpdater: Model config file updates for process isolation
    - TaskDistributor: Task-to-process assignment and metadata injection
    - WorkerEnvironmentConfig: Worker process environment variable setup
    - SequentialExecutionStrategy: Single-process execution implementation
    - ProcessPoolExecutionStrategy: Multiprocessing.Pool implementation
    - MPIExecutionStrategy: MPI-based distributed execution implementation
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from symfluence.core.mixins import ConfigMixin

from .parallel import (
    ConfigurationUpdater,
    DirectoryManager,
    MPIExecutionStrategy,
    PersistentMPIExecutionStrategy,
    ProcessPoolExecutionStrategy,
    SequentialExecutionStrategy,
    TaskDistributor,
    WorkerEnvironmentConfig,
)

logger = logging.getLogger(__name__)


class ParallelExecutionMixin(ConfigMixin):
    """Mixin providing unified parallel execution infrastructure for model optimizers.

    Orchestrates distributed model evaluations across multiple processes during calibration.
    Supports three execution paradigms seamlessly: sequential (debugging), multiprocessing
    (shared-memory systems), and MPI (HPC clusters). Manages process isolation through
    dedicated directories, automatic configuration file updates, and environment setup.

    This class implements the Mixin Pattern, designed to be inherited by optimizer classes
    that need parallel execution capabilities. Delegates to specialized strategy classes
    (SequentialExecutionStrategy, ProcessPoolExecutionStrategy, MPIExecutionStrategy) for
    actual execution, maintaining clean separation of concerns.

    Key Responsibilities:

        1. Strategy Selection & Execution:
           - Detects execution environment (MPI, single machine, single process)
           - Selects optimal execution strategy: Sequential → ProcessPool → MPI
           - Provides graceful fallback: MPI → ProcessPool → Sequential on failure
           - Exposes unified execute_batch() interface regardless of strategy

        2. Process Directory Management:
           - Creates process-specific directories to prevent file conflicts
           - Manages {base_dir}/parallel/{process_id}/ directory scheme
           - Copies base settings to all process directories
           - Provides cleanup of parallel directories after execution

        3. Configuration File Updates:
           - Updates SUMMA fileManager.txt paths for process isolation
           - Updates MizuRoute control file paths for routing
           - Synchronizes configuration across all process directories
           - Handles calibration period time settings

        4. Task Distribution:
           - Round-robin assignment of tasks to processes
           - Injects process-specific directory paths into task dictionaries
           - Maintains task order for result collection

        5. Worker Environment Setup:
           - Configures worker process environment variables
           - Handles GDAL configuration, Python paths, library paths
           - Applies environment settings across execution strategies

    Attributes:

        config (Dict[str, Any]): Configuration dictionary with NUM_PROCESSES and paths
        logger (logging.Logger): Logger instance for execution logging
        project_dir (Path): Project base directory (from class using this mixin)

        Lazy-initialized properties (created on-demand):
        - _directory_manager: DirectoryManager instance
        - _config_updater: ConfigurationUpdater instance
        - _task_distributor: TaskDistributor instance
        - _worker_env_config: WorkerEnvironmentConfig instance

    Properties:

        num_processes (int): Number of parallel processes from config.NUM_PROCESSES.
            Default 1 (sequential). > 1 enables parallel execution.

        use_parallel (bool): True if num_processes > 1. Controls strategy selection.

        max_workers (int): min(num_processes, cpu_count()).
            Prevents oversubscription on multi-core systems.

        is_mpi_run (bool): True if MPI environment detected
            (OMPI_COMM_WORLD_RANK or PMI_RANK in environment).

    Required Mixin Attributes:

        Classes using this mixin MUST provide:
        - self.config: Dict[str, Any] with 'NUM_PROCESSES' key
        - self.logger: logging.Logger instance
        - self.project_dir: Path to project directory

    Workflow Methods:

        setup_parallel_processing(base_dir, model_name, exp_id):
            Creates parallel directory structure for each process.
            Returns: Dict mapping process_id → directory paths

        copy_base_settings(source_settings, parallel_dirs, model_name):
            Copies model settings files to all process directories.

        update_file_managers(parallel_dirs, model_name, exp_id):
            Updates SUMMA fileManager.txt with process-specific paths.
            Handles settingsPath, outputPath, outFilePrefix updates.

        update_mizuroute_controls(parallel_dirs, model_name, exp_id):
            Updates MizuRoute control file with process-specific paths.
            Updates <input_dir>, <output_dir>, <ancil_dir>, <fname_qsim>.

        distribute_tasks(tasks, parallel_dirs):
            Assigns tasks to processes round-robin.
            Returns tasks with process_id and directory info added.

        execute_batch(tasks, worker_func, max_workers):
            Main execution method. Selects strategy and executes tasks.
            Returns: List of result dictionaries with scores and errors.

        execute_batch_ordered(tasks, worker_func, max_workers):
            Execution with guaranteed result ordering (uses ProcessPool).
            Returns: Results in same order as input tasks.

        cleanup_parallel_processing(parallel_dirs):
            Optional cleanup of parallel directories.

        setup_worker_environment() → Dict[str, str]:
            Returns environment variables for worker processes.

        apply_worker_environment():
            Applies environment variables to current process.

    Execution Strategies:

        Sequential (num_processes=1):
            - Single-process serial execution
            - No parallelization overhead
            - For debugging or single-task runs
            - Implementation: SequentialExecutionStrategy

        ProcessPool (num_processes > 1, single machine):
            - Python multiprocessing.Pool for shared-memory systems
            - Default strategy for workstations/small servers
            - Spawns worker processes for each task
            - Configurable pool size (max_workers)
            - Implementation: ProcessPoolExecutionStrategy

        MPI (num_processes > 1, HPC environment):
            - MPI-based distributed execution across multiple nodes
            - Master rank (0) distributes tasks to worker ranks
            - Requires MPI environment (Slurm/PBS/LSF job submission)
            - Auto-detected via OMPI_COMM_WORLD_RANK, PMI_RANK environment variables
            - Fallback to ProcessPool if MPI initialization fails
            - Implementation: MPIExecutionStrategy

    Example Usage:

        >>> class MyOptimizer(ParallelExecutionMixin):
        ...     def __init__(self, config, logger, project_dir):
        ...         from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        ...         self.logger = logger
        ...         self.project_dir = project_dir
        ...
        ...     def run_calibration(self):
        ...         # Setup parallel execution
        ...         parallel_dirs = self.setup_parallel_processing(
        ...             base_dir=self.project_dir / 'parallel',
        ...             model_name='SUMMA',
        ...             experiment_id='exp_001'
        ...         )
        ...
        ...         # Prepare model configs
        ...         self.copy_base_settings(settings_src, parallel_dirs, 'SUMMA')
        ...         self.update_file_managers(parallel_dirs, 'SUMMA', 'exp_001')
        ...
        ...         # Create tasks (parameter sets to evaluate)
        ...         tasks = [
        ...             {'param_id': i, 'params': params_set_i}
        ...             for i, params_set_i in enumerate(parameter_sets)
        ...         ]
        ...
        ...         # Distribute and execute
        ...         tasks = self.distribute_tasks(tasks, parallel_dirs)
        ...         results = self.execute_batch(tasks, self.evaluate_single_task)
        ...
        ...         # Cleanup
        ...         self.cleanup_parallel_processing(parallel_dirs)
        ...
        ...         return results

    MPI Deployment Example:

        >>> # Slurm job script: calibrate.slurm
        >>> #!/bin/bash
        >>> #SBATCH --nodes 2
        >>> #SBATCH --ntasks 16
        >>> #SBATCH --time 04:00:00
        >>> srun python calibrate.py --config config.yaml
        >>> # srun sets MPI environment variables detected by is_mpi_run

    Error Handling:

        - MPI Execution Failure: Automatically falls back to ProcessPool
        - ProcessPool Failure: Falls back to Sequential execution
        - Task Execution Failure: Individual task error captured, other tasks continue
        - Failed results: {'individual_id': ..., 'score': None, 'error': 'message'}
        - Comprehensive logging with traceback for debugging

    Performance Considerations:

        - Process Overhead: ProcessPool/MPI have startup overhead (~0.5-1 sec)
          Use for runs with many tasks or expensive worker functions
        - Optimal Task Count: min(num_tasks, num_processes * 4) for load balancing
        - Memory Usage: Each process duplicates worker memory (plan accordingly)
        - MPI Scaling: Efficient up to ~256 ranks on typical HPC systems
        - I/O Bottleneck: Ensure parallel_dirs on fast storage for good scaling

    Configuration:

        config.NUM_PROCESSES: int (default 1)
            - 1: Sequential execution
            - 2-8: ProcessPool on workstation
            - 8+: MPI on HPC cluster (with proper job submission)

    References:

        - Multiprocessing Documentation: https://docs.python.org/3/library/multiprocessing.html
        - mpi4py: https://mpi4py.readthedocs.io/
        - Slurm Job Scheduler: https://slurm.schedmd.com/
        - Process Pool Pattern: Gang of Four design patterns
        - Mixin Pattern: Gang of Four design patterns
        - Strategy Pattern: Gang of Four design patterns

    See Also:

        - DirectoryManager: Low-level parallel directory management
        - ConfigurationUpdater: Model configuration file updates
        - TaskDistributor: Task-to-process distribution logic
        - BaseModelOptimizer: Example of class using this mixin
        - SequentialExecutionStrategy: Single-process execution implementation
        - ProcessPoolExecutionStrategy: Multiprocessing implementation
        - MPIExecutionStrategy: MPI-based execution implementation
    """

    # =========================================================================
    # Lazy initialization of helper classes
    # =========================================================================

    @property
    def _directory_manager(self) -> DirectoryManager:
        """Get or create directory manager."""
        if not hasattr(self, '__directory_manager'):
            self.__directory_manager = DirectoryManager(self.logger)
        return self.__directory_manager

    @property
    def _config_updater(self) -> ConfigurationUpdater:
        """Get or create configuration updater."""
        if not hasattr(self, '__config_updater'):
            self.__config_updater = ConfigurationUpdater(self.config, self.logger)
        return self.__config_updater

    @property
    def _task_distributor(self) -> TaskDistributor:
        """Get or create task distributor."""
        if not hasattr(self, '__task_distributor'):
            self.__task_distributor = TaskDistributor(self.num_processes)
        return self.__task_distributor

    @property
    def _worker_env_config(self) -> WorkerEnvironmentConfig:
        """Get or create worker environment config."""
        if not hasattr(self, '__worker_env_config'):
            self.__worker_env_config = WorkerEnvironmentConfig()
        return self.__worker_env_config

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_processes(self) -> int:
        """Get number of processes to use for parallel execution."""
        return max(1, self._get_config_value(lambda: self.config.system.num_processes, default=1, dict_key='NUM_PROCESSES'))

    @property
    def use_parallel(self) -> bool:
        """Check if parallel execution is enabled."""
        return self.num_processes > 1

    @property
    def max_workers(self) -> int:
        """Get maximum number of worker processes."""
        return min(self.num_processes, mp.cpu_count())

    @property
    def is_mpi_run(self) -> bool:
        """Check if running under MPI."""
        return "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ

    # =========================================================================
    # Directory setup (delegates to DirectoryManager)
    # =========================================================================

    def setup_parallel_processing(
        self,
        base_dir: Path,
        model_name: str,
        experiment_id: str
    ) -> Dict[int, Dict[str, Path]]:
        """
        Setup parallel processing directories for each process.

        Creates process-specific directories to avoid file conflicts during
        parallel model evaluations.

        Args:
            base_dir: Base directory for parallel processing
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping process IDs to their directory paths
        """
        return self._directory_manager.setup_parallel_directories(
            base_dir, model_name, experiment_id, self.num_processes
        )

    def copy_base_settings(
        self,
        source_settings_dir: Path,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str
    ) -> None:
        """
        Copy base settings to each parallel process directory.

        Args:
            source_settings_dir: Source settings directory
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model
        """
        self._directory_manager.copy_base_settings(
            source_settings_dir, parallel_dirs, model_name
        )

    def cleanup_parallel_processing(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]]
    ) -> None:
        """
        Cleanup parallel processing directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
        """
        self._directory_manager.cleanup(parallel_dirs)

    # =========================================================================
    # Configuration updates (delegates to ConfigurationUpdater)
    # =========================================================================

    def update_file_managers(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str,
        experiment_id: str,
        file_manager_name: str = 'fileManager.txt'
    ) -> None:
        """
        Update file manager paths in process-specific directories.

        Updates settingsPath, outputPath, outFilePrefix, and simulation times
        to point to process-specific directories and use calibration period.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            file_manager_name: Name of the file manager file (default: 'fileManager.txt')
        """
        self._config_updater.update_file_managers(
            parallel_dirs, model_name, experiment_id, file_manager_name
        )

    def update_mizuroute_controls(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str,
        experiment_id: str,
        control_file_name: str = 'mizuroute.control'
    ) -> None:
        """
        Update mizuRoute control file paths in process-specific directories.

        Updates <input_dir>, <output_dir>, <ancil_dir>, <case_name>, and <fname_qsim>
        to point to process-specific directories instead of global directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            control_file_name: Name of the control file (default: 'mizuroute.control')
        """
        self._config_updater.update_mizuroute_controls(
            parallel_dirs, model_name, experiment_id, control_file_name
        )

    # =========================================================================
    # Task distribution (delegates to TaskDistributor)
    # =========================================================================

    def distribute_tasks(
        self,
        tasks: List[Dict[str, Any]],
        parallel_dirs: Optional[Dict[int, Dict[str, Path]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Distribute tasks across processes.

        Assigns each task to a process and updates the task with
        process-specific directory paths.

        Args:
            tasks: List of task dictionaries
            parallel_dirs: Optional process-specific directories

        Returns:
            List of tasks with process assignments
        """
        return self._task_distributor.distribute(tasks, parallel_dirs)

    # =========================================================================
    # Batch execution (uses execution strategies)
    # =========================================================================

    # =========================================================================
    # Persistent MPI strategy lifecycle
    # =========================================================================

    def _get_or_create_persistent_mpi_strategy(
        self, worker_func: Callable, max_workers: int
    ) -> PersistentMPIExecutionStrategy:
        """Return the persistent MPI strategy, starting it on first call."""
        if not hasattr(self, '_ParallelExecutionMixin__persistent_mpi'):
            self.__persistent_mpi = None

        if self.__persistent_mpi is not None and self.__persistent_mpi.is_alive:
            return self.__persistent_mpi

        # Previous strategy died or doesn't exist — (re)create
        if self.__persistent_mpi is not None:
            self.logger.warning("Persistent MPI workers died — restarting")
            self.__persistent_mpi.shutdown()

        strategy = PersistentMPIExecutionStrategy(
            self.project_dir, self.num_processes, self.logger,
        )
        strategy.startup(worker_func=worker_func, max_workers=max_workers)
        self.__persistent_mpi = strategy
        return strategy

    def _shutdown_mpi_strategy(self) -> None:
        """Shut down the persistent MPI strategy if it is running."""
        if hasattr(self, '_ParallelExecutionMixin__persistent_mpi') and self.__persistent_mpi is not None:
            self.__persistent_mpi.shutdown()
            self.__persistent_mpi = None

    def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks using MPI if parallel, otherwise sequentially.

        When running under MPI with multiple processes, uses a persistent
        worker pool that stays alive across batches to avoid repeated Python
        import storms on shared filesystems (Lustre IOPS mitigation).

        Falls back through: PersistentMPI -> spawn-per-batch MPI ->
        ProcessPool -> error results.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results from task execution
        """
        if max_workers is None:
            max_workers = self.max_workers

        if self.use_parallel and len(tasks) > 1:
            # --- Try persistent MPI first (no repeated Python imports) ---
            try:
                strategy = self._get_or_create_persistent_mpi_strategy(
                    worker_func, max_workers,
                )
                return strategy.execute(tasks, worker_func, max_workers)
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"Persistent MPI execution failed: {e}")
                self._shutdown_mpi_strategy()

            # --- Fallback: spawn-per-batch MPI ---
            try:
                self.logger.info("Falling back to spawn-per-batch MPI...")
                strategy = MPIExecutionStrategy(
                    self.project_dir, self.num_processes, self.logger
                )
                return strategy.execute(tasks, worker_func, max_workers)
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"MPI batch execution failed: {e}")

            # --- Fallback: ProcessPool ---
            try:
                self.logger.info("Falling back to ProcessPool execution...")
                strategy = ProcessPoolExecutionStrategy(self.logger)
                return strategy.execute(tasks, worker_func, max_workers)
            except Exception as e2:  # noqa: BLE001 — must-not-raise contract
                self.logger.error(f"ProcessPool fallback also failed: {e2}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Return empty results with errors for all tasks
                return [
                    {
                        'individual_id': task.get('individual_id', i),
                        'score': None,
                        'error': str(e2)
                    }
                    for i, task in enumerate(tasks)
                ]
        else:
            # Sequential execution for a single process or single task
            strategy = SequentialExecutionStrategy(self.logger)
            return strategy.execute(tasks, worker_func, max_workers)

    def execute_batch_ordered(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks and return results in the same order as input.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results in the same order as input tasks
        """
        if max_workers is None:
            max_workers = self.max_workers

        strategy = ProcessPoolExecutionStrategy(self.logger)
        return strategy.execute(tasks, worker_func, max_workers)

    # =========================================================================
    # Environment setup (delegates to WorkerEnvironmentConfig)
    # =========================================================================

    def setup_worker_environment(self) -> Dict[str, str]:
        """
        Setup environment variables for worker processes.

        Returns:
            Dictionary of environment variables to set
        """
        return self._worker_env_config.get_environment()

    def apply_worker_environment(self) -> None:
        """Apply worker environment variables to current process."""
        self._worker_env_config.apply_to_current_process()

    # =========================================================================
    # Legacy method for backward compatibility
    # =========================================================================

    def _create_mpi_worker_script(
        self,
        script_path: Path,
        tasks_file: Path,
        results_file: Path,
        worker_module: str,
        worker_function: str
    ) -> None:
        """
        Create the MPI worker script file.

        Note: This method is kept for backward compatibility.
        New code should use MPIExecutionStrategy directly.
        """
        strategy = MPIExecutionStrategy(
            self.project_dir, self.num_processes, self.logger
        )
        strategy._create_worker_script(
            script_path, tasks_file, results_file, worker_module, worker_function
        )
