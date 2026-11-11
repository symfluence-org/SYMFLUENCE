# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MPI Execution Strategy

Executes tasks using MPI (mpirun) for distributed computing.
"""
from __future__ import annotations

import logging
import os

# Security rationale: Used for trusted internal MPI task serialization
import pickle  # nosec B403
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..worker_environment import WorkerEnvironmentConfig
from .base import ExecutionStrategy

#: How often the coordinator wakes while waiting on an MPI batch.
_WAIT_POLL_SECONDS = 15.0


def symfluence_import_root() -> Path:
    """Directory to put on ``sys.path`` so ``import symfluence`` resolves here.

    Derived from the imported package rather than by counting ``.parent``s
    from this file: the count was short, so MPI workers got
    ``src/symfluence/core`` on their path — exposing ``config``, ``metrics``,
    ``logging_utils`` ... as top-level modules that shadow unrelated packages.
    """
    import symfluence

    return Path(symfluence.__file__).resolve().parent.parent


def batch_deadline_seconds(
    n_tasks: int, n_workers: int, model_timeout_s: Optional[float] = None
) -> float:
    """Wall-clock bound for one MPI batch.

    At least 7200 s and 300 s per task (the historical rule); when the model
    timeout is known, also enough for every task queued on a rank to use its
    full budget.
    """
    deadline = float(max(7200, n_tasks * 300))
    if model_timeout_s:
        waves = -(-max(1, n_tasks) // max(1, n_workers))  # ceil
        deadline = max(deadline, waves * float(model_timeout_s) + 300.0)
    return deadline


class MPIExecutionStrategy(ExecutionStrategy):
    """
    MPI execution strategy.

    Uses mpirun to distribute tasks across multiple processes.
    Generates a temporary worker script and communicates via pickle files.
    """

    def __init__(
        self,
        project_dir: Path,
        num_processes: int,
        logger: logging.Logger = None
    ):
        """
        Initialize MPI execution strategy.

        Args:
            project_dir: Project directory for temporary files
            num_processes: Number of MPI processes to use
            logger: Optional logger instance
        """
        self.project_dir = project_dir
        self.num_processes = num_processes
        self.logger = logger or logging.getLogger(__name__)
        self.worker_env = WorkerEnvironmentConfig()

    @property
    def name(self) -> str:
        """Strategy identifier for logging and selection."""
        return "mpi"

    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks using MPI.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results from task execution

        Raises:
            RuntimeError: If MPI execution fails
        """
        work_dir = self.project_dir / "temp_mpi"
        work_dir.mkdir(exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]
        tasks_file = work_dir / f'mpi_tasks_{unique_id}.pkl'
        results_file = work_dir / f'mpi_results_{unique_id}.pkl'
        worker_script = work_dir / f'mpi_worker_{unique_id}.py'

        # Get worker module and function name
        worker_module, worker_function = self._get_worker_info(worker_func)

        cleanup_files = True
        try:
            self.logger.debug(
                f"MPI batch: {len(tasks)} tasks, worker={worker_module}.{worker_function}"
            )

            # Save tasks to file
            with open(tasks_file, 'wb') as f:
                pickle.dump(tasks, f)

            # Create worker script
            self._create_worker_script(
                worker_script, tasks_file, results_file,
                worker_module, worker_function
            )
            if os.name != 'nt':
                worker_script.chmod(0o755)

            # Determine number of processes
            num_procs = min(max_workers, self.num_processes, len(tasks))

            # Find Python executable
            python_exe = self._find_python_executable()

            # Setup environment
            mpi_env = self._build_mpi_environment()

            # Try available launchers in preference order, falling back on failure
            launchers = self._detect_mpi_launchers()
            last_error = None

            for launcher in launchers:
                mpi_cmd = self._build_mpi_command(
                    python_exe, num_procs, worker_script, tasks_file, results_file,
                    launcher=launcher,
                )

                self.logger.debug(f"MPI command: {' '.join(mpi_cmd)}")

                result = self._run_mpi_batch(mpi_cmd, mpi_env, tasks, num_procs)
                self._log_mpi_output(result)

                if result.returncode == 0 and results_file.exists():
                    with open(results_file, 'rb') as f:
                        # Security rationale: Loading trusted MPI results
                        results = pickle.load(f)  # nosec B301
                    self.logger.debug(f"MPI completed: {len(results)} results")
                    return results

                # Record failure and try next launcher
                last_error = (
                    f"MPI launcher '{launcher}' failed (rc={result.returncode})"
                )
                self.logger.warning(
                    f"{last_error}, stderr: {(result.stderr or '')[:200]}"
                )
                if len(launchers) > 1:
                    self.logger.info(
                        "Falling back to next available launcher..."
                    )
                # Clean stale results before retry
                if results_file.exists():
                    results_file.unlink()

            # All launchers failed
            cleanup_files = False
            raise RuntimeError(
                f"All MPI launchers failed. Last error: {last_error}"
            )

        finally:
            if cleanup_files:
                self._cleanup_files([tasks_file, results_file, worker_script])

    def _get_worker_info(self, worker_func: Callable) -> tuple:
        """Get module and function name from callable."""
        if hasattr(worker_func, '__module__'):
            return worker_func.__module__, worker_func.__name__
        return (
            "symfluence.optimization.workers.summa_parallel_workers",
            "_evaluate_parameters_worker_safe"
        )

    def _run_mpi_batch(
        self,
        mpi_cmd: List[str],
        mpi_env: Dict[str, str],
        tasks: List[Dict[str, Any]],
        num_procs: int,
    ) -> subprocess.CompletedProcess:
        """Run one MPI batch to completion, keeping the eval watchdog informed.

        The launcher runs in its own process group, registered with the
        watchdog, so neither a batch deadline nor a watchdog kill leaves ranks
        behind. The wait loop beats the watchdog: a batch legitimately takes
        many evaluations' worth of time, and its own deadline bounds a hang.
        """
        from symfluence.core.calibration.eval_watchdog import (
            get_watchdog,
            model_timeout_from_tasks,
        )
        from symfluence.core.process_exec import kill_process_group

        model_timeout = model_timeout_from_tasks(tasks)
        watchdog = get_watchdog()
        watchdog.ensure_covers(model_timeout)
        deadline_s = batch_deadline_seconds(len(tasks), num_procs, model_timeout)

        group_kwargs: Dict[str, Any] = {} if os.name == 'nt' else {'start_new_session': True}
        proc = subprocess.Popen(
            mpi_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=mpi_env,
            **group_kwargs,
        )
        watchdog.register_child_group(proc.pid)
        start = time.monotonic()
        try:
            while True:
                watchdog.beat()
                try:
                    stdout, stderr = proc.communicate(timeout=_WAIT_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - start > deadline_s:
                        kill_process_group(proc.pid)
                        proc.kill()
                        stdout, stderr = proc.communicate()
                        return subprocess.CompletedProcess(
                            mpi_cmd, -9, stdout,
                            (stderr or '') + f"\nMPI batch exceeded its {deadline_s:.0f}s deadline and was killed",
                        )
        except BaseException:
            # Interrupted (KeyboardInterrupt, ...): the launcher is in its own
            # session, so nothing else will stop it.
            kill_process_group(proc.pid)
            proc.kill()
            raise
        finally:
            watchdog.unregister_child_group(proc.pid)
            watchdog.beat()
        return subprocess.CompletedProcess(mpi_cmd, proc.returncode, stdout, stderr)

    def _find_python_executable(self) -> str:
        """Python for the MPI workers: this interpreter, unless overridden.

        Workers must import the same packages as the coordinator, so the
        default is ``sys.executable``. ``$SYMFLUENCE_PYTHON`` is the explicit
        override. (This used to prefer any ``~/venv/bin/python`` that happened
        to exist over the interpreter actually running the calibration.)
        """
        explicit = os.environ.get('SYMFLUENCE_PYTHON')
        if explicit and os.access(explicit, os.X_OK):
            self.logger.info(f"Using SYMFLUENCE_PYTHON override: {explicit}")
            return explicit
        return sys.executable

    @staticmethod
    def _detect_mpi_launchers() -> list:
        """Return available MPI launchers in preference order.

        mpirun is preferred because it handles PMI negotiation internally,
        while srun requires matching PMI/PMIx versions between SLURM and
        OpenMPI — a common source of failures on HPC systems (e.g., Anvil).
        """
        available = [c for c in ("mpirun", "mpiexec", "srun") if shutil.which(c)]
        if not available:
            raise RuntimeError(
                "No MPI launcher found. Install OpenMPI, MS-MPI, or run under SLURM."
            )
        return available

    def _build_mpi_command(
        self,
        python_exe: str,
        num_processes: int,
        worker_script: Path,
        tasks_file: Path,
        results_file: Path,
        launcher: str = None,
    ) -> list:
        """Build the MPI launch command with platform-aware launcher detection."""
        if launcher is None:
            launcher = self._detect_mpi_launchers()[0]
        self.logger.debug(f"Using MPI launcher: {launcher}")

        cmd = [launcher]

        # Only OpenMPI's mpirun supports -x for env forwarding;
        # srun and mpiexec (MS-MPI) do not.
        if launcher == "mpirun":
            cmd.extend([
                '-x', 'OMP_NUM_THREADS',
                '-x', 'HDF5_USE_FILE_LOCKING',
                '-x', 'MKL_NUM_THREADS',
            ])

        cmd.extend([
            '-n', str(num_processes),
            python_exe,
            str(worker_script),
            str(tasks_file),
            str(results_file)
        ])
        return cmd

    def _build_mpi_environment(self) -> Dict[str, str]:
        """Build environment for MPI execution."""
        mpi_env = os.environ.copy()

        # Add the directory that contains the symfluence package to PYTHONPATH
        src_path = str(symfluence_import_root())
        current_pythonpath = mpi_env.get('PYTHONPATH', '')
        if current_pythonpath:
            mpi_env['PYTHONPATH'] = f"{src_path}{os.pathsep}{current_pythonpath}"
        else:
            mpi_env['PYTHONPATH'] = src_path

        # Add worker environment variables
        mpi_env.update(self.worker_env.get_environment())

        # OpenMPI settings
        if 'OMPI_MCA_' not in mpi_env:
            mpi_env['OMPI_MCA_pls_rsh_agent'] = 'ssh'

        self.logger.debug(f"MPI environment - PYTHONPATH: {mpi_env.get('PYTHONPATH')}")

        return mpi_env

    def _log_mpi_output(self, result: subprocess.CompletedProcess) -> None:
        """Log MPI execution output."""
        self.logger.debug(f"MPI returncode: {result.returncode}")

        if result.stdout:
            self.logger.debug(f"MPI stdout: {result.stdout[:1000]}")
        if result.stderr:
            self.logger.debug(f"MPI stderr: {result.stderr[:1000]}")

        if result.returncode != 0:
            self.logger.error(f"MPI execution failed (returncode={result.returncode})")
            self.logger.error(
                f"MPI stdout: {result.stdout[:2000] if result.stdout else 'empty'}"
            )
            self.logger.error(
                f"MPI stderr: {result.stderr[:2000] if result.stderr else 'empty'}"
            )

    def _cleanup_files(self, files: List[Path]) -> None:
        """Clean up temporary files."""
        for file_path in files:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

    def _create_worker_script(
        self,
        script_path: Path,
        tasks_file: Path,
        results_file: Path,
        worker_module: str,
        worker_function: str
    ) -> None:
        """Create the MPI worker script file."""
        # Directory that contains the symfluence package
        src_path = symfluence_import_root()

        script_content = f'''#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path
from mpi4py import MPI
import logging

# Add symfluence src to path
sys.path.insert(0, r"{str(src_path)}")

# Worker logging via the shared protocol helpers ([P##] rank-tagged handler;
# rebound to the actual rank inside main()).
from symfluence.core.logging_utils import get_worker_logger, silence_third_party

logger = get_worker_logger(0, level=logging.INFO)
silence_third_party()

try:
    from {worker_module} import {worker_function}
except ImportError as e:
    logger.error(f"Failed to import worker function: {{e}}")
    logger.error(f"sys.path = {{sys.path}}")
    sys.exit(1)

def main():
    """MPI worker main function."""
    global logger
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = get_worker_logger(rank, level=logging.INFO)

    tasks_file = Path(sys.argv[1])
    results_file = Path(sys.argv[2])

    if rank == 0:
        # Master process - load all tasks
        try:
            with open(tasks_file, 'rb') as f:
                all_tasks = pickle.load(f)
        except (ValueError, RuntimeError, IOError) as e:
            logger.error(f"Master failed to load tasks: {{e}}")
            all_tasks = []

        logger.info(f"Rank 0: Loaded {{len(all_tasks)}} tasks")

        # Distribute tasks by proc_id to avoid race conditions
        from collections import defaultdict
        tasks_by_proc = defaultdict(list)
        for task in all_tasks:
            proc_id = task.get('proc_id', 0)
            assigned_rank = proc_id % size
            tasks_by_proc[assigned_rank].append(task)

        logger.info(f"Rank 0: Distributed tasks by proc_id - {{{{r: len(tasks_by_proc[r]) for r in range(size)}}}}")
        all_results = []

        for worker_rank in range(size):
            worker_tasks = tasks_by_proc[worker_rank]

            if worker_rank == 0:
                my_tasks = worker_tasks
                logger.info(f"Rank 0: Processing {{len(my_tasks)}} tasks locally")
            else:
                logger.info(f"Rank 0: Sending {{len(worker_tasks)}} tasks to rank {{worker_rank}}")
                comm.send(worker_tasks, dest=worker_rank, tag=1)

        # Process rank 0 tasks
        for i, task in enumerate(my_tasks):
            try:
                worker_result = {worker_function}(task)
                all_results.append(worker_result)
            except (ValueError, RuntimeError, IOError) as e:
                logger.error(f"Rank 0: Task {{i}} failed: {{e}}")
                error_result = {{
                    'individual_id': task.get('individual_id', -1),
                    'params': task.get('params', {{}}),
                    'score': None,
                    'error': f'Rank 0 error: {{str(e)}}'
                }}
                all_results.append(error_result)

        # Collect results from workers
        for worker_rank in range(1, size):
            try:
                logger.info(f"Rank 0: Waiting for results from rank {{worker_rank}}")
                worker_results = comm.recv(source=worker_rank, tag=2)
                logger.info(f"Rank 0: Received {{len(worker_results)}} results from rank {{worker_rank}}")
                all_results.extend(worker_results)
            except (ValueError, RuntimeError, IOError) as e:
                logger.error(f"Error receiving from worker {{worker_rank}}: {{e}}")

        # Save results
        logger.info(f"Rank 0: Saving {{len(all_results)}} results to {{results_file}}")
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"Rank 0: Results saved successfully")

    else:
        # Worker process
        logger.info(f"Rank {{rank}}: Waiting for tasks from rank 0")
        try:
            my_tasks = comm.recv(source=0, tag=1)
            logger.info(f"Rank {{rank}}: Received {{len(my_tasks)}} tasks")

            my_results = []

            for i, task in enumerate(my_tasks):
                logger.info(f"Rank {{rank}}: Processing task {{i+1}}/{{len(my_tasks)}}")
                try:
                    worker_result = {worker_function}(task)
                    my_results.append(worker_result)
                except (ValueError, RuntimeError, IOError) as e:
                    logger.error(f"Rank {{rank}}: Task {{i}} failed: {{e}}")
                    error_result = {{
                        'individual_id': task.get('individual_id', -1),
                        'params': task.get('params', {{}}),
                        'score': None,
                        'error': f'Rank {{rank}} error: {{str(e)}}'
                    }}
                    my_results.append(error_result)

            logger.info(f"Rank {{rank}}: Sending {{len(my_results)}} results back to rank 0")
            comm.send(my_results, dest=0, tag=2)
            logger.info(f"Rank {{rank}}: Results sent successfully")

        except (ValueError, RuntimeError, IOError) as e:
            logger.error(f"Worker {{rank}} failed: {{e}}")

if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
