# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Persistent MPI Execution Strategy

Launches MPI worker processes once and keeps them alive across multiple
execute() calls, communicating via pickle files in a local temp directory
with file-based signaling.  This eliminates the repeated Python-import
metadata storms that cause Lustre IOPS spikes when fresh processes are
spawned every batch, while avoiding the fragility of piping data through
mpirun's stdin/stdout relay.
"""
from __future__ import annotations

import atexit
import logging
import os

# Security rationale: Used for trusted internal MPI task serialization
import pickle  # nosec B403
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..worker_environment import WorkerEnvironmentConfig
from .base import ExecutionStrategy
from .mpi import batch_deadline_seconds, symfluence_import_root

#: How often the coordinator's wait loop beats the eval watchdog.
_BEAT_INTERVAL_SECONDS = 15.0

#: Signal used for the unblockable second-stage kill of the worker group.
#:
#: ``signal.SIGKILL`` is POSIX-only: on Windows the attribute does not exist, so
#: merely *evaluating* ``signal.SIGKILL`` at a call site raises AttributeError —
#: before any ``os.name`` guard downstream can help. Windows' equivalent of an
#: unblockable kill is TerminateProcess, which ``Popen.kill()`` issues, so the
#: signal value is irrelevant there and SIGTERM stands in as the sentinel.
#:
#: Escalation level is therefore carried by ``_signal_group(hard=True)``, NOT by
#: comparing against this constant: on Windows it equals SIGTERM, so an equality
#: test would treat the graceful first stage as a hard kill and collapse the two
#: stages into one.
HARD_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


class PersistentMPIExecutionStrategy(ExecutionStrategy):
    """Persistent MPI execution strategy.

    Workers are launched once during ``startup()`` and kept alive for all
    subsequent ``execute()`` calls.  Communication between the coordinator
    and the MPI broker (rank 0) uses pickle files in a local temp directory
    with atomic file-based signaling:

    1. Coordinator writes ``tasks.pkl`` then creates ``tasks.ready``
    2. Broker polls for ``tasks.ready``, reads ``tasks.pkl``, distributes
       tasks to worker ranks via MPI, gathers results
    3. Broker writes ``results.pkl`` then creates ``results.ready``
    4. Coordinator polls for ``results.ready``, reads ``results.pkl``

    The temp directory uses ``$SLURM_TMPDIR`` (node-local ``/tmp``) when
    available, so no Lustre traffic is generated for task/result exchange.
    """

    def __init__(
        self,
        project_dir: Path,
        num_processes: int,
        logger: logging.Logger = None,
    ):
        self.project_dir = project_dir
        self.num_processes = num_processes
        self.logger = logger or logging.getLogger(__name__)
        self.worker_env = WorkerEnvironmentConfig()

        # Populated by startup()
        self._process: Optional[subprocess.Popen] = None
        self._worker_script: Optional[Path] = None
        self._comm_dir: Optional[Path] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # Last-resort reaper, registered while the pool is alive. Without it an
        # interpreter that exits without calling shutdown() leaves the ranks
        # running: they are re-parented to init and keep burning a core each.
        self._atexit_hook: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # ExecutionStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mpi_persistent"

    @property
    def is_persistent(self) -> bool:
        return True

    def startup(
        self,
        worker_func: Callable = None,
        max_workers: int = None,
    ) -> None:
        """Launch the persistent MPI worker pool."""
        if self._process is not None:
            self.logger.warning("startup() called but workers are already running")
            return

        if worker_func is None:
            raise ValueError("worker_func is required for startup()")

        num_procs = max_workers or self.num_processes
        worker_module, worker_function = self._get_worker_info(worker_func)

        # Create communication directory on local storage
        self._comm_dir = self._create_comm_dir()
        self.logger.info(f"Persistent MPI comm dir: {self._comm_dir}")

        # Create worker script on local scratch when available to avoid
        # NFS reads when MPI ranks load the script at launch.
        slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
        if slurm_tmpdir:
            work_dir = Path(slurm_tmpdir) / "symfluence_mpi_scripts"
        else:
            work_dir = self.project_dir / "temp_mpi"
        work_dir.mkdir(parents=True, exist_ok=True)

        uid = uuid.uuid4().hex[:8]
        self._worker_script = work_dir / f"mpi_persistent_worker_{uid}.py"
        self._create_persistent_worker_script(
            self._worker_script, worker_module, worker_function,
            str(self._comm_dir),
        )
        if os.name != 'nt':
            self._worker_script.chmod(0o755)

        python_exe = self._find_python_executable()
        mpi_env = self._build_mpi_environment()
        launchers = self._detect_mpi_launchers()

        last_error = None
        for launcher in launchers:
            cmd = self._build_launch_command(
                launcher, python_exe, num_procs, self._worker_script,
            )
            self.logger.info(
                f"Starting persistent MPI workers ({num_procs} ranks) "
                f"via {launcher}"
            )
            self.logger.debug(f"Command: {' '.join(str(c) for c in cmd)}")

            try:
                # Own process group: signalling the launcher alone does not
                # reliably reach the ranks it spawned, so a terminate() could
                # return cleanly while the workers kept running (orphaned to
                # init). With a new session we can signal the whole group.
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=mpi_env,
                    **self._process_group_kwargs(),
                )
                # Drain stderr in a background thread
                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr, daemon=True,
                )
                self._stderr_thread.start()

                # Verify the process started by waiting briefly
                try:
                    rc = self._process.wait(timeout=2)
                    stderr_snippet = ""
                    if self._process.stderr:
                        try:
                            stderr_snippet = self._process.stderr.read(2000).decode(
                                errors="replace"
                            )
                        except (OSError, ValueError):
                            pass
                    last_error = (
                        f"MPI process exited immediately (rc={rc}): "
                        f"{stderr_snippet[:500]}"
                    )
                    self.logger.warning(last_error)
                    self._process = None
                    continue
                except subprocess.TimeoutExpired:
                    pass  # Good — still running

                self._register_atexit_reaper()
                # os._exit (the eval watchdog) skips atexit; let the watchdog
                # kill this group itself if it ever fires.
                self._watchdog_register(self._process.pid)
                self.logger.info("Persistent MPI workers started successfully")
                return

            except FileNotFoundError:
                last_error = f"Launcher '{launcher}' not found"
                self.logger.warning(last_error)
                self._process = None
                continue

        raise RuntimeError(
            f"Failed to start persistent MPI workers. Last error: {last_error}"
        )

    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int,
    ) -> List[Dict[str, Any]]:
        """Send a batch of tasks to the persistent workers and return results."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(
                "Persistent MPI workers are not running. "
                "Call startup() first or the process has died."
            )

        comm_dir = self._comm_dir

        # Clean any stale signal files
        for f in ('tasks.ready', 'results.ready', 'results.pkl'):
            p = comm_dir / f
            if p.exists():
                p.unlink()

        # Write tasks to local temp directory
        tasks_file = comm_dir / 'tasks.pkl'
        with open(tasks_file, 'wb') as f:
            pickle.dump(tasks, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Signal the broker that tasks are ready
        (comm_dir / 'tasks.ready').write_text(str(len(tasks)))
        self.logger.debug(f"Sent {len(tasks)} tasks to persistent workers")

        # Wait for results (poll for results.ready)
        results_signal = comm_dir / 'results.ready'
        results_file = comm_dir / 'results.pkl'
        poll_interval = 0.5  # seconds
        # Generous deadline: every task queued on a rank may use the model's
        # full configured timeout. Rank 0 is the broker, so N-1 ranks work.
        from symfluence.core.calibration.eval_watchdog import (
            get_watchdog,
            model_timeout_from_tasks,
        )
        model_timeout = model_timeout_from_tasks(tasks)
        watchdog = get_watchdog()
        watchdog.ensure_covers(model_timeout)
        n_workers = max(1, min(max_workers or self.num_processes, self.num_processes) - 1)
        timeout = batch_deadline_seconds(len(tasks), n_workers, model_timeout)
        start = time.monotonic()
        last_beat = start

        while not results_signal.exists():
            now = time.monotonic()
            if now - last_beat >= _BEAT_INTERVAL_SECONDS:
                # The coordinator is alive and waiting; this loop's own
                # deadline bounds a hung batch. Without the beat every batch
                # longer than the watchdog limit killed the run (exit 75).
                watchdog.beat()
                last_beat = now
            if now - start > timeout:
                # Kill the ranks: the batch is abandoned, and they would keep
                # writing into the per-process directories.
                self._signal_group(HARD_KILL_SIGNAL, hard=True)
                self._handle_dead_process()
                raise RuntimeError(
                    f"Timed out waiting for MPI worker results "
                    f"after {timeout:.0f}s"
                )
            if self._process.poll() is not None:
                self._handle_dead_process()
                raise RuntimeError("Persistent MPI workers died during execution")
            time.sleep(poll_interval)
        watchdog.beat()

        # Read results
        with open(results_file, 'rb') as f:
            # Security rationale: Trusted internal data
            results = pickle.load(f)  # nosec B301

        # Clean up for next batch
        for f in ('tasks.pkl', 'tasks.ready', 'results.pkl', 'results.ready'):
            p = comm_dir / f
            if p.exists():
                p.unlink()

        self.logger.debug(
            f"Persistent MPI batch complete: {len(results)} results "
            f"for {len(tasks)} tasks"
        )
        return results

    def shutdown(self) -> None:
        """Terminate the persistent MPI worker pool gracefully."""
        # Drop the reaper first: _handle_dead_process() clears _process when the
        # pool dies on its own, and returning early below would otherwise leave
        # the hook registered against a pid this strategy no longer owns.
        self._unregister_atexit_reaper()

        if self._process is None:
            return

        self.logger.info("Shutting down persistent MPI workers...")

        try:
            if self._process.poll() is None:
                # Preferred: poison file, so ranks finish the current task and
                # exit on their own.
                if self._comm_dir:
                    try:
                        (self._comm_dir / 'shutdown').write_text('stop')
                    except OSError as exc:
                        self.logger.warning(f"Could not write shutdown signal: {exc}")
                try:
                    self._process.wait(timeout=30)
                    self.logger.info("MPI workers shut down gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(
                        "MPI workers did not exit in 30 s, terminating process group"
                    )
                    self._signal_group(signal.SIGTERM)
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning("Still alive after SIGTERM, killing")
                        self._signal_group(HARD_KILL_SIGNAL, hard=True)
                        try:
                            self._process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.logger.error(
                                "MPI worker group survived SIGKILL; ranks may "
                                "still be running"
                            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.logger.error(f"Error during MPI shutdown: {exc}")
        finally:
            self._watchdog_unregister(self._process.pid)
            self._process = None

        # Clean up temp files
        if self._worker_script and self._worker_script.exists():
            try:
                self._worker_script.unlink()
            except OSError:
                pass
            self._worker_script = None

        if self._comm_dir and self._comm_dir.exists():
            try:
                shutil.rmtree(self._comm_dir, ignore_errors=True)
            except OSError:
                pass
            self._comm_dir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_group_kwargs() -> Dict[str, Any]:
        """Popen kwargs that put the launcher in its own process group."""
        if os.name == 'nt':
            flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
            return {'creationflags': flags} if flags else {}
        return {'start_new_session': True}

    def _signal_group(self, sig: int, *, hard: bool = False) -> None:
        """Signal the whole worker process group, falling back to the launcher.

        The ranks are children of the launcher, not of this process, so
        signalling ``self._process`` alone can leave them running. ``start_new
        _session=True`` at launch makes the launcher a group leader, so its pid
        doubles as the group id.
        """
        if self._process is None:
            return
        pid = self._process.pid
        if os.name != 'nt':
            try:
                os.killpg(os.getpgid(pid), sig)
                return
            except (ProcessLookupError, PermissionError, OSError) as exc:
                self.logger.debug(f"killpg({pid}, {sig}) failed: {exc}; signalling pid")
        try:
            # Windows has no process-group signalling and no SIGKILL; the stage
            # is selected by the caller rather than inferred from `sig`, which
            # is SIGTERM for both stages here.
            if hard:
                self._process.kill()
            else:
                self._process.terminate()
        except (OSError, ValueError) as exc:
            self.logger.debug(f"Signalling pid {pid} failed: {exc}")

    def _register_atexit_reaper(self) -> None:
        """Reap the pool if the interpreter exits before shutdown() runs."""
        if self._atexit_hook is not None:
            return

        def _reap() -> None:
            proc = self._process
            if proc is None or proc.poll() is not None:
                return
            self._signal_group(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_group(HARD_KILL_SIGNAL, hard=True)

        self._atexit_hook = _reap
        atexit.register(_reap)

    def _unregister_atexit_reaper(self) -> None:
        if self._atexit_hook is not None:
            atexit.unregister(self._atexit_hook)
            self._atexit_hook = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _handle_dead_process(self) -> None:
        rc = self._process.poll() if self._process else None
        self.logger.error(f"Persistent MPI process died (returncode={rc})")
        if self._process is not None:
            self._watchdog_unregister(self._process.pid)
        self._process = None

    @staticmethod
    def _watchdog_register(pid: Any) -> None:
        from symfluence.core.calibration.eval_watchdog import get_watchdog
        if isinstance(pid, int):
            get_watchdog().register_child_group(pid)

    @staticmethod
    def _watchdog_unregister(pid: Any) -> None:
        from symfluence.core.calibration.eval_watchdog import get_watchdog
        if isinstance(pid, int):
            get_watchdog().unregister_child_group(pid)

    def _drain_stderr(self) -> None:
        try:
            for line in self._process.stderr:
                if isinstance(line, bytes):
                    line = line.decode(errors='replace')
                line = line.rstrip('\n')
                if line:
                    self.logger.info(f"[MPI worker] {line}")
        except (ValueError, OSError):
            pass

    @staticmethod
    def _create_comm_dir() -> Path:
        """Create the communication directory on node-local storage."""
        # Prefer SLURM_TMPDIR (node-local /tmp), fall back to system temp
        base = os.environ.get('SLURM_TMPDIR') or tempfile.gettempdir()
        comm_dir = Path(base) / f"symfluence_mpi_comm_{uuid.uuid4().hex[:8]}"
        comm_dir.mkdir(parents=True, exist_ok=True)
        return comm_dir

    @staticmethod
    def _get_worker_info(worker_func: Callable) -> tuple:
        if hasattr(worker_func, '__module__'):
            return worker_func.__module__, worker_func.__name__
        return (
            "symfluence.optimization.workers.summa_parallel_workers",
            "_evaluate_parameters_worker_safe",
        )

    def _find_python_executable(self) -> str:
        """Find the best Python executable, preferring node-local storage.

        Search order (first match wins):
        1. $SYMFLUENCE_PYTHON — explicit override for HPC job scripts
        2. Venv in $SLURM_TMPDIR — avoids NFS metadata IOPS
        3. sys.executable — the interpreter running the calibration
        """
        venv_bin = "Scripts" if os.name == "nt" else "bin"
        slurm_tmpdir = os.environ.get('SLURM_TMPDIR')

        # 1. Explicit override
        explicit = os.environ.get('SYMFLUENCE_PYTHON')
        if explicit and os.access(explicit, os.X_OK):
            self.logger.info(f"Using SYMFLUENCE_PYTHON override: {explicit}")
            return explicit

        # 2. Venv copied to node-local scratch (eliminates NFS metadata storms)
        if slurm_tmpdir:
            for name in ("python3", "python"):
                scratch_py = Path(slurm_tmpdir) / "venv" / venv_bin / name
                if scratch_py.exists() and os.access(str(scratch_py), os.X_OK):
                    self.logger.info(
                        f"Using node-local scratch Python: {scratch_py} "
                        f"(avoids NFS metadata IOPS)"
                    )
                    return str(scratch_py)

        # 3. This interpreter. Workers must import the same packages as the
        # coordinator. (This used to fall through to any ``venv`` next to the
        # package or ``~/venv/bin/python`` that happened to exist, in
        # preference to the interpreter actually running the calibration.)
        python_exe = sys.executable

        # Warn if final choice is on shared storage in a SLURM environment
        if slurm_tmpdir and not self._is_local_path(python_exe, slurm_tmpdir):
            self.logger.warning(
                f"Python executable ({python_exe}) is on shared storage. "
                f"This causes NFS metadata IOPS spikes from module discovery. "
                f"Copy venv to $SLURM_TMPDIR/venv or set SYMFLUENCE_PYTHON "
                f"to a node-local path."
            )

        return python_exe

    @staticmethod
    def _is_local_path(path: str, slurm_tmpdir: Optional[str] = None) -> bool:
        """Check whether a path is on node-local storage."""
        # Security rationale: path check, not file creation
        local_prefixes = ["/tmp"]  # nosec B108
        if slurm_tmpdir:
            local_prefixes.append(slurm_tmpdir)
        return any(path.startswith(p) for p in local_prefixes)

    @staticmethod
    def _detect_mpi_launchers() -> list:
        available = [
            c for c in ("mpirun", "mpiexec", "srun") if shutil.which(c)
        ]
        if not available:
            raise RuntimeError(
                "No MPI launcher found. "
                "Install OpenMPI, MS-MPI, or run under SLURM."
            )
        return available

    def _build_launch_command(
        self,
        launcher: str,
        python_exe: str,
        num_processes: int,
        worker_script: Path,
    ) -> list:
        # Rank 0 is a dedicated broker; ranks 1..N-1 are workers.
        # We use the requested num_processes (not +1) to stay within
        # SLURM slot limits.  This gives N-1 task workers.
        cmd = [launcher]
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
        ])
        return cmd

    def _build_mpi_environment(self) -> Dict[str, str]:
        mpi_env = os.environ.copy()
        src_path = str(symfluence_import_root())
        current_pythonpath = mpi_env.get('PYTHONPATH', '')
        if current_pythonpath:
            mpi_env['PYTHONPATH'] = f"{src_path}{os.pathsep}{current_pythonpath}"
        else:
            mpi_env['PYTHONPATH'] = src_path
        mpi_env.update(self.worker_env.get_environment())
        if 'OMPI_MCA_' not in mpi_env:
            mpi_env['OMPI_MCA_pls_rsh_agent'] = 'ssh'
        return mpi_env

    # ------------------------------------------------------------------
    # Worker script generation
    # ------------------------------------------------------------------

    def _create_persistent_worker_script(
        self,
        script_path: Path,
        worker_module: str,
        worker_function: str,
        comm_dir: str,
    ) -> None:
        """Generate the long-lived MPI worker script.

        The generated script:
        - Imports all modules once at startup.
        - Rank 0 acts as a broker: polls for task files, fans out
          to worker ranks via MPI, gathers results, writes result files.
        - Ranks 1..N-1 loop waiting for tasks from rank 0.
        - A ``shutdown`` file signals graceful exit.
        """
        src_path = symfluence_import_root()

        script_content = f'''#!/usr/bin/env python3
# Auto-generated persistent MPI worker script — do not edit.
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pickle
import sys
import time
import logging
from pathlib import Path

# ------------------------------------------------------------------
# Logging (stderr only)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger("mpi_persistent_worker")

_rank = "?"

# ------------------------------------------------------------------
# Path setup & imports (happens ONCE)
# ------------------------------------------------------------------
sys.path.insert(0, r"{str(src_path)}")

from symfluence.core.logging_utils import silence_third_party
silence_third_party()

from mpi4py import MPI

try:
    from {worker_module} import {worker_function} as _worker_func
except ImportError as exc:
    logger.error("Failed to import worker function: %s", exc)
    logger.error("sys.path = %s", sys.path)
    MPI.COMM_WORLD.Abort(1)

COMM_DIR = Path(r"{comm_dir}")


# ------------------------------------------------------------------
# Main loops
# ------------------------------------------------------------------

def _log(msg, *args, level=logging.INFO):
    """Log with rank prefix."""
    logger.log(level, f"[Rank {{_rank}}] {{msg}}", *args)


def broker_loop(comm, rank, size):
    """Rank-0 broker: poll for task files, distribute via MPI, write results.

    Rank 0 is a **dedicated** broker — it does not execute tasks itself.
    This avoids a deadlock where worker ranks block on comm.send while
    rank 0 is still busy running its own task and hasn't posted the
    matching comm.recv yet.
    """
    tasks_signal = COMM_DIR / "tasks.ready"
    shutdown_signal = COMM_DIR / "shutdown"
    tasks_file = COMM_DIR / "tasks.pkl"
    results_file = COMM_DIR / "results.pkl"
    results_signal = COMM_DIR / "results.ready"
    num_workers = size - 1  # ranks 1..N-1

    _log("Broker ready (dedicated, %d workers), polling %s", num_workers, COMM_DIR)

    while True:
        # Poll for tasks or shutdown
        while not tasks_signal.exists():
            if shutdown_signal.exists():
                _log("Shutdown signal received")
                _broadcast_shutdown(comm, size)
                return
            time.sleep(0.1)

        # Read tasks
        with open(tasks_file, "rb") as f:
            tasks = pickle.load(f)

        _log("Received batch of %d tasks", len(tasks))

        # Distribute tasks to worker ranks only (1..N-1)
        tasks_per_rank = _distribute_tasks(tasks, num_workers)

        for i, dest in enumerate(range(1, size)):
            comm.send(tasks_per_rank[i], dest=dest, tag=1)

        _log("Tasks distributed, waiting for results...")

        # Gather results from all workers (no own tasks to block on)
        all_results = []
        for src in range(1, size):
            worker_results = comm.recv(source=src, tag=2)
            all_results.extend(worker_results)

        # Write results and signal
        with open(results_file, "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Remove tasks signal before creating results signal
        if tasks_signal.exists():
            tasks_signal.unlink()

        results_signal.write_text(str(len(all_results)))

        _log("Sent %d results back to coordinator", len(all_results))


def worker_loop(comm, rank):
    """Ranks 1..N-1: receive tasks, execute, return results."""
    while True:
        tasks = comm.recv(source=0, tag=1)

        if tasks is None:
            _log("Received shutdown sentinel")
            return

        results = _run_tasks(tasks, rank)
        comm.send(results, dest=0, tag=2)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _distribute_tasks(tasks, num_workers):
    """Round-robin tasks to worker ranks, respecting proc_id when present."""
    buckets = [[] for _ in range(num_workers)]
    for task in tasks:
        dest = task.get("proc_id", 0) % num_workers
        buckets[dest].append(task)
    return buckets


def _run_tasks(tasks, rank):
    """Execute a list of tasks and return results (never raises)."""
    results = []
    for i, task in enumerate(tasks):
        ind_id = task.get("individual_id", "?")
        proc_id = task.get("proc_id", "?")
        summa_dir = task.get("summa_dir", "?")
        _log("Starting task %d (ind=%s, proc=%s, summa_dir=%s)",
             i, ind_id, proc_id, summa_dir)
        try:
            result = _worker_func(task)
            score = result.get("score") if isinstance(result, dict) else "?"
            _log("Completed task %d (ind=%s, score=%s)", i, ind_id, score)
            results.append(result)
        except Exception as exc:
            _log("Task %d failed: %s", i, exc, level=logging.ERROR)
            results.append({{
                "individual_id": task.get("individual_id", -1),
                "params": task.get("params", {{}}),
                "score": None,
                "error": f"Rank {{rank}} error: {{exc}}",
            }})
    return results


def _broadcast_shutdown(comm, size):
    """Tell all worker ranks to exit."""
    for dest in range(1, size):
        comm.send(None, dest=dest, tag=1)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    global _rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    _rank = rank

    _log("Worker started (size=%d)", size)

    try:
        if rank == 0:
            broker_loop(comm, rank, size)
        else:
            worker_loop(comm, rank)
    except Exception as exc:
        _log("Fatal error: %s", exc, level=logging.ERROR)
        import traceback
        traceback.print_exc(file=sys.stderr)
        comm.Abort(1)
    finally:
        _log("Exiting")


if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
