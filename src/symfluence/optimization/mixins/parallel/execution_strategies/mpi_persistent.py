# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Persistent MPI Execution Strategy

Launches MPI worker processes once and keeps them alive across multiple
execute() calls, communicating via length-prefixed pickle over stdin/stdout
pipes.  This eliminates the repeated Python-import metadata storms that
cause Lustre IOPS spikes when fresh processes are spawned every batch.
"""

import logging
import os
import pickle  # nosec B403 - Used for trusted internal MPI task serialization
import shutil
import struct
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..worker_environment import WorkerEnvironmentConfig
from .base import ExecutionStrategy

# ---------------------------------------------------------------------------
# Framing helpers (shared between coordinator and worker script)
# ---------------------------------------------------------------------------

def _write_frame(stream, data: bytes) -> None:
    """Write a length-prefixed frame to a binary stream."""
    stream.write(struct.pack('>Q', len(data)))
    stream.write(data)
    stream.flush()


def _read_frame(stream) -> bytes:
    """Read a length-prefixed frame from a binary stream."""
    header = _read_exact(stream, 8)
    if not header:
        raise EOFError("Connection closed while reading frame header")
    length = struct.unpack('>Q', header)[0]
    return _read_exact(stream, length)


def _read_exact(stream, n: int) -> bytes:
    """Read exactly *n* bytes from *stream*."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return b''
            raise EOFError(f"Expected {n} bytes, got {len(buf)}")
        buf.extend(chunk)
    return bytes(buf)


class PersistentMPIExecutionStrategy(ExecutionStrategy):
    """Persistent MPI execution strategy.

    Workers are launched once during ``startup()`` and kept alive for all
    subsequent ``execute()`` calls.  Communication happens over pipes,
    not the shared filesystem, so only the initial Python import touches
    Lustre — and that happens exactly once.
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
        self._stderr_thread: Optional[threading.Thread] = None

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
        """Launch the persistent MPI worker pool.

        Args:
            worker_func: The worker callable (used to resolve module/function).
            max_workers: Number of MPI ranks to launch.
        """
        if self._process is not None:
            self.logger.warning("startup() called but workers are already running")
            return

        if worker_func is None:
            raise ValueError("worker_func is required for startup()")

        num_procs = max_workers or self.num_processes
        worker_module, worker_function = self._get_worker_info(worker_func)

        work_dir = self.project_dir / "temp_mpi"
        work_dir.mkdir(exist_ok=True)

        uid = uuid.uuid4().hex[:8]
        self._worker_script = work_dir / f"mpi_persistent_worker_{uid}.py"
        self._create_persistent_worker_script(
            self._worker_script, worker_module, worker_function,
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
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=mpi_env,
                )
                # Drain stderr in a background thread to prevent deadlocks
                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr, daemon=True,
                )
                self._stderr_thread.start()

                # Verify the process started by waiting briefly
                try:
                    rc = self._process.wait(timeout=2)
                    # If it exited within 2 s, startup failed
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
                    # Good — still running after 2 s
                    pass

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

        # Serialize tasks and send via stdin pipe
        payload = pickle.dumps(tasks, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            _write_frame(self._process.stdin, payload)
        except (BrokenPipeError, OSError) as exc:
            self._handle_dead_process()
            raise RuntimeError(f"Failed to send tasks to MPI workers: {exc}") from exc

        # Read results from stdout pipe
        try:
            result_bytes = _read_frame(self._process.stdout)
        except (EOFError, OSError) as exc:
            self._handle_dead_process()
            raise RuntimeError(
                f"Failed to read results from MPI workers: {exc}"
            ) from exc

        results = pickle.loads(result_bytes)  # nosec B301 - Trusted internal data
        self.logger.debug(
            f"Persistent MPI batch complete: {len(results)} results "
            f"for {len(tasks)} tasks"
        )
        return results

    def shutdown(self) -> None:
        """Terminate the persistent MPI worker pool gracefully."""
        if self._process is None:
            return

        self.logger.info("Shutting down persistent MPI workers...")

        try:
            if self._process.poll() is None:
                # Send shutdown sentinel (empty payload)
                shutdown_payload = pickle.dumps(
                    None, protocol=pickle.HIGHEST_PROTOCOL,
                )
                try:
                    _write_frame(self._process.stdin, shutdown_payload)
                except (BrokenPipeError, OSError):
                    pass  # Process may have already exited

                try:
                    self._process.wait(timeout=30)
                    self.logger.info("MPI workers shut down gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(
                        "MPI workers did not exit in 30 s, terminating"
                    )
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.logger.error(f"Error during MPI shutdown: {exc}")
        finally:
            self._process = None

        # Clean up the worker script
        if self._worker_script and self._worker_script.exists():
            try:
                self._worker_script.unlink()
            except OSError:
                pass
            self._worker_script = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        """Check if the persistent MPI process is still running."""
        return self._process is not None and self._process.poll() is None

    def _handle_dead_process(self) -> None:
        """Log and clean up after a dead MPI process."""
        rc = self._process.poll() if self._process else None
        self.logger.error(
            f"Persistent MPI process died (returncode={rc})"
        )
        self._process = None

    def _drain_stderr(self) -> None:
        """Read stderr in background to prevent pipe deadlock."""
        try:
            for line in self._process.stderr:
                if isinstance(line, bytes):
                    line = line.decode(errors='replace')
                line = line.rstrip('\n')
                if line:
                    self.logger.debug(f"[MPI worker] {line}")
        except (ValueError, OSError):
            pass  # Stream closed

    # ------------------------------------------------------------------
    # Reused from MPIExecutionStrategy (kept DRY via delegation where
    # possible, duplicated where signatures diverge)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_worker_info(worker_func: Callable) -> tuple:
        if hasattr(worker_func, '__module__'):
            return worker_func.__module__, worker_func.__name__
        return (
            "symfluence.optimization.workers.summa_parallel_workers",
            "_evaluate_parameters_worker_safe",
        )

    def _find_python_executable(self) -> str:
        python_exe = sys.executable
        venv_bin = "Scripts" if os.name == "nt" else "bin"
        pkg_root = Path(__file__).parent.parent.parent.parent.parent.parent
        venv_paths = [
            pkg_root / "venv" / venv_bin / "python",
            pkg_root / "venv" / venv_bin / "python3",
            pkg_root / "venv" / venv_bin / "python3.11",
            Path.home() / "venv" / venv_bin / "python",
            Path.home() / "venv" / venv_bin / "python3",
        ]
        for venv_path in venv_paths:
            if venv_path.exists():
                python_exe = str(venv_path)
                self.logger.info(f"Using venv Python: {python_exe}")
                break
        return python_exe

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
        src_path = str(Path(__file__).parent.parent.parent.parent.parent)
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
    ) -> None:
        """Generate the long-lived MPI worker script.

        The generated script:
        - Imports all modules once at startup.
        - Rank 0 acts as a broker: reads task batches from stdin, fans out
          to worker ranks via MPI, gathers results, writes them to stdout.
        - Ranks 1..N-1 loop waiting for tasks from rank 0.
        - A ``None`` payload signals graceful shutdown.
        - All logging goes to stderr; stdout is reserved for the binary
          framing protocol.
        """
        src_path = Path(__file__).parent.parent.parent.parent.parent

        script_content = f'''#!/usr/bin/env python3
# Auto-generated persistent MPI worker script — do not edit.
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pickle
import struct
import sys
import logging

# ------------------------------------------------------------------
# Protect stdout: redirect Python-level stdout to stderr so that
# library code (e.g., Fortran wrappers) cannot corrupt the binary
# framing protocol on fd 1.
# ------------------------------------------------------------------
_protocol_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
sys.stdout = sys.stderr  # all print() now goes to stderr

# ------------------------------------------------------------------
# Logging (stderr only)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [Rank %(rank)s] %(message)s',
    stream=sys.stderr,
    defaults={{"rank": "?"}},
)
logger = logging.getLogger("mpi_persistent_worker")

for noisy in (
    'rasterio', 'fiona', 'boto3', 'botocore',
    'matplotlib', 'urllib3', 's3transfer',
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ------------------------------------------------------------------
# Path setup & imports (happens ONCE)
# ------------------------------------------------------------------
sys.path.insert(0, r"{str(src_path)}")

from mpi4py import MPI

try:
    from {worker_module} import {worker_function} as _worker_func
except ImportError as exc:
    logger.error("Failed to import worker function: %s", exc)
    logger.error("sys.path = %s", sys.path)
    MPI.COMM_WORLD.Abort(1)


# ------------------------------------------------------------------
# Framing helpers (mirror coordinator side)
# ------------------------------------------------------------------

def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return b""
            raise EOFError(f"Expected {{n}} bytes, got {{len(buf)}}")
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(stream):
    header = _read_exact(stream, 8)
    if not header:
        raise EOFError("stdin closed")
    length = struct.unpack(">Q", header)[0]
    return _read_exact(stream, length)


def _write_frame(stream, data):
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def broker_loop(comm, rank, size):
    """Rank-0 broker: relay between coordinator pipes and MPI workers."""
    stdin_buf = os.fdopen(sys.stdin.fileno(), "rb", buffering=0)

    while True:
        # 1. Read a task batch from the coordinator via stdin
        try:
            raw = _read_frame(stdin_buf)
        except EOFError:
            logger.info("stdin closed — shutting down")
            _broadcast_shutdown(comm, size)
            return

        tasks = pickle.loads(raw)

        # None is the shutdown sentinel
        if tasks is None:
            logger.info("Received shutdown sentinel")
            _broadcast_shutdown(comm, size)
            return

        logger.info("Received batch of %d tasks", len(tasks))

        # 2. Distribute tasks across ranks (including self)
        tasks_per_rank = _distribute_tasks(tasks, size)

        for dest in range(1, size):
            comm.send(tasks_per_rank[dest], dest=dest, tag=1)

        # 3. Process rank-0's own share
        my_results = _run_tasks(tasks_per_rank[0], rank)

        # 4. Gather results from workers
        all_results = list(my_results)
        for src in range(1, size):
            worker_results = comm.recv(source=src, tag=2)
            all_results.extend(worker_results)

        # 5. Send results back to coordinator via stdout
        result_bytes = pickle.dumps(all_results, protocol=pickle.HIGHEST_PROTOCOL)
        _write_frame(_protocol_stdout, result_bytes)

        logger.info("Sent %d results back to coordinator", len(all_results))


def worker_loop(comm, rank):
    """Ranks 1..N-1: receive tasks, execute, return results."""
    while True:
        tasks = comm.recv(source=0, tag=1)

        # None is the shutdown sentinel
        if tasks is None:
            logger.info("Received shutdown sentinel")
            return

        results = _run_tasks(tasks, rank)
        comm.send(results, dest=0, tag=2)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _distribute_tasks(tasks, size):
    """Round-robin tasks to ranks, respecting proc_id when present."""
    buckets = [[] for _ in range(size)]
    for task in tasks:
        dest = task.get("proc_id", 0) % size
        buckets[dest].append(task)
    return buckets


def _run_tasks(tasks, rank):
    """Execute a list of tasks and return results (never raises)."""
    results = []
    for i, task in enumerate(tasks):
        try:
            result = _worker_func(task)
            results.append(result)
        except Exception as exc:
            logger.error("Rank %d task %d failed: %s", rank, i, exc)
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Update the logging filter so %(rank)s works
    for handler in logging.root.handlers:
        old_fmt = handler.formatter
        handler.setFormatter(
            logging.Formatter(
                fmt=f"[%(asctime)s] [%(levelname)s] [Rank {{rank}}] %(message)s",
            )
        )

    logger.info("Worker started (size=%d)", size)

    try:
        if rank == 0:
            broker_loop(comm, rank, size)
        else:
            worker_loop(comm, rank)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        comm.Abort(1)
    finally:
        logger.info("Exiting")


if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
