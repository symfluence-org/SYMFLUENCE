# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Subprocess execution mixin for model runners.

Provides execute_subprocess (canonical), execute_model_subprocess (deprecated),
and run_with_retry for all model runners via mixin inheritance.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from symfluence.core.modeling.execution.model_executor import (
    ExecutionResult,
    augment_conda_library_paths,
)
from symfluence.core.process_exec import run as run_subprocess

#: Backstop wall-clock bound applied when a caller passes no explicit timeout.
#: ``timeout=None`` used to mean "wait forever", and 11 of the 19 call sites
#: relied on that default — so any model binary that wedged took its workflow
#: with it. Observed on Windows: fuse.exe stopped after writing its output
#: NetCDF header and sat at 0% CPU while the Python parent blocked in
#: ``subprocess.run`` for 37h, holding a reproduction run slot with nothing
#: logged and nothing failed.
#:
#: This is deliberately generous rather than tight: it exists to bound a hang,
#: not to police slow-but-legitimate runs (large domains and long calibrations
#: can run for many hours). Paths that know their own budget should still pass
#: an explicit ``timeout`` — see FUSE_TIMEOUT. Set the environment variable to
#: 0 to restore unbounded behaviour.
DEFAULT_SUBPROCESS_TIMEOUT = 86400  # 24h


def _default_timeout() -> Optional[int]:
    """Resolve the backstop timeout, honouring SYMFLUENCE_SUBPROCESS_TIMEOUT."""
    raw = os.environ.get('SYMFLUENCE_SUBPROCESS_TIMEOUT')
    if raw is None:
        return DEFAULT_SUBPROCESS_TIMEOUT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SUBPROCESS_TIMEOUT
    return value if value > 0 else None  # 0/negative == explicit opt-out


class SubprocessExecutionMixin:
    """Mixin providing subprocess execution methods for model runners."""

    def execute_subprocess(
        self,
        command: Union[List[str], str],
        log_file: Path,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        shell: bool = False,
        check: bool = True,
        capture_output: bool = False,
        success_message: Optional[str] = None,
        error_context: Optional[Dict[str, Any]] = None,
        success_log_level: int = logging.INFO
    ) -> ExecutionResult:
        """Execute a subprocess with standardized logging and error handling.

        This is the canonical execution method for all model runners. Returns
        an ``ExecutionResult`` with structured information about the run.

        Args:
            command: Command to execute (list or string)
            log_file: Path to write stdout/stderr
            cwd: Working directory for execution
            env: Environment variables (merged with os.environ)
            timeout: Timeout in seconds. None applies DEFAULT_SUBPROCESS_TIMEOUT
                as a backstop against a wedged binary hanging the workflow
                forever; pass an explicit value when the path knows its budget.
            shell: Whether to use shell execution
            check: Raise exception on non-zero exit code
            capture_output: Return stdout/stderr in result metadata
            success_message: Custom message to log on success
            error_context: Additional context to include in error logs
            success_log_level: Log level for success message (default INFO)

        Returns:
            ExecutionResult with success status, return code, and metadata

        Raises:
            subprocess.TimeoutExpired: If check=True and the timeout is
                exceeded (with check=False a failed ExecutionResult with
                return_code -1 is returned instead). The child's whole
                process group is killed either way.
            subprocess.CalledProcessError: If check=True and process fails
        """
        start_time = time.time()
        if timeout is None:
            timeout = _default_timeout()

        # Merge environment variables
        run_env = os.environ.copy()
        augment_conda_library_paths(run_env)
        if env:
            run_env.update(env)

        # Ensure log directory exists
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Log execution start
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        self.logger.debug(f"Executing: {cmd_str}")
        if cwd:
            self.logger.debug(f"Working directory: {cwd}")

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                result = run_subprocess(
                    command,
                    check=False,  # We handle the check ourselves
                    stdin=subprocess.DEVNULL,
                    stdout=f if not capture_output else subprocess.PIPE,
                    stderr=subprocess.STDOUT if not capture_output else subprocess.PIPE,
                    cwd=cwd,
                    env=run_env,
                    shell=shell,  # nosec B602,B604
                    text=True,
                    timeout=timeout,
                    # mpirun / shell wrappers leave grandchildren behind when
                    # only the direct child is killed on timeout.
                    kill_group=True,
                )

            duration = time.time() - start_time

            # Build result
            exec_result = ExecutionResult(
                success=(result.returncode == 0),
                return_code=result.returncode,
                log_file=log_file,
                duration_seconds=duration,
                metadata={}
            )

            if capture_output:
                exec_result.metadata['stdout'] = result.stdout
                exec_result.metadata['stderr'] = result.stderr

            # Log outcome
            if result.returncode == 0:
                msg = success_message or f"Process completed successfully in {duration:.1f}s"
                self.logger.log(success_log_level, msg)
            else:
                # Log at WARNING so a non-zero exit is visible at the default log
                # level, not hidden until the user reruns with --debug. This is the
                # most user-facing diagnostic gap (e.g. "SUMMA appears to exit
                # silently").
                self.logger.warning(f"Process exited with code {result.returncode}; see log file: {log_file}")
                exec_result.error_message = f"Exit code: {result.returncode}"

                # Log error context
                if error_context:
                    for key, value in error_context.items():
                        self.logger.debug(f"  {key}: {value}")

                if check:
                    raise subprocess.CalledProcessError(
                        result.returncode, command
                    )

            return exec_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.logger.error(f"Process timed out after {timeout}s; see log file: {log_file}")
            if check:
                # A timeout is a failed run. Returning a result here let
                # callers that rely on check=True (and ignore the return
                # value) carry on and score whatever output already existed.
                raise
            return ExecutionResult(
                success=False,
                return_code=-1,
                log_file=log_file,
                duration_seconds=duration,
                error_message=f"Timeout after {timeout}s"
            )

    def execute_model_subprocess(
        self,
        command: Union[List[str], str],
        log_file: Path,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = False,
        check: bool = True,
        timeout: Optional[int] = None,
        success_message: str = "Model execution completed successfully",
        success_log_level: int = logging.INFO,
        error_context: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Execute model subprocess (deprecated — use execute_subprocess instead).

        Backward-compatible wrapper returning ``subprocess.CompletedProcess``.

        .. deprecated::
            Use :meth:`execute_subprocess` which returns ``ExecutionResult``.
        """
        warnings.warn(
            "execute_model_subprocess() is deprecated, use execute_subprocess() instead. "
            "execute_subprocess() returns an ExecutionResult with richer metadata.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            # Merge environment variables
            run_env = os.environ.copy()
            augment_conda_library_paths(run_env)
            if env:
                run_env.update(env)

            # Ensure log directory exists
            self.ensure_dir(log_file.parent)

            # Execute subprocess
            self.logger.debug(f"Executing command: {' '.join(command)}")
            with open(log_file, 'w', encoding='utf-8') as f:
                # Security rationale: shell mode for trusted model executables
                result = run_subprocess(
                    command,
                    check=check,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env=run_env,
                    shell=shell,  # nosec B602,B604
                    text=True,
                    timeout=timeout
                )

            if result.returncode == 0:
                self.logger.log(success_log_level, success_message)
            else:
                self.logger.debug(f"Process exited with code {result.returncode}")

            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"Model execution failed with return code {e.returncode}"
            self.logger.error(error_msg)

            # Log error context if provided
            if error_context:
                for key, value in error_context.items():
                    self.logger.error(f"{key}: {value}")

            # Read and include last lines from log file for better diagnostics
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_lines = f.readlines()
                        last_lines = log_lines[-20:] if len(log_lines) > 20 else log_lines
                        if last_lines:
                            self.logger.error("Last 20 lines from log file:")
                            for line in last_lines:
                                self.logger.error(f"  {line.rstrip()}")
                except Exception as read_error:  # noqa: BLE001 — model execution resilience
                    self.logger.error(f"Could not read log file: {read_error}", exc_info=True)

            self.logger.error(f"Full log available at: {log_file}")
            raise

        except subprocess.TimeoutExpired:
            self.logger.error(f"Process timeout after {timeout} seconds")
            self.logger.error(f"See log file for details: {log_file}")
            raise

    def run_with_retry(
        self,
        command: Union[List[str], str],
        log_file: Path,
        max_attempts: int = 3,
        retry_delay: int = 5,
        **kwargs
    ) -> ExecutionResult:
        """Execute subprocess with automatic retry on failure.

        Args:
            command: Command to execute
            log_file: Path for log file
            max_attempts: Maximum retry attempts
            retry_delay: Seconds between retries
            **kwargs: Additional arguments for execute_subprocess

        Returns:
            ExecutionResult from final attempt
        """
        last_result = ExecutionResult(
            success=False,
            return_code=-1,
            error_message="Execution failed to start or all attempts failed",
            metadata={'command': str(command)}
        )

        for attempt in range(1, max_attempts + 1):
            self.logger.debug(f"Attempt {attempt}/{max_attempts}")

            # Add attempt number to log file name
            attempt_log = log_file.with_suffix(f".attempt{attempt}.log")

            result = self.execute_subprocess(
                command=command,
                log_file=attempt_log,
                check=False,
                **kwargs
            )

            if result.success:
                return result

            last_result = result

            if attempt < max_attempts:
                self.logger.warning(f"Attempt {attempt} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        return last_result
