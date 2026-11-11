# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SLURM job management mixin for model runners.

Provides SLURM script generation, job submission, monitoring, GRU-parallel
execution, and unified execute_in_mode dispatch.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

from symfluence.core.modeling.execution.model_executor import (
    ExecutionMode,
    ExecutionResult,
    SlurmJobConfig,
)
from symfluence.core.process_exec import run as run_subprocess

#: States that are not a verdict yet: the job left ``squeue`` but accounting
#: has not recorded how it ended ('' = no record at all).
_SLURM_UNSETTLED_STATES = frozenset({
    "", "UNKNOWN", "PENDING", "RUNNING", "COMPLETING", "CONFIGURING",
    "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SUSPENDED",
    "STAGE_OUT", "SIGNALING",
})

#: How many extra polls to give accounting to settle before giving up.
SLURM_STATUS_RETRIES = 10


class SlurmExecutionMixin:
    """Mixin providing SLURM execution methods for model runners."""

    def is_slurm_available(self) -> bool:
        """Check if SLURM is available on this system."""
        return shutil.which("sbatch") is not None

    def create_slurm_script(
        self,
        config: SlurmJobConfig,
        commands: List[str],
        setup_commands: Optional[List[str]] = None,
        cleanup_commands: Optional[List[str]] = None
    ) -> str:
        """Generate a SLURM batch script.

        Args:
            config: SlurmJobConfig with job parameters
            commands: Main commands to execute
            setup_commands: Commands to run before main commands
            cleanup_commands: Commands to run after main commands

        Returns:
            Complete SLURM batch script as string
        """
        lines = ["#!/bin/bash"]

        # Core directives
        lines.append(f"#SBATCH --job-name={config.job_name}")
        lines.append(f"#SBATCH --time={config.time_limit}")
        lines.append(f"#SBATCH --mem={config.memory}")
        lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")

        # Optional directives
        if config.partition:
            lines.append(f"#SBATCH --partition={config.partition}")
        if config.account:
            lines.append(f"#SBATCH --account={config.account}")
        if config.array_size is not None:
            lines.append(f"#SBATCH --array=0-{config.array_size}")
        if config.output_pattern:
            lines.append(f"#SBATCH --output={config.output_pattern}")
        if config.error_pattern:
            lines.append(f"#SBATCH --error={config.error_pattern}")

        # Additional directives
        for key, value in config.additional_directives.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Job info for debugging
        lines.extend([
            "# Print job info for debugging",
            'echo "Starting job at $(date)"',
            'echo "Running on host: $(hostname)"',
            'echo "SLURM Job ID: $SLURM_JOB_ID"',
        ])

        if config.array_size is not None:
            lines.append('echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"')

        lines.append("")

        # Setup commands
        if setup_commands:
            lines.append("# Setup")
            lines.extend(setup_commands)
            lines.append("")

        # Main commands
        lines.append("# Main execution")
        lines.extend(commands)
        lines.append("")

        # Cleanup commands
        if cleanup_commands:
            lines.append("# Cleanup")
            lines.extend(cleanup_commands)
            lines.append("")

        lines.append('echo "Job completed at $(date)"')

        return "\n".join(lines)

    def submit_slurm_job(
        self,
        script_path: Path,
        wait: bool = False,
        poll_interval: int = 60,
        max_wait_time: int = 3600
    ) -> ExecutionResult:
        """Submit a SLURM job and optionally wait for completion.

        Args:
            script_path: Path to SLURM batch script
            wait: If True, monitor job until completion
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait before returning

        Returns:
            ExecutionResult with job_id and completion status
        """
        if not self.is_slurm_available():
            raise RuntimeError("SLURM 'sbatch' command not found")

        start_time = time.time()

        try:
            # Submit job
            result = run_subprocess(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )

            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted SLURM job: {job_id}")

            if not wait:
                # Submitted, NOT finished. 'success' here only means sbatch
                # accepted the job; callers must check slurm_job_finished()
                # before consuming outputs.
                return ExecutionResult(
                    success=True,
                    return_code=0,
                    job_id=job_id,
                    metadata={'status': 'submitted', 'finished': False}
                )

            # Monitor job
            return self.monitor_slurm_job(
                job_id=job_id,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time
            )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SLURM submission failed: {e.stderr}")
            return ExecutionResult(
                success=False,
                return_code=e.returncode,
                error_message=e.stderr,
                duration_seconds=time.time() - start_time
            )

    @staticmethod
    def slurm_job_finished(result: ExecutionResult) -> bool:
        """True when *result* describes a job that ran to successful completion.

        ``submit_slurm_job(wait=False)`` returns ``success=True`` as soon as
        sbatch accepts the job; that result must not be mistaken for a
        finished run (e.g. to start merging outputs).
        """
        return bool(result.success) and result.metadata.get('status') != 'submitted'

    def monitor_slurm_job(
        self,
        job_id: str,
        poll_interval: int = 60,
        max_wait_time: int = 3600,
        on_status_change: Optional[Callable[[str, str], None]] = None
    ) -> ExecutionResult:
        """Monitor a SLURM job until completion or timeout.

        Args:
            job_id: SLURM job ID to monitor
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait
            on_status_change: Callback(job_id, new_status) on status change

        Returns:
            ExecutionResult with final job status
        """
        start_time = time.time()
        last_status = None
        unsettled_checks = 0

        self.logger.info(f"Monitoring SLURM job {job_id}")

        while (time.time() - start_time) < max_wait_time:
            try:
                # Check if job is still in queue
                queue_result = run_subprocess(
                    ["squeue", "-j", job_id, "-h"],
                    capture_output=True,
                    text=True
                )

                if not queue_result.stdout.strip():
                    # Job no longer in queue, check final status
                    status = self._get_slurm_job_status(job_id)
                    duration = time.time() - start_time

                    if status == "COMPLETED":
                        self.logger.info(f"Job {job_id} completed successfully")
                        return ExecutionResult(
                            success=True,
                            return_code=0,
                            job_id=job_id,
                            duration_seconds=duration,
                            metadata={'slurm_status': status}
                        )

                    if status in _SLURM_UNSETTLED_STATES:
                        # Accounting lags the queue: the job has left squeue
                        # but sacct has no terminal record yet. Ask again a
                        # bounded number of times rather than guessing.
                        unsettled_checks += 1
                        if unsettled_checks <= SLURM_STATUS_RETRIES:
                            self.logger.debug(
                                f"Job {job_id} left the queue but its final state is "
                                f"not recorded yet ({status or 'no record'}); "
                                f"retry {unsettled_checks}/{SLURM_STATUS_RETRIES}"
                            )
                            time.sleep(poll_interval)
                            continue
                        status = status or "UNKNOWN"
                        message = (
                            f"SLURM job {job_id} left the queue but no final state "
                            f"could be read from sacct/scontrol after "
                            f"{SLURM_STATUS_RETRIES} retries (last: {status}); "
                            f"not treating it as successful"
                        )
                    else:
                        # Anything other than COMPLETED on any task is a
                        # failure: FAILED, CANCELLED, TIMEOUT, NODE_FAIL,
                        # OUT_OF_MEMORY, PREEMPTED, BOOT_FAIL, DEADLINE, ...
                        message = f"SLURM status: {status}"

                    self.logger.error(f"Job {job_id} did not complete: {message}")
                    return ExecutionResult(
                        success=False,
                        return_code=1,
                        job_id=job_id,
                        duration_seconds=duration,
                        error_message=message,
                        metadata={'slurm_status': status}
                    )
                else:
                    # Job still running - parse status
                    unsettled_checks = 0
                    current_status = self._parse_squeue_output(queue_result.stdout)

                    if current_status != last_status:
                        self.logger.info(f"Job {job_id}: {current_status}")
                        if on_status_change:
                            on_status_change(job_id, current_status)
                        last_status = current_status

            except subprocess.SubprocessError as e:
                self.logger.warning(f"Error checking job status: {e}")

            time.sleep(poll_interval)

        # Timeout: stop the job. Leaving it running means it keeps writing
        # into an output directory the caller has already reported as failed
        # (and may reuse).
        self.logger.warning(
            f"Timeout after {max_wait_time}s waiting for job {job_id}; cancelling it"
        )
        self._cancel_slurm_job(job_id)
        return ExecutionResult(
            success=False,
            return_code=-1,
            job_id=job_id,
            duration_seconds=time.time() - start_time,
            error_message=f"Timeout after {max_wait_time}s (job cancelled)",
            metadata={'slurm_status': 'MONITOR_TIMEOUT'}
        )

    def _cancel_slurm_job(self, job_id: str) -> None:
        """Best-effort ``scancel`` of a job (all array tasks)."""
        try:
            run_subprocess(["scancel", job_id], capture_output=True, text=True)
        except (subprocess.SubprocessError, OSError) as e:
            self.logger.warning(f"Could not cancel SLURM job {job_id}: {e}")

    @staticmethod
    def _normalise_slurm_state(raw: str) -> str:
        """'CANCELLED by 1000' -> 'CANCELLED'; 'CANCELLED+' -> 'CANCELLED'."""
        tokens = raw.strip().split()
        return tokens[0].rstrip('+').upper() if tokens else ''

    def _get_slurm_job_states(self, job_id: str) -> List[str]:
        """Final state of every task of a job (one entry per array task).

        Reads ``sacct -X`` (allocations only, so the .batch/.extern steps do
        not appear) and falls back to ``scontrol show job`` on clusters
        without accounting. Empty when neither has a record.
        """
        states: List[str] = []
        try:
            result = run_subprocess(
                ["sacct", "-X", "-j", job_id, "-o", "State", "-n", "--parsable2"],
                capture_output=True,
                text=True
            )
            states = [
                self._normalise_slurm_state(line.split('|')[0])
                for line in (result.stdout or '').splitlines()
            ]
            states = [state for state in states if state]
        except (subprocess.SubprocessError, OSError):
            states = []
        if states:
            return states

        try:
            result = run_subprocess(
                ["scontrol", "show", "job", job_id, "-o"],
                capture_output=True,
                text=True
            )
        except (subprocess.SubprocessError, OSError):
            return []
        for line in (result.stdout or '').splitlines():
            for token in line.split():
                if token.startswith('JobState='):
                    state = self._normalise_slurm_state(token.split('=', 1)[1])
                    if state:
                        states.append(state)
        return states

    def _get_slurm_job_status(self, job_id: str) -> str:
        """Aggregate final status of a SLURM job across all of its tasks.

        Returns ``COMPLETED`` only when every task completed. Otherwise the
        first failing state, or — when nothing has failed but some task has no
        terminal record yet — that unsettled state ('' when there is no record
        at all).
        """
        states = self._get_slurm_job_states(job_id)
        if not states:
            return ""
        failed = [
            state for state in states
            if state != "COMPLETED" and state not in _SLURM_UNSETTLED_STATES
        ]
        if failed:
            return failed[0]
        unsettled = [state for state in states if state in _SLURM_UNSETTLED_STATES]
        if unsettled:
            return unsettled[0]
        return "COMPLETED"

    def _parse_squeue_output(self, output: str) -> str:
        """Parse squeue output to get human-readable status."""
        lines = output.strip().split('\n')
        pending = sum(1 for line in lines if 'PENDING' in line or ' PD ' in line)
        running = sum(1 for line in lines if 'RUNNING' in line or ' R ' in line)
        return f"{running} running, {pending} pending"

    def create_gru_parallel_script(
        self,
        model_exe: Path,
        file_manager: Path,
        log_dir: Path,
        total_grus: int,
        grus_per_job: int,
        job_name: str,
        time_limit: str = "03:00:00",
        memory: str = "4G"
    ) -> str:
        """Create a SLURM script for GRU-parallel execution.

        Args:
            model_exe: Path to model executable
            file_manager: Path to file manager/control file
            log_dir: Directory for log files
            total_grus: Total number of GRUs to process
            grus_per_job: Number of GRUs per array task
            job_name: SLURM job name
            time_limit: Time limit per task
            memory: Memory per task

        Returns:
            Complete SLURM batch script
        """
        n_array_jobs = max(1, -(-total_grus // grus_per_job)) - 1  # Ceiling div, 0-based

        slurm_config = SlurmJobConfig(
            job_name=job_name,
            time_limit=time_limit,
            memory=memory,
            array_size=n_array_jobs,
            output_pattern=str(log_dir / f"{job_name}_%A_%a.out"),
            error_pattern=str(log_dir / f"{job_name}_%A_%a.err")
        )

        setup = [
            f"mkdir -p {log_dir}",
            "",
            "# Calculate GRU range for this task",
            f"gru_start=$(( ({grus_per_job} * $SLURM_ARRAY_TASK_ID) + 1 ))",
            f"gru_end=$(( gru_start + {grus_per_job} - 1 ))",
            "",
            f"if [ $gru_end -gt {total_grus} ]; then",
            f"    gru_end={total_grus}",
            "fi",
            "",
            'echo "Processing GRUs $gru_start to $gru_end"',
        ]

        commands = [
            "for gru in $(seq $gru_start $gru_end); do",
            '    echo "Starting GRU $gru"',
            f"    {model_exe} -g $gru 1 -m {file_manager}",
            "    exit_code=$?",
            "    if [ $exit_code -ne 0 ]; then",
            '        echo "Model failed for GRU $gru with exit code $exit_code"',
            "        exit 1",
            "    fi",
            '    echo "Completed GRU $gru"',
            "done",
        ]

        return self.create_slurm_script(
            config=slurm_config,
            commands=commands,
            setup_commands=setup
        )

    def estimate_optimal_grus_per_job(
        self,
        total_grus: int,
        min_jobs: int = 10,
        max_jobs: int = 500,
        ideal_grus_per_job: int = 50
    ) -> int:
        """Estimate optimal number of GRUs per SLURM job.

        Args:
            total_grus: Total GRUs in domain
            min_jobs: Minimum number of jobs to create
            max_jobs: Maximum jobs (prevent queue flooding)
            ideal_grus_per_job: Target GRUs per job

        Returns:
            Optimal GRUs per job
        """
        if total_grus <= min_jobs:
            return 1

        if total_grus <= ideal_grus_per_job * min_jobs:
            return max(1, total_grus // min_jobs)

        ideal_jobs = total_grus // ideal_grus_per_job

        if ideal_jobs <= max_jobs:
            grus_per_job = ideal_grus_per_job
        else:
            grus_per_job = -(-total_grus // max_jobs)

        # Scale up for very large domains
        if total_grus > 10000:
            scale_factor = min(3.0, total_grus / 10000)
            grus_per_job = int(grus_per_job * scale_factor)

        return min(grus_per_job, total_grus)

    def execute_in_mode(
        self,
        mode: ExecutionMode,
        command: Union[List[str], str],
        log_file: Path,
        slurm_config: Optional[SlurmJobConfig] = None,
        wait_for_slurm: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """Execute command in specified mode (local or SLURM).

        Args:
            mode: Execution mode (LOCAL or SLURM)
            command: Command to execute
            log_file: Path for log file
            slurm_config: SLURM configuration (required for SLURM mode)
            wait_for_slurm: Wait for SLURM job completion
            **kwargs: Additional arguments for execute_subprocess

        Returns:
            ExecutionResult
        """
        if mode == ExecutionMode.LOCAL:
            return self.execute_subprocess(command, log_file, **kwargs)

        elif mode in (ExecutionMode.SLURM, ExecutionMode.SLURM_ARRAY):
            if not slurm_config:
                raise ValueError("slurm_config required for SLURM execution")

            if not self.is_slurm_available():
                self.logger.warning("SLURM not available, falling back to local execution")
                return self.execute_subprocess(command, log_file, **kwargs)

            # Create script. A list command is ONE command line (argv), not
            # one script line per element; cwd/env must travel into the job.
            command_line = shlex.join(command) if isinstance(command, list) else command
            setup: List[str] = []
            for key, value in (kwargs.get('env') or {}).items():
                setup.append(f"export {key}={shlex.quote(str(value))}")
            if kwargs.get('cwd'):
                setup.append(f"cd {shlex.quote(str(kwargs['cwd']))} || exit 1")
            script_content = self.create_slurm_script(
                config=slurm_config,
                commands=[command_line],
                setup_commands=setup or None,
            )

            # Write script
            script_path = log_file.parent / f"{slurm_config.job_name}.sh"
            script_path.write_text(script_content, encoding='utf-8')
            if os.name != 'nt':
                script_path.chmod(0o755)

            # Submit
            return self.submit_slurm_job(
                script_path=script_path,
                wait=wait_for_slurm
            )

        else:
            raise ValueError(f"Unknown execution mode: {mode}")
