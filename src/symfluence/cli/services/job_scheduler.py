# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Job scheduler service for SLURM and other HPC schedulers.

Handles job script generation, submission, and monitoring.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..console import Console
from .base import BaseService

# Placeholder a ``--template`` file may contain to position the SYMFLUENCE
# command block; without it the block is appended to the template.
TEMPLATE_COMMAND_PLACEHOLDER = "{{SYMFLUENCE_COMMAND}}"

# ``#SBATCH`` lines are parsed by sbatch, not by the shell, so quoting does
# not protect them. Values are restricted to characters that cannot end the
# directive or start a new line.
_SBATCH_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:+@%=/-]+$")
_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.+@%=-]+$")

# How often / how long to ask sacct for a finished job's state before giving
# up (accounting records can lag the queue by a few seconds).
_SACCT_ATTEMPTS = 6
_SACCT_RETRY_S = 5.0


def _sbatch_value(option: str, value: Any, pattern: "re.Pattern[str]" = _SBATCH_VALUE_RE) -> str:
    """Validate a value destined for an ``#SBATCH`` directive."""
    text = str(value)
    if not pattern.match(text):
        raise ValueError(
            f"Invalid value for SLURM option {option}: {text!r} "
            "(allowed: letters, digits and _ . , : + @ % = / -)"
        )
    return text


class JobSchedulerService(BaseService):
    """
    Service for managing HPC job submissions.

    Handles:
    - SLURM job script generation
    - Job submission
    - Job monitoring
    - Environment detection (HPC vs laptop)
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the JobSchedulerService.

        Args:
            console: Console instance for output. If None, uses global console.
        """
        super().__init__(console=console)
        # Result dict of the most recent submit_slurm_job call (job id, final
        # state when --wait was used); lets the CLI tell a submission failure
        # from a job that was submitted and then failed.
        self.last_result: Dict[str, Any] = {}

    def handle_slurm_job_submission(
        self,
        execution_plan: Dict[str, Any],
        symfluence_instance: Optional[Any] = None,
    ) -> bool:
        """
        Handle SLURM job submission workflow.

        Args:
            execution_plan: Execution plan from CLI manager
            symfluence_instance: SYMFLUENCE instance (optional)

        Returns:
            bool: Success status
        """
        try:
            slurm_options = execution_plan.get("slurm_options", {})

            if not slurm_options.get("job_account"):
                self._console.warning(
                    "No SLURM account specified. This may be required on your system."
                )
                self._console.indent(
                    "Use --account to specify an account if job submission fails."
                )

            result = self.submit_slurm_job(execution_plan, symfluence_instance)

            self.last_result = result
            if result.get("dry_run", False):
                return True
            if result.get("success", False):
                self._console.newline()
                self._console.success("SLURM job submission successful!")
                if not slurm_options.get("submit_and_wait", False):
                    self._console.info("Job is running in background. Monitor with:")
                    self._console.indent(f"squeue -j {result['job_id']}")
                    self._console.indent(f"tail -f SYMFLUENCE_*_{result['job_id']}.out")
                return True
            elif result.get("job_id") and result.get("final_state"):
                self._console.error(
                    f"SLURM job {result['job_id']} did not complete successfully "
                    f"(state: {result['final_state']}, exit code: "
                    f"{result.get('job_exit_code') or 'unknown'})"
                )
                return False
            else:
                self._console.error("Job submission failed")
                return False

        except (KeyError, RuntimeError, OSError, ValueError, TypeError, subprocess.SubprocessError) as e:
            self._console.error(f"Error in SLURM job submission: {str(e)}")
            return False

    def submit_slurm_job(
        self, execution_plan: Dict[str, Any], symfluence_instance=None
    ) -> Dict[str, Any]:
        """
        Submit a SLURM job for the execution plan.

        Args:
            execution_plan: Execution plan dictionary
            symfluence_instance: Optional SYMFLUENCE instance

        Returns:
            Dict with submission results including job_id

        Raises:
            RuntimeError: If SLURM is not available or submission fails
        """
        self._console.panel("Preparing SLURM Job Submission", style="blue")

        if not execution_plan.get("dry_run", False) and not self._check_slurm_available():
            raise RuntimeError("SLURM commands (sbatch) not available on this system")

        slurm_options = execution_plan.get("slurm_options", {})

        # Auto-generate job name if not provided
        if not slurm_options.get("job_name"):
            if symfluence_instance and hasattr(symfluence_instance, "config"):
                domain = symfluence_instance.config.get("DOMAIN_NAME", "symfluence")
            else:
                domain = "symfluence"

            mode = execution_plan.get("mode", "workflow")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slurm_options["job_name"] = f"{domain}_{mode}_{timestamp}"

        self._console.info(f"Job Name: {slurm_options['job_name']}")
        self._console.info(f"Time Limit: {slurm_options['job_time']}")
        self._console.info(f"Memory: {slurm_options['job_memory']}")
        self._console.info(f"Tasks: {slurm_options['job_ntasks']}")

        if slurm_options.get("job_account"):
            self._console.info(f"Account: {slurm_options['job_account']}")
        if slurm_options.get("job_partition"):
            self._console.info(f"Partition: {slurm_options['job_partition']}")

        config_file = execution_plan.get("config_file", "./config.yaml")

        script_content = self._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        if execution_plan.get("dry_run", False):
            self._console.info("[DRY RUN] SLURM script that would be submitted:")
            self._console.rule()
            for line in script_content.splitlines():
                self._console.info(line)
            self._console.rule()
            self._console.info("[DRY RUN] No script was written and no job was submitted")
            return {
                "success": True,
                "dry_run": True,
                "job_id": None,
                "job_name": slurm_options["job_name"],
            }

        script_path = Path(f"SYMFLUENCE_{slurm_options['job_name']}.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        self._console.info(f"SLURM script created: {script_path}")

        try:
            self._console.newline()
            self._console.info("Submitting job...")
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            job_id = result.stdout.strip().split()[-1]

            if not job_id:
                raise RuntimeError("Could not extract job ID from sbatch output")

            self._console.success("Job submitted successfully!")
            self._console.info(f"Job ID: {job_id}")

            submission_result = {
                "success": True,
                "job_id": job_id,
                "script_path": script_path,
                "job_name": slurm_options["job_name"],
                "submission_time": datetime.now().isoformat(),
            }

            if slurm_options.get("submit_and_wait", False):
                self._console.newline()
                self._console.info(f"Waiting for job {job_id} to complete...")
                state, job_exit_code = self._monitor_slurm_job(job_id)
                submission_result["final_state"] = state
                submission_result["job_exit_code"] = job_exit_code
                # Leaving the queue is not success: FAILED, TIMEOUT,
                # CANCELLED, OUT_OF_MEMORY, ... all leave it too.
                submission_result["success"] = state == "COMPLETED"

            return submission_result

        except subprocess.CalledProcessError as e:
            self._console.error(f"Error submitting job: {e}")
            self._console.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job: {e}") from e

    def _create_symfluence_slurm_script(
        self,
        execution_plan: Dict[str, Any],
        slurm_options: Dict[str, Any],
        config_file: str,
    ) -> str:
        """Create SLURM script content for SYMFLUENCE workflow."""
        job_mode = execution_plan.get("job_mode", "workflow")
        job_steps = execution_plan.get("job_steps", [])
        settings = execution_plan.get("settings", {})

        def _quote_args(args):
            return " ".join(shlex.quote(str(arg)) for arg in args)

        if job_mode == "individual_steps":
            steps = [str(step) for step in job_steps if step]
            if steps:
                if len(steps) == 1:
                    symfluence_cmd = f"symfluence workflow step {shlex.quote(steps[0])}"
                else:
                    symfluence_cmd = f"symfluence workflow steps {_quote_args(steps)}"
            else:
                symfluence_cmd = "symfluence workflow run"

            if config_file:
                symfluence_cmd += f" --config {shlex.quote(config_file)}"

            if settings.get("force_rerun", False):
                symfluence_cmd += " --force-rerun"
        elif job_mode == "pour_point_setup":
            pour_point_info = execution_plan.get("pour_point", {})
            symfluence_cmd = (
                f"symfluence project pour-point {shlex.quote(str(pour_point_info.get('coordinates')))} "
                f"--domain-name {shlex.quote(str(pour_point_info.get('domain_name')))} "
                f"--definition {shlex.quote(str(pour_point_info.get('domain_definition_method')))}"
            )
            if pour_point_info.get("bounding_box_coords"):
                symfluence_cmd += (
                    f" --bounding-box {shlex.quote(str(pour_point_info['bounding_box_coords']))}"
                )
            if pour_point_info.get("experiment_id"):
                symfluence_cmd += (
                    f" --experiment-id {shlex.quote(str(pour_point_info['experiment_id']))}"
                )
        else:
            workflow_args = [str(arg) for arg in job_steps if arg]
            if workflow_args and workflow_args[0] == "symfluence":
                workflow_args = workflow_args[1:]

            if workflow_args:
                symfluence_cmd = f"symfluence {_quote_args(workflow_args)}"
            else:
                symfluence_cmd = "symfluence workflow run"

            if config_file and "--config" not in workflow_args:
                symfluence_cmd += f" --config {shlex.quote(config_file)}"

            if settings.get("force_rerun", False) and "--force-rerun" not in workflow_args:
                symfluence_cmd += " --force-rerun"
            if settings.get("continue_on_error", False) and "--continue-on-error" not in workflow_args:
                symfluence_cmd += " --continue-on-error"

        if settings.get("debug", False) and "--debug" not in symfluence_cmd:
            symfluence_cmd += " --debug"

        job_name = _sbatch_value("--name", slurm_options["job_name"], _JOB_NAME_RE)

        command_block = f"""
echo "Python environment: $(which python)"
echo "SYMFLUENCE directory: $(pwd)"
echo ""

# Run SYMFLUENCE workflow
echo "Starting SYMFLUENCE workflow..."
echo "Command:" {shlex.quote(symfluence_cmd)}
echo ""

exit_code=0
{symfluence_cmd} || exit_code=$?

echo ""
echo "=========================================="
echo "SYMFLUENCE Job Completed"
echo "Exit code: $exit_code"
echo "Finished: $(date)"
echo "=========================================="

exit $exit_code
"""

        template_path = slurm_options.get("slurm_template")
        if template_path:
            # --template: the user's own batch header and environment setup
            # (#SBATCH directives, module loads, activation). Only the
            # SYMFLUENCE command block is generated.
            template_file = Path(template_path)
            if not template_file.is_file():
                raise FileNotFoundError(f"SLURM template not found: {template_path}")
            template = template_file.read_text(encoding="utf-8")
            if TEMPLATE_COMMAND_PLACEHOLDER in template:
                return template.replace(TEMPLATE_COMMAND_PLACEHOLDER, command_block)
            return template.rstrip("\n") + "\n" + command_block

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=SYMFLUENCE_{job_name}_%j.out
#SBATCH --error=SYMFLUENCE_{job_name}_%j.err
#SBATCH --time={_sbatch_value("--time", slurm_options['job_time'])}
#SBATCH --ntasks={_sbatch_value("--tasks", slurm_options['job_ntasks'])}
#SBATCH --mem={_sbatch_value("--memory", slurm_options['job_memory'])}"""

        if slurm_options.get("job_account"):
            script += f"\n#SBATCH --account={_sbatch_value('--account', slurm_options['job_account'])}"
        if slurm_options.get("job_partition"):
            script += f"\n#SBATCH --partition={_sbatch_value('--partition', slurm_options['job_partition'])}"
        if slurm_options.get("job_nodes") and slurm_options["job_nodes"] > 1:
            script += f"\n#SBATCH --nodes={_sbatch_value('--nodes', slurm_options['job_nodes'])}"

        script += f"""

# Job information
echo "=========================================="
echo "SYMFLUENCE SLURM Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name:" {shlex.quote(job_name)}
echo "Node: $HOSTNAME"
echo "Started: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
"""

        # Environment activation is emitted only when asked for; otherwise the
        # job inherits the submitting shell's environment (sbatch's default).
        modules = slurm_options.get("job_modules")
        conda_env = slurm_options.get("conda_env")
        if modules or conda_env:
            script += '\n# Load modules and environment\necho "Loading environment..."\n'
        if modules:
            script += f"module restore {shlex.quote(str(modules))}\n"
        if conda_env:
            # Batch shells are non-interactive: `conda activate` needs the
            # shell hook that an interactive shell gets from `conda init`.
            script += 'eval "$(conda shell.bash hook)"\n'
            script += f"conda activate {shlex.quote(str(conda_env))}\n"

        # Strict mode only from here on: module/conda activation scripts
        # routinely reference unset variables and would trip `set -u`.
        script += "\nset -euo pipefail\n"
        script += command_block
        return script

    def _check_slurm_available(self) -> bool:
        """Check if SLURM commands are available."""
        return shutil.which("sbatch") is not None

    def _monitor_slurm_job(self, job_id: str, poll_interval: float = 60.0) -> Tuple[str, str]:
        """Wait for a SLURM job to leave the queue, then report how it ended.

        Returns:
            ``(state, exit_code)`` from accounting, e.g. ``("COMPLETED", "0:0")``
            or ``("FAILED", "1:0")``. The state is ``"UNKNOWN"`` when sacct
            has no record of the job.
        """
        self._console.info(f"Monitoring job {job_id}...")
        while True:
            try:
                result = subprocess.run(
                    ["squeue", "-j", job_id, "-h"],
                    capture_output=True,
                    text=True,
                )
                if not result.stdout.strip():
                    break
                lines = result.stdout.strip().split("\n")
                if lines:
                    status_info = lines[0].split()
                    if len(status_info) >= 5:
                        status = status_info[4]
                        self._console.info(f"Job status: {status}")
            except subprocess.SubprocessError as e:
                self._console.warning(f"Error checking job status: {e}")
            time.sleep(poll_interval)

        # An empty queue only says the job is no longer pending or running.
        state, exit_code = self._final_job_state(job_id)
        if state == "COMPLETED":
            self._console.success(f"Job {job_id} completed (exit code {exit_code})")
        else:
            self._console.error(
                f"Job {job_id} ended in state {state} (exit code {exit_code or 'unknown'})"
            )
        return state, exit_code

    def _final_job_state(self, job_id: str) -> Tuple[str, str]:
        """Ask ``sacct`` for a finished job's state and exit code.

        Retries briefly while accounting has no record yet. States such as
        ``CANCELLED by 1234`` are reduced to their first word.
        """
        for attempt in range(_SACCT_ATTEMPTS):
            try:
                result = subprocess.run(
                    ["sacct", "-j", job_id, "-X", "-o", "State,ExitCode",
                     "--parsable2", "-n"],
                    capture_output=True,
                    text=True,
                )
                lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
            except (subprocess.SubprocessError, OSError) as e:
                self._console.warning(f"Error querying sacct for job {job_id}: {e}")
                lines = []
            if lines:
                state_field, _, exit_code = lines[0].partition("|")
                state = state_field.strip().split(" ")[0].rstrip("+").upper()
                if state and state not in ("PENDING", "RUNNING", "COMPLETING",
                                           "CONFIGURING", "REQUEUED", "RESIZING",
                                           "SUSPENDED"):
                    return state, exit_code.strip()
            if attempt < _SACCT_ATTEMPTS - 1:
                time.sleep(_SACCT_RETRY_S)
        return "UNKNOWN", ""

    def detect_environment(self) -> str:
        """
        Detect whether we're running on HPC or a personal computer.

        Returns:
            str: 'hpc' or 'laptop'
        """
        hpc_schedulers = ["sbatch", "qsub", "bsub"]
        for scheduler in hpc_schedulers:
            if shutil.which(scheduler):
                return "hpc"

        hpc_env_vars = [
            "SLURM_CLUSTER_NAME",
            "SLURM_JOB_ID",
            "PBS_JOBID",
            "SGE_CLUSTER_NAME",
            "LOADL_STEP_ID",
        ]

        for env_var in hpc_env_vars:
            if env_var in os.environ:
                return "hpc"

        if Path("/scratch").exists():
            return "hpc"

        return "laptop"


# Backward compatibility alias
JobScheduler = JobSchedulerService
