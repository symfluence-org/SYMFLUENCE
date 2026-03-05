# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TauDEM command execution and MPI orchestration.

Provides TauDEM command execution with MPI support and retry logic.
Eliminates code duplication across GeofabricDelineator and LumpedWatershedDelineator.

Handles:
- MPI launcher detection (srun or mpirun)
- Module load commands
- Retry logic on failure
- Command prefixing logic

Refactored from geofabric_utils.py (2026-01-01)
"""

import os
import shlex
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Union


class TauDEMExecutor:
    """
    Executes TauDEM commands with MPI support and retry logic.

    This class manages TauDEM command execution across different HPC environments,
    automatically detecting whether to use srun (SLURM) or mpirun (generic MPI).
    """

    def __init__(self, config: Dict[str, Any], logger: Any, taudem_dir: str):
        """
        Initialize TauDEM executor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            taudem_dir: Path to TauDEM binary directory
        """
        self.config = config
        self.logger = logger
        self.taudem_dir = taudem_dir
        self.num_processes = config.get('NUM_PROCESSES', 1)
        self.max_retries = config.get('MAX_RETRIES', 3)
        self.retry_delay = config.get('RETRY_DELAY', 5)

        # Add TauDEM to PATH
        os.environ['PATH'] = f"{os.environ['PATH']}:{taudem_dir}"

        # Ensure LD_LIBRARY_PATH includes GDAL libs for TauDEM's runtime dependency.
        # GDAL may come from conda (CONDA_PREFIX/lib) or system HPC modules
        # (module load gdal sets LD_LIBRARY_PATH in the shell).
        # We check conda first, then verify GDAL is actually discoverable.
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            conda_lib = os.path.join(conda_prefix, 'lib')
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if conda_lib not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib

    @staticmethod
    def _is_openmpi() -> bool:
        """Check if the available mpirun is OpenMPI."""
        try:
            result = subprocess.run(
                ["mpirun", "--version"],
                capture_output=True, text=True, timeout=5
            )
            return "Open MPI" in result.stdout or "open-mpi" in result.stdout.lower()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _srun_has_pmi() -> bool:
        """Check if srun supports PMIx/PMI2 for MPI launch."""
        try:
            result = subprocess.run(
                ["srun", "--mpi=list"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout + result.stderr
            return "pmix" in output.lower() or "pmi2" in output.lower()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_mpi_command(self) -> Optional[str]:
        """
        Detect the best available MPI launcher.

        Respects MPI_LAUNCHER config override. Otherwise uses smart detection:
        - Checks for bundled mpirun in taudem_dir (npm dist/bin/ layout)
        - If OpenMPI is detected, always use mpirun (OpenMPI's own launcher)
          since srun requires SLURM PMI support that OpenMPI often lacks.
        - If srun is available AND has PMI support, use srun (Intel MPI, MPICH).
        - Falls back to mpirun, then None.

        Returns:
            Path to mpirun, 'srun', 'mpirun', or None if no MPI launcher found
        """
        override = self.config.get('MPI_LAUNCHER')
        if override and shutil.which(override):
            return override

        # Check for bundled mpirun next to TauDEM tools (npm layout)
        bundled = os.path.join(self.taudem_dir, 'mpirun')
        if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
            self.logger.debug("Using bundled MPI launcher: %s", bundled)
            # Set OPAL_PREFIX so OpenMPI finds its data files
            prefix = os.path.dirname(self.taudem_dir)
            os.environ.setdefault('OPAL_PREFIX', prefix)
            return bundled

        has_srun = shutil.which("srun") is not None
        has_mpirun = shutil.which("mpirun") is not None

        if has_mpirun and self._is_openmpi():
            self.logger.debug("OpenMPI detected — using mpirun (srun requires PMI support)")
            return "mpirun"

        if has_srun and self._srun_has_pmi():
            return "srun"

        if has_mpirun:
            return "mpirun"

        if has_srun:
            return "srun"

        return None

    def get_mpi_command(self) -> Optional[str]:
        """
        Get the MPI launcher command for parallel TauDEM execution.

        Detects available MPI implementations (mpiexec, mpirun) on the system
        and returns the appropriate launcher command.

        Returns:
            Optional[str]: MPI launcher command (e.g., 'mpiexec -n 4') or None
                if MPI is not available or configured.
        """
        return self._get_mpi_command()

    @staticmethod
    def _strip_mpi_prefix(parts: List[str]) -> List[str]:
        """
        Strip a leading MPI launcher prefix from a tokenized command.

        Handles the common pattern: mpirun/srun -n <N> <command> ...
        """
        if not parts:
            return parts

        if parts[0] in {"mpirun", "srun"}:
            if len(parts) >= 3 and parts[1] in {"-n", "-np"}:
                return parts[3:]
            return parts[1:]

        return parts

    def run_command(self, command: str, retry: bool = True) -> None:
        """
        Run a TauDEM command with MPI support and retry logic.

        Handles several scenarios:
        1. Commands with module load (e.g., "module load taudem && pitremove")
        2. Commands already prefixed with mpirun/srun
        3. Regular commands that need MPI prefix

        Args:
            command: TauDEM command to execute
            retry: Enable retry on failure (default: True)

        Raises:
            subprocess.CalledProcessError: If command fails after all retries
        """
        detected_mpi = self._get_mpi_command()
        run_cmds: List[Optional[str]] = [detected_mpi] if detected_mpi else [None]
        if detected_mpi:
            # If srun is primary, try mpirun as fallback before no-MPI
            if detected_mpi == "srun" and shutil.which("mpirun"):
                run_cmds.append("mpirun")
            run_cmds.append(None)

        last_err: Optional[subprocess.CalledProcessError] = None
        max_attempts = self.max_retries if retry else 1

        for run_cmd in run_cmds:
            if run_cmd is None and detected_mpi is not None:
                self.logger.info(f"Trying without {detected_mpi}...")

            for attempt in range(max_attempts):
                try:
                    # Check if command already has MPI prefix to avoid double prefixing
                    has_mpi_prefix = any(cmd in command for cmd in ["mpirun", "srun"])

                    # Determine if shell execution is required
                    # module load is a shell function and requires shell=True
                    needs_shell = "module load" in command

                    if run_cmd and needs_shell:
                        # Handle commands with module load specially - requires shell=True
                        # Security note: module load commands require shell execution
                        # as 'module' is a shell function, not an executable
                        parts = command.split(" && ")
                        if len(parts) == 2:
                            module_part = parts[0]
                            actual_cmd = parts[1]
                            if not has_mpi_prefix:
                                full_command: Union[str, List[str]] = f"{module_part} && {run_cmd} -n {self.num_processes} {actual_cmd}"
                            else:
                                full_command = command
                        else:
                            full_command = command
                    elif run_cmd and not has_mpi_prefix:
                        # Add MPI prefix for regular commands - use list format
                        # Export LD_LIBRARY_PATH to MPI child processes so TauDEM
                        # can find GDAL libs (from conda env or HPC module load)
                        if run_cmd == "mpirun":
                            full_command = [run_cmd, "-x", "LD_LIBRARY_PATH", "-n", str(self.num_processes)] + shlex.split(command)
                        elif run_cmd == "srun":
                            # srun: use --export=ALL to propagate LD_LIBRARY_PATH
                            # to compute nodes (not all SLURM clusters do this by default)
                            full_command = [run_cmd, "--export=ALL", "-n", str(self.num_processes)] + shlex.split(command)
                        else:
                            full_command = [run_cmd, "-n", str(self.num_processes)] + shlex.split(command)
                    elif has_mpi_prefix:
                        # Command already has MPI prefix - parse with shlex
                        full_command = shlex.split(command)
                        if run_cmd is None:
                            full_command = self._strip_mpi_prefix(full_command)
                    else:
                        # No MPI launcher available - parse with shlex
                        full_command = shlex.split(command)

                    self.logger.debug(f"Running command: {full_command}")
                    result = subprocess.run(
                        full_command,
                        check=True,
                        shell=needs_shell,  # nosec B602 - shell controlled by MPI config
                        capture_output=True,
                        text=True
                    )
                    self.logger.debug(f"Command output: {result.stdout}")
                    return

                except subprocess.CalledProcessError as e:
                    last_err = e
                    self.logger.error(f"Error executing command: {full_command}")
                    self.logger.error(f"Error details: {e.stderr}")

                    if attempt < max_attempts - 1 and retry:
                        self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)

        if last_err is not None:
            raise last_err
