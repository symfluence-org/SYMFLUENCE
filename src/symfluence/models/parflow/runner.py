# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Model Runner

Executes the ParFlow binary from a prepared simulation directory.
ParFlow reads its .pfidb database file and writes .pfb binary output.
Rainfall forcing is applied via OverlandFlow BC in the .pfidb (monthly cycle).
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.mpi_utils import find_mpirun
from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("PARFLOW", method_name="run_parflow")
class ParFlowRunner(BaseModelRunner):
    """
    Runs ParFlow via direct parflow invocation.

    Handles:
    - Executable path resolution
    - Input file copying to simulation directory
    - Model execution (parflow <runname>)
    - MPI support for parallel execution
    - Output verification (*.pfb files)
    """


    MODEL_NAME = "PARFLOW"
    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.settings_dir = self.project_dir / "settings" / "PARFLOW"

    def _get_parflow_executable(self) -> Path:
        """Get the ParFlow executable path."""
        return self.get_model_executable(
            install_path_key='PARFLOW_INSTALL_PATH',
            default_install_subpath='installs/parflow',
            default_exe_name='parflow',
            typed_exe_accessor=lambda: (
                self.config.model.parflow.exe
                if self.config.model and self.config.model.parflow
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.parflow.timeout,
            default=3600,
        )

    def _get_num_procs(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.parflow.num_procs,
            default=1,
        )

    def _get_parflow_dir(self) -> Optional[str]:
        """Get PARFLOW_DIR for environment variable."""
        pf_dir = self._get_config_value(
            lambda: self.config.model.parflow.parflow_dir,
            default=None,
            dict_key='PARFLOW_DIR',
        )
        if pf_dir and pf_dir != 'default':
            return str(pf_dir)
        # Infer from executable path
        try:
            exe = self._get_parflow_executable()
            return str(exe.parent.parent)
        except Exception:  # noqa: BLE001 — model execution resilience
            return None

    def _get_runname(self) -> str:
        """Get the ParFlow run name from settings."""
        runname_file = self.settings_dir / 'runname.txt'
        if runname_file.exists():
            return runname_file.read_text().strip()

        # Look for .pfidb files
        pfidb_files = list(self.settings_dir.glob("*.pfidb"))
        if pfidb_files:
            return pfidb_files[0].stem

        return self._get_config_value(
            lambda: self.config.domain.name,
            default='parflow_run',
        )

    def run_parflow(self, sim_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute ParFlow.

        Args:
            sim_dir: Optional override for simulation directory. If None,
                     uses standard output path.

        Returns:
            Path to output directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        logger.debug(f"Running ParFlow for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "ParFlow model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            # Setup output directory
            if sim_dir is None:
                self.output_dir = (
                    self.project_dir / "simulations"
                    / self.config.domain.experiment_id / "PARFLOW"
                )
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            pf_exe = self._get_parflow_executable()
            logger.debug(f"Using ParFlow executable: {pf_exe}")

            # Get run name
            runname = self._get_runname()
            logger.debug(f"ParFlow run name: {runname}")

            # Copy input files to simulation directory
            self._setup_sim_directory(self.output_dir)

            # Build command
            num_procs = self._get_num_procs()
            if num_procs > 1:
                mpirun = find_mpirun(pf_exe)
                if mpirun is None:
                    raise ModelExecutionError(
                        "MPI launcher (mpirun/mpiexec) not found. "
                        "Install OpenMPI or use the npm package which bundles it."
                    )
                cmd = [mpirun, '-np', str(num_procs), str(pf_exe), runname]
            else:
                cmd = [str(pf_exe), runname]

            logger.debug(f"Executing ParFlow from: {self.output_dir}")

            env = os.environ.copy()

            # Set PARFLOW_DIR if available
            pf_dir = self._get_parflow_dir()
            if pf_dir:
                env['PARFLOW_DIR'] = pf_dir

            timeout = self._get_timeout()

            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.stdout:
                logger.debug(f"ParFlow stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"ParFlow stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"ParFlow execution returned code {result.returncode}")
                logger.error(
                    f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
                )
                # Check for ParFlow log file
                log_file = self.output_dir / f"{runname}.out.log"
                if log_file.exists():
                    log_content = log_file.read_text()
                    error_lines = [
                        ln for ln in log_content.splitlines()
                        if 'error' in ln.lower() or 'failed' in ln.lower()
                    ]
                    if error_lines:
                        logger.error(f"ParFlow log errors: {error_lines[-5:]}")

                raise ModelExecutionError(
                    f"ParFlow execution failed with return code {result.returncode}"
                )

            logger.debug("ParFlow execution completed successfully")
            self._verify_output(runname)

            return self.output_dir

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Copy all ParFlow input files to simulation directory."""
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"ParFlow settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )

        # Remove stale .pfb output from previous runs (avoids file count
        # confusion when DumpInterval changes between runs)
        for old_pfb in sim_dir.glob('*.pfb'):
            old_pfb.unlink()

        # Copy all ParFlow input files (.pfidb, runname.txt, etc.)
        # Skip .npy reference files (monthly_rainfall.npy etc.)
        for src in self.settings_dir.iterdir():
            if src.is_file() and src.suffix != '.npy':
                shutil.copy2(src, sim_dir / src.name)
                logger.debug(f"Copied {src.name} to simulation directory")

    def _verify_output(self, runname: str) -> None:
        """Verify ParFlow produced valid output files."""
        # Check for pressure .pfb files
        press_files = list(self.output_dir.glob(f"{runname}.out.press.*.pfb"))
        satur_files = list(self.output_dir.glob(f"{runname}.out.satur.*.pfb"))

        if not press_files:
            raise RuntimeError(
                f"ParFlow did not produce expected pressure .pfb output "
                f"in {self.output_dir}"
            )

        for f in press_files[:1]:
            if f.stat().st_size == 0:
                raise RuntimeError(f"ParFlow pressure output file is empty: {f}")

        logger.debug(
            f"Verified ParFlow output: {len(press_files)} pressure file(s), "
            f"{len(satur_files)} saturation file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for ParFlow execution."""
        return self.run_parflow(**kwargs)
