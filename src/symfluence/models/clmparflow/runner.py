# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Model Runner

Executes the ParFlow-CLM binary from a prepared simulation directory.
Same execution pattern as standalone ParFlow, but reads from
settings/CLMPARFLOW/ and verifies both .pfb and CLM output files.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.mpi_utils import find_mpirun
from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_runner("CLMPARFLOW", method_name="run_clmparflow")
class CLMParFlowRunner(BaseModelRunner):
    """
    Runs ParFlow-CLM via direct parflow invocation.

    Same as ParFlowRunner but uses CLMPARFLOW settings directory
    and verifies CLM output files in addition to standard .pfb files.
    """


    MODEL_NAME = "CLMPARFLOW"
    _logged_setup = False

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.settings_dir = self.project_dir / "settings" / "CLMPARFLOW"

    def _get_parflow_executable(self) -> Path:
        """Get the CLMParFlow executable path."""
        return self.get_model_executable(
            install_path_key='CLMPARFLOW_INSTALL_PATH',
            default_install_subpath='installs/clmparflow',
            default_exe_name='parflow',
            typed_exe_accessor=lambda: (
                self.config.model.clmparflow.exe
                if self.config.model and self.config.model.clmparflow
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.clmparflow.timeout,
            default=7200,
        )

    def _get_num_procs(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.clmparflow.num_procs,
            default=1,
        )

    def _get_parflow_dir(self) -> Optional[str]:
        """Get PARFLOW_DIR for environment variable."""
        pf_dir = self._get_config_value(
            lambda: self.config.model.clmparflow.parflow_dir,
            default=None,
            dict_key='CLMPARFLOW_DIR',
        )
        if pf_dir and pf_dir != 'default':
            return str(pf_dir)
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

        pfidb_files = list(self.settings_dir.glob("*.pfidb"))
        if pfidb_files:
            return pfidb_files[0].stem

        return self._get_config_value(
            lambda: self.config.domain.name,
            default='clmparflow_run',
        )

    def run_clmparflow(self, sim_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute ParFlow-CLM.

        Args:
            sim_dir: Optional override for simulation directory.

        Returns:
            Path to output directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        with symfluence_error_handler(
            "CLMParFlow model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            if sim_dir is None:
                self.output_dir = (
                    self.project_dir / "simulations"
                    / self.config.domain.experiment_id / "CLMPARFLOW"
                )
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            pf_exe = self._get_parflow_executable()
            runname = self._get_runname()

            if not CLMParFlowRunner._logged_setup:
                logger.info(f"Running CLMParFlow for domain: {self.config.domain.name}")
                logger.info(f"Using CLMParFlow executable: {pf_exe}")
                logger.info(f"CLMParFlow run name: {runname}")
                CLMParFlowRunner._logged_setup = True

            self._setup_sim_directory(self.output_dir)

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

            logger.debug(f"Executing CLMParFlow from: {self.output_dir}")

            env = os.environ.copy()

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
                logger.debug(f"CLMParFlow stdout: {result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"CLMParFlow stderr: {result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"CLMParFlow execution returned code {result.returncode}")
                logger.error(
                    f"stderr: {result.stderr[-2000:] if result.stderr else 'none'}"
                )
                log_file = self.output_dir / f"{runname}.out.log"
                if log_file.exists():
                    log_content = log_file.read_text()
                    error_lines = [
                        ln for ln in log_content.splitlines()
                        if 'error' in ln.lower() or 'failed' in ln.lower()
                    ]
                    if error_lines:
                        logger.error(f"CLMParFlow log errors: {error_lines[-5:]}")

                raise ModelExecutionError(
                    f"CLMParFlow execution failed with return code {result.returncode}"
                )

            logger.debug("CLMParFlow execution completed successfully")
            self._verify_output(runname)

            return self.output_dir

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Copy CLMParFlow settings and forcing files to simulation directory."""
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"CLMParFlow settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )

        for old_pfb in sim_dir.glob('*.pfb'):
            old_pfb.unlink()

        # Copy settings (pfidb, drv_*.dat, runname.txt, etc.)
        for src in self.settings_dir.iterdir():
            if src.is_file() and src.suffix != '.npy':
                shutil.copy2(src, sim_dir / src.name)
                logger.debug(f"Copied {src.name} to simulation directory")

        # Copy forcing files (forcing.1d, etc.) from data/forcing/CLMPARFLOW_input/
        forcing_input_dir = resolve_data_subdir(
            self.project_dir, 'forcing'
        ) / 'CLMPARFLOW_input'
        if forcing_input_dir.exists():
            for src in forcing_input_dir.iterdir():
                if src.is_file() and src.suffix != '.npy':
                    shutil.copy2(src, sim_dir / src.name)
                    logger.debug(f"Copied forcing {src.name} to simulation directory")

    def _verify_output(self, runname: str) -> None:
        """Verify CLMParFlow produced valid output files."""
        press_files = list(self.output_dir.glob(f"{runname}.out.press.*.pfb"))
        satur_files = list(self.output_dir.glob(f"{runname}.out.satur.*.pfb"))

        if not press_files:
            raise RuntimeError(
                f"CLMParFlow did not produce expected pressure .pfb output "
                f"in {self.output_dir}"
            )

        for f in press_files[:1]:
            if f.stat().st_size == 0:
                raise RuntimeError(f"CLMParFlow pressure output file is empty: {f}")

        # Check for CLM output files (optional — CLM may not write .C.pfb in all configs)
        clm_files = list(self.output_dir.glob(f"{runname}.out.clm_output.*.C.pfb"))

        logger.debug(
            f"Verified CLMParFlow output: {len(press_files)} pressure, "
            f"{len(satur_files)} saturation, {len(clm_files)} CLM output file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for CLMParFlow execution."""
        return self.run_clmparflow(**kwargs)
