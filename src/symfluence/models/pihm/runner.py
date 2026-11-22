# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Model Runner

Executes MM-PIHM from a prepared simulation directory.

MM-PIHM expects the following directory structure:
    {sim_dir}/
        input/{project_name}/{project_name}.*   -- all input files
        output/{project_name}/                   -- output directory (created)

The binary is invoked as:
    pihm [-o output/{project_name}] {project_name}

from {sim_dir} as the working directory.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base.base_runner import BaseModelRunner
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

logger = logging.getLogger(__name__)

# Must match PIHMPreProcessor.PROJECT_NAME
PROJECT_NAME = "pihm_lumped"


@R.runners.add("PIHM")
class PIHMRunner(BaseModelRunner):
    """
    Runs MM-PIHM via direct invocation.

    Handles:
    - Executable path resolution
    - Simulation directory setup with correct MM-PIHM layout:
          sim_dir/input/{project_name}/  -- input files copied from settings/PIHM
          sim_dir/output/{project_name}/ -- created for model output
    - Model execution: pihm [-o output/{project_name}] {project_name}
    - Output verification (.rivflx, .gw, .surf, .recharge files)
    """


    MODEL_NAME = "PIHM"
    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)
        self.settings_dir = self.project_dir / "settings" / "PIHM"

    def _get_pihm_executable(self) -> Path:
        """Get the Flux-PIHM executable path.

        Uses flux-pihm (Noah LSM build) by default for proper
        energy-balance ET computation. Falls back to pihm if
        flux-pihm is not found.
        """
        return self.get_model_executable(
            install_path_key='PIHM_INSTALL_PATH',
            default_install_subpath='installs/pihm',
            default_exe_name='flux-pihm',
            typed_exe_accessor=lambda: (
                self.config.model.pihm.exe
                if self.config.model and self.config.model.pihm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _get_timeout(self) -> int:
        return self._get_config_value(lambda: self.config.model.pihm.timeout, dict_key='PIHM_TIMEOUT')

    def run_pihm(self, sim_dir: Optional[Path] = None, **kwargs) -> Optional[Path]:
        """
        Execute MM-PIHM.

        Sets up the simulation directory with the correct MM-PIHM layout,
        runs the model, and verifies output.

        Args:
            sim_dir: Optional override for simulation directory.

        Returns:
            Path to simulation directory on success.

        Raises:
            ModelExecutionError: If execution fails.
        """
        logger.info(f"Running MM-PIHM for domain: {self.config.domain.name}")

        with symfluence_error_handler(
            "PIHM model execution",
            logger,
            error_type=ModelExecutionError,
        ):
            if sim_dir is None:
                self.output_dir = self.get_simulation_dir('PIHM', self.config.domain.experiment_id)
            else:
                self.output_dir = sim_dir

            self.output_dir.mkdir(parents=True, exist_ok=True)

            pihm_exe = self._get_pihm_executable()
            logger.info(f"Using PIHM executable: {pihm_exe}")

            # Set up the MM-PIHM directory structure
            self._setup_sim_directory(self.output_dir)

            # MM-PIHM command: pihm [-o <name>] <project_name>
            # PIHM internally prepends "output/" to the -o arg, so pass just the name.
            # Run from sim_dir which contains input/ and output/ subdirs.
            cmd = [str(pihm_exe), "-o", PROJECT_NAME, PROJECT_NAME]
            logger.info(
                f"Executing MM-PIHM: {' '.join(cmd)} "
                f"(cwd={self.output_dir})"
            )

            env = os.environ.copy()
            timeout = self._get_timeout()

            result = run_subprocess(
                cmd,
                cwd=str(self.output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.stdout:
                logger.debug(f"PIHM stdout:\n{result.stdout[-2000:]}")
            if result.stderr:
                logger.debug(f"PIHM stderr:\n{result.stderr[-2000:]}")

            if result.returncode != 0:
                logger.error(f"PIHM execution returned code {result.returncode}")
                if result.stderr:
                    logger.error(
                        f"stderr: {result.stderr[-2000:]}"
                    )
                if result.stdout:
                    logger.error(
                        f"stdout (last 2000 chars): {result.stdout[-2000:]}"
                    )
                raise ModelExecutionError(
                    f"PIHM execution failed with return code {result.returncode}"
                )

            logger.info("MM-PIHM execution completed successfully")
            self._verify_output()

            return self.output_dir

    def _setup_sim_directory(self, sim_dir: Path) -> None:
        """Set up the MM-PIHM simulation directory structure.

        Creates:
            sim_dir/input/{project_name}/  -- copies all input files here
            sim_dir/output/{project_name}/ -- empty, for model output

        The preprocessor writes files to settings/PIHM/ with the project name
        prefix. This method copies them into the correct MM-PIHM layout.
        """
        if not self.settings_dir.exists():
            raise ModelExecutionError(
                f"PIHM settings directory not found: {self.settings_dir}. "
                "Run preprocessing first."
            )

        # Create MM-PIHM directory structure
        input_dir = sim_dir / "input" / PROJECT_NAME
        output_dir = sim_dir / "output" / PROJECT_NAME
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy all input files from settings/PIHM/ to input/{project_name}/
        file_count = 0
        for src in self.settings_dir.iterdir():
            if src.is_file() and src.name.startswith(PROJECT_NAME):
                dst = input_dir / src.name
                shutil.copy2(src, dst)
                logger.debug(f"Copied {src.name} -> input/{PROJECT_NAME}/")
                file_count += 1

        if file_count == 0:
            raise ModelExecutionError(
                f"No input files found in {self.settings_dir} "
                f"with prefix '{PROJECT_NAME}'. Run preprocessing first."
            )

        # Copy global lookup tables from PIHM install dir to input/
        # MM-PIHM expects vegprmt.tbl, co2.txt, ndep.txt, epc/ at input/ level
        pihm_exe = self._get_pihm_executable()
        # The install dir is the MM-PIHM repo root containing input/
        # Try exe's directory first (exe may be in repo root), then parent
        install_input = pihm_exe.parent / "input"
        if not install_input.exists():
            install_input = pihm_exe.parent.parent / "input"
        base_input = sim_dir / "input"
        global_files = ["vegprmt.tbl", "co2.txt", "ndep.txt"]
        for gf in global_files:
            src = install_input / gf
            dst = base_input / gf
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                logger.debug(f"Copied global table {gf}")
        # Copy epc/ directory
        epc_src = install_input / "epc"
        epc_dst = base_input / "epc"
        if epc_src.is_dir() and not epc_dst.exists():
            shutil.copytree(epc_src, epc_dst)
            logger.debug("Copied epc/ directory")

        logger.info(
            f"Set up MM-PIHM directory: {file_count} input files in "
            f"input/{PROJECT_NAME}/, output -> output/{PROJECT_NAME}/"
        )

    def _verify_output(self) -> None:
        """Verify MM-PIHM produced valid output files.

        Checks for key output files in output/{project_name}/:
            *.rivflx* -- river fluxes
            *.surf*   -- surface water
            *.gw*     -- groundwater
            *.recharge* -- recharge
        """
        pihm_output_dir = self.output_dir / "output" / PROJECT_NAME
        if not pihm_output_dir.exists():
            raise RuntimeError(
                f"PIHM output directory not found: {pihm_output_dir}"
            )

        # MM-PIHM output naming: pihm_lumped.river.flx1.txt, pihm_lumped.gw.txt, etc.
        rivflx_files = list(pihm_output_dir.glob("*.river.flx*.txt"))
        gw_files = list(pihm_output_dir.glob("*.gw.txt"))
        surf_files = list(pihm_output_dir.glob("*.surf.txt"))
        recharge_files = list(pihm_output_dir.glob("*.recharge.txt"))

        all_outputs = rivflx_files + gw_files + surf_files + recharge_files

        if not all_outputs:
            # List what IS in the output directory for diagnostics
            existing = list(pihm_output_dir.iterdir())
            if existing:
                logger.warning(
                    f"Output directory contains: "
                    f"{[f.name for f in existing[:20]]}"
                )
            raise RuntimeError(
                f"MM-PIHM did not produce expected output in {pihm_output_dir}. "
                "Expected .river.flx, .gw, .surf, or .recharge files."
            )

        logger.info(
            f"Verified MM-PIHM output in {pihm_output_dir}: "
            f"{len(rivflx_files)} rivflx, {len(gw_files)} gw, "
            f"{len(surf_files)} surf, {len(recharge_files)} recharge file(s)"
        )

    def run(self, **kwargs) -> Optional[Path]:
        """Alternative entry point for PIHM execution."""
        return self.run_pihm(**kwargs)
