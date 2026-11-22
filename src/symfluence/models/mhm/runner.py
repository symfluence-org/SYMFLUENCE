# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM Model Runner.

Executes the mHM (mesoscale Hydrological Model) using prepared input files.
mHM is run from within the settings directory where the namelists reside.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('MHM')
class MHMRunner(BaseModelRunner):
    """Runner for the mHM model."""

    MODEL_NAME = "MHM"

    def _setup_model_specific_paths(self) -> None:
        """Set up mHM-specific paths.

        Uses standard paths:
            self.setup_dir   -> {project_dir}/settings/MHM
            self.forcing_path -> {project_dir}/data/forcing/MHM_input
        """
        self.setup_dir = self.project_dir / "settings" / self.model_name
        self.forcing_path = self.project_forcing_dir / f"{self.model_name}_input"

        self.mhm_exe = self.get_model_executable(
            install_path_key='MHM_INSTALL_PATH',
            default_install_subpath='installs/mhm',
            default_exe_name='mhm',
            typed_exe_accessor=lambda: (
                self.config.model.mhm.exe
                if self.config.model and self.config.model.mhm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build mHM execution command.

        mHM reads namelists (mhm.nml, mrm.nml) from its working directory.
        """
        return [str(self.mhm_exe)]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from settings directory where namelists reside."""
        return self.setup_dir

    def _get_run_timeout(self) -> int:
        """mHM timeout from config."""
        return self._get_config_value(lambda: self.config.model.mhm.timeout, dict_key='MHM_TIMEOUT')
