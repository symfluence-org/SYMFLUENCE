# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Model Runner.

Executes the CRHM (Cold Regions Hydrological Model) using prepared input files.
CRHM is driven by a .prj project file and reads forcing from a .obs file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('CRHM')
class CRHMRunner(BaseModelRunner):
    """Runner for the CRHM model."""

    MODEL_NAME = "CRHM"

    def _setup_model_specific_paths(self) -> None:
        """Set up CRHM-specific paths."""
        self.crhm_forcing_dir = self.project_forcing_dir / "CRHM_input"
        self.settings_dir = self.project_dir / "settings" / "CRHM"

        self.crhm_exe = self.get_model_executable(
            install_path_key='CRHM_INSTALL_PATH',
            default_install_subpath='installs/crhm',
            default_exe_name='crhm',
            typed_exe_accessor=lambda: (
                self.config.model.crhm.exe
                if self.config.model and self.config.model.crhm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build CRHM execution command.

        CRHM takes the project file as a positional argument.
        ``--obs_file_directory`` points to the obs files when cwd
        differs from the project file location.
        """
        project_file_name = self._get_config_value(
            lambda: self.config.model.crhm.project_file,
            dict_key='CRHM_PROJECT_FILE',
        )
        project_file = self.settings_dir / project_file_name
        obs_dir = str(self.crhm_forcing_dir) + os.sep
        output_file = self.output_dir / "crhm_output.txt"

        return [
            str(self.crhm_exe),
            '--obs_file_directory', obs_dir,
            '-o', str(output_file),
            '-p', '30',
            str(project_file),
        ]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from output directory."""
        return self.output_dir

    def _get_run_timeout(self) -> int:
        """CRHM timeout from config."""
        return self._get_config_value(lambda: self.config.model.crhm.timeout, dict_key='CRHM_TIMEOUT')
