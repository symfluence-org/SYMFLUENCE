# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Model Runner.

Executes the PRMS model using prepared input files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('PRMS')
class PRMSRunner(BaseModelRunner):
    """Runner for the PRMS model."""

    MODEL_NAME = "PRMS"

    def _setup_model_specific_paths(self) -> None:
        """Set up PRMS-specific paths."""
        self.setup_dir = self.project_dir / "settings" / self.model_name
        self.settings_dir = self.setup_dir

        self.prms_exe = self.get_model_executable(
            install_path_key='PRMS_INSTALL_PATH',
            default_install_subpath='installs/prms',
            default_exe_name='prms',
            typed_exe_accessor=lambda: (
                self.config.model.prms.exe
                if self.config.model and self.config.model.prms
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build PRMS execution command with control file."""
        control_name = self._get_config_value(
            lambda: self.config.model.prms.control_file,
            dict_key='PRMS_CONTROL_FILE',
        )
        control_file = self.settings_dir / control_name
        return [str(self.prms_exe), '-C', str(control_file)]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from output directory."""
        return self.output_dir

    def _get_run_timeout(self) -> int:
        """PRMS timeout from config."""
        return self._get_config_value(lambda: self.config.model.prms.timeout, dict_key='PRMS_TIMEOUT')
