# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Model Runner.

Executes the GSFLOW binary which internally couples PRMS and MODFLOW-NWT.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('GSFLOW')
class GSFLOWRunner(BaseModelRunner):
    """Runner for the GSFLOW coupled model."""

    MODEL_NAME = "GSFLOW"

    def _setup_model_specific_paths(self) -> None:
        """Set up GSFLOW-specific paths."""
        self.setup_dir = self.get_config_path(
            'SETTINGS_GSFLOW_PATH',
            'settings/GSFLOW/'
        )

        self.gsflow_exe = self.get_model_executable(
            install_path_key='GSFLOW_INSTALL_PATH',
            default_install_subpath='installs/gsflow/bin',
            exe_name_key='GSFLOW_EXE',
            default_exe_name='gsflow',
            typed_exe_accessor=lambda: (
                self.config.model.gsflow.exe
                if self.config.model and self.config.model.gsflow
                else None
            ),
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build GSFLOW execution command."""
        control_file = self._get_config_value(
            lambda: self.config.model.gsflow.control_file,
            dict_key='GSFLOW_CONTROL_FILE',
        )
        control_path = self.setup_dir / control_file
        return [str(self.gsflow_exe), str(control_path)]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from settings directory."""
        return self.setup_dir

    def _get_run_environment(self):
        """Suppress macOS malloc logging."""
        return {'MallocStackLogging': '0'}

    def _get_run_timeout(self) -> int:
        """GSFLOW timeout from config."""
        return self._get_config_value(lambda: self.config.model.gsflow.timeout, dict_key='GSFLOW_TIMEOUT')
