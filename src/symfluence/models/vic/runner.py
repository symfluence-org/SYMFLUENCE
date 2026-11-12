# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC Model Runner.

Executes the VIC model using prepared input files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('VIC')
class VICRunner(BaseModelRunner):
    """Runner for the VIC model (image or classic driver)."""

    MODEL_NAME = "VIC"

    def _setup_model_specific_paths(self) -> None:
        """Set up VIC-specific paths."""
        self.settings_dir = self.project_dir / "settings" / "VIC"

        # Determine executable name from driver config
        driver = self._get_config_value(lambda: self.config.model.vic.driver, dict_key='VIC_DRIVER')
        default_exe = 'vic_image.exe' if driver == 'image' else 'vic_classic.exe'

        self.vic_exe = self.get_model_executable(
            install_path_key='VIC_INSTALL_PATH',
            default_install_subpath='installs/vic',
            default_exe_name=default_exe,
            typed_exe_accessor=lambda: (
                self.config.model.vic.exe
                if self.config.model and self.config.model.vic
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build VIC execution command with global parameter file."""
        global_file_name = self._get_config_value(
            lambda: self.config.model.vic.global_param_file,
            dict_key='VIC_GLOBAL_PARAM_FILE',
        )
        global_param = self.settings_dir / global_file_name
        return [str(self.vic_exe), '-g', str(global_param)]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from output directory."""
        return self.output_dir

    def _get_run_timeout(self) -> int:
        """VIC timeout from config."""
        return self._get_config_value(lambda: self.config.model.vic.timeout, dict_key='VIC_TIMEOUT')
