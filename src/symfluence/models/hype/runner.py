# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE model runner.

Handles HYPE model execution and run-time management.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('HYPE')
class HYPERunner(BaseModelRunner):  # type: ignore[misc]
    """Runner for the HYPE model."""

    MODEL_NAME = "HYPE"

    def _setup_model_specific_paths(self) -> None:
        """Set up HYPE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "HYPE"

        self.hype_exe = self.get_model_executable(
            install_path_key='HYPE_INSTALL_PATH',
            default_install_subpath='installs/hype/bin',
            exe_name_key='HYPE_EXE',
            default_exe_name='hype',
            typed_exe_accessor=lambda: self.typed_config.model.hype.exe if (self.typed_config and self.typed_config.model.hype) else None,
            must_exist=True
        )

    def _get_output_dir(self) -> Path:
        """HYPE uses custom output path resolution.

        During calibration the worker pre-sets ``output_dir`` to the
        per-iteration directory.  Honour that override instead of
        re-deriving the path from config (which may be a flat dict).
        """
        if hasattr(self, '_output_dir_override') and self._output_dir_override is not None:
            return self._output_dir_override
        try:
            experiment_id = self.config.domain.experiment_id
            return self.get_config_path('EXPERIMENT_OUTPUT_HYPE', f"simulations/{experiment_id}/HYPE")
        except (AttributeError, KeyError):
            if self.output_dir is not None:
                return self.output_dir
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='default', dict_key='EXPERIMENT_ID')
            return self.get_simulation_dir('HYPE', experiment_id)

    def _build_run_command(self) -> Optional[List[str]]:
        """Build HYPE execution command."""
        return [
            str(self.hype_exe),
            str(self.setup_dir).rstrip('/') + '/'
        ]

    def _get_expected_outputs(self) -> List[str]:
        """HYPE output location is controlled by resultdir in info.txt,
        not by the runner's output_dir. Skip file verification here;
        the postprocessor and calibration worker check outputs directly."""
        return []
