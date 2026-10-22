# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Model Runner.

Executes CLM5 (cesm.exe) with prepared namelists and input files.
At runtime, CIME is bypassed -- cesm.exe is invoked directly from
the run directory where namelists are placed.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('CLM')
class CLMRunner(BaseModelRunner):
    """Runner for CLM5 via direct cesm.exe invocation."""

    MODEL_NAME = "CLM"

    def run(self, **kwargs) -> Optional[Path]:
        """Propagate native execution failure instead of reporting an empty success."""
        result = super().run(**kwargs)
        if result is None:
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError("CLM did not produce a successful native run; inspect its log")
        return result

    def _setup_model_specific_paths(self) -> None:
        """Set up CLM-specific paths."""
        self.setup_dir = self.project_dir / "settings" / self.model_name
        self.settings_dir = self.setup_dir

        self.clm_exe = self.get_model_executable(
            install_path_key='CLM_INSTALL_PATH',
            default_install_subpath='installs/clm',
            default_exe_name='cesm.exe',
            typed_exe_accessor=lambda: (
                self.config.model.clm.exe
                if self.config.model and self.config.model.clm
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build CLM execution command."""
        return [str(self.clm_exe)]

    def _prepare_run(self) -> None:
        """Clean stale output and copy NUOPC runtime files to run directory."""
        self._cleanup_stale_output(self.output_dir)
        self._setup_run_directory(self.output_dir)

    def _get_run_cwd(self) -> Optional[Path]:
        """CLM reads namelists from cwd."""
        return self.output_dir

    def _get_run_environment(self) -> Optional[Dict[str, str]]:
        """Disable macOS nano malloc zone (ESMF false heap corruption)."""
        return {'MallocNanoZone': '0'}

    def _get_run_timeout(self) -> int:
        """CLM timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.clm.timeout,
            default=3600,
        )

    def _cleanup_stale_output(self, run_dir: Path) -> None:
        """Remove stale output/restart/rpointer files before a fresh run.

        CMEPS treats the run as a continuation (not cold start) when
        rpointer files are present, which causes SIGTRAP on macOS ARM64
        when the restart state conflicts with the configured start date.
        """
        for pattern in [
            '*.clm2.h0.*.nc', '*.clm2.r.*.nc', '*.clm2.rh0.*.nc',
            '*.cpl.r.*.nc', '*.datm.r.*.nc', 'rpointer.*', '*.log',
        ]:
            for f in run_dir.glob(pattern):
                f.unlink()

    def _setup_run_directory(self, run_dir: Path) -> None:
        """Copy all NUOPC runtime files to run directory."""
        runtime_files = [
            'nuopc.runconfig', 'nuopc.runseq', 'fd.yaml',
            'datm_in', 'datm.streams.xml', 'lnd_in',
            'drv_in', 'drv_flds_in', 'CASEROOT',
            'user_nl_clm',
        ]
        # Create timing dirs that cesm.exe expects
        (run_dir / 'timing' / 'checkpoints').mkdir(parents=True, exist_ok=True)
        # use_init_interp writes ./init_generated_files/finidat_interp_dest.nc
        # (fixed path in CTSM 5.3, not a namelist item); without it CLM aborts
        (run_dir / 'init_generated_files').mkdir(parents=True, exist_ok=True)

        for name in runtime_files:
            src = self.settings_dir / name
            if src.exists():
                shutil.copy2(src, run_dir / name)
                self.logger.debug(f"Copied {name} to run directory")
            else:
                self.logger.warning(f"Runtime file not found: {src}")
