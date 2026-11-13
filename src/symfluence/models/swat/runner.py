# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Model Runner.

Executes the SWAT model using prepared TxtInOut input files.
SWAT runs from within its output directory and produces output.rch
as its primary output file.

At run time the runner assembles a working directory by copying
settings from ``settings/SWAT/`` and forcing from
``data/forcing/SWAT_input/`` into the simulation output directory.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


def _fsync_directory(target_dir: Path) -> None:
    """Best-effort flush of a directory's entries to disk.

    Opening a directory with ``os.open(..., O_RDONLY)`` and calling
    ``os.fsync`` is a POSIX-only technique used to guarantee that files
    freshly copied into ``target_dir`` are durable before the Fortran
    executable reads them (this prevents a SIGBUS on arm64).

    On Windows a directory cannot be opened this way — ``os.open`` raises
    ``PermissionError`` ([Errno 13]) on a directory path — and the durability
    concern does not apply, so this is a no-op there.  Any other platform
    that cannot fsync a directory (e.g. some network filesystems) is also
    tolerated silently.
    """
    if os.name == 'nt':
        return
    try:
        fd = os.open(str(target_dir), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@R.runners.add('SWAT')
class SWATRunner(BaseModelRunner):
    """Runner for the SWAT model."""

    MODEL_NAME = "SWAT"

    def _setup_model_specific_paths(self) -> None:
        """Set up SWAT-specific paths."""
        # Standard layout: settings in settings/SWAT/, forcing in data/forcing/SWAT_input/
        self.swat_settings_dir = self.project_dir / 'settings' / 'SWAT'
        self.swat_forcing_dir = self.project_forcing_dir / 'SWAT_input'

        self.swat_exe = self.get_model_executable(
            install_path_key='SWAT_INSTALL_PATH',
            default_install_subpath='installs/swat',
            default_exe_name='swat_rel.exe',
            typed_exe_accessor=lambda: (
                self.config.model.swat.exe
                if self.config.model and self.config.model.swat
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _prepare_run(self) -> None:
        """Assemble TxtInOut in the output directory from settings + forcing."""
        self._assemble_txtinout(self.output_dir)

    def _assemble_txtinout(self, target_dir: Path) -> None:
        """Copy settings and forcing files into a single working directory.

        After copying, removes stale output files from any prior run and
        forces an fsync so the Fortran executable sees complete files.

        Args:
            target_dir: Directory to assemble TxtInOut into.
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        # Remove stale output from prior runs so SWAT starts clean
        for pattern in ('output.*', '*.out'):
            for stale in target_dir.glob(pattern):
                stale.unlink(missing_ok=True)

        # Copy settings files (file.cio, .bsn, .sub, .hru, etc.)
        if self.swat_settings_dir.exists():
            for f in self.swat_settings_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, target_dir / f.name)

        # Copy forcing files (.pcp, .tmp)
        if self.swat_forcing_dir.exists():
            for f in self.swat_forcing_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, target_dir / f.name)

        # Fsync the directory so all copied files are flushed to disk
        # before the Fortran executable reads them (prevents SIGBUS on arm64)
        _fsync_directory(target_dir)

        # Verify the critical control file exists
        cio = target_dir / 'file.cio'
        if not cio.exists():
            raise FileNotFoundError(
                f"file.cio not found in {target_dir} after assembly"
            )

    def _get_run_environment(self) -> Optional[Dict[str, str]]:
        """Set single-threaded environment for Fortran runtime."""
        return {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
        }

    def _build_run_command(self) -> Optional[List[str]]:
        """Build SWAT execution command.

        SWAT reads file.cio from its working directory.
        """
        return [str(self.swat_exe)]

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from the output directory (assembled TxtInOut)."""
        return self.output_dir

    def _get_expected_outputs(self) -> List[str]:
        """Expect output.rch in output directory."""
        return ['output.rch']

    def _get_run_timeout(self) -> int:
        """SWAT timeout from config."""
        return self._get_config_value(lambda: self.config.model.swat.timeout, dict_key='SWAT_TIMEOUT')
