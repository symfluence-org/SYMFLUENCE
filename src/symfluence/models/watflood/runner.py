# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD Model Runner.

Executes WATFLOOD from a working directory (copy exe pattern, like MESH).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.process_exec import run as run_subprocess

logger = logging.getLogger(__name__)


class WATFLOODRunner(BaseModelRunner):
    """Runner for the WATFLOOD model.

    WATFLOOD is executed from a working directory containing all input files
    and the executable (copy-exe pattern, similar to MESH).
    """


    MODEL_NAME = "WATFLOOD"
    def run(self, **kwargs) -> Optional[Path]:
        """Run WATFLOOD model."""
        try:
            settings_dir = self.project_dir / 'WATFLOOD_input' / 'settings'
            output_dir = self.project_dir / 'WATFLOOD_output'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            watflood_exe = self._get_executable()
            if not watflood_exe.exists():
                raise ModelExecutionError(f"WATFLOOD executable not found: {watflood_exe}")

            # Copy executable to working directory (MESH pattern)
            work_exe = settings_dir / watflood_exe.name
            if not work_exe.exists() or work_exe.stat().st_mtime < watflood_exe.stat().st_mtime:
                shutil.copy2(watflood_exe, work_exe)
                work_exe.chmod(0o755)

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            timeout = self._get_config_value(
                lambda: self.config.model.watflood.timeout,
                default=3600,
                dict_key='WATFLOOD_TIMEOUT'
            )

            stdout_file = output_dir / 'watflood_stdout.log'
            stderr_file = output_dir / 'watflood_stderr.log'

            # Run from settings directory
            cmd = [str(work_exe)]
            logger.info(f"Running WATFLOOD: {' '.join(cmd)} in {settings_dir}")

            with open(stdout_file, 'w') as stdout_f, \
                 open(stderr_file, 'w') as stderr_f:
                result = run_subprocess(
                    cmd,
                    cwd=str(settings_dir),
                    env=env,
                    input=b"\n" * 256,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout
                )

            if result.returncode != 0:
                raise ModelExecutionError(f"WATFLOOD failed with return code {result.returncode}")

            self._check_native_logs(stdout_file, stderr_file)

            # Copy outputs to output directory
            self._collect_outputs(settings_dir, output_dir)

            if self._verify_output(output_dir):
                return output_dir
            raise ModelExecutionError("WATFLOOD did not produce verified output")

        except subprocess.TimeoutExpired:
            raise ModelExecutionError(f"WATFLOOD timed out after {timeout}s")
        except ModelExecutionError:
            raise
        except Exception as e:  # noqa: BLE001 — model execution resilience
            raise ModelExecutionError(f"Error running WATFLOOD: {e}") from e

    @staticmethod
    def _check_native_logs(*paths: Path) -> None:
        """Fortran STOP can return zero even when CHARM aborts its run."""
        for path in paths:
            text = path.read_text(errors='replace')
            text = re.sub(r'(?im)^\s*STOP CHARM: normal ending\s*$', '', text)
            if re.search(r'(?im)^\s*(?:STOP\b|PAUSE\b|.*\bprogram aborted\b|Fortran runtime error:)', text):
                raise ModelExecutionError(f"WATFLOOD native failure reported in {path}")

    def _get_executable(self) -> Path:
        """Get WATFLOOD executable path."""
        install_path = self._get_config_value(
            lambda: self.config.model.watflood.install_path,
            default='default',
            dict_key='WATFLOOD_INSTALL_PATH'
        )
        exe_name = self._get_config_value(
            lambda: self.config.model.watflood.exe,
            default='watflood',
            dict_key='WATFLOOD_EXE'
        )

        if install_path == 'default':
            return self.data_dir / "installs" / "watflood" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _collect_outputs(self, settings_dir: Path, output_dir: Path) -> None:
        """Copy WATFLOOD outputs to the standard output directory."""
        for pattern in ['*.tb0', '*.csv', '*.out']:
            for f in settings_dir.glob(pattern):
                if f.is_file():
                    shutil.copy2(f, output_dir / f.name)

    def _verify_output(self, output_dir: Path) -> bool:
        """Verify WATFLOOD produced output files."""
        output_files = (
            list(output_dir.glob('*.tb0')) +
            list(output_dir.glob('*.csv'))
        )
        if not output_files:
            logger.error("No WATFLOOD output files produced")
            return False
        logger.info(f"WATFLOOD output verified: {len(output_files)} files")
        return True
