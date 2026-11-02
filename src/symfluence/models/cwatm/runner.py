# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM Model Runner."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.registries import R


@R.runners.add('CWATM')
class CWatMRunner(BaseModelRunner):
    """Runner for CWatM (Community Water Model).

    CWatM is a pure-Python model (no PCRaster dependency) invoked as:
        python run_cwatm.py settings.ini
    """

    MODEL_NAME = "CWATM"

    def _setup_model_specific_paths(self) -> None:
        self.settings_dir = self.project_dir / "settings" / "CWATM"

        install_path = self._get_config_value(
            lambda: self.config.model.cwatm.install_path,
            default='default',
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )
        self.cwatm_dir = resolve_install_dir(install_path, data_dir, 'installs/cwatm')

        exe_name = self._get_config_value(
            lambda: self.config.model.cwatm.exe,
            default='run_cwatm.py',
        )
        self.runner_script = self.cwatm_dir / exe_name

    def run(self, **kwargs) -> Optional[Path]:
        config_file = self._get_config_value(
            lambda: self.config.model.cwatm.config_file,
            default='settings.ini',
        )
        ini_path = self.settings_dir / config_file
        if not ini_path.exists():
            raise FileNotFoundError(f"CWatM config not found: {ini_path}")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.cwatm_dir)

        cmd = [sys.executable, str(self.runner_script), str(ini_path), '-q']

        timeout = self._get_config_value(
            lambda: self.config.model.cwatm.timeout,
            default=14400,
        )

        log_dir = self.get_simulation_dir('CWATM', self._get_experiment_id()) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running CWatM for domain: {self.domain_name}")
        result = self.execute_subprocess(
            command=cmd,
            log_file=log_dir / f"cwatm_run_{self._timestamp()}.log",
            cwd=self.settings_dir,
            env=env,
            timeout=timeout,
        )
        if not result.success:
            self.logger.error(f"CWatM simulation failed (rc={result.returncode})")

        output_dir = self.get_simulation_dir('CWATM', self._get_experiment_id())
        return output_dir if result.success else None

    def _get_experiment_id(self) -> str:
        return self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default="run_1",
        )

    @staticmethod
    def _timestamp() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_expected_outputs(self) -> List[str]:
        return ['discharge_daily.nc']
