# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB Model Runner."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.registries import R
from symfluence.models.lisflood.runner import _find_pcraster_site_packages


@R.runners.add('PCRGLOBWB')
class PCRGLOBWBRunner(BaseModelRunner):
    """Runner for the PCR-GLOBWB 2.0 model.

    PCR-GLOBWB is a Python/PCRaster-based model. If PCRaster is not in
    the current environment, the runner searches for a conda environment
    with PCRaster and injects its site-packages via PYTHONPATH (same
    strategy as LISFLOOD).
    """

    MODEL_NAME = "PCRGLOBWB"

    def _setup_model_specific_paths(self) -> None:
        self.settings_dir = self.project_dir / "settings" / "PCRGLOBWB"

        install_path = self._get_config_value(
            lambda: self.config.model.pcrglobwb.install_path,
            default='default',
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )
        self.pcrglobwb_dir = resolve_install_dir(install_path, data_dir, 'installs/pcrglobwb')

        exe_name = self._get_config_value(
            lambda: self.config.model.pcrglobwb.exe,
            default='deterministic_runner.py',
        )
        self.runner_script = self.pcrglobwb_dir / exe_name

    def run(self, **kwargs) -> Optional[Path]:
        config_file = self._get_config_value(
            lambda: self.config.model.pcrglobwb.config_file,
            default='setup.ini',
        )
        ini_path = self.settings_dir / config_file
        if not ini_path.exists():
            raise FileNotFoundError(f"PCR-GLOBWB config not found: {ini_path}")

        env = dict(os.environ)

        # Inject PCR-GLOBWB source into PYTHONPATH
        pcrglobwb_pythonpath = str(self.pcrglobwb_dir)

        # Discover PCRaster (reuses LISFLOOD's 5-tier search)
        pcraster_site = _find_pcraster_site_packages()
        if pcraster_site:
            env["PYTHONPATH"] = f"{pcraster_site}:{pcrglobwb_pythonpath}"
            pcraster_prefix = str(Path(pcraster_site).parent.parent.parent)
            env["CONDA_PREFIX"] = pcraster_prefix
            # PCR-GLOBWB calls CLI tools like `mapattr` — add pcraster bin to PATH
            pcraster_bin = str(Path(pcraster_prefix) / "bin")
            env["PATH"] = f"{pcraster_bin}:{env.get('PATH', '')}"
        else:
            env["PYTHONPATH"] = pcrglobwb_pythonpath

        cmd = [sys.executable, str(self.runner_script), str(ini_path)]

        timeout = self._get_config_value(
            lambda: self.config.model.pcrglobwb.timeout,
            default=14400,
        )

        log_dir = self.get_simulation_dir('PCRGLOBWB', self._get_experiment_id()) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running PCRGLOBWB for domain: {self.domain_name}")
        result = self.execute_subprocess(
            command=cmd,
            log_file=log_dir / f"pcrglobwb_run_{self._timestamp()}.log",
            cwd=self.settings_dir,
            env=env,
            timeout=timeout,
        )
        if not result.success:
            self.logger.error(f"PCRGLOBWB simulation failed (rc={result.returncode})")

        output_dir = self.get_simulation_dir('PCRGLOBWB', self._get_experiment_id())
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
        return ['discharge_dailyTot_output.nc']
