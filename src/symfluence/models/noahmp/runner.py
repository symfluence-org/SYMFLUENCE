# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP Model Runner.

Drives the standalone noah-owp-modular executable (NOAA-OWP).
The model reads a Fortran namelist (``namelist.input``), ingests ASCII
forcing, and writes a single NetCDF output file.

Reference repository: https://github.com/NOAA-OWP/noah-owp-modular
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.runners.add('NOAHMP')
class NoahMPRunner(BaseModelRunner):
    """Runner for the standalone Noah-MP land surface model.

    noah-owp-modular is a 1-D column Fortran model invoked as:
        ./noah_owp_modular.exe
    from the directory containing ``namelist.input``.
    """

    MODEL_NAME = "NOAHMP"

    def _setup_model_specific_paths(self) -> None:
        self.settings_dir = self.project_dir / "settings" / "NOAHMP"

        install_path = self._get_config_value(
            lambda: self.config.model.noahmp.install_path,
            default='default',
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )
        self.noahmp_dir = resolve_install_dir(install_path, data_dir, 'installs/noah-owp-modular')

        exe_name = self._get_config_value(
            lambda: self.config.model.noahmp.exe,
            default='noah_owp_modular.exe',
        )
        self.executable = self.noahmp_dir / 'run' / exe_name

    def run(self, **kwargs) -> Optional[Path]:
        namelist_file = self._get_config_value(
            lambda: self.config.model.noahmp.namelist_file,
            default='namelist.input',
        )
        namelist_path = self.settings_dir / namelist_file
        if not namelist_path.exists():
            raise FileNotFoundError(f"Noah-MP namelist not found: {namelist_path}")

        run_dir = self.settings_dir
        exe_path = self.executable.resolve()
        if not exe_path.exists():
            raise FileNotFoundError(
                f"Noah-MP executable not found: {exe_path}. "
                "Run 'symfluence install NOAHMP' to build it."
            )

        env = dict(os.environ)
        ld_path = env.get('LD_LIBRARY_PATH', '')
        netcdf_lib = self._find_netcdf_lib()
        if netcdf_lib:
            env['LD_LIBRARY_PATH'] = f"{netcdf_lib}:{ld_path}" if ld_path else netcdf_lib

        cmd = [str(exe_path)]

        timeout = self._get_config_value(
            lambda: self.config.model.noahmp.timeout,
            default=7200,
        )

        log_dir = self.get_simulation_dir('NOAHMP', self._get_experiment_id()) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running Noah-MP for domain: {self.domain_name}")
        result = self.execute_subprocess(
            command=cmd,
            log_file=log_dir / f"noahmp_run_{self._timestamp()}.log",
            cwd=run_dir,
            env=env,
            timeout=timeout,
        )
        if not result.success:
            self.logger.error(f"Noah-MP simulation failed (rc={result.returncode})")

        output_dir = self.get_simulation_dir('NOAHMP', self._get_experiment_id())
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

    def _find_netcdf_lib(self) -> Optional[str]:
        """Locate NetCDF Fortran library path for LD_LIBRARY_PATH."""
        import shutil
        nf_config = shutil.which('nf-config')
        if nf_config:
            import subprocess
            try:
                result = run_subprocess(
                    [nf_config, '--prefix'],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    lib_dir = Path(result.stdout.strip()) / 'lib'
                    if lib_dir.is_dir():
                        return str(lib_dir)
            except (OSError, subprocess.SubprocessError):
                pass
        return None

    def _get_expected_outputs(self) -> List[str]:
        return ['output.nc']
