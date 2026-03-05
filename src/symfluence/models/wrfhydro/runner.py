# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WRF-Hydro Model Runner.

Executes the WRF-Hydro model using prepared input files.
"""

import re
import shutil
from pathlib import Path
from typing import List, Optional

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mpi_utils import find_mpirun
from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_runner('WRFHYDRO')
class WRFHydroRunner(BaseModelRunner):
    """Runner for the WRF-Hydro model (Noah-MP + routing)."""

    MODEL_NAME = "WRFHYDRO"

    def _setup_model_specific_paths(self) -> None:
        """Set up WRF-Hydro-specific paths."""
        self.settings_dir = self.project_dir / "settings" / "WRFHYDRO"
        self.routing_dir = self.settings_dir / "routing"

        self.wrfhydro_exe = self.get_model_executable(
            install_path_key='WRFHYDRO_INSTALL_PATH',
            default_install_subpath='installs/wrfhydro',
            default_exe_name='wrf_hydro.exe',
            typed_exe_accessor=lambda: (
                self.config.model.wrfhydro.exe
                if self.config.model and self.config.model.wrfhydro
                else None
            ),
            candidates=['bin', 'Run', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        """Build WRF-Hydro MPI execution command."""
        mpi_procs = self._get_config_value(
            lambda: self.config.compute.mpi_processes,
            default=1,
        )
        mpirun = find_mpirun(self.wrfhydro_exe)
        if mpirun is None:
            raise ModelExecutionError(
                "MPI launcher (mpirun/mpiexec) not found. "
                "Install OpenMPI or use the npm package which bundles it."
            )
        return [mpirun, '-np', str(mpi_procs), str(self.wrfhydro_exe)]

    def _prepare_run(self) -> None:
        """Copy namelists, symlink domain/routing/TBL files to output dir."""
        run_dir = self.output_dir

        # Copy HRLDAS and hydro namelists
        namelist_name = self._get_config_value(
            lambda: self.config.model.wrfhydro.namelist_file,
            default='namelist.hrldas',
        )
        hydro_name = self._get_config_value(
            lambda: self.config.model.wrfhydro.hydro_namelist,
            default='hydro.namelist',
        )
        for name in [namelist_name, hydro_name]:
            src = self.settings_dir / name
            if src.exists():
                shutil.copy2(src, run_dir / src.name)

        # Symlink domain/routing NetCDF files
        for nc_file in ['wrfinput_d01.nc', 'Fulldom_hires.nc', 'Route_Link.nc',
                        'soil_properties.nc', 'GWBUCKPARM.nc']:
            for search_dir in [self.settings_dir, self.routing_dir]:
                src_file = search_dir / nc_file
                if src_file.exists():
                    dest = run_dir / nc_file
                    if not (dest.exists() or dest.is_symlink()):
                        dest.symlink_to(src_file.resolve())
                    break

        # Symlink .TBL lookup tables — prefer settings_dir (staged by preprocessor)
        tbl_found = False
        for tbl_file in self.settings_dir.glob('*.TBL'):
            dest = run_dir / tbl_file.name
            if not (dest.exists() or dest.is_symlink()):
                dest.symlink_to(tbl_file.resolve())
            tbl_found = True

        # Fallback: try install directory if preprocessor didn't stage them
        if not tbl_found:
            wrfhydro_install = self.wrfhydro_exe.parent.parent
            tbl_dirs = [wrfhydro_install / 'Run', wrfhydro_install / 'run',
                        self.wrfhydro_exe.parent]
            for tbl_dir in tbl_dirs:
                if tbl_dir.exists():
                    for tbl_file in tbl_dir.glob('*.TBL'):
                        dest = run_dir / tbl_file.name
                        if not (dest.exists() or dest.is_symlink()):
                            dest.symlink_to(tbl_file.resolve())

        # Patch OUTDIR in namelist.hrldas to point to this run directory
        hrldas_copy = run_dir / namelist_name
        if hrldas_copy.exists():
            content = hrldas_copy.read_text(encoding='utf-8')
            content = re.sub(
                r"(OUTDIR\s*=\s*)'[^']*'",
                rf"\g<1>'{run_dir}'",
                content
            )
            hrldas_copy.write_text(content, encoding='utf-8')

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from output directory."""
        return self.output_dir

    def _get_run_timeout(self) -> int:
        """WRF-Hydro timeout from config."""
        return self._get_config_value(
            lambda: self.config.model.wrfhydro.timeout,
            default=7200,
        )
