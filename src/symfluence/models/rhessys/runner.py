# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Model Runner.

Executes the RHESSys model using prepared input files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.runners.add('RHESSys')
class RHESSysRunner(BaseModelRunner):
    """Runner for the RHESSys model with optional WMFire support."""

    MODEL_NAME = "RHESSys"

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)
        self.wmfire_enabled = self._check_wmfire_enabled()

    def _setup_model_specific_paths(self) -> None:
        """Set up RHESSys-specific paths."""
        self.setup_dir = self.project_dir / "settings" / self.MODEL_NAME
        self.worldfiles_dir = self.setup_dir / "worldfiles"
        self.tecfiles_dir = self.setup_dir / "tecfiles"
        self.defs_dir = self.setup_dir / "defs"
        self.routing_dir = self.setup_dir / "routing"
        self.fire_dir = self.setup_dir / "fire"

        self.rhessys_exe = self.get_model_executable(
            install_path_key='RHESSYS_INSTALL_PATH',
            default_install_subpath='installs/rhessys',
            default_exe_name='rhessys',
            typed_exe_accessor=lambda: (
                self.config.model.rhessys.installation.exe_name
                if self.config.model and self.config.model.rhessys
                   and self.config.model.rhessys.installation
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _check_wmfire_enabled(self) -> bool:
        """Check if WMFire fire spread is enabled."""
        try:
            if hasattr(self.config.model.rhessys, 'use_wmfire'):
                return self.config.model.rhessys.use_wmfire
            if hasattr(self.config.model.rhessys, 'use_vmfire'):
                return self.config.model.rhessys.use_vmfire
        except AttributeError:
            pass
        return False

    def _binary_supports_subsurfacegw(self) -> bool:
        """Check if the RHESSys binary was built with SYMFLUENCE patches."""
        try:
            # Run with a deliberately invalid worldfile — the binary will fail,
            # but if it chokes on '-subsurfacegw' specifically, it's unpatched.
            result = run_subprocess(
                [str(self.rhessys_exe), '-subsurfacegw', '-w', '/dev/null'],
                capture_output=True, text=True, timeout=5,
            )
            stderr = result.stderr + result.stdout
            if 'invalid' in stderr.lower() and 'subsurfacegw' in stderr.lower():
                self.logger.warning(
                    "RHESSys binary does not support -subsurfacegw "
                    "(rebuild with: symfluence binary install rhessys --patched)"
                )
                return False
            return True
        except Exception:  # noqa: BLE001 — model execution resilience
            return False

    def _get_wmfire_resolution(self) -> int:
        """Get WMFire grid resolution from config or use default."""
        try:
            if (hasattr(self.config.model.rhessys, 'wmfire') and
                self.config.model.rhessys.wmfire is not None):
                return self.config.model.rhessys.wmfire.grid_resolution
        except AttributeError:
            pass
        return 30

    def _build_run_command(self) -> Optional[List[str]]:
        """Build RHESSys command with all required flags."""
        cmd: List[str] = [str(self.rhessys_exe)]

        # World file and header
        world_file = self.worldfiles_dir / f"{self.config.domain.name}.world"
        header_file = self.worldfiles_dir / f"{self.config.domain.name}.world.hdr"
        if world_file.exists():
            cmd.extend(["-w", str(world_file)])
            if header_file.exists():
                cmd.extend(["-whdr", str(header_file)])

        # TEC file
        tec_file = self.tecfiles_dir / f"{self.config.domain.name}.tec"
        if tec_file.exists():
            cmd.extend(["-t", str(tec_file)])

        # Output prefix
        output_prefix = self.output_dir / "rhessys"
        cmd.extend(["-pre", str(output_prefix)])

        # Start and end dates
        start_date = pd.to_datetime(self._get_config_value(
            lambda: self.config.domain.time_start))
        end_date = pd.to_datetime(self._get_config_value(
            lambda: self.config.domain.time_end))
        cmd.extend(["-st", str(start_date.year), str(start_date.month),
                    str(start_date.day), "1"])
        cmd.extend(["-ed", str(end_date.year), str(end_date.month),
                    str(end_date.day), "1"])

        # Basin output
        cmd.extend(["-b"])

        # Grow mode (Farquhar photosynthesis)
        use_grow_mode = self._get_config_value(
            lambda: self.config.model.rhessys.use_grow_mode,
            dict_key='RHESSYS_USE_GROW_MODE',
        )
        if use_grow_mode:
            cmd.extend(["-g"])

        # Vegetation scaling flags
        cmd.extend(["-sv", "1.0", "1.0"])
        cmd.extend(["-svalt", "1.0", "1.0"])

        # Longwave radiation flag
        cmd.extend(["-longwaveevap"])

        # Fire spread if WMFire is enabled
        if self.wmfire_enabled:
            patch_grid = self.fire_dir / "patch_grid.txt"
            dem_grid = self.fire_dir / "dem_grid.txt"
            if patch_grid.exists() and dem_grid.exists():
                resolution = self._get_wmfire_resolution()
                cmd.extend(["-firespread", str(resolution),
                            str(patch_grid), str(dem_grid)])
            else:
                raise RuntimeError(
                    f"WMFire is enabled but fire grid files not found. "
                    f"Expected: {patch_grid} and {dem_grid}. "
                    f"Run preprocessing first or set RHESSYS_USE_WMFIRE: false"
                )

        # Routing
        routing_file = self.routing_dir / f"{self.config.domain.name}.routing"
        if routing_file.exists():
            cmd.extend(["-r", str(routing_file)])

        # Subgrid variability
        std_scale = self._get_config_value(lambda: self.config.model.rhessys.std_scale, dict_key='RHESSYS_STD_SCALE')
        if std_scale > 0:
            cmd.extend(["-stdev", str(std_scale)])

        # Groundwater store and subsurface-to-GW recharge pathway
        cmd.extend(["-gw", "1.0", "1.0"])

        # -subsurfacegw requires SYMFLUENCE-patched build (SYMFLUENCE_PATCHED=1).
        # Without the patch, RHESSys rejects the unknown flag and exits rc=1.
        use_subsurface_gw = self._get_config_value(
            lambda: self.config.model.rhessys.use_subsurface_gw,
            dict_key='RHESSYS_USE_SUBSURFACE_GW',
        )
        if use_subsurface_gw and self._binary_supports_subsurfacegw():
            cmd.extend(["-subsurfacegw"])

        return cmd

    def _get_run_cwd(self) -> Optional[Path]:
        """Run from output directory."""
        return self.output_dir

    def _get_run_environment(self) -> Optional[Dict[str, str]]:
        """Add library paths for RHESSys and WMFire."""
        lib_paths = []
        rhessys_bin_dir = self.rhessys_exe.parent
        rhessys_lib_dir = rhessys_bin_dir.parent / "lib"
        wmfire_lib_dir = Path(self.config.system.data_dir) / "installs" / "wmfire" / "lib"

        for lib_dir in [rhessys_bin_dir, rhessys_lib_dir, wmfire_lib_dir]:
            if lib_dir.exists():
                lib_paths.append(str(lib_dir))

        if not lib_paths:
            return None

        lib_path_str = ":".join(lib_paths)
        if sys.platform == "darwin":
            return {'DYLD_LIBRARY_PATH': lib_path_str}
        return {'LD_LIBRARY_PATH': lib_path_str}

    def _get_run_timeout(self) -> int:
        """RHESSys timeout."""
        return 7200
