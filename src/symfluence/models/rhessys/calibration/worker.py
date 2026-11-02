# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Worker

Worker implementation for RHESSys model optimization.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker, run_worker_task
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.logging_utils import log_once
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_executable
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('RHESSys')
class RHESSysWorker(BaseWorker):
    """
    Worker for RHESSys model calibration.

    Handles parameter application, RHESSys execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RHESSys worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities
    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to RHESSys definition files.

        Args:
            params: Parameter values to apply
            settings_dir: RHESSys settings directory (contains defs/ subdirectory)
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Applying RHESSys parameters to {settings_dir}")

            # Separate worldfile params from def file params
            worldfile_param_names = {'precip_lapse_rate'}
            def_params = {}
            self._pending_worldfile_params = {}
            for pname, pval in params.items():
                if pname in worldfile_param_names:
                    self._pending_worldfile_params[pname] = pval
                else:
                    def_params[pname] = pval

            # The settings_dir should contain a 'defs' subdirectory
            defs_dir = settings_dir / 'defs'
            if not defs_dir.exists():
                self.logger.error(
                    f"RHESSys defs directory not found: {defs_dir}. "
                    "This indicates that RHESSys input files were not correctly copied to the worker directory. "
                    "Check that 'settings/RHESSys/defs' exists in the domain directory."
                )
                return False

            # Log available def files for debugging
            def_files = list(defs_dir.glob('*.def'))
            self.logger.debug(f"Found {len(def_files)} def files in {defs_dir}: {[f.name for f in def_files]}")

            # Update definition files with new parameters
            result = self._update_def_files(defs_dir, def_params)

            return result

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying RHESSys parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_def_files(self, defs_dir: Path, params: Dict[str, float]) -> bool:
        """
        Update RHESSys definition files with new parameter values.

        Args:
            defs_dir: Path to defs directory
            params: Parameters to update

        Returns:
            True if successful
        """
        import re

        # Parameters that live in the worldfile (not def files)
        # These are applied after the worldfile is copied in _build_command
        # Note: precip_lapse_rate handling is done in _build_command via -p flag
        _ = {'precip_lapse_rate'}  # Worldfile params - documented for reference

        # Mapping from parameter names to definition files
        # NOTE: gw_loss_coeff is in hillslope.def, sat_to_gw_coeff is in soil.def
        # (verified from RHESSys .params output files)
        PARAM_FILE_MAP = {
            'sat_to_gw_coeff': 'soil.def',      # Saturated zone to GW recharge coefficient
            'gw_loss_coeff': 'hillslope.def',   # Groundwater loss/baseflow coefficient (slow)
            'gw_loss_fast_coeff': 'hillslope.def',  # Fast groundwater loss coefficient
            'gw_loss_fast_threshold': 'hillslope.def',  # Threshold storage for fast flow (m)
            'n_routing_power': 'basin.def',
            'psi_air_entry': 'basin.def',
            'pore_size_index': 'basin.def',
            'porosity_0': 'soil.def',
            'porosity_decay': 'soil.def',
            'Ksat_0': 'soil.def',
            'Ksat_0_v': 'soil.def',
            'm': 'soil.def',
            'm_z': 'soil.def',
            'soil_depth': 'soil.def',
            'active_zone_z': 'soil.def',
            'snow_melt_Tcoef': 'soil.def',
            'snow_water_capacity': 'soil.def',
            'maximum_snow_energy_deficit': 'soil.def',
            'max_snow_temp': 'zone.def',
            'min_rain_temp': 'zone.def',
            'epc.max_lai': 'stratum.def',
            'epc.gl_smax': 'stratum.def',
            'epc.gl_c': 'stratum.def',
            'epc.vpd_open': 'stratum.def',
            'epc.vpd_close': 'stratum.def',
            'theta_mean_std_p1': 'soil.def',
            'theta_mean_std_p2': 'soil.def',
        }

        # Group parameters by file
        params_by_file: Dict[str, Dict[str, float]] = {}
        for param_name, value in params.items():
            def_file_name = PARAM_FILE_MAP.get(param_name)
            if def_file_name:
                if def_file_name not in params_by_file:
                    params_by_file[def_file_name] = {}
                params_by_file[def_file_name][param_name] = value
            else:
                log_once(self.logger, logging.WARNING, key=f'rhessys-unknown-param-{param_name}',
                         message=f"Unknown RHESSys parameter '{param_name}' - not in PARAM_FILE_MAP")

        total_params_updated = 0

        # Update each file
        for def_file_name, file_params in params_by_file.items():
            def_file = defs_dir / def_file_name
            if not def_file.exists():
                log_once(self.logger, logging.WARNING, key=f'rhessys-def-missing-{def_file}',
                         message=f"Definition file not found: {def_file}")
                continue

            with open(def_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Log first few lines of def file for debugging (first time only)
            if total_params_updated == 0:
                self.logger.debug(f"RHESSys def file sample ({def_file_name}): {lines[:3]}")

            updated_lines = []
            params_matched = set()

            for line in lines:
                updated = False
                for param_name, value in file_params.items():
                    # Match: value<whitespace>param_name (allow trailing comments)
                    # Pattern matches: optional start ^, float group(1), whitespace group(2), param_name group(3), remaining group(4)
                    pattern = rf'^([\d\.\-\+eE]+)(\s+)({re.escape(param_name)})(\s.*|)$'
                    match = re.match(pattern, line)
                    if match:
                        old_value = match.group(1)
                        new_line = f"{value:.6f}{match.group(2)}{match.group(3)}{match.group(4)}\n"
                        # Strip double newlines if they happen
                        new_line = new_line.replace('\n\n', '\n')
                        updated_lines.append(new_line)
                        params_matched.add(param_name)
                        self.logger.debug(f"  RHESSys param {param_name}: {old_value} -> {value:.6f}")
                        updated = True
                        break
                if not updated:
                    updated_lines.append(line)

            with open(def_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # VERIFICATION: Re-read file to confirm writes succeeded
            with open(def_file, 'r', encoding='utf-8') as f:
                verify_content = f.read()
            for param_name, value in file_params.items():
                expected_str = f"{value:.6f}"
                if expected_str not in verify_content:
                    self.logger.error(
                        f"WRITE VERIFICATION FAILED: {param_name}={expected_str} not found in {def_file_name}!"
                    )

            # Check for unmatched parameters - use INFO level to ensure visibility
            unmatched = set(file_params.keys()) - params_matched
            if unmatched:
                log_once(
                    self.logger, logging.INFO,
                    key=f'rhessys-params-unmatched-{def_file_name}',
                    message=(
                        f"RHESSys params NOT FOUND in {def_file_name}: {unmatched}. "
                        f"Looking for pattern: 'value<whitespace>param_name'"
                    ),
                )

            total_params_updated += len(params_matched)
            self.logger.debug(f"Updated {len(params_matched)}/{len(file_params)} params in {def_file_name}")

        # Warn if no parameters were actually updated
        if total_params_updated == 0:
            self.logger.error(
                "CRITICAL: No RHESSys parameters were updated! "
                "Calibration will not work. Check def file format and parameter names."
            )
            return False

        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run RHESSys model.

        Args:
            config: Configuration dictionary
            settings_dir: RHESSys settings directory
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            # Get paths
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f'domain_{domain_name}'
            rhessys_input_dir = project_dir / 'settings' / 'RHESSys'

            # Use sim_dir for output if provided
            rhessys_output_dir = Path(kwargs.get('sim_dir', output_dir))
            rhessys_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files to ensure fresh model run
            self._cleanup_stale_output(rhessys_output_dir)

            # Log cleanup status
            output_file = rhessys_output_dir / 'rhessys_basin.daily'
            self.logger.debug(f"After cleanup, output file exists: {output_file.exists()}")

            # Get executable
            rhessys_exe = self._get_rhessys_executable(config, data_dir)
            if not rhessys_exe.exists():
                self.logger.error(f"RHESSys executable not found: {rhessys_exe}")
                return False

            # Build command
            cmd = self._build_command(
                rhessys_exe,
                config,
                rhessys_input_dir,
                settings_dir,
                rhessys_output_dir
            )

            # Set library path for WMFire
            env = os.environ.copy()
            lib_paths = []
            rhessys_bin_dir = rhessys_exe.parent
            # libwmfire.so ships inside the RHESSys install tree (installs/rhessys/lib),
            # NOT under installs/wmfire/lib. Mirror RHESSysRunner._get_run_environment():
            # without this dir on the search path, the binary fails at dynamic-link time
            # ("libwmfire.so: cannot open shared object file") on every calibration eval.
            rhessys_lib_dir = rhessys_bin_dir.parent / "lib"
            wmfire_lib_dir = data_dir / "installs" / "wmfire" / "lib"

            for lib_dir in [rhessys_bin_dir, rhessys_lib_dir, wmfire_lib_dir]:
                if lib_dir.exists():
                    lib_paths.append(str(lib_dir))

            if lib_paths:
                lib_path_str = ":".join(lib_paths)
                if sys.platform == "darwin":
                    env["DYLD_LIBRARY_PATH"] = f"{lib_path_str}:{env.get('DYLD_LIBRARY_PATH', '')}"
                else:
                    env["LD_LIBRARY_PATH"] = f"{lib_path_str}:{env.get('LD_LIBRARY_PATH', '')}"

            # Run RHESSys with timeout to catch parameter combinations that cause hangs
            import time as time_module
            # Normal runs complete in <1s; 120s timeout catches problematic parameters
            # that cause numerical instability or runaway iteration in RHESSys.
            timeout_seconds = config.get('RHESSYS_CALIBRATION_TIMEOUT', 120)
            run_start = time_module.time()

            # Write stdout/stderr to files to avoid pipe buffer issues
            stdout_file = rhessys_output_dir / 'rhessys_stdout.log'
            stderr_file = rhessys_output_dir / 'rhessys_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = run_subprocess(
                        cmd,
                        cwd=str(rhessys_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout_seconds
                    )
            except subprocess.TimeoutExpired:
                params = kwargs.get('params', {})
                param_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
                self.logger.warning(
                    f"RHESSys timed out after {timeout_seconds}s - skipping trial. "
                    f"Parameters: {param_str}"
                )
                return False
            run_time = time_module.time() - run_start
            self.logger.debug(f"RHESSys completed in {run_time:.2f}s")

            # Read stderr for logging if needed
            result.stderr = ""
            if stderr_file.exists():
                try:
                    result.stderr = stderr_file.read_text(encoding='utf-8')
                except Exception:  # noqa: BLE001 — calibration resilience
                    pass

            # Log any stderr output (even on success) for debugging
            if result.stderr:
                stderr_text = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                self.logger.debug(f"RHESSys stderr: {stderr_text[:500]}")

            if result.returncode != 0:
                stderr_text = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                self._last_error = f"RHESSys failed: {stderr_text[-500:]}"
                self.logger.error(self._last_error)
                return False

            # Verify output exists and log details
            output_file = rhessys_output_dir / 'rhessys_basin.daily'
            if not output_file.exists():
                self._last_error = "No basin output file produced"
                self.logger.error(f"Expected output at {output_file}")
                return False

            # Log output file size and modification time to verify fresh output
            file_stat = output_file.stat()
            file_size = file_stat.st_size
            mod_time = file_stat.st_mtime
            self.logger.debug(
                f"RHESSys output: {output_file.name}, size={file_size} bytes, "
                f"mtime={mod_time:.3f}, run_time={run_time:.2f}s"
            )

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running RHESSys: {e}", exc_info=True)
            return False

    def _binary_supports_subsurfacegw(self, exe: Path) -> bool:
        """Check if the RHESSys binary was built with SYMFLUENCE patches."""
        cache_key = '_subsurfacegw_supported'
        if hasattr(self, cache_key):
            return getattr(self, cache_key)
        try:
            # Put the install lib dir on the library path so a binary linked
            # against libwmfire (the patched build) can load during this probe.
            # Without it the probe fails with "libwmfire.so: cannot open shared
            # object file", the flag is silently dropped, and calibration runs
            # WITHOUT the subsurface-to-GW recharge pathway — silently capping
            # skill (observed: KGE 0.365 instead of ~0.83, matching an unpatched
            # binary). run_model already sets this path; the probe must too.
            env = os.environ.copy()
            lib_dir = exe.parent.parent / 'lib'
            if lib_dir.is_dir():
                for _var in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
                    env[_var] = str(lib_dir) + (
                        os.pathsep + env[_var] if env.get(_var) else '')
            result = run_subprocess(
                [str(exe), '-subsurfacegw', '-w', '/dev/null'],
                capture_output=True, text=True, timeout=5, check=False, env=env,
            )
            stderr = result.stderr + result.stdout
            # A binary that could not even load its shared libs tells us nothing
            # about flag support — do NOT treat that as "supported" (it would pass
            # the flag to a binary that then also fails at run time).
            if 'cannot open shared object' in stderr.lower() or 'error while loading' in stderr.lower():
                raise ModelExecutionError(
                    f"RHESSys probe could not load libraries: {stderr.strip()[:200]}"
                )
            supported = not ('invalid' in stderr.lower() and 'subsurfacegw' in stderr.lower())
            if not supported:
                self.logger.debug(
                    "RHESSys binary does not support -subsurfacegw; "
                    "calibrating without subsurface-to-GW recharge pathway"
                )
            else:
                self.logger.debug("RHESSys binary supports -subsurfacegw (SYMFLUENCE-patched)")
            setattr(self, cache_key, supported)
            return supported
        except Exception:  # noqa: BLE001 — calibration resilience
            setattr(self, cache_key, False)
            return False

    def _get_rhessys_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get RHESSys executable path."""
        install_path = resolve_config_value(config, lambda: config.model.rhessys.install_path, 'default', 'RHESSYS_INSTALL_PATH')
        exe_name = resolve_config_value(config, lambda: config.model.rhessys.exe, 'rhessys', 'RHESSYS_EXE')
        # A directory install_path gets exe_name appended; otherwise it is the full path
        exe = resolve_install_executable(install_path, data_dir, 'installs/rhessys/bin', exe_name)
        # Windows builds produce <name>.exe even when the config declares a bare
        # name; fall back to the .exe variant when the bare path is absent (else
        # every eval reports "executable not found" -> 100% crash rate).
        if exe.suffix != '.exe':
            exe_win = exe.with_suffix('.exe')
            if exe_win.exists():
                return exe_win
        return exe

    def _build_command(
        self,
        exe: Path,
        config: Dict[str, Any],
        rhessys_input_dir: Path,
        settings_dir: Path,
        output_dir: Path
    ) -> list:
        """Build RHESSys command line."""
        domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')

        # Use worker-specific defs directory if available
        worker_defs_dir = settings_dir / 'defs'

        # Strategy: Copy world file to settings_dir and ensure header matches.
        # This ensures RHESSys finds the modified header file (which points to modified defs)
        # by looking for <world_file>.hdr in the same directory, avoiding unsupported flags.
        original_world = rhessys_input_dir / 'worldfiles' / f'{domain_name}.world'
        worker_world = settings_dir / f'{domain_name}.world'
        original_hdr = rhessys_input_dir / 'worldfiles' / f'{domain_name}.world.hdr'
        worker_hdr = settings_dir / f'{domain_name}.world.hdr'

        if worker_defs_dir.exists():
            # ALWAYS copy world file to ensure initial conditions are up-to-date
            # (previous versions only copied if not exists, causing stale initial conditions)
            if original_world.exists():
                import shutil
                shutil.copy2(original_world, worker_world)
                self.logger.debug(f"Copied/updated world file to {worker_world}")

                # Apply any worldfile-level calibration parameters (e.g. precip_lapse_rate)
                if hasattr(self, '_pending_worldfile_params') and self._pending_worldfile_params:
                    self._apply_worldfile_params(worker_world, self._pending_worldfile_params)

            # ALWAYS recreate the header to ensure paths are correct
            # (previous versions only created if not exists, which could cause stale paths)
            if original_hdr.exists():
                # Copy and modify header to point to worker defs
                self._create_worker_header(original_hdr, worker_hdr, worker_defs_dir, rhessys_input_dir)
                self.logger.debug(f"Created/updated worker header: {worker_hdr}")
        else:
            log_once(self.logger, logging.WARNING, key=f'rhessys-defs-dir-missing-{worker_defs_dir}',
                     message=f"Worker defs directory does not exist: {worker_defs_dir}")

        # Parse dates
        start_str = resolve_config_value(config, lambda: config.domain.time_start, '2004-01-01 01:00', 'EXPERIMENT_TIME_START')
        end_str = resolve_config_value(config, lambda: config.domain.time_end, '2004-12-31 23:00', 'EXPERIMENT_TIME_END')
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        # Use worker world if it was created successfully, otherwise fallback to original
        world_to_use = worker_world if worker_world.exists() else original_world

        tecfile = rhessys_input_dir / 'tecfiles' / f'{domain_name}.tec'
        routing = rhessys_input_dir / 'routing' / f'{domain_name}.routing'

        # Get scaling parameters from config if available (default to None to match Runner)
        s1 = config.get('RHESSYS_S1')
        s2 = config.get('RHESSYS_S2')
        s3 = config.get('RHESSYS_S3')
        gw1 = config.get('RHESSYS_GW1')
        gw2 = config.get('RHESSYS_GW2')

        # Log which world file is being used and verify header exists
        expected_hdr = Path(str(world_to_use) + '.hdr')
        self.logger.debug(
            f"RHESSys command setup: world={world_to_use}, "
            f"header_exists={expected_hdr.exists()}, "
            f"worker_defs_exists={worker_defs_dir.exists()}"
        )
        if expected_hdr.exists():
            # Log first 2 lines of header to verify paths
            with open(expected_hdr, 'r', encoding='utf-8') as f:
                hdr_content = f.read()
                hdr_lines = hdr_content.split('\n')[:4]
            self.logger.debug(f"Header file content (first 4 lines): {hdr_lines}")

            # CRITICAL: Verify header points to worker defs, not original defs.
            # The header is written with forward-slash def paths (see
            # _create_worker_header), so compare against the POSIX form.
            worker_defs_str = worker_defs_dir.as_posix()
            if worker_defs_str in hdr_content:
                self.logger.debug(f"Header CORRECTLY points to worker defs: {worker_defs_str}")
            else:
                self.logger.error(
                    f"HEADER MISMATCH! Expected defs path '{worker_defs_str}' not found in header! "
                    f"This means RHESSys will read WRONG def files!"
                )

        # CRITICAL DIAGNOSTIC: Verify actual def file content before RHESSys runs
        soil_def = worker_defs_dir / 'soil.def'
        if soil_def.exists():
            with open(soil_def, 'r', encoding='utf-8') as f:
                soil_lines = f.readlines()
            # Show lines 5-12 which should contain Ksat_0 and m parameters
            self.logger.debug("BEFORE RUN soil.def lines 5-12:")
            for i, line in enumerate(soil_lines[4:12], start=5):
                self.logger.debug(f"  Line {i}: {repr(line)}")

        # Build header file path - MUST explicitly specify with -whdr flag
        # RHESSys doesn't always auto-detect <worldfile>.hdr
        header_file = Path(str(world_to_use) + '.hdr')

        cmd = [
            str(exe),
            '-w', str(world_to_use),
            '-whdr', str(header_file),  # CRITICAL: Explicitly specify header file!
            '-t', str(tecfile),
            '-r', str(routing),
            '-st', str(start_date.year), str(start_date.month), str(start_date.day), '1',
            '-ed', str(end_date.year), str(end_date.month), str(end_date.day), '1',
            '-pre', 'rhessys',
        ]

        # Basin output without grow mode. The Jarvis conductance model (non-grow)
        # uses gs = gl_smax * f(vpd) * f(psi) * LAI, giving the optimizer direct
        # control over ET magnitude through gl_smax calibration. Grow mode's Farquhar
        # model constrains gs by nitrogen/CO2, making gl_smax ineffective as a lever.
        cmd.extend(['-b'])

        if gw1 is not None and gw2 is not None:
            cmd.extend(['-gw', str(gw1), str(gw2)])

        if s1 is not None and s2 is not None and s3 is not None:
            cmd.extend(['-s', str(s1), str(s2), str(s3)])

        # Vegetation scaling flags — required for correct model physics.
        # Even at 1.0, these activate RHESSys internal code paths that affect
        # stomatal conductance and canopy processes.
        cmd.extend(['-sv', '1.0', '1.0'])
        cmd.extend(['-svalt', '1.0', '1.0'])

        # Longwave radiation in evaporation — must match runner for consistent
        # water balance between calibration and production runs.
        cmd.extend(['-longwaveevap'])

        # Fire spread if WMFire is enabled
        wmfire_enabled = resolve_config_value(config, lambda: config.model.rhessys.use_wmfire, False, 'RHESSYS_USE_WMFIRE')
        if wmfire_enabled:
            fire_dir = rhessys_input_dir / "fire"
            patch_grid = fire_dir / "patch_grid.txt"
            dem_grid = fire_dir / "dem_grid.txt"
            if patch_grid.exists() and dem_grid.exists():
                resolution = resolve_config_value(config, lambda: config.model.rhessys.wmfire.grid_resolution, 30, 'WMFIRE_GRID_RESOLUTION')
                cmd.extend(["-firespread", str(resolution), str(patch_grid), str(dem_grid)])
                self.logger.debug(f"WMFire fire spread enabled: {resolution}m resolution")
            else:
                log_once(
                    self.logger, logging.WARNING,
                    key=f'rhessys-wmfire-grids-missing-{fire_dir}',
                    message=(
                        f"WMFire is enabled but fire grid files not found at {fire_dir}. "
                        "Fire spread will be disabled for calibration."
                    ),
                )

        # Subgrid variability for lumped mode (-stdev enables variance-based return flow)
        # This uses normal distribution around mean sat_deficit to generate partial saturation
        std_scale = resolve_config_value(config, lambda: config.model.rhessys.std_scale, 1.0, 'RHESSYS_STD_SCALE')
        if std_scale > 0:
            cmd.extend(["-stdev", str(std_scale)])
            self.logger.debug(f"Subgrid variability enabled with std_scale={std_scale}")

        # Subsurface-to-GW recharge pathway (SYMFLUENCE Patch 1)
        # Requires SYMFLUENCE-patched build (symfluence binary install rhessys --patched)
        # Same switch as the runner (RHESSYS_USE_SUBSURFACE_GW); previously the
        # worker always added the flag when the binary supported it.
        use_subsurface_gw = resolve_config_value(
            config, lambda: config.model.rhessys.use_subsurface_gw, True, 'RHESSYS_USE_SUBSURFACE_GW')
        if use_subsurface_gw and self._binary_supports_subsurfacegw(exe):
            cmd.extend(["-subsurfacegw"])

        return cmd

    def _apply_worldfile_params(self, world_file: Path, params: Dict[str, float]):
        """Apply calibration parameters to the worldfile (e.g. precip_lapse_rate)."""
        import re
        with open(world_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            replaced = False
            for param_name, value in params.items():
                pattern = rf'^(\s*)([\d\.\-\+eE]+)(\s+)({re.escape(param_name)})(\s*\n?)$'
                match = re.match(pattern, line)
                if match:
                    new_line = f"{match.group(1)}{value:.8f}{match.group(3)}{match.group(4)}{match.group(5)}"
                    updated_lines.append(new_line)
                    self.logger.debug(f"Worldfile param {param_name}: {match.group(2)} -> {value:.8f}")
                    replaced = True
                    break
            if not replaced:
                updated_lines.append(line)

        with open(world_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

    def _create_worker_header(
        self,
        original_hdr: Path,
        worker_hdr: Path,
        worker_defs_dir: Path,
        rhessys_input_dir: Path
    ):
        """Create a worker-specific header file pointing to worker defs.

        Uses line-by-line replacement to handle any .def file path, regardless
        of how the original path was normalized or constructed.
        """
        with open(original_hdr, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        worker_defs_str = str(worker_defs_dir)
        new_lines = []
        replacements = 0

        for line in lines:
            stripped = line.strip()
            # Match lines that are file paths ending in .def. Accept either path
            # separator: on native Windows the preprocessor writes the header
            # with backslash paths (C:\...\defs\basin.def), so a '/'-only check
            # would never fire and the header would keep pointing at the original
            # (non-worker) defs -- causing a HEADER MISMATCH and RHESSys reading
            # the wrong def files.
            if stripped.endswith('.def') and ('/' in stripped or '\\' in stripped):
                # Extract just the filename (e.g. basin.def). PureWindowsPath
                # splits on both separators so a POSIX or Windows path both work.
                def_filename = PureWindowsPath(stripped).name
                # Emit the def path with FORWARD slashes even on Windows. RHESSys
                # (construct_*_defaults.c) derives the "<pre>_<name>.params" output
                # filename by tokenising this path on '/' only; a backslash path is
                # never split, so the whole "C:\...\soil.def" becomes the basename
                # and RHESSys tries to open "rhessys_C:\...soil.params" -- an
                # invalid name (embedded ':' / '\\') that aborts the run. A
                # forward-slash path tokenises correctly and Windows fopen() still
                # opens it.
                new_path = (worker_defs_dir / def_filename).as_posix()
                new_lines.append(new_path + '\n')
                replacements += 1
                self.logger.debug(f"Header path: {stripped} -> {new_path}")
            else:
                new_lines.append(line)

        if replacements == 0:
            log_once(
                self.logger, logging.WARNING,
                key='rhessys-header-no-def-paths',
                message=(
                    f"No .def path replacements made in header file. "
                    f"Header content: '{(''.join(lines))[:500]}...'"
                ),
            )
        else:
            self.logger.debug(
                f"Header: replaced {replacements} .def paths to point to {worker_defs_str}"
            )

        with open(worker_hdr, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        self.logger.debug(f"Created worker header at {worker_hdr}")

    def _cleanup_stale_output(self, output_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Remove stale RHESSys output files before a new model run.

        This ensures that each calibration iteration produces fresh output
        and prevents reusing stale results from previous runs.

        Args:
            output_dir: Directory containing RHESSys output files
            config: Optional config dict to get experiment_id for nested path cleanup
        """
        # RHESSys output file patterns to clean up
        output_patterns = [
            'rhessys_basin.daily',
            'rhessys_basin.hourly',
            'rhessys_basin.monthly',
            'rhessys_basin.yearly',
        ]

        files_removed = 0

        # Clean at the direct output directory level
        for pattern in output_patterns:
            direct_file = output_dir / pattern
            if direct_file.exists():
                try:
                    direct_file.unlink()
                    files_removed += 1
                    self.logger.debug(f"Removed stale output: {direct_file}")
                except (OSError, IOError) as e:
                    log_once(self.logger, logging.WARNING, key=f'rhessys-stale-file-{direct_file}',
                                  message=f"Could not remove stale file {direct_file}: {e}")

        # Clean the RHESSys-specific output file with wildcard
        for file_path in output_dir.glob('rhessys_*.daily'):
            try:
                file_path.unlink()
                files_removed += 1
            except (OSError, IOError) as e:
                log_once(self.logger, logging.WARNING, key=f'rhessys-stale-file-{file_path}',
                          message=f"Could not remove stale file {file_path}: {e}")

        for file_path in output_dir.glob('rhessys_*.hourly'):
            try:
                file_path.unlink()
                files_removed += 1
            except (OSError, IOError) as e:
                log_once(self.logger, logging.WARNING, key=f'rhessys-stale-file-{file_path}',
                          message=f"Could not remove stale file {file_path}: {e}")

        if files_removed > 0:
            self.logger.debug(f"Cleaned up {files_removed} stale RHESSys output files from {output_dir}")

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from RHESSys output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Get simulation directory
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Read RHESSys output - robustly locate the file
            # RHESSys produces output in a nested structure: [sim_dir]/simulations/[experiment_id]/RHESSys/
            experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, 'run_1', 'EXPERIMENT_ID')
            possible_paths = [
                sim_dir / 'rhessys_basin.daily',
                sim_dir / 'simulations' / experiment_id / 'RHESSys' / 'rhessys_basin.daily',
            ]

            sim_file = None
            for path in possible_paths:
                if path.exists():
                    sim_file = path
                    break

            if not sim_file:
                # Last resort: try recursive glob
                found = list(sim_dir.glob('**/rhessys_basin.daily'))
                if found:
                    sim_file = found[0]

            if not sim_file:
                self.logger.error(f"rhessys_basin.daily not found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'rhessys_basin.daily not found'}

            sim_df = pd.read_csv(sim_file, sep=r'\s+', header=0)
            self.logger.debug(f"Read {len(sim_df)} rows from {sim_file}")

            # Get streamflow in mm/day
            # Use unrouted 'streamflow' for calibration. The 'routedstreamflow'
            # column applies a gamma-function routing that artificially damps
            # peak flows in single-patch (lumped) models where there is no
            # spatial network to route through. This causes low variability
            # ratio (alpha) in KGE. For distributed models, routedstreamflow
            # would be more appropriate but for lumped calibration, the raw
            # hillslope streamflow is the correct signal.
            if 'streamflow' in sim_df.columns:
                streamflow_mm = sim_df['streamflow'].values
            elif 'routedstreamflow' in sim_df.columns:
                streamflow_mm = sim_df['routedstreamflow'].values
            else:
                self.logger.error("No streamflow column found in basin.daily")
                return {'kge': self.penalty_score, 'error': 'No streamflow column'}

            # Convert to m³/s using catchment area from shared utility
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f'domain_{domain_name}'

            # Cache catchment area to avoid repeated shapefile reads
            cache_key = f"_cached_area_{domain_name}"
            if hasattr(self, cache_key):
                area_km2 = getattr(self, cache_key)
            else:
                area_km2 = self._streamflow_metrics.get_catchment_area(
                    config, project_dir, domain_name, source='shapefile'
                )
                setattr(self, cache_key, area_km2)
                self.logger.debug(f"Cached catchment area: {area_km2:.2f} km²")
            area_m2 = area_km2 * 1e6  # Convert km² to m²

            # Q (m³/s) = Q (mm/day) * area (m²) / 86400 / 1000
            streamflow_m3s = streamflow_mm * area_m2 / 86400 / 1000

            # Check for NaN values in simulation
            nan_count = pd.isna(streamflow_m3s).sum()
            if nan_count > 0:
                log_once(
                    self.logger, logging.WARNING,
                    key='rhessys-output-nan',
                    message=f"RHESSys output contains {nan_count} NaN values out of {len(streamflow_m3s)} timesteps",
                )

            # Check for zero discharge (model didn't produce runoff)
            if streamflow_m3s.sum() == 0:
                log_once(
                    self.logger, logging.WARNING,
                    key='rhessys-zero-streamflow',
                    message="RHESSys simulation produced zero streamflow - check model parameters",
                )
                return {'kge': self.penalty_score, 'error': 'Zero streamflow from model'}

            # Create dates
            sim_dates = pd.to_datetime(
                sim_df.apply(
                    lambda r: f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d}",
                    axis=1
                )
            )
            sim_series = pd.Series(streamflow_m3s, index=sim_dates)

            # Load observations - cache to avoid repeated CSV reads
            obs_cache_key = f"_cached_obs_{domain_name}"
            if hasattr(self, obs_cache_key):
                obs_values, obs_index = getattr(self, obs_cache_key)
            else:
                obs_values, obs_index = self._streamflow_metrics.load_observations(
                    config, project_dir, domain_name, resample_freq='D'
                )
                if obs_values is not None:
                    setattr(self, obs_cache_key, (obs_values, obs_index))
                    self.logger.debug(f"Cached {len(obs_values)} observations")
            if obs_values is None:
                self.logger.error("Observations not found for metric calculation")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Align and calculate metrics
            try:
                # Parse calibration period if specified
                calib_period_tuple = None
                calib_period_str = resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD')
                if calib_period_str:
                    try:
                        start_str, end_str = calib_period_str.split(',')
                        calib_period_tuple = (start_str.strip(), end_str.strip())
                    except (ValueError, IndexError) as e:
                        log_once(self.logger, logging.WARNING, key='rhessys-calib-period-parse',
                                 message=f"Could not parse calibration period '{calib_period_str}': {e}")

                # Auto-exclude spinup: if no calibration period is set, skip the
                # first year of simulation to avoid initial condition transients
                if calib_period_tuple is None and len(sim_dates) > 365:
                    spinup_days = int(config.get('RHESSYS_SPINUP_DAYS', 365))
                    spinup_end = sim_dates.min() + pd.Timedelta(days=spinup_days)
                    calib_period_tuple = (
                        spinup_end.strftime('%Y-%m-%d'),
                        sim_dates.max().strftime('%Y-%m-%d')
                    )
                    log_once(
                        self.logger, logging.INFO,
                        key='rhessys-auto-spinup',
                        message=(
                            f"No CALIBRATION_PERIOD set; auto-excluding {spinup_days}-day spinup. "
                            f"Evaluating from {calib_period_tuple[0]} to {calib_period_tuple[1]}"
                        ),
                    )

                # Let StreamflowMetrics handle alignment and period filtering
                obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                    sim_series, obs_series, calibration_period=calib_period_tuple
                )

                # Log streamflow statistics for diagnostics
                self.logger.debug(
                    f"RHESSys metrics: n={len(obs_aligned)}, "
                    f"obs_mean={np.nanmean(obs_aligned):.2f}, obs_std={np.nanstd(obs_aligned):.2f}, "
                    f"sim_mean={np.nanmean(sim_aligned):.2f}, sim_std={np.nanstd(sim_aligned):.2f}"
                )

                results = self._streamflow_metrics.calculate_metrics(
                    obs_aligned, sim_aligned, metrics=['kge', 'nse']
                )
                return results

            except ValueError as e:
                self.logger.error(f"Alignment error: {e}")
                return {'kge': self.penalty_score, 'error': str(e)}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating RHESSys metrics: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_rhessys_parameters_worker(task_data)


def _evaluate_rhessys_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        RHESSysWorker,
        task_data,
        extra_env={'HDF5_DISABLE_VERSION_CHECK': '1'},
        start_jitter=(0.1, 0.8),
        error_prefix='Critical RHESSys worker exception',
    )
