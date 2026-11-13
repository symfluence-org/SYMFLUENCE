# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Worker

Worker implementation for PRMS model optimization.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    run_worker_task,
)
from symfluence.core.calibration.workers.streamflow_worker import StreamflowWorker
from symfluence.core.exceptions import ConfigurationError, ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_executable
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('PRMS')
class PRMSWorker(StreamflowWorker):
    """
    Worker for PRMS model calibration.

    Handles parameter application to PRMS parameter files, PRMS execution,
    and metric calculation from statvar output.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to PRMS parameter file.

        Copies fresh parameter files from the original settings/PRMS location,
        then updates values using the parameter manager.

        Args:
            params: Parameter values to apply
            settings_dir: PRMS settings directory containing .dat files
            **kwargs: Additional arguments (config)

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying PRMS parameters to {settings_dir}")

            config = kwargs.get('config', self.config) or {}
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'PRMS'

            if original_settings_dir.exists() and original_settings_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_settings_dir.glob('*.dat'):
                    shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(f"Copied PRMS files from {original_settings_dir} to {settings_dir}")
                # Rewrite absolute paths in control.dat to point to worker dir
                self._update_control_dat_paths(settings_dir, config)
            elif not settings_dir.exists():
                self.logger.error(f"PRMS settings directory not found: {settings_dir} "
                                  f"(original also missing: {original_settings_dir})")
                return False

            # Find parameter file
            param_file_name = resolve_config_value(
                config,
                lambda: config.model.prms.parameter_file,
                dict_key='PRMS_PARAMETER_FILE',
            )
            param_file = settings_dir / param_file_name
            if not param_file.exists():
                # Try any .dat file that isn't control or data
                for f in settings_dir.glob('*.dat'):
                    if 'control' not in f.name and 'data' not in f.name:
                        param_file = f
                        break

            if not param_file.exists():
                self.logger.error(f"PRMS parameter file not found in {settings_dir}")
                return False

            params = self._enforce_constraints(params)
            return self._update_parameter_file(param_file, params)

        except ConfigurationError:
            raise
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying PRMS parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_parameter_file(self, param_file: Path, params: Dict[str, float]) -> bool:
        """
        Update PRMS parameter file with new values.

        Args:
            param_file: Path to params.dat
            params: Parameters to update

        Returns:
            True if successful

        Raises:
            ConfigurationError: a parameter has no block in the file (e.g.
                ``soil_rechr_max`` under PRMS5, ``rain_cbh_adj`` in station
                mode); it would otherwise be calibrated without any effect.
        """
        try:
            from ..parameter_file import update_parameter_file
            update_parameter_file(param_file, params)
            return True

        except ConfigurationError:
            raise
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating PRMS parameter file: {e}", exc_info=True)
            return False

    @staticmethod
    def _enforce_constraints(params: Dict[str, float]) -> Dict[str, float]:
        """Enforce PRMS cross-parameter constraints.

        PRMS crashes (rc=255) when:
        - soil_rechr_max > soil_moist_max (recharge zone exceeds total storage)
        - tmax_allsnow >= tmax_allrain (snow threshold above rain threshold)
        """
        p = dict(params)
        if 'soil_rechr_max' in p and 'soil_moist_max' in p:
            if p['soil_rechr_max'] > p['soil_moist_max']:
                p['soil_rechr_max'] = p['soil_moist_max'] * 0.9
        if 'tmax_allsnow' in p and 'tmax_allrain' in p:
            if p['tmax_allsnow'] >= p['tmax_allrain']:
                p['tmax_allsnow'] = p['tmax_allrain'] - 2.0
        if 'soil2gw_max' in p and 'soil_moist_max' in p:
            if p['soil2gw_max'] > p['soil_moist_max']:
                p['soil2gw_max'] = p['soil_moist_max'] * 0.5
        return p

    def _update_control_dat_paths(self, settings_dir: Path, config: Dict[str, Any]) -> None:
        """
        Rewrite param_file path in control.dat to point to the worker's directory.

        Only param_file needs rewriting — data_file points to the shared
        forcing directory (unchanged between iterations) and output paths
        (stat_var_file, csv_output_file) use relative filenames.

        Args:
            settings_dir: Worker's PRMS settings directory
            config: Configuration dictionary
        """
        control_file = resolve_config_value(
            config,
            lambda: config.model.prms.control_file,
            dict_key='PRMS_CONTROL_FILE',
        )
        control_path = settings_dir / control_file
        if not control_path.exists():
            return

        try:
            content = control_path.read_bytes().decode('utf-8', errors='replace')
            blocks = content.split('####\n')

            updated_blocks = []
            for block in blocks:
                lines = block.strip().split('\n')
                if lines and lines[0].strip() == 'param_file':
                    if len(lines) >= 4:
                        old_path = Path(lines[-1].strip())
                        new_path = settings_dir / old_path.name
                        lines[-1] = str(new_path)
                        self.logger.debug(f"Updated control.dat param_file: {old_path.name} -> {new_path}")
                    updated_block = '\n'.join(lines) + '\n'
                else:
                    updated_block = block
                updated_blocks.append(updated_block)

            control_path.write_text('####\n'.join(updated_blocks), encoding='utf-8')
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Could not update control.dat paths: {e}", exc_info=True)

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run PRMS model.

        PRMS is executed with: prms -C control.dat
        from the settings directory.

        Args:
            config: Configuration dictionary
            settings_dir: PRMS settings directory (with .dat files)
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))

            prms_output_dir = Path(kwargs.get('sim_dir', output_dir))
            prms_output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            prms_exe = self._get_prms_executable(config, data_dir)
            if not prms_exe.exists():
                self.logger.error(f"PRMS executable not found: {prms_exe}")
                return False

            # Get control file
            control_file = resolve_config_value(
                config,
                lambda: config.model.prms.control_file,
                dict_key='PRMS_CONTROL_FILE',
            )
            control_path = settings_dir / control_file
            if not control_path.exists():
                self.logger.error(f"PRMS control file not found: {control_path}")
                return False

            # Build command
            cmd = [str(prms_exe), '-C', str(control_path)]

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            timeout = resolve_config_value(config, lambda: config.model.prms.timeout, dict_key='PRMS_TIMEOUT')

            stdout_file = prms_output_dir / 'prms_stdout.log'
            stderr_file = prms_output_dir / 'prms_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = run_subprocess(
                        cmd,
                        cwd=str(prms_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"PRMS timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"PRMS failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output - check output dir first, then settings fallback
            output_files = list(prms_output_dir.glob('statvar*'))
            if not output_files:
                output_files = list(settings_dir.glob('statvar*'))
            if not output_files:
                output_files = (
                    list(prms_output_dir.glob('*.csv')) +
                    list(settings_dir.glob('*.csv'))
                )

            if not output_files:
                self._last_error = "No statvar output files produced"
                self.logger.error(self._last_error)
                return False

            tolerance = resolve_config_value(config, lambda: config.model.prms.water_balance_tolerance_mm, None, 'PRMS_WATER_BALANCE_TOLERANCE_MM')
            if tolerance is not None:
                from ..balance import check_water_balance
                check_water_balance(output_files[0], settings_dir / resolve_config_value(
                    config,
                    lambda: config.model.prms.parameter_file,
                    dict_key='PRMS_PARAMETER_FILE',
                ),
                                    float(tolerance), bool(resolve_config_value(
                                        config,
                                        lambda: config.model.prms.glacier_flag,
                                        dict_key='PRMS_GLACIER_FLAG',
                                    )))
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running PRMS: {e}", exc_info=True)
            return False

    def _get_prms_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get PRMS executable path."""
        install_path = resolve_config_value(
            config,
            lambda: config.model.prms.install_path,
            dict_key='PRMS_INSTALL_PATH',
        )
        exe_name = resolve_config_value(config, lambda: config.model.prms.exe, dict_key='PRMS_EXE')

        exe = resolve_install_executable(install_path, data_dir, 'installs/prms/bin', exe_name)
        # Windows builds produce <name>.exe even when the config declares a bare
        # name; fall back to the .exe variant when the bare path is absent (else
        # every eval reports "executable not found" -> 100% crash rate).
        if exe.suffix != '.exe':
            exe_win = exe.with_suffix('.exe')
            if exe_win.exists():
                return exe_win
        return exe

    def _find_statvar(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Path:
        """This trial's statvar file: the run directory, then the settings directories."""
        sim_dir = Path(kwargs.get('sim_dir', output_dir))
        search_dirs = [sim_dir]
        if kwargs.get('settings_dir'):
            search_dirs.append(Path(kwargs['settings_dir']))
        search_dirs.append(self._project_location(config)[0] / 'settings' / 'PRMS')
        for directory in search_dirs:
            matches = list(directory.glob('statvar*'))
            if matches:
                return matches[0]
        raise ModelExecutionError(f"No PRMS statvar files found in {sim_dir}")

    def read_simulated_streamflow(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Optional[pd.Series]:
        """Basin streamflow (m3/s) from the PRMS statvar file.

        Uses the shared reader (:mod:`symfluence.models.prms.statvar`) so
        calibration and the postprocessor's results CSV read the same
        variable (basin_cfs, else seg_outflow) from the same column.
        """
        from symfluence.models.prms.statvar import read_statvar_streamflow
        statvar_file = self._find_statvar(output_dir, config, **kwargs)
        return read_statvar_streamflow(statvar_file).rename('PRMS_discharge_cms')

    def additional_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        *,
        project_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Snow-cover and glacier mass-balance scores when PRMS_CRYOSPHERE_CONSTRAINTS is on."""
        if not resolve_config_value(
            config,
            lambda: config.model.prms.cryosphere_constraints,
            dict_key='PRMS_CRYOSPHERE_CONSTRAINTS',
        ):
            return {}
        from ..cryosphere import calculate_constraints
        settings_dir = kwargs.get('settings_dir')
        parameter_file = Path(settings_dir) / 'params.dat' if settings_dir else project_dir / 'settings/PRMS/params.dat'
        return calculate_constraints(
            self._find_statvar(output_dir, config, **kwargs), parameter_file, project_dir,
            resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD'),
            resolve_config_value(
                config,
                lambda: config.model.prms.snow_cover_error_scale,
                dict_key='PRMS_SNOW_COVER_ERROR_SCALE',
            ),
            resolve_config_value(
                config,
                lambda: config.model.prms.glacier_mb_error_scale,
                dict_key='PRMS_GLACIER_MB_ERROR_SCALE',
            ))

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_prms_parameters_worker(task_data)


def _evaluate_prms_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        PRMSWorker,
        task_data,
        model_name='PRMS',
        extra_env=MACOS_MALLOC_QUIET_ENV,
    )
