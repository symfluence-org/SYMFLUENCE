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
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.constants import ModelDefaults
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('PRMS')
class PRMSWorker(BaseWorker):
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

    _streamflow_metrics = StreamflowMetrics()

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
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
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
            param_file_name = config.get('PRMS_PARAMETER_FILE', 'params.dat')
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

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying PRMS parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_parameter_file(self, param_file: Path, params: Dict[str, float]) -> bool:
        """
        Update PRMS parameter file with new values.

        Parses the ####-delimited block format and replaces value lines
        for each calibration parameter.

        Args:
            param_file: Path to params.dat
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            from ..derived_parameters import apply_derived_parameters
            content = param_file.read_text(encoding='utf-8')
            content, params = apply_derived_parameters(content, params)
            blocks = content.split('####\n')

            updated_blocks = []
            for block in blocks:
                updated_block = block
                for param_name, value in params.items():
                    lines = block.strip().split('\n')
                    for line in lines:
                        if line.strip() == param_name:
                            updated_block = self._replace_block_values(
                                block, param_name, value
                            )
                            self.logger.debug(
                                f"Updated {param_name} = {value:.6f}"
                            )
                            break
                updated_blocks.append(updated_block)

            param_file.write_text('####\n'.join(updated_blocks), encoding='utf-8')
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating PRMS parameter file: {e}", exc_info=True)
            return False

    def _replace_block_values(self, block: str, param_name: str, value: float) -> str:
        """Replace value lines in a PRMS parameter block.

        Preserves leading/trailing whitespace in the block so that
        #### delimiters remain properly separated when blocks are rejoined.
        """
        stripped = block.strip()
        lines = stripped.split('\n')

        param_idx = None
        for i, line in enumerate(lines):
            if line.strip() == param_name:
                param_idx = i
                break

        if param_idx is None:
            return block

        # Structure: name, ndim, dim_name, dim_size, type_code, values...
        try:
            dim_size = int(lines[param_idx + int(lines[param_idx + 1]) + 2].strip())
        except (IndexError, ValueError):
            dim_size = 1

        value_start = param_idx + int(lines[param_idx + 1]) + 4
        formatted = f"{value:.6f}"

        new_lines = lines[:value_start]
        for _ in range(dim_size):
            new_lines.append(formatted)

        remaining = value_start + dim_size
        if remaining < len(lines):
            new_lines.extend(lines[remaining:])

        result = '\n'.join(new_lines)
        # Preserve the trailing newline so #### delimiters stay on their own line
        if block.endswith('\n'):
            result += '\n'
        return result

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
        control_file = config.get('PRMS_CONTROL_FILE', 'control.dat')
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
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            prms_output_dir = Path(kwargs.get('sim_dir', output_dir))
            prms_output_dir.mkdir(parents=True, exist_ok=True)

            # Get executable
            prms_exe = self._get_prms_executable(config, data_dir)
            if not prms_exe.exists():
                self.logger.error(f"PRMS executable not found: {prms_exe}")
                return False

            # Get control file
            control_file = config.get('PRMS_CONTROL_FILE', 'control.dat')
            control_path = settings_dir / control_file
            if not control_path.exists():
                self.logger.error(f"PRMS control file not found: {control_path}")
                return False

            # Build command
            cmd = [str(prms_exe), '-C', str(control_path)]

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            timeout = config.get('PRMS_TIMEOUT', 3600)

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

            tolerance = config.get('PRMS_WATER_BALANCE_TOLERANCE_MM')
            if tolerance is not None:
                from ..balance import check_water_balance
                check_water_balance(output_files[0], settings_dir / config.get('PRMS_PARAMETER_FILE', 'params.dat'),
                                    float(tolerance), bool(config.get('PRMS_GLACIER_FLAG', False)))
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running PRMS: {e}", exc_info=True)
            return False

    def _get_prms_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get PRMS executable path."""
        install_path = config.get('PRMS_INSTALL_PATH', 'default')
        exe_name = config.get('PRMS_EXE', 'prms')

        if install_path == 'default':
            exe = data_dir / "installs" / "prms" / "bin" / exe_name
        else:
            install_path = Path(install_path)
            exe = install_path / exe_name if install_path.is_dir() else install_path
        # Windows builds produce <name>.exe even when the config declares a bare
        # name; fall back to the .exe variant when the bare path is absent (else
        # every eval reports "executable not found" -> 100% crash rate).
        if exe.suffix != '.exe':
            exe_win = exe.with_suffix('.exe')
            if exe_win.exists():
                return exe_win
        return exe

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from PRMS output.

        Extracts streamflow from statvar output, loads observations,
        aligns time series, and computes KGE/NSE.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            settings_dir = kwargs.get('settings_dir', None)

            # Find statvar output - search multiple locations
            output_files = list(sim_dir.glob('statvar*'))

            if not output_files and settings_dir:
                output_files = list(Path(settings_dir).glob('statvar*'))

            if not output_files:
                domain_name = config.get('DOMAIN_NAME', '')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                prms_settings = data_dir / f'domain_{domain_name}' / 'settings' / 'PRMS'
                if prms_settings.exists():
                    output_files = list(prms_settings.glob('statvar*'))

            if not output_files:
                self.logger.error(f"No PRMS statvar files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            statvar_file = output_files[0]

            # Extract streamflow from statvar
            sim_series = self._extract_streamflow_from_statvar(statvar_file)
            if sim_series is None or len(sim_series) == 0:
                return {'kge': self.penalty_score, 'error': 'No streamflow data extracted'}

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Parse calibration period
            cal_period_str = config.get('CALIBRATION_PERIOD', '')
            cal_period_tuple = None
            if cal_period_str and ',' in str(cal_period_str):
                parts = str(cal_period_str).split(',')
                cal_period_tuple = (parts[0].strip(), parts[1].strip())

            # Align and calculate metrics
            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series, calibration_period=cal_period_tuple
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            if config.get('PRMS_CRYOSPHERE_CONSTRAINTS', False):
                from ..cryosphere import calculate_constraints
                parameter_file = Path(settings_dir) / 'params.dat' if settings_dir else project_dir / 'settings/PRMS/params.dat'
                results.update(calculate_constraints(
                    statvar_file, parameter_file, project_dir, cal_period_str,
                    config.get('PRMS_SNOW_COVER_ERROR_SCALE', 0.2),
                    config.get('PRMS_GLACIER_MB_ERROR_SCALE', 1.0)))
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating PRMS metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    def _extract_streamflow_from_statvar(self, statvar_file: Path) -> Optional[pd.Series]:
        """
        Extract streamflow from PRMS statvar text file.

        The statvar file has lines like:
        ``2 year month day hour min sec nstep seg_outflow_1 hru_actet_1 ...``

        Args:
            statvar_file: Path to statvar.dat

        Returns:
            Time series of streamflow in m3/s, or None on failure
        """
        try:
            lines = statvar_file.read_text(encoding='utf-8').strip().split('\n')

            dates = []
            values = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 9:
                    try:
                        year = int(parts[1])
                        month = int(parts[2])
                        day = int(parts[3])
                        # seg_outflow is typically the first output variable (index 7)
                        streamflow = float(parts[7])
                        date = pd.Timestamp(year=year, month=month, day=day)
                        dates.append(date)
                        # PRMS seg_outflow is in cfs, convert to cms
                        values.append(streamflow * 0.0283168)
                    except (ValueError, IndexError):
                        continue

            if dates:
                return pd.Series(values, index=dates, name='PRMS_discharge_cms')

            return None

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error extracting PRMS streamflow: {e}", exc_info=True)
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_prms_parameters_worker(task_data)


def _evaluate_prms_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    import os
    import random
    import signal
    import time
    import traceback

    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MallocStackLogging': '0',
    })

    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = PRMSWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'PRMS worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
