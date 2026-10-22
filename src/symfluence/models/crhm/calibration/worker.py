# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Worker

Worker implementation for CRHM model optimization.
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
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('CRHM')
class CRHMWorker(BaseWorker):
    """
    Worker for CRHM model calibration.

    Handles parameter application to .prj files, CRHM execution,
    and metric calculation from CSV output.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize CRHM worker.

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
        Apply parameters to CRHM project file.

        Args:
            params: Parameter values to apply
            settings_dir: CRHM settings directory (contains .prj file)
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying CRHM parameters to {settings_dir}")

            # Always copy fresh from the original CRHM_input location
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'CRHM'

            if original_settings_dir.exists() and original_settings_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_settings_dir.glob('*'):
                    if f.is_file():
                        shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(f"Copied CRHM files from {original_settings_dir} to {settings_dir}")
            elif not settings_dir.exists():
                self.logger.error(f"CRHM settings directory not found: {settings_dir} "
                                  f"(original also missing: {original_settings_dir})")
                return False

            # Find project file
            prj_file_name = config.get('CRHM_PROJECT_FILE', 'model.prj')
            prj_file = settings_dir / prj_file_name
            if not prj_file.exists():
                # Try to find any .prj file
                prj_files = list(settings_dir.glob('*.prj'))
                if prj_files:
                    prj_file = prj_files[0]
                else:
                    self.logger.error(f"CRHM project file not found in {settings_dir}")
                    return False

            # Keep nested recharge storage within total soil storage, and
            # initialize within each trial's capacities during model spinup.
            params = dict(params)
            if 'soil_moist_max' in params and 'soil_rechr_max' in params:
                params['soil_rechr_max'] = min(params['soil_rechr_max'], params['soil_moist_max'])
                params['soil_moist_init'] = 0.5 * params['soil_moist_max']
                params['soil_rechr_init'] = 0.5 * params['soil_rechr_max']
            if 'Sdmax' in params:
                params['Sdinit'] = 0.0
            if 'tmax_allrain' in params and 'tmax_allsnow' in params:
                params['tmax_allsnow'] = min(params['tmax_allsnow'], params['tmax_allrain'] - 0.1)
            return self._update_prj_file(prj_file, params)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying CRHM parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_prj_file(self, prj_file: Path, params: Dict[str, float]) -> bool:
        """
        Update CRHM project file (.prj) with new parameter values.

        In the native CRHM .prj format each parameter is stored as a
        two-line block inside the ``Parameters:`` section::

            <module> <param_name> [<min to max>]
            <value(s)>

        This method scans for header lines that reference a calibrated
        parameter and replaces the *following* value line.

        Args:
            prj_file: Path to project file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            content = prj_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            new_lines = []

            # Build a set of parameter names for fast lookup
            param_set = set(params.keys())
            skip_next = False
            pending_param: Optional[str] = None

            for line in lines:
                if skip_next and pending_param is not None:
                    # This line holds the old value(s) -- replace it
                    value = params[pending_param]
                    if abs(value) < 0.001 and value != 0:
                        formatted = f"{value:.8e}"
                    elif abs(value) > 9999:
                        formatted = f"{value:.2f}"
                    else:
                        formatted = f"{value:.6f}"
                    new_lines.append(' '.join([formatted] * max(1, len(line.split()))))
                    self.logger.debug(f"Updated {pending_param} = {formatted}")
                    skip_next = False
                    pending_param = None
                    continue

                # Check if this is a parameter header line:
                # ``<module> <param_name> [<min to max>]``
                stripped = line.strip()
                parts = stripped.split()
                if len(parts) >= 2 and parts[1] in param_set:
                    pending_param = parts[1]
                    skip_next = True

                new_lines.append(line)

            # Save
            temp_file = prj_file.with_suffix('.prj.tmp')
            temp_file.write_text('\n'.join(new_lines), encoding='utf-8')
            temp_file.replace(prj_file)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating CRHM project file: {e}", exc_info=True)
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run CRHM model.

        Args:
            config: Configuration dictionary
            settings_dir: CRHM settings directory
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            # Use sim_dir for output if provided
            crhm_output_dir = Path(kwargs.get('sim_dir', output_dir))
            crhm_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files
            self._cleanup_stale_output(crhm_output_dir)

            # Get executable
            crhm_exe = self._get_crhm_executable(config, data_dir)
            if not crhm_exe.exists():
                self.logger.error(f"CRHM executable not found: {crhm_exe}")
                return False

            # Get project file
            prj_file_name = config.get('CRHM_PROJECT_FILE', 'model.prj')
            prj_file = settings_dir / prj_file_name
            if not prj_file.exists():
                prj_files = list(settings_dir.glob('*.prj'))
                if prj_files:
                    prj_file = prj_files[0]
                else:
                    self.logger.error(f"Project file not found: {prj_file}")
                    return False

            # Build command: crhm [options] <project_file>
            # The project file is a positional argument.  -f is output format,
            # not the project file flag.  Use --obs_file_directory so CRHM
            # can locate forcing (.obs) files, which live in
            # data/forcing/CRHM_input (separate from settings/CRHM).
            domain_name = config.get('DOMAIN_NAME', '')
            project_dir = data_dir / f'domain_{domain_name}'
            forcing_dir = resolve_data_subdir(project_dir, 'forcing') / 'CRHM_input'
            obs_dir = str(forcing_dir) + os.sep
            cmd = [str(crhm_exe), '--obs_file_directory', obs_dir, str(prj_file)]

            # Set environment
            env = os.environ.copy()

            # Run with timeout
            timeout = config.get('CRHM_TIMEOUT', 300)

            stdout_file = crhm_output_dir / 'crhm_stdout.log'
            stderr_file = crhm_output_dir / 'crhm_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = run_subprocess(
                        cmd,
                        cwd=str(crhm_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"CRHM timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"CRHM failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output (CRHM produces .txt or .csv files)
            output_files = list(crhm_output_dir.glob('CRHM_output*.txt'))
            if not output_files:
                output_files = list(crhm_output_dir.glob('*.csv'))
            if not output_files:
                output_files = list(crhm_output_dir.glob('*.txt'))
            if not output_files:
                self._last_error = "No output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running CRHM: {e}", exc_info=True)
            return False

    def _get_crhm_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get CRHM executable path."""
        install_path = config.get('CRHM_INSTALL_PATH', 'default')
        exe_name = config.get('CRHM_EXE', 'crhm')

        if install_path == 'default':
            exe = data_dir / "installs" / "crhm" / "bin" / exe_name
        else:
            install_path = Path(install_path)
            exe = install_path / exe_name if install_path.is_dir() else install_path
        # Windows builds produce <name>.exe even when the config declares a bare
        # name; the calibration worker (unlike the runner) looked only for the
        # bare path, so every eval reported "executable not found" and returned a
        # penalty -> 100% crash rate. Fall back to the .exe variant when present.
        if exe.suffix != '.exe':
            exe_win = exe.with_suffix('.exe')
            if exe_win.exists():
                return exe_win
        return exe

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale CRHM output files."""
        for pattern in ['*.csv', '*.txt', '*.log']:
            for file_path in output_dir.glob(pattern):
                try:
                    file_path.unlink()
                except (OSError, IOError):
                    pass

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from CRHM output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Find output file (CRHM produces .txt or .csv)
            output_files = list(sim_dir.glob('CRHM_output*.txt'))
            if not output_files:
                output_files = list(sim_dir.glob('*output*.csv'))
            if not output_files:
                output_files = list(sim_dir.glob('*.csv'))
            if not output_files:
                output_files = list(sim_dir.glob('*.txt'))

            # Also search settings_dir if provided
            settings_dir = kwargs.get('settings_dir')
            if not output_files and settings_dir:
                sd = Path(settings_dir)
                for subdir in [sd / 'output', sd]:
                    output_files = list(subdir.glob('CRHM_output*.txt'))
                    if not output_files:
                        output_files = list(subdir.glob('*.csv'))
                    if output_files:
                        break

            if not output_files:
                self.logger.error(f"No CRHM output files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_file = output_files[0]

            # Extract streamflow - handle both CSV and tab-separated TXT
            if sim_file.suffix == '.txt':
                df = pd.read_csv(sim_file, sep='\t', parse_dates=True, index_col=0,
                                 skiprows=[1], encoding='latin-1')  # Skip units row
            else:
                df = pd.read_csv(sim_file, parse_dates=True, index_col=0,
                                 encoding='latin-1')

            # Find flow column (CRHM uses 'basinflow(1)' for basin outlet flow)
            flow_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'basinflow' in col_lower or 'basin_flow' in col_lower:
                    flow_col = col
                    break
            if flow_col is None:
                for col in ['flow', 'Flow', 'discharge', 'Discharge', 'Q', 'flow_cms']:
                    if col in df.columns:
                        flow_col = col
                        break
            if flow_col is None:
                flow_cols = [c for c in df.columns if 'flow' in c.lower()]
                if flow_cols:
                    flow_col = flow_cols[0]

            if flow_col is None:
                return {'kge': self.penalty_score, 'error': 'No flow variable in output'}

            sim_series = df[flow_col].astype(float)
            sim_series.index = pd.to_datetime(sim_series.index)

            # CRHM basinflow is in m^3/interval; resample to daily mean (m^3/s)
            # Interval is typically 1 hour, so m^3/hr -> convert to m^3/s = /3600
            interval_seconds = 3600  # Default 1-hour interval
            if len(sim_series) > 1:
                dt = (sim_series.index[1] - sim_series.index[0]).total_seconds()
                if dt > 0:
                    interval_seconds = dt
            sim_series = sim_series / interval_seconds  # Convert to m^3/s
            # Native CRHM output is labelled at the end of each interval.
            sim_series.index = sim_series.index - pd.Timedelta(seconds=interval_seconds)
            sim_series = sim_series.resample('D').mean()

            # Skip warmup period
            warmup_days = int(config.get('CRHM_WARMUP_DAYS', 365))
            if warmup_days > 0 and len(sim_series) > warmup_days:
                sim_series = sim_series.iloc[warmup_days:]
                self.logger.debug(f"Skipped {warmup_days} warmup days for metric calculation")

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

            # Restrict scoring to the configured calibration window; the final
            # evaluation swaps EVALUATION_PERIOD into this key to score the
            # held-out window from the same full-period run.
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
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating CRHM metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_crhm_parameters_worker(task_data)


def _evaluate_crhm_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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

    # Set up signal handler
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    # Force single-threaded execution
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    })

    # Small random delay
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = CRHMWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'CRHM worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
