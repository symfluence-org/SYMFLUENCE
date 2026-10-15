# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Worker.

Worker implementation for GSFLOW model optimization.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.constants import ModelDefaults
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('GSFLOW')
class GSFLOWWorker(BaseWorker):
    """Worker for GSFLOW model calibration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        """Apply parameters to GSFLOW PRMS params.dat and MODFLOW UPW package."""
        try:
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_setup_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'GSFLOW'

            # Copy entire settings tree (including modflow/ subdir) to sim dir
            if original_setup_dir.exists() and original_setup_dir.resolve() != settings_dir.resolve():
                if settings_dir.exists():
                    shutil.rmtree(settings_dir)
                shutil.copytree(original_setup_dir, settings_dir)

            # Copy forcing data.dat into settings_dir so the control file (run
            # with cwd=settings_dir, data_file='data.dat') finds it locally.
            # The forcing lives under <domain>/forcing/GSFLOW_input (the
            # preprocessor's forcing_dir); fall back to the legacy data/forcing
            # layout for older datastores.
            domain_dir = data_dir / f'domain_{domain_name}'
            for fdir in (domain_dir / 'forcing' / 'GSFLOW_input',
                         domain_dir / 'data' / 'forcing' / 'GSFLOW_input'):
                data_file = fdir / 'data.dat'
                if data_file.exists():
                    shutil.copy2(data_file, settings_dir / 'data.dat')
                    break

            params = self._enforce_constraints(params)
            self._trial_params = dict(params)
            from ..parameter_files import apply_gsflow_parameters
            apply_gsflow_parameters(settings_dir, params, config.get('GSFLOW_PARAMETER_FILE', 'params.dat'))
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying GSFLOW parameters: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    @staticmethod
    def _enforce_constraints(params: Dict[str, float]) -> Dict[str, float]:
        """Enforce GSFLOW cross-parameter physical constraints.

        Mirrors PRMS worker constraints for parameters active in coupled mode:
        - soil_rechr_max <= soil_moist_max (if both present, e.g. PRMS-only mode)
        - tmax_allsnow < tmax_allrain (if both present)
        """
        p = dict(params)
        if 'soil_rechr_max' in p and 'soil_moist_max' in p:
            if p['soil_rechr_max'] > p['soil_moist_max']:
                p['soil_rechr_max'] = p['soil_moist_max'] * 0.9
        if 'tmax_allsnow' in p and 'tmax_allrain' in p:
            if p['tmax_allsnow'] >= p['tmax_allrain']:
                p['tmax_allsnow'] = p['tmax_allrain'] - 2.0
        return p

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Run GSFLOW model; with GSFLOW_EVALUATION_LOG set, append one JSON line per run."""
        self._last_error = ''
        start = time.monotonic()
        ok = self._run_model(config, settings_dir, output_dir, **kwargs)
        log_path = config.get('GSFLOW_EVALUATION_LOG')
        if log_path:
            record = dict(time=pd.Timestamp.now().isoformat(timespec='seconds'), run_dir=str(settings_dir),
                          ok=ok, seconds=round(time.monotonic() - start, 1), error=self._last_error or None,
                          params=getattr(self, '_trial_params', None))
            try:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
            except OSError as e:
                self.logger.warning(f'Could not write GSFLOW evaluation log {log_path}: {e}')
        return ok

    def _run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            sim_dir.mkdir(parents=True, exist_ok=True)

            # Create output subdirectories
            for sub in ('modflow', 'prms'):
                (sim_dir / sub).mkdir(parents=True, exist_ok=True)

            # Localize control/NAM file paths for this trial:
            # data_file → local data.dat (copied in apply_parameters)
            # output paths → absolute paths into sim_dir
            control_file = config.get('GSFLOW_CONTROL_FILE', 'gsflow.control')
            self._localize_control_paths(settings_dir, sim_dir, control_file)
            self._localize_nam_paths(settings_dir, sim_dir)

            install_path = config.get('GSFLOW_INSTALL_PATH', 'default')
            exe_name = config.get('GSFLOW_EXE', 'gsflow')
            if install_path == 'default':
                gsflow_exe = data_dir / "installs" / "gsflow" / "bin" / exe_name
            else:
                p = Path(install_path)
                gsflow_exe = p / exe_name if p.is_dir() else p

            # Windows builds land as <name>.exe even though the declared name
            # is bare — mirror BaseModelRunner._exe_name_variants.
            if not gsflow_exe.exists() and sys.platform == 'win32' and not gsflow_exe.suffix:
                windows_exe = gsflow_exe.with_name(f"{gsflow_exe.name}.exe")
                if windows_exe.exists():
                    gsflow_exe = windows_exe

            if not gsflow_exe.exists():
                self.logger.error(f"GSFLOW executable not found: {gsflow_exe}")
                return False

            control_path = settings_dir / control_file
            if not control_path.exists():
                self.logger.error(f"GSFLOW control file not found: {control_path}")
                return False

            cmd = [str(gsflow_exe), str(control_path)]
            env = os.environ.copy()
            env['MallocStackLogging'] = '0'
            timeout = config.get('GSFLOW_TIMEOUT', 7200)

            try:
                with open(sim_dir / 'gsflow_stdout.log', 'w') as out, \
                     open(sim_dir / 'gsflow_stderr.log', 'w') as err:
                    result = run_subprocess(
                        cmd, cwd=str(settings_dir), env=env,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self._last_error = f'GSFLOW timed out after {timeout}s'
                self.logger.warning(self._last_error)
                return False

            if result.returncode != 0:
                self._last_error = f"GSFLOW failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Check for output in sim_dir (control file paths point here)
            output_files = (
                list(sim_dir.glob('statvar*')) +
                list(sim_dir.glob('*.csv')) +
                list(sim_dir.glob('gsflow.*'))
            )
            if not output_files:
                self._last_error = "No GSFLOW output files produced"
                self.logger.error(self._last_error)
                return False
            budget_file = sim_dir / 'gsflow.csv'
            if budget_file.exists() and self._extract_streamflow_from_csv(budget_file) is None:
                self._last_error = "Invalid GSFLOW coupled budget; refusing to score finite outlet flow alone"
                self.logger.error(self._last_error)
                return False
            budget_log = sim_dir / 'gsflow.out'
            if budget_log.exists():
                import re
                failures = re.findall(r'Number of non-convergence:\s*(\d+)', budget_log.read_text())
                if any(int(n) > 0 for n in failures):
                    self._last_error = f'GSFLOW reported non-converged time steps ({max(int(n) for n in failures)})'
                    self.logger.warning(self._last_error)
                    return False
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running GSFLOW: {e}", exc_info=True)
            return False

    def _localize_control_paths(self, settings_dir: Path, sim_dir: Path,
                                 control_file: str) -> None:
        """Rewrite control file paths for a calibration trial.

        Sets data_file to local data.dat (copied by apply_parameters)
        and output paths to absolute paths in sim_dir.
        """
        control_path = settings_dir / control_file
        if not control_path.exists():
            return

        path_updates = {
            'data_file': 'data.dat',
            'gsflow_output_file': str(sim_dir / 'gsflow.out'),
            'model_output_file': str(sim_dir / 'prms' / 'gsflow_prms.out'),
            'csv_output_file': str(sim_dir / 'gsflow.csv'),
            'stat_var_file': str(sim_dir / 'statvar.dat'),
        }

        content = control_path.read_text(encoding='utf-8')
        for key, new_value in path_updates.items():
            content = self._replace_control_value(content, key, new_value)
        control_path.write_text(content, encoding='utf-8')

    def _localize_nam_paths(self, settings_dir: Path, sim_dir: Path) -> None:
        """Rewrite NAM file output paths for a calibration trial."""
        nam_file = settings_dir / 'modflow' / 'modflow.nam'
        if not nam_file.exists():
            return

        lines = nam_file.read_text(encoding='utf-8').split('\n')
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('LIST'):
                parts = stripped.split(None, 2)
                new_lines.append(
                    f"LIST  {parts[1]}  {sim_dir}/modflow/gsflow_mf.list")
            elif stripped.startswith('DATA(BINARY)'):
                parts = stripped.split(None, 2)
                new_lines.append(
                    f"DATA(BINARY)  {parts[1]}  {sim_dir}/modflow/heads.out")
            else:
                new_lines.append(line)
        nam_file.write_text('\n'.join(new_lines), encoding='utf-8')

    @staticmethod
    def _replace_control_value(content: str, key: str, new_value: str) -> str:
        """Replace a single value in GSFLOW ####-delimited control file."""
        marker = f'####\n{key}\n'
        idx = content.find(marker)
        if idx < 0:
            return content

        # Skip past key line
        after_key = idx + len(marker)
        # Skip nvals line
        nvals_end = content.index('\n', after_key) + 1
        # Skip dtype line
        dtype_end = content.index('\n', nvals_end) + 1
        # Find end of value line
        value_end = content.index('\n', dtype_end)

        return content[:dtype_end] + new_value + content[value_end:]

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate metrics from GSFLOW output.

        Args:
            output_dir: Directory containing model outputs.
            config: Configuration dictionary.
            **kwargs: Optional overrides.
                period: Comma-separated "start, end" date string to override
                    the default CALIBRATION_PERIOD (used for evaluation-period
                    metrics in split-sample validation).
                sim_dir: Alternative simulation output directory.
                settings_dir: Fallback directory for statvar files.
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            settings_dir = kwargs.get('settings_dir', None)

            # In COUPLED GSFLOW the basin outlet flow is routed through MODFLOW
            # SFR and reported as StreamOut_Q in gsflow.csv (m3/day); the PRMS
            # statvar 'basin_cfs' is NOT the routed streamflow (near-zero in
            # coupled mode), so the CSV is the authoritative source.
            sim_series = None
            csv_files = (list(sim_dir.glob('gsflow.csv'))
                         or list(sim_dir.glob('gsflow_*.csv'))
                         or list(sim_dir.glob('*.csv')))
            if csv_files:
                sim_series = self._extract_streamflow_from_csv(csv_files[0])
                if sim_series is None or len(sim_series) == 0:
                    return {'kge': self.penalty_score, 'error': 'Invalid coupled streamflow/budget CSV'}

            if sim_series is None or len(sim_series) == 0:
                output_files = list(sim_dir.glob('statvar*'))
                if not output_files and settings_dir:
                    output_files = list(Path(settings_dir).glob('statvar*'))
                if output_files:
                    sim_series = self._extract_streamflow_from_statvar(output_files[0])

            if sim_series is None or len(sim_series) == 0:
                return {'kge': self.penalty_score, 'error': 'No streamflow data'}

            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Allow explicit period override (for evaluation-period metrics)
            period_str = kwargs.get('period', '') or config.get('CALIBRATION_PERIOD', '')
            period_tuple = None
            if period_str and ',' in str(period_str):
                parts = str(period_str).split(',')
                period_tuple = (parts[0].strip(), parts[1].strip())

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series, calibration_period=period_tuple
            )
            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            if config.get('GSFLOW_CRYOSPHERE_CONSTRAINTS', False):
                # Same snow-cover / glacier mass-balance scores as PRMS (GSFLOW
                # embeds PRMS); always scored over the calibration period.
                from symfluence.models.prms.cryosphere import calculate_constraints
                statvar = next(iter(sorted(sim_dir.glob('statvar*'))), None)
                param_dir = Path(settings_dir) if settings_dir else project_dir / 'settings' / 'GSFLOW'
                if statvar is None:
                    return {'kge': self.penalty_score, 'error': 'No statvar for cryosphere constraints'}
                results.update(calculate_constraints(
                    statvar, param_dir / config.get('GSFLOW_PARAMETER_FILE', 'params.dat'), project_dir,
                    config.get('CALIBRATION_PERIOD', ''), config.get('GSFLOW_SNOW_COVER_ERROR_SCALE', 0.2),
                    config.get('GSFLOW_GLACIER_MB_ERROR_SCALE', 1.0)))
            return results
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating GSFLOW metrics: {e}", exc_info=True)
            return {'kge': self.penalty_score, 'error': str(e)}

    def _extract_streamflow_from_csv(self, csv_file: Path) -> Optional[pd.Series]:
        """Extract basin outlet streamflow from the GSFLOW CSV budget file.

        Uses StreamOut_Q (basin streamflow leaving the SFR network, m3/day)
        and converts to m3/s. This is the routed streamflow in COUPLED mode.
        """
        try:
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns]
            date_col = df.columns[0]
            flow_col = next((c for c in df.columns
                             if c.lower() in ('streamout_q', 'streamflow_out',
                                              'basinstreamflow_q')), None)
            if flow_col is None:
                return None
            for column in [c for c in df if c == flow_col or c.endswith(('_S', '_Q'))]:
                if column in df and not np.isfinite(pd.to_numeric(df[column], errors='coerce')).all():
                    self.logger.error(f"Non-finite GSFLOW budget variable {column} in {csv_file}")
                    return None
            idx = pd.to_datetime(df[date_col])
            series = pd.Series(pd.to_numeric(df[flow_col], errors='coerce').values,
                               index=idx, name='GSFLOW_discharge_cms') / 86400.0
            return series.dropna()
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading GSFLOW CSV {csv_file}: {e}",
                              exc_info=True)
            return None

    def _extract_streamflow_from_statvar(self, statvar_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from GSFLOW statvar file."""
        try:
            lines = statvar_file.read_text(encoding='utf-8').strip().split('\n')
            dates, values = [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 8:
                    try:
                        year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                        # basin_cfs is the first statvar (index 7 after
                        # id year month day hour min sec)
                        streamflow_cfs = float(parts[7])
                        dates.append(pd.Timestamp(year=year, month=month, day=day))
                        values.append(streamflow_cfs * 0.0283168)  # cfs → cms
                    except (ValueError, IndexError):
                        continue
            if dates:
                return pd.Series(values, index=dates, name='GSFLOW_discharge_cms')
            return None
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error extracting GSFLOW streamflow: {e}", exc_info=True)
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_gsflow_parameters_worker(task_data)


def _evaluate_gsflow_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for process pool execution."""
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
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MallocStackLogging': '0',
    })
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = GSFLOWWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'GSFLOW worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
