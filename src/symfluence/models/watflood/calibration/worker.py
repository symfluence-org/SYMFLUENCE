# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD Worker.

Worker implementation for WATFLOOD model optimization via Wine execution.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    BaseWorker,
    run_worker_task,
)
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('WATFLOOD')
class WATFLOODWorker(BaseWorker):
    """Worker for WATFLOOD model calibration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        """Apply parameters to WATFLOOD .par file."""
        import re
        try:
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_dir = data_dir / f'domain_{domain_name}' / 'WATFLOOD_input' / 'settings'

            par_file = config.get('WATFLOOD_PAR_FILE', 'bow.par')
            if original_dir.exists() and original_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                # Copy the WATFLOOD settings tree (subdirectories included) when the
                # parameter file isn't already staged here. Key on the .par file
                # rather than the basin/ dir: a stale, empty basin/ left by a
                # previous failed run must not silently skip the copy.
                have_par = ((settings_dir / 'basin' / par_file).exists()
                            or (settings_dir / par_file).exists())
                if not have_par:
                    for item in original_dir.iterdir():
                        dest = settings_dir / item.name
                        if item.is_dir():
                            if not dest.exists():
                                shutil.copytree(item, dest)
                        else:
                            shutil.copy2(item, dest)

            # Find .par file (check basin/ subdirectory too)
            par_path = settings_dir / 'basin' / par_file
            if not par_path.exists():
                par_path = settings_dir / par_file
            if not par_path.exists():
                # Search for any .par file
                par_files = list(settings_dir.rglob('*.par'))
                if par_files:
                    par_path = par_files[0]
                else:
                    self.logger.error(f"No .par file found in {settings_dir}")
                    return False

            from symfluence.models.watflood.parameters import PAR_KEYWORD_MAP

            content = par_path.read_text(encoding='utf-8')
            for param_name, value in params.items():
                # Map calibration name to par file keyword (parser format)
                par_keyword = PAR_KEYWORD_MAP.get(param_name, param_name)
                # Parser-format lines: `:keyword, value,` or `:keyword, value`
                pattern = re.compile(
                    rf'^(:{re.escape(par_keyword)},\s*)([-+]?[\d.eE+\-]+)',
                    re.MULTILINE | re.IGNORECASE
                )
                content = pattern.sub(
                    lambda m, _value=value: f"{m.group(1)}{_value:.3E}", content  # type: ignore[misc]
                )

            par_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying WATFLOOD parameters: {e}", exc_info=True)
            return False

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Run WATFLOOD model (native binary or Wine for .exe)."""
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            sim_dir.mkdir(parents=True, exist_ok=True)

            install_path = config.get('WATFLOOD_INSTALL_PATH', 'default')
            exe_name = config.get('WATFLOOD_EXE', 'watflood')
            if install_path == 'default':
                install_dir = data_dir / "installs" / "watflood" / "bin"
            else:
                p = Path(install_path)
                install_dir = p if p.is_dir() else p.parent

            watflood_exe = install_dir / exe_name
            if not watflood_exe.exists():
                self.logger.error(f"WATFLOOD executable not found: {watflood_exe}")
                return False

            is_native = not exe_name.lower().endswith('.exe')

            if is_native:
                # Native binary — run directly
                work_exe = settings_dir / exe_name
                if not work_exe.exists():
                    shutil.copy2(watflood_exe, work_exe)
                    work_exe.chmod(0o755)
                cmd = [str(work_exe)]
            else:
                # Windows binary — use Wine
                for f in install_dir.iterdir():
                    if f.suffix.lower() in ('.exe', '.dll'):
                        dest = settings_dir / f.name
                        if not dest.exists():
                            shutil.copy2(f, dest)

                wine_cmd = self._find_wine()
                if wine_cmd is None:
                    self.logger.error("Wine not found (needed for .exe binary)")
                    return False
                cmd = [wine_cmd, exe_name]

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'
            if not is_native:
                env['WINEDEBUG'] = '-all'
            timeout = config.get('WATFLOOD_TIMEOUT', 3600)

            try:
                with open(sim_dir / 'watflood_stdout.log', 'w') as out, \
                     open(sim_dir / 'watflood_stderr.log', 'w') as err:
                    # CHARM issues an unconditional `read(*,*)` "hit any key"
                    # prompt on the first event (sub.f, id<=1). With no stdin it
                    # hits EOF and the read crashes; feed blank lines so it
                    # continues non-interactively.
                    result = run_subprocess(
                        cmd, cwd=str(settings_dir), env=env,
                        input=b'\n' * 256, stdout=out, stderr=err,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"WATFLOOD timed out after {timeout}s")
                return False

            # WATFLOOD may exit non-zero but still produce valid output
            if result.returncode != 0:
                self.logger.warning(f"WATFLOOD exited with code {result.returncode}")
            # Collect outputs from settings_dir and results/
            for src in [settings_dir, settings_dir / 'results']:
                if src.exists():
                    for pattern in ['*.tb0', '*.csv', 'spl*', 'resin*']:
                        for f in src.glob(pattern):
                            if f.is_file():
                                shutil.copy2(f, sim_dir / f.name)

            output_files = (
                list(sim_dir.glob('*.tb0')) +
                list(sim_dir.glob('*.csv')) +
                list(sim_dir.glob('spl*'))
            )
            if not output_files:
                self._last_error = "No WATFLOOD output files produced"
                self.logger.error(self._last_error)
                return False
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running WATFLOOD: {e}", exc_info=True)
            return False

    @staticmethod
    def _find_wine() -> Optional[str]:
        """Find Wine executable."""
        for candidate in ['wine', '/opt/homebrew/bin/wine', '/usr/local/bin/wine']:
            try:
                run_subprocess(
                    [candidate, '--version'],
                    capture_output=True, timeout=10
                )
                return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate metrics from WATFLOOD output."""
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Prioritize CHARM daily CSV and numbered streamflow tb0 files
            charm_csv = sim_dir / 'CHARM_dly.csv'
            str_tb0 = sorted(sim_dir.glob('[0-9]*_str.tb0'))
            if charm_csv.exists():
                output_files = [charm_csv]
            elif str_tb0:
                output_files = str_tb0
            else:
                tb0_files = list(sim_dir.glob('spl*.tb0')) + list(sim_dir.glob('resin*.tb0'))
                csv_files = list(sim_dir.glob('*.csv'))
                output_files = tb0_files + csv_files

            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_series = self._extract_streamflow(output_files[0])
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
            cal_period_str = config.get('CALIBRATION_PERIOD', '')
            cal_period_tuple = None
            if cal_period_str and ',' in str(cal_period_str):
                parts = str(cal_period_str).split(',')
                cal_period_tuple = (parts[0].strip(), parts[1].strip())

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series, calibration_period=cal_period_tuple
            )
            return self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating WATFLOOD metrics: {e}", exc_info=True)
            return {'kge': self.penalty_score, 'error': str(e)}

    def _extract_streamflow(self, output_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from WATFLOOD output."""
        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                # Drop unnamed empty columns from trailing commas
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                for col in df.columns:
                    if any(v in col.lower() for v in ['_sim', 'qsim', 'qo', 'flow']):
                        return df[col]
            elif output_file.suffix == '.tb0':
                lines = output_file.read_text(encoding='utf-8').strip().split('\n')
                data_start = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith((':', '#')) and stripped[0].isdigit():
                        data_start = i
                        break
                dates, values = [], []
                for line in lines[data_start:]:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            value = float(parts[4])
                            dates.append(pd.Timestamp(year=year, month=month, day=day))
                            values.append(value)
                        except (ValueError, IndexError):
                            continue
                if dates:
                    return pd.Series(values, index=dates, name='WATFLOOD_discharge_cms')
            return None
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error extracting WATFLOOD streamflow: {e}", exc_info=True)
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_watflood_parameters_worker(task_data)


def _evaluate_watflood_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for process pool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        WATFLOODWorker,
        task_data,
        model_name='WATFLOOD',
        extra_env=MACOS_MALLOC_QUIET_ENV,
    )
