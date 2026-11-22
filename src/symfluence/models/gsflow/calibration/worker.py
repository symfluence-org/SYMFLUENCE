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

import pandas as pd

from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    run_worker_task,
)
from symfluence.core.calibration.workers.streamflow_worker import StreamflowWorker
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_executable
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('GSFLOW')
class GSFLOWWorker(StreamflowWorker):
    """Worker for GSFLOW model calibration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        """Apply parameters to GSFLOW PRMS params.dat and MODFLOW UPW package."""
        try:
            config = kwargs.get('config', self.config) or {}
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
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
            apply_gsflow_parameters(settings_dir, params, resolve_config_value(
                config,
                lambda: config.model.gsflow.parameter_file,
                dict_key='GSFLOW_PARAMETER_FILE',
            ))
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
        log_path = resolve_config_value(config, lambda: config.model.gsflow.evaluation_log, None, 'GSFLOW_EVALUATION_LOG')
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
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            sim_dir.mkdir(parents=True, exist_ok=True)

            # Create output subdirectories
            for sub in ('modflow', 'prms'):
                (sim_dir / sub).mkdir(parents=True, exist_ok=True)

            # Localize control/NAM file paths for this trial:
            # data_file → local data.dat (copied in apply_parameters)
            # output paths → absolute paths into sim_dir
            control_file = resolve_config_value(
                config,
                lambda: config.model.gsflow.control_file,
                dict_key='GSFLOW_CONTROL_FILE',
            )
            self._localize_control_paths(settings_dir, sim_dir, control_file)
            self._localize_nam_paths(settings_dir, sim_dir)

            install_path = resolve_config_value(
                config,
                lambda: config.model.gsflow.install_path,
                dict_key='GSFLOW_INSTALL_PATH',
            )
            exe_name = resolve_config_value(config, lambda: config.model.gsflow.exe, dict_key='GSFLOW_EXE')
            gsflow_exe = resolve_install_executable(install_path, data_dir, 'installs/gsflow/bin', exe_name)

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
            timeout = resolve_config_value(config, lambda: config.model.gsflow.timeout, dict_key='GSFLOW_TIMEOUT')

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

    def read_simulated_streamflow(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Optional[pd.Series]:
        """Routed outlet streamflow (m3/s) for this trial.

        In COUPLED GSFLOW the basin outlet flow is routed through MODFLOW
        SFR and reported as StreamOut_Q in gsflow.csv (m3/day); the PRMS
        statvar 'basin_cfs' is NOT the routed streamflow (near-zero in
        coupled mode), so the CSV is the authoritative source. Shared
        with the postprocessor (models.gsflow.extractor).
        """
        from symfluence.models.gsflow.extractor import extract_gsflow_streamflow
        sim_dir = Path(kwargs.get('sim_dir', output_dir))
        return extract_gsflow_streamflow(sim_dir, kwargs.get('settings_dir'))

    def additional_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        *,
        project_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Snow-cover and glacier mass-balance scores when GSFLOW_CRYOSPHERE_CONSTRAINTS is on.

        Same scores as PRMS (GSFLOW embeds PRMS); always scored over the
        calibration period, also when ``period`` overrides the streamflow
        window.
        """
        if not resolve_config_value(
            config,
            lambda: config.model.gsflow.cryosphere_constraints,
            dict_key='GSFLOW_CRYOSPHERE_CONSTRAINTS',
        ):
            return {}
        from symfluence.models.prms.cryosphere import calculate_constraints
        sim_dir = Path(kwargs.get('sim_dir', output_dir))
        settings_dir = kwargs.get('settings_dir')
        statvar = next(iter(sorted(sim_dir.glob('statvar*'))), None)
        if statvar is None:
            raise ModelExecutionError('No statvar for cryosphere constraints')
        param_dir = Path(settings_dir) if settings_dir else project_dir / 'settings' / 'GSFLOW'
        return calculate_constraints(
            statvar,
            param_dir / resolve_config_value(
                config,
                lambda: config.model.gsflow.parameter_file,
                dict_key='GSFLOW_PARAMETER_FILE',
            ),
            project_dir,
            resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD'),
            resolve_config_value(
                config,
                lambda: config.model.gsflow.snow_cover_error_scale,
                dict_key='GSFLOW_SNOW_COVER_ERROR_SCALE',
            ),
            resolve_config_value(
                config,
                lambda: config.model.gsflow.glacier_mb_error_scale,
                dict_key='GSFLOW_GLACIER_MB_ERROR_SCALE',
            ))

    def _extract_streamflow_from_csv(self, csv_file: Path) -> Optional[pd.Series]:
        """Basin outlet streamflow (m3/s) from the GSFLOW budget CSV."""
        from symfluence.models.gsflow.extractor import read_gsflow_csv_streamflow
        try:
            return read_gsflow_csv_streamflow(csv_file)
        except (OSError, ValueError, KeyError) as e:
            self.logger.error(f"Error reading GSFLOW CSV {csv_file}: {e}", exc_info=True)
            return None

    def _extract_streamflow_from_statvar(self, statvar_file: Path) -> Optional[pd.Series]:
        """Streamflow (m3/s) from a GSFLOW/PRMS statvar file (shared parser)."""
        from symfluence.models.prms.statvar import read_statvar_streamflow
        try:
            return read_statvar_streamflow(statvar_file).rename('GSFLOW_discharge_cms')
        except (OSError, ModelExecutionError) as e:
            self.logger.error(f"Error extracting GSFLOW streamflow: {e}", exc_info=True)
            return None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_gsflow_parameters_worker(task_data)


def _evaluate_gsflow_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for process pool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        GSFLOWWorker,
        task_data,
        model_name='GSFLOW',
        extra_env=MACOS_MALLOC_QUIET_ENV,
    )
