# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM calibration worker."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

from .parameter_manager import CWatMParameterManager


@R.workers.add('CWATM')
class CWatMWorker(BaseWorker):
    """Calibration worker for CWatM."""

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        original_dir = self._get_original_settings_dir()
        self._setup_process_dir(original_dir, settings_dir)

        output_dir = kwargs.get('output_dir', settings_dir / 'output')
        self._patch_ini(settings_dir, output_dir)

        pm = CWatMParameterManager(self.config, self.logger, settings_dir)
        return pm.update_model_files(params, settings_dir)

    def run_model(self, config: Dict, settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        ini_path = settings_dir / 'settings.ini'
        if not ini_path.exists():
            self.logger.error(f"INI not found: {ini_path}")
            return False

        env = dict(os.environ)
        install_path = resolve_config_value(config, lambda: config.model.cwatm.install_path, 'default', 'CWATM_INSTALL_PATH')
        data_dir = resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR')
        cwatm_dir = resolve_install_dir(install_path, data_dir, 'installs/cwatm')

        env['PYTHONPATH'] = str(cwatm_dir)

        exe_name = resolve_config_value(config, lambda: config.model.cwatm.exe, 'run_cwatm.py', 'CWATM_EXE')
        cmd = [sys.executable, str(cwatm_dir / exe_name), str(ini_path), '-q']

        timeout = int(resolve_config_value(config, lambda: config.model.cwatm.timeout, 14400, 'CWATM_TIMEOUT'))
        try:
            result = run_subprocess(
                cmd, cwd=str(settings_dir), env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                self.logger.warning(f"CWatM failed (rc={result.returncode})")
                if result.stderr:
                    self.logger.debug(result.stderr[-500:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.logger.warning(f"CWatM timed out after {timeout}s")
            return False

    def calculate_metrics(self, output_dir: Path, config: Dict, **kwargs) -> Dict[str, float]:
        try:
            sim = self._load_simulated_streamflow(output_dir)
            obs = self._load_observations(config)

            if sim is None or obs is None or len(sim) == 0 or len(obs) == 0:
                return {'KGE': -999.0, 'NSE': -999.0}

            cal_period = resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD')
            if cal_period:
                parts = cal_period.split(',')
                if len(parts) == 2:
                    sim = sim[parts[0].strip():parts[1].strip()]
                    obs = obs[parts[0].strip():parts[1].strip()]

            combined = pd.concat([sim.rename('sim'), obs.rename('obs')], axis=1).dropna()
            if len(combined) < 30:
                return {'KGE': -999.0, 'NSE': -999.0}

            return self._streamflow_metrics.calculate_metrics(
                combined['obs'].values, combined['sim'].values
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.logger.error(f"Metric calculation failed: {e}")
            return {'KGE': -999.0, 'NSE': -999.0}

    def _load_simulated_streamflow(self, output_dir: Path) -> Optional[pd.Series]:
        import xarray as xr

        for pattern in ['discharge_daily.nc', 'dis_outlet_daily.nc', '*discharge*.nc']:
            matches = list(output_dir.glob(pattern))
            if matches:
                ds = xr.open_dataset(matches[0])
                for v in ['discharge', 'dis_outlet', 'Qsim']:
                    if v in ds.data_vars:
                        data = ds[v]
                        spatial_dims = [d for d in data.dims if d not in ['time']]
                        if spatial_dims:
                            data = data.max(dim=spatial_dims)
                        series = data.to_series()
                        ds.close()
                        return series
                ds.close()

        self.logger.error(f"No discharge output found in {output_dir}")
        return None

    def _load_observations(self, config: Dict) -> Optional[pd.Series]:
        data_dir = config.get('SYMFLUENCE_DATA_DIR', '.')
        domain = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
        obs_dir = Path(data_dir) / f'domain_{domain}' / 'observations' / 'streamflow' / 'preprocessed'
        if not obs_dir.exists():
            return None

        csv_files = sorted(obs_dir.glob('*.csv'))
        if not csv_files:
            return None

        df = pd.read_csv(csv_files[0], parse_dates=[0], index_col=0)
        col = 'discharge_cms' if 'discharge_cms' in df.columns else df.columns[0]
        return df[col].dropna()

    def _get_original_settings_dir(self) -> Path:
        data_dir = self.config.get('SYMFLUENCE_DATA_DIR', '.')
        domain = resolve_config_value(self.config, lambda: self.config.domain.name, '', 'DOMAIN_NAME')
        return Path(data_dir) / f'domain_{domain}' / 'settings' / 'CWATM'

    def _setup_process_dir(self, original_dir: Path, settings_dir: Path) -> None:
        """Symlink static files, copy settings.ini for patching."""
        settings_dir.mkdir(parents=True, exist_ok=True)

        # INI: always copy fresh
        ini_src = original_dir / 'settings.ini'
        ini_dst = settings_dir / 'settings.ini'
        if ini_src.exists():
            shutil.copy2(ini_src, ini_dst)

        # Forcing: symlink
        forcing_dst = settings_dir / 'forcing'
        forcing_src = original_dir / 'forcing'
        if not forcing_dst.exists() and forcing_src.exists():
            forcing_dst.symlink_to(forcing_src)

    def _patch_ini(self, settings_dir: Path, output_dir: Path) -> None:
        """Redirect output directory in settings.ini."""
        import re

        ini_path = settings_dir / 'settings.ini'
        if not ini_path.exists():
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        content = ini_path.read_text()
        content = re.sub(r'(?m)^OUT_Dir\s*=.*$', f'OUT_Dir = {output_dir}', content)
        content = re.sub(r'(?m)^PathOut\s*=.*$', f'PathOut = {output_dir}', content)
        ini_path.write_text(content)
