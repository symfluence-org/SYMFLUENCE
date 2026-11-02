# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB calibration worker."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R
from symfluence.models.lisflood.runner import _find_pcraster_site_packages

from .parameter_manager import PCRGLOBWBParameterManager


@R.workers.add('PCRGLOBWB')
class PCRGLOBWBWorker(BaseWorker):
    """Calibration worker for PCR-GLOBWB."""

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        routing_params = {k: v for k, v in params.items() if k.startswith('ROUTE_')}
        model_params = {k: v for k, v in params.items() if not k.startswith('ROUTE_')}
        self._routing_params = routing_params

        original_dir = self._get_original_settings_dir()

        # First call: set up process dir with symlinks (fast, ~0s)
        # Subsequent calls: only update the calibrated .map files
        self._setup_process_dir(original_dir, settings_dir)

        # Patch INI output dir to process-specific location
        output_dir = kwargs.get('output_dir', settings_dir / 'output')
        self._patch_ini_output(settings_dir, output_dir)

        # Update only the calibrated parameters
        pm = PCRGLOBWBParameterManager(self.config, self.logger, settings_dir)
        return pm.update_model_files(model_params, settings_dir)

    def _setup_process_dir(self, original_dir: Path, settings_dir: Path) -> None:
        """Set up process isolation directory using symlinks for speed.

        Static files (forcing, clone map, non-calibrated .map files) are
        symlinked to the original. Only the INI and calibrated .map files
        are real copies that get modified each iteration.
        """
        settings_dir.mkdir(parents=True, exist_ok=True)
        target_params = settings_dir / 'parameters'

        # Clone map: symlink (never changes)
        clone_dst = settings_dir / 'clone.map'
        clone_src = original_dir / 'clone.map'
        if not clone_dst.exists() and clone_src.exists():
            clone_dst.symlink_to(clone_src)

        # Forcing: symlink entire directory (never changes)
        forcing_dst = settings_dir / 'forcing'
        forcing_src = original_dir / 'forcing'
        if not forcing_dst.exists() and forcing_src.exists():
            forcing_dst.symlink_to(forcing_src)

        # INI: always copy fresh (gets patched each iteration)
        ini_src = original_dir / 'setup.ini'
        ini_dst = settings_dir / 'setup.ini'
        if ini_src.exists():
            shutil.copy2(ini_src, ini_dst)

        # Parameters directory: symlink static files, copy calibrated ones
        original_params = original_dir / 'parameters'
        if not original_params.exists():
            return

        target_params.mkdir(parents=True, exist_ok=True)

        calibrated_files = {f'{p}.map' for p in PCRGLOBWBParameterManager.MAP_FILE_PARAMS}

        for src_file in original_params.iterdir():
            dst_file = target_params / src_file.name
            if dst_file.exists():
                if src_file.name in calibrated_files:
                    # Re-copy calibrated files (will be overwritten by parameter manager)
                    if dst_file.is_symlink():
                        dst_file.unlink()
                    else:
                        continue  # already a real file from previous iteration
                else:
                    continue  # static symlink already exists

            if src_file.name in calibrated_files:
                shutil.copy2(src_file, dst_file)
            else:
                dst_file.symlink_to(src_file)

    def run_model(self, config: Dict, settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        ini_path = settings_dir / 'setup.ini'
        if not ini_path.exists():
            self.logger.error(f"INI not found: {ini_path}")
            return False

        env = dict(os.environ)

        # PCR-GLOBWB install path
        install_path = resolve_config_value(config, lambda: config.model.pcrglobwb.install_path, 'default', 'PCRGLOBWB_INSTALL_PATH')
        data_dir = resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR')
        pcrglobwb_dir = resolve_install_dir(install_path, data_dir, 'installs/pcrglobwb')

        env['PYTHONPATH'] = str(pcrglobwb_dir)

        # Discover PCRaster
        pcraster_site = _find_pcraster_site_packages()
        if pcraster_site:
            env['PYTHONPATH'] = f"{pcraster_site}:{env['PYTHONPATH']}"
            pcraster_prefix = str(Path(pcraster_site).parent.parent.parent)
            env['CONDA_PREFIX'] = pcraster_prefix
            env['PATH'] = f"{pcraster_prefix}/bin:{env.get('PATH', '')}"

        exe_name = resolve_config_value(config, lambda: config.model.pcrglobwb.exe, 'deterministic_runner.py', 'PCRGLOBWB_EXE')
        runner_script = pcrglobwb_dir / exe_name
        cmd = [sys.executable, str(runner_script), str(ini_path)]

        timeout = int(resolve_config_value(config, lambda: config.model.pcrglobwb.timeout, 14400, 'PCRGLOBWB_TIMEOUT'))
        try:
            result = run_subprocess(
                cmd, cwd=str(settings_dir), env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                self.logger.warning(f"PCR-GLOBWB failed (rc={result.returncode})")
                if result.stderr:
                    self.logger.debug(result.stderr[-500:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.logger.warning(f"PCR-GLOBWB timed out after {timeout}s")
            return False

    def calculate_metrics(self, output_dir: Path, config: Dict, **kwargs) -> Dict[str, float]:
        try:
            sim = self._load_simulated_streamflow(output_dir, config)
            obs = self._load_observations(config)

            if sim is None or obs is None or len(sim) == 0 or len(obs) == 0:
                return {'KGE': -999.0, 'NSE': -999.0}

            # Resample sub-daily to daily if needed
            if hasattr(sim.index, 'freq') and sim.index.freq and sim.index.freq.n < 86400:
                sim = sim.resample('D').mean()

            # Filter to calibration period
            cal_period = resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD')
            if cal_period:
                parts = cal_period.split(',')
                if len(parts) == 2:
                    sim = sim[parts[0].strip():parts[1].strip()]
                    obs = obs[parts[0].strip():parts[1].strip()]

            # Align
            combined = pd.concat([sim.rename('sim'), obs.rename('obs')], axis=1).dropna()
            if len(combined) < 30:
                self.logger.warning(f"Only {len(combined)} aligned data points")
                return {'KGE': -999.0, 'NSE': -999.0}

            metrics = self._streamflow_metrics.calculate_metrics(
                combined['obs'].values, combined['sim'].values
            )
            return metrics
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Metric calculation failed: {e}")
            return {'KGE': -999.0, 'NSE': -999.0}

    def _load_simulated_streamflow(self, output_dir: Path, config: Dict) -> Optional[pd.Series]:
        """Load discharge from PCR-GLOBWB NetCDF output."""
        import xarray as xr

        # Search in netcdf/ subdirectory (PCR-GLOBWB convention)
        for search_dir in [output_dir / 'netcdf', output_dir]:
            if not search_dir.exists():
                continue
            for pattern in ['discharge_dailyTot_output.nc', 'discharge_*_output.nc']:
                matches = list(search_dir.glob(pattern))
                if matches:
                    ds = xr.open_dataset(matches[0])
                    q_var = None
                    for v in ['discharge', 'Qsim', 'Q']:
                        if v in ds.data_vars:
                            q_var = ds[v]
                            break
                    if q_var is None:
                        ds.close()
                        continue

                    spatial_dims = [d for d in q_var.dims if d not in ['time']]
                    if spatial_dims:
                        streamflow = q_var.max(dim=spatial_dims)
                    else:
                        streamflow = q_var
                    series = streamflow.to_series()
                    ds.close()

                    # Apply post-hoc routing if available
                    if hasattr(self, '_routing_params') and self._routing_params:
                        series = self._apply_routing(series, self._routing_params)

                    return series

        self.logger.error(f"No discharge output found in {output_dir}")
        return None

    def _load_observations(self, config: Dict) -> Optional[pd.Series]:
        """Load observed streamflow (model-ready store first, CSV fallback)."""
        data_dir = config.get('SYMFLUENCE_DATA_DIR', '.')
        domain = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
        project_dir = Path(data_dir) / f'domain_{domain}'
        values, index = self._streamflow_metrics.load_observations(
            config, project_dir, domain, resample_freq=None,
        )
        if values is None or index is None:
            self.logger.error("Observations not found (store or preprocessed CSV)")
            return None
        return pd.Series(values, index=index, name='discharge_cms').dropna()

    @staticmethod
    def _apply_routing(q: pd.Series, params: Dict[str, float]) -> pd.Series:
        """Two-store linear reservoir routing for lumped mode smoothing."""
        alpha = params.get('ROUTE_ALPHA', 0.5)
        beta = params.get('ROUTE_BETA', 0.98)
        split = params.get('ROUTE_SPLIT', 0.5)
        baseflow = params.get('ROUTE_BASEFLOW', 0.0)

        q_in = q.values
        n = len(q_in)
        s_fast = np.zeros(n)
        s_slow = np.zeros(n)

        for i in range(1, n):
            s_fast[i] = alpha * s_fast[i - 1] + split * q_in[i]
            s_slow[i] = beta * s_slow[i - 1] + (1.0 - split) * q_in[i]

        q_out = (1.0 - alpha) * s_fast + (1.0 - beta) * s_slow + baseflow
        return pd.Series(np.maximum(q_out, 0.0), index=q.index, name=q.name)

    def _get_original_settings_dir(self) -> Path:
        data_dir = self.config.get('SYMFLUENCE_DATA_DIR', '.')
        domain = resolve_config_value(self.config, lambda: self.config.domain.name, '', 'DOMAIN_NAME')
        return Path(data_dir) / f'domain_{domain}' / 'settings' / 'PCRGLOBWB'

    def _patch_ini_output(self, settings_dir: Path, output_dir: Path) -> None:
        """Redirect output and parameter paths to process-specific directory."""
        import configparser
        ini_path = settings_dir / 'setup.ini'
        if not ini_path.exists():
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        original_params = str(self._get_original_settings_dir() / 'parameters')
        process_params = str(settings_dir / 'parameters')

        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ini_path)

        if 'globalOptions' in ini:
            ini['globalOptions']['outputDir'] = str(output_dir)

        # Rewrite all parameter paths from original → process-specific
        for section in ini.sections():
            for key, val in ini[section].items():
                if original_params in val:
                    ini[section][key] = val.replace(original_params, process_params)

        with open(ini_path, 'w') as f:
            ini.write(f)
