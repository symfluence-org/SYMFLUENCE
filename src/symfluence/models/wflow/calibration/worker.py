# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow Worker."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xarray as xr

from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    run_worker_task,
)
from symfluence.core.calibration.workers.streamflow_worker import StreamflowWorker
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.modeling.utilities.periods import split_period
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R
from symfluence.models.wflow.lumped_discharge import is_runoff_flux_output, lumped_discharge_cms


@R.workers.add('WFLOW')
class WflowWorker(StreamflowWorker):
    """Worker for Wflow model calibration."""

    def __init__(self, config=None, logger=None):
        super().__init__(config, logger)
        self._routing_params = None  # Populated per iteration by apply_parameters


    def apply_parameters(self, params, settings_dir, **kwargs):
        try:
            # Routing params are applied post-hoc to the lumped runoff series.
            # They still go to the parameter manager, which records them beside
            # the staticmaps for the postprocessor.
            self._routing_params = {k: v for k, v in params.items() if k.startswith('ROUTE_')}

            config = kwargs.get('config', self.config) or {}
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            original_settings = data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW'
            settings_dir = Path(settings_dir)
            staticmaps_name = resolve_config_value(config, lambda: config.model.wflow.staticmaps_file, 'wflow_staticmaps.nc', 'WFLOW_STATICMAPS_FILE')

            # Copy staticmaps from original settings to process directory
            original_file = original_settings / staticmaps_name
            target_file = settings_dir / staticmaps_name
            if original_file.exists() and original_file != target_file:
                shutil.copy2(original_file, target_file)

            # Copy and patch the TOML config for process-specific paths
            config_file = resolve_config_value(config, lambda: config.model.wflow.config_file, 'wflow_sbm.toml', 'WFLOW_CONFIG_FILE')
            original_toml = original_settings / config_file
            target_toml = settings_dir / config_file
            out_dir = kwargs.get('output_dir') or kwargs.get('proc_output_dir')
            # Patch when running in an isolated process dir (target != source) or
            # whenever an explicit output dir is given (final evaluation runs in
            # the main settings dir, so target == source but we still need to
            # redirect dir_output so the evaluator can find the run).
            if original_toml.exists() and (original_toml != target_toml or out_dir):
                self._patch_toml(original_toml, target_toml, settings_dir, out_dir)

            from .parameter_manager import WflowParameterManager
            pm = WflowParameterManager(config, self.logger, settings_dir)
            return pm.update_model_files(params, settings_dir)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to apply Wflow parameters: {e}")
            return False

    def _patch_toml(self, source_toml, target_toml, settings_dir, output_dir):
        """Copy TOML and patch path_static and dir_output for process isolation.

        Paths are written with forward slashes: backslashes are escapes in TOML
        basic strings, and Wflow accepts forward slashes on every platform.
        """
        content = source_toml.read_text()
        # Point path_static to the process-specific staticmaps
        content = re.sub(
            r'(path_static\s*=\s*)"[^"]*"',
            lambda m: f'{m.group(1)}"{(settings_dir / "wflow_staticmaps.nc").as_posix()}"',
            content,
        )
        # Redirect output to process-specific dir if available
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            content = re.sub(
                r'(dir_output\s*=\s*)"[^"]*"',
                lambda m: f'{m.group(1)}"{output_dir.as_posix()}"',
                content,
            )
            content = re.sub(
                r'(path\s*=\s*)"(?:[^"]*/)?output\.csv"',
                lambda m: f'{m.group(1)}"{(output_dir / "output.csv").as_posix()}"',
                content,
            )
            # Remove state I/O paths for calibration (cold start each iteration)
            # Keep [state] and [state.variables] sections — Wflow requires them
            content = re.sub(r'path_input\s*=\s*"[^"]*"\n?', '', content)
            content = re.sub(r'path_output\s*=\s*"[^"]*"\n?', '', content)
        target_toml.write_text(content)

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        try:
            install_path = resolve_config_value(config, lambda: config.model.wflow.install_path, 'default', 'WFLOW_INSTALL_PATH')
            data_dir = resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR')
            exe_path = resolve_install_dir(install_path, data_dir, 'installs/wflow/bin') / 'wflow_cli'
            config_file = resolve_config_value(config, lambda: config.model.wflow.config_file, 'wflow_sbm.toml', 'WFLOW_CONFIG_FILE')
            timeout = int(resolve_config_value(config, lambda: config.model.wflow.timeout, 7200, 'WFLOW_TIMEOUT'))
            if not exe_path.exists():
                self.logger.error(f"wflow_cli not found at {exe_path}")
                return False
            # Final evaluation applies parameters through the parameter manager,
            # bypassing apply_parameters(). Always bind native output paths to
            # the requested run directory before launching Wflow.
            settings_dir = Path(settings_dir).resolve()
            output_dir = Path(output_dir).resolve()
            toml_path = settings_dir / config_file
            self._patch_toml(toml_path, toml_path, settings_dir, output_dir)
            if resolve_config_value(config, lambda: config.model.wflow.calibration_only_period, False, 'WFLOW_CALIBRATION_ONLY_PERIOD'):
                end = (config['EXPERIMENT_TIME_END'] if kwargs.get('final_evaluation', False)
                       else split_period(config['CALIBRATION_PERIOD'], allow_extra=True, strict=True)[1])
                native_end = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
                content = re.sub(r'^endtime\s*=.*$', f'endtime = {native_end:%Y-%m-%dT00:00:00}',
                                 toml_path.read_text(), flags=re.MULTILINE)
                toml_path.write_text(content)
            command = [str(exe_path), str(toml_path)]
            result = run_subprocess(
                command,
                cwd=str(settings_dir), capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0 and result.stderr:
                self.logger.warning(f"Wflow stderr: {result.stderr[-500:]}")
            if result.returncode == 0 and resolve_config_value(config, lambda: config.model.wflow.spatial_mode, None, 'WFLOW_SPATIAL_MODE') == 'distributed':
                from ..balance import check
                check(output_dir)
                import tomllib
                native = tomllib.loads(toml_path.read_text())
                expected = pd.date_range(pd.Timestamp(native['time']['starttime']) + pd.Timedelta(days=1),
                                         pd.Timestamp(native['time']['endtime']), freq='D')
                actual = pd.DatetimeIndex(pd.read_csv(output_dir / 'output.csv', usecols=['time']).time)
                if not actual.equals(expected):
                    raise ModelExecutionError('Incomplete Wflow daily output coverage')
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Wflow execution failed: {e}")
            return False

    def read_simulated_streamflow(self, output_dir, config, **kwargs):
        """This trial's outlet discharge (m3/s), daily means when Wflow ran sub-daily."""
        sim = self._load_simulated_streamflow(Path(output_dir), config, self._routing_params or None)
        return None if sim is None else self.daily_mean_if_subdaily(sim)

    def additional_metrics(self, output_dir, config, *, project_dir, sim, obs, **kwargs):
        """Seasonal bias (when a COMPOSITE_METRIC component) and cryosphere constraints."""
        metrics: Dict[str, Any] = {}
        if any(str(k).upper() == 'SEASONAL_BIAS_SCORE'
               for k in (resolve_config_value(config, lambda: config.optimization.composite_metric, None, 'COMPOSITE_METRIC') or {})):
            from .seasonal import seasonal_bias_score
            metrics['seasonal_bias_score'] = seasonal_bias_score(sim, obs)
        if resolve_config_value(config, lambda: config.model.wflow.cryosphere_constraints, False, 'WFLOW_CRYOSPHERE_CONSTRAINTS'):
            from ..cryosphere import calculate_constraints

            # Spatial geometry is immutable during calibration; parameter updates
            # cannot change the areas used to weight snow/ice observations.
            metrics.update(calculate_constraints(
                output_dir, project_dir / 'settings/WFLOW', project_dir,
                resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD'),
                resolve_config_value(config, lambda: config.model.wflow.snow_cover_error_scale, .2, 'WFLOW_SNOW_COVER_ERROR_SCALE'),
                resolve_config_value(config, lambda: config.model.wflow.glacier_mb_error_scale, 1., 'WFLOW_GLACIER_MB_ERROR_SCALE'),
                resolve_config_value(config, lambda: config.model.wflow.snow_cover_threshold_mm, 5., 'WFLOW_SNOW_COVER_THRESHOLD_MM')))
        return metrics

    def _load_simulated_streamflow(self, output_dir: Path, config: dict = None, routing_params: dict = None):
        """Load Q from Wflow CSV or netCDF output.

        For lumped models (``Q`` = soil_surface_water__net_runoff_volume_flux)
        the discharge is the full outflow of the cell: net runoff plus the
        river-fraction runoff and lateral subsurface flow columns, converted
        from mm/dt to m³/s using basin area and the TOML's model step, then
        the linear reservoir routing (see ``lumped_discharge``).
        """
        q = None
        table = None
        # Try CSV first (primary format from [output.csv] TOML section)
        csv_matches = list(output_dir.glob('output*.csv'))
        if csv_matches:
            df = pd.read_csv(csv_matches[0], parse_dates=[0], index_col=0, on_bad_lines='skip')
            table = df
            for col in ['Q', 'Q_av', 'q_av']:
                if col in df.columns:
                    q = df[col]
                    break
            if q is None and len(df.columns) == 1:
                q = df.iloc[:, 0]
        # Fallback: netCDF
        if q is None:
            nc_matches = list(output_dir.glob('output*.nc'))
            if nc_matches:
                ds = xr.open_dataset(nc_matches[0])
                for var in ['Q', 'Q_av', 'q_av']:
                    if var in ds.data_vars:
                        q_var = ds[var]
                        spatial_dims = [d for d in q_var.dims if d not in ['time']]
                        q = q_var.max(dim=spatial_dims).to_series() if spatial_dims else q_var.to_series()
                        break
                ds.close()
        if q is None:
            return None
        # Lumped runoff flux (mm per model step) -> m³/s, then the post-hoc
        # routing; the postprocessor goes through the same function.
        toml_path = self._find_run_toml(output_dir, config)
        if toml_path is not None and is_runoff_flux_output(toml_path) and config:
            # The whole table when there is one: the lumped outflow is the sum
            # of several columns (net runoff, river fraction, subsurface).
            q = lumped_discharge_cms(
                table if table is not None and q.name in table.columns else q,
                toml_path, self._staticmaps_path(config), routing_params)
        from symfluence.models.wflow.extractor import is_distributed_wflow, to_period_start_labels
        if q is not None and is_distributed_wflow(config):
            q = to_period_start_labels(q)
        return q

    def _find_run_toml(self, output_dir: Path, config: dict = None):
        """The run's TOML: beside the outputs, else the domain settings."""
        toml_name = resolve_config_value(config, lambda: config.model.wflow.config_file, 'wflow_sbm.toml', 'WFLOW_CONFIG_FILE') if config else 'wflow_sbm.toml'
        # Search multiple possible locations for the TOML
        search_dirs = [output_dir, output_dir.parent / 'settings', output_dir.parent]
        # Also check the original domain settings (works during DDS calibration)
        if config:
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            search_dirs.append(data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW')
        for settings_dir in search_dirs:
            toml_path = settings_dir / toml_name
            if toml_path.exists():
                return toml_path
        return None

    def _is_runoff_flux_output(self, output_dir: Path, config: dict = None):
        """Check if the TOML output uses net_runoff_volume_flux (mm/dt units)."""
        toml_path = self._find_run_toml(output_dir, config)
        return toml_path is not None and is_runoff_flux_output(toml_path)

    @staticmethod
    def _staticmaps_path(config) -> Path:
        domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
        data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
        staticmaps_name = resolve_config_value(config, lambda: config.model.wflow.staticmaps_file, 'wflow_staticmaps.nc', 'WFLOW_STATICMAPS_FILE')
        return data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW' / staticmaps_name

    @staticmethod
    def evaluate_worker_function(task_data):
        """Static worker function for process pool execution."""
        return _evaluate_wflow_parameters_worker(task_data)


def _evaluate_wflow_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for process pool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        WflowWorker,
        task_data,
        model_name='Wflow',
        extra_env=MACOS_MALLOC_QUIET_ENV,
    )
