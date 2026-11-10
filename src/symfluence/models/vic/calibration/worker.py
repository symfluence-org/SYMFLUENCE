# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC Worker

Worker implementation for VIC model optimization.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    BaseWorker,
    run_worker_task,
)
from symfluence.core.exceptions import ModelExecutionError, ValidationError
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.modeling.utilities.periods import split_period
from symfluence.core.path_resolver import resolve_install_executable
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R
from symfluence.models.vic.constants import target_snow_constants


@R.workers.add('VIC')
class VICWorker(BaseWorker):
    """
    Worker for VIC model calibration.

    Handles parameter application, VIC execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize VIC worker.

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
        Apply parameters to VIC parameter file.

        Args:
            params: Parameter values to apply
            settings_dir: VIC settings directory (contains parameters/ subdirectory)
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying VIC parameters to {settings_dir}")

            # The settings_dir should contain a 'parameters' subdirectory
            params_dir = settings_dir / 'parameters'

            # Always copy fresh from the original settings location
            # to ensure dimensions match the current domain file
            config = kwargs.get('config', self.config) or {}
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            original_params_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'VIC' / 'parameters'

            if original_params_dir.exists():
                params_dir.mkdir(parents=True, exist_ok=True)
                for f in original_params_dir.glob('*.nc'):
                    dest = params_dir / f.name
                    if f.resolve() != dest.resolve():
                        shutil.copy2(f, dest)
                self.logger.debug(f"Copied VIC parameters from {original_params_dir} to {params_dir}")
            elif not params_dir.exists():
                self.logger.error(f"VIC parameters directory not found: {params_dir} "
                                  f"(original also missing: {original_params_dir})")
                return False

            # Find parameter file
            params_file = params_dir / 'vic_params.nc'
            if not params_file.exists():
                # Try to find any .nc file
                nc_files = list(params_dir.glob('*params*.nc'))
                if nc_files:
                    params_file = nc_files[0]
                else:
                    self.logger.error(f"VIC parameter file not found in {params_dir}")
                    return False

            # Update parameter file
            return self._update_params_file(params_file, params)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying VIC parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_params_file(self, params_file: Path, params: Dict[str, float]) -> bool:
        """
        Update VIC parameter NetCDF file with new values.

        Args:
            params_file: Path to parameter NetCDF file
            params: Parameters to update

        Returns:
            True if successful
        """
        # Parameter to variable mapping
        PARAM_VAR_MAP = {
            'infilt': 'infilt',
            'Ds': 'Ds',
            'Dsmax': 'Dsmax',
            'Ws': 'Ws',
            'c': 'c',
            'depth1': 'depth',
            'depth2': 'depth',
            'depth3': 'depth',
            'Ksat': 'Ksat',
            'expt': 'expt',
            'Wcr_FRACT': 'Wcr_FRACT',
            'Wpwp_FRACT': 'Wpwp_FRACT',
            'snow_rough': 'snow_rough',
            'max_snow_albedo': 'max_snow_albedo',
        }

        # Special parameters handled outside the standard loop
        SPECIAL_PARAMS = {'elev_offset', 'Wpwp_ratio', 'min_rain_temp', 'max_snow_temp',
                          'Ksat_decay', 'expt_increase'}

        LAYER_PARAMS = {
            'depth1': 0,
            'depth2': 1,
            'depth3': 2,
        }

        try:
            ds = xr.open_dataset(params_file)
            ds = ds.load()

            # Derive Wpwp_FRACT from Wpwp_ratio × Wcr_FRACT (smooth reparameterization)
            if 'Wpwp_ratio' in params and 'Wcr_FRACT' in params:
                params['Wpwp_FRACT'] = params['Wpwp_ratio'] * params['Wcr_FRACT']

            # Validate parameter combinations before applying
            validation_error = self._validate_params(params)
            if validation_error:
                ds.close()
                self.logger.debug(f"Parameter validation failed: {validation_error}")
                return False

            # Store snow constants for the worker-local VIC constants file
            self._global_file_params = {}
            if 'min_rain_temp' in params:
                self._global_file_params['SNOW_MIN_RAIN_TEMP'] = params['min_rain_temp']
            if 'max_snow_temp' in params:
                self._global_file_params['SNOW_MAX_SNOW_TEMP'] = params['max_snow_temp']

            # Handle elev_offset: shift snow band elevations to control snowmelt timing
            # Positive offset raises band elevations → cooler bands → delayed snowmelt
            # Negative offset lowers band elevations → warmer bands → earlier snowmelt
            if 'elev_offset' in params and 'elevation' in ds:
                offset = params['elev_offset']
                for band in range(ds['elevation'].shape[0]):
                    mask = ~np.isnan(ds['elevation'].values[band])
                    ds['elevation'].values[band][mask] += offset
                self.logger.debug(f"Applied elev_offset = {offset:.0f}m to snow band elevations")

                # Recalculate Pfactor to match shifted elevations
                if 'Pfactor' in ds and 'AreaFract' in ds:
                    config = self.config or {}
                    pfactor_per_km = float(resolve_config_value(config, lambda: config.model.vic.pfactor_per_km, 0.0005, 'VIC_PFACTOR_PER_KM'))
                    elevs = ds['elevation'].values
                    areas = ds['AreaFract'].values
                    mean_elev = float(np.nansum(elevs * areas) / np.nansum(areas))
                    for band in range(ds['Pfactor'].shape[0]):
                        band_elev = ds['elevation'].values[band]
                        mask = ~np.isnan(band_elev)
                        ds['Pfactor'].values[band][mask] = np.clip(
                            1.0 + pfactor_per_km * (band_elev[mask] - mean_elev),
                            0.5, 2.0,
                        )
                    self.logger.debug(
                        f"Recalculated Pfactor for shifted elevations "
                        f"(pfactor_per_km={pfactor_per_km}, mean_elev={mean_elev:.0f}m)"
                    )

            # Apply layer-specific Ksat with depth decay
            if 'Ksat' in params and 'Ksat' in ds:
                base_ksat = params['Ksat']
                ksat_decay = params.get('Ksat_decay', 1.0)
                for layer in range(ds['Ksat'].shape[0]):
                    layer_ksat = base_ksat * (ksat_decay ** layer)
                    mask = ~np.isnan(ds['Ksat'].values[layer])
                    ds['Ksat'].values[layer][mask] = layer_ksat
                self.logger.debug(
                    f"Updated Ksat: layer0={base_ksat:.1f}, decay={ksat_decay:.3f}"
                )

            # Apply layer-specific expt with depth increase
            if 'expt' in params and 'expt' in ds:
                base_expt = params['expt']
                expt_inc = params.get('expt_increase', 0.0)
                for layer in range(ds['expt'].shape[0]):
                    layer_expt = base_expt + expt_inc * layer
                    mask = ~np.isnan(ds['expt'].values[layer])
                    ds['expt'].values[layer][mask] = layer_expt
                self.logger.debug(
                    f"Updated expt: layer0={base_expt:.1f}, increase={expt_inc:.2f}/layer"
                )

            # Parameters already handled above — skip in generic loop
            LAYER_SPECIFIC_PARAMS = {'Ksat', 'expt'}

            for param_name, value in params.items():
                if param_name in SPECIAL_PARAMS or param_name in LAYER_SPECIFIC_PARAMS:
                    continue

                var_name = PARAM_VAR_MAP.get(param_name)
                if var_name is None or var_name not in ds:
                    continue

                if param_name in LAYER_PARAMS:
                    layer_idx = LAYER_PARAMS[param_name]
                    if 'nlayer' in ds[var_name].dims:
                        mask = ~np.isnan(ds[var_name].values[layer_idx])
                        ds[var_name].values[layer_idx][mask] = value
                else:
                    if len(ds[var_name].dims) == 2:
                        mask = ~np.isnan(ds[var_name].values)
                        ds[var_name].values[mask] = value
                    elif len(ds[var_name].dims) == 3:
                        for layer in range(ds[var_name].shape[0]):
                            mask = ~np.isnan(ds[var_name].values[layer])
                            ds[var_name].values[layer][mask] = value
                    else:
                        ds[var_name].values = value

                self.logger.debug(f"Updated {param_name} = {value}")

            # Always set init_moist to 50% of current soil capacity.
            # This ensures consistent initial conditions regardless of depth changes
            # during calibration, and prevents init_moist from exceeding max capacity
            # (which crashes VIC).
            if 'init_moist' in ds and 'depth' in ds and 'bulk_density' in ds and 'soil_density' in ds:
                for layer in range(ds['depth'].shape[0]):
                    depth_vals = ds['depth'].values[layer]
                    bulk_vals = ds['bulk_density'].values[layer]
                    soil_vals = ds['soil_density'].values[layer]

                    valid = ~np.isnan(depth_vals)
                    porosity = 1.0 - bulk_vals[valid] / soil_vals[valid]
                    max_moist = depth_vals[valid] * porosity * 1000.0
                    # Set init_moist to 50% of current max capacity
                    ds['init_moist'].values[layer][valid] = 0.5 * max_moist

                self.logger.debug("Set init_moist to 50% of current soil capacity")

            # Save
            temp_file = params_file.with_suffix('.nc.tmp')
            ds.to_netcdf(temp_file)
            ds.close()
            temp_file.replace(params_file)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating VIC parameters: {e}", exc_info=True)
            return False

    def _validate_params(self, params: Dict[str, float]) -> Optional[str]:
        """
        Validate parameter combinations before applying to VIC.

        Returns None if valid, or an error string if invalid.
        """
        # Wpwp_FRACT must be strictly less than Wcr_FRACT
        if 'Wpwp_FRACT' in params and 'Wcr_FRACT' in params:
            if params['Wpwp_FRACT'] >= params['Wcr_FRACT']:
                return f"Wpwp_FRACT ({params['Wpwp_FRACT']:.4f}) >= Wcr_FRACT ({params['Wcr_FRACT']:.4f})"

        # min_rain_temp must be less than max_snow_temp
        if 'min_rain_temp' in params and 'max_snow_temp' in params:
            if params['min_rain_temp'] >= params['max_snow_temp']:
                return f"min_rain_temp ({params['min_rain_temp']:.2f}) >= max_snow_temp ({params['max_snow_temp']:.2f})"

        # Soil depths must be positive
        for depth_key in ['depth1', 'depth2', 'depth3']:
            if depth_key in params and params[depth_key] <= 0:
                return f"{depth_key} ({params[depth_key]:.4f}) must be positive"

        # Dsmax must be positive
        if 'Dsmax' in params and params['Dsmax'] <= 0:
            return f"Dsmax ({params['Dsmax']:.4f}) must be positive"

        return None

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run VIC model.

        Args:
            config: Configuration dictionary
            settings_dir: VIC settings directory
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f'domain_{domain_name}'
            vic_settings_dir = project_dir / 'settings' / 'VIC'

            # Use sim_dir for output if provided
            vic_output_dir = Path(kwargs.get('sim_dir', output_dir))
            vic_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files
            self._cleanup_stale_output(vic_output_dir)

            # Get executable
            vic_exe = self._get_vic_executable(config, data_dir)
            if not vic_exe.exists():
                self.logger.error(f"VIC executable not found: {vic_exe}")
                return False

            # Get or create global parameter file
            global_param_file = self._get_or_create_global_file(
                config, vic_settings_dir, settings_dir, vic_output_dir
            )

            if not global_param_file.exists():
                self.logger.error(f"Global parameter file not found: {global_param_file}")
                return False

            # Build command
            cmd = [str(vic_exe), '-g', str(global_param_file)]

            # Set environment
            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            # Run with timeout
            timeout = resolve_config_value(config, lambda: config.model.vic.timeout, 300, 'VIC_TIMEOUT')

            stdout_file = vic_output_dir / 'vic_stdout.log'
            stderr_file = vic_output_dir / 'vic_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = run_subprocess(
                        cmd,
                        cwd=str(vic_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"VIC timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"VIC failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output
            output_files = list(vic_output_dir.glob('*.nc'))
            if not output_files:
                self._last_error = "No output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running VIC: {e}", exc_info=True)
            return False

    def _get_vic_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get VIC executable path."""
        install_path = resolve_config_value(config, lambda: config.model.vic.install_path, 'default', 'VIC_INSTALL_PATH')
        exe_name = resolve_config_value(config, lambda: config.model.vic.exe, 'vic_image.exe', 'VIC_EXE')

        return resolve_install_executable(install_path, data_dir, 'installs/vic/bin', exe_name)

    def _get_or_create_global_file(
        self,
        config: Dict[str, Any],
        vic_settings_dir: Path,
        settings_dir: Path,
        output_dir: Path
    ) -> Path:
        """Get or create a worker-specific global parameter file."""
        # First try to find existing global file
        global_file_name = resolve_config_value(config, lambda: config.model.vic.global_param_file, 'vic_global.txt', 'VIC_GLOBAL_PARAM_FILE')
        original_global = vic_settings_dir / global_file_name

        if not original_global.exists():
            self.logger.warning(f"Original global file not found: {original_global}")
            return original_global

        # Create worker-specific version with updated output path
        worker_global = settings_dir / global_file_name

        with open(original_global, 'r', encoding='utf-8') as f:
            content = f.read()

        content = target_snow_constants(
            content, vic_settings_dir, settings_dir,
            getattr(self, '_global_file_params', {}),
        )
        new_lines = []
        for line in content.splitlines():
            parts = line.split()
            key = parts[0] if parts else ''
            if key == 'RESULT_DIR':
                new_lines.append(f'RESULT_DIR {output_dir}')
            elif key == 'PARAMETERS':
                worker_params = settings_dir / 'parameters' / 'vic_params.nc'
                new_lines.append(f'PARAMETERS {worker_params}' if worker_params.exists() else line)
            else:
                new_lines.append(line)

        with open(worker_global, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

        return worker_global

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale VIC output files."""
        for pattern in ['*.nc', '*.log']:
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
        Calculate metrics from VIC output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Find output file
            output_files = list(sim_dir.glob('*fluxes*.nc'))
            if not output_files:
                output_files = list(sim_dir.glob('vic_output*.nc'))
            if not output_files:
                output_files = list(sim_dir.glob('*.nc'))

            if not output_files:
                self.logger.error(f"No VIC output files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_file = output_files[0]

            # Extract streamflow
            ds = xr.open_dataset(sim_file)

            # Basin-mean runoff + baseflow depth: the same area-weighted
            # aggregation the postprocessor uses for the results CSV.
            from symfluence.models.vic.extractor import (
                basin_mean_vic_runoff,
                load_vic_cell_area,
            )
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f'domain_{domain_name}'
            domain_file = resolve_config_value(
                config, lambda: config.model.vic.domain_file, 'vic_domain.nc', 'VIC_DOMAIN_FILE'
            )
            cell_area = load_vic_cell_area(project_dir / 'settings' / 'VIC' / 'parameters' / domain_file)
            try:
                total_runoff = basin_mean_vic_runoff(ds, cell_area).values
            except ModelExecutionError:
                ds.close()
                return {'kge': self.penalty_score, 'error': 'No runoff variable'}

            times = pd.to_datetime(ds['time'].values)
            ds.close()

            # Convert to m³/s using catchment area
            area_km2 = self._streamflow_metrics.get_catchment_area(
                config, project_dir, domain_name, source='shapefile'
            )
            area_m2 = area_km2 * 1e6

            # mm/day -> m³/s
            streamflow_m3s = total_runoff * area_m2 / 1000 / 86400
            sim_series = pd.Series(streamflow_m3s, index=times)

            # Skip warmup period to avoid spinup artifacts contaminating metrics.
            # VIC starts with INIT_STATE FALSE, so the first N days are unreliable.
            warmup_days = int(config.get('VIC_WARMUP_DAYS', 365))
            if warmup_days > 0 and len(sim_series) > warmup_days:
                sim_series = sim_series.iloc[warmup_days:]
                self.logger.debug(f"Skipped {warmup_days} warmup days for metric calculation")

            # Load observations
            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            period = kwargs.get('period') or resolve_config_value(config, lambda: config.domain.calibration_period, None, 'CALIBRATION_PERIOD') or config.get('domain', {}).get('calibration_period')
            calibration_period = None
            if period:
                parts = split_period(period)
                if parts is None:
                    raise ValidationError('Invalid VIC scoring period')
                start, end = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
                if pd.isna(start) or pd.isna(end) or start > end:
                    raise ValidationError('Invalid VIC scoring period')
                calibration_period = (start, end)

            # Align and calculate metrics
            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series, calibration_period=calibration_period
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating VIC metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_vic_parameters_worker(task_data)


def _evaluate_vic_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        VICWorker,
        task_data,
        model_name='VIC',
        extra_env=MACOS_MALLOC_QUIET_ENV,
    )
