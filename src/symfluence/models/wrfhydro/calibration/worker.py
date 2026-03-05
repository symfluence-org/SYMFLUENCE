# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WRF-Hydro Worker

Worker implementation for WRF-Hydro model optimization.
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.constants import ModelDefaults
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask


@OptimizerRegistry.register_worker('WRFHYDRO')
class WRFHydroWorker(BaseWorker):
    """
    Worker for WRF-Hydro model calibration.

    Handles parameter application to Fortran namelists, WRF-Hydro execution,
    and metric calculation from CHRTOUT output.
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
        Apply parameters to WRF-Hydro namelist files.

        Copies fresh namelists from the original WRFHydro_input location,
        then updates values in hydro.namelist and namelist.hrldas.

        Args:
            params: Parameter values to apply
            settings_dir: WRF-Hydro settings directory
            **kwargs: Additional arguments (config)

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying WRF-Hydro parameters to {settings_dir}")

            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'WRFHYDRO'

            if original_settings_dir.exists() and original_settings_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                # Namelists must be copied (they get modified per trial)
                for pattern in ['*.namelist', 'namelist.*']:
                    for f in original_settings_dir.glob(pattern):
                        shutil.copy2(f, settings_dir / f.name)
                # TBL files: copy calibrated ones, symlink the rest
                calibrated_tbls = {'SOILPARM.TBL', 'GENPARM.TBL'}
                for f in original_settings_dir.glob('*.TBL'):
                    dest = settings_dir / f.name
                    if dest.exists() or dest.is_symlink():
                        dest.unlink()
                    if f.name in calibrated_tbls:
                        shutil.copy2(f, dest)
                    else:
                        dest.symlink_to(f.resolve())
                # Domain NC files don't change — symlink them
                for f in original_settings_dir.glob('*.nc'):
                    dest = settings_dir / f.name
                    if dest.exists() or dest.is_symlink():
                        dest.unlink()
                    dest.symlink_to(f.resolve())
                self.logger.debug(
                    f"Set up WRF-Hydro trial from {original_settings_dir} "
                    f"(symlinked static files, copied calibrated files)"
                )
            elif not settings_dir.exists():
                self.logger.error(
                    f"WRF-Hydro settings directory not found: {settings_dir} "
                    f"(original also missing: {original_settings_dir})"
                )
                return False

            # Separate params by target file
            from ..parameters import WRFHYDRO_PARAM_TARGETS

            hydro_params = {}
            hrldas_params = {}
            soilparm_params = {}
            genparm_params = {}

            for param_name, value in params.items():
                target_info = WRFHYDRO_PARAM_TARGETS.get(param_name, {})
                target = target_info.get('target', 'hydro_namelist')
                if target == 'hrldas_namelist':
                    hrldas_params[param_name] = value
                elif target == 'soilparm_tbl':
                    soilparm_params[param_name] = (value, target_info)
                elif target == 'genparm_tbl':
                    genparm_params[param_name] = (value, target_info)
                else:
                    hydro_params[param_name] = value

            success = True

            # Update hydro.namelist
            hydro_file = config.get('WRFHYDRO_HYDRO_NAMELIST', 'hydro.namelist')
            hydro_path = settings_dir / hydro_file
            if hydro_params and hydro_path.exists():
                if not self._update_namelist_file(hydro_path, hydro_params):
                    success = False

            # Update namelist.hrldas
            hrldas_file = config.get('WRFHYDRO_NAMELIST_FILE', 'namelist.hrldas')
            hrldas_path = settings_dir / hrldas_file
            if hrldas_params and hrldas_path.exists():
                if not self._update_namelist_file(hrldas_path, hrldas_params):
                    success = False

            # Update SOILPARM.TBL for soil hydraulic parameters
            if soilparm_params:
                soilparm_path = settings_dir / 'SOILPARM.TBL'
                if soilparm_path.exists():
                    if not self._update_soilparm_tbl(
                        soilparm_path, soilparm_params, config
                    ):
                        success = False

            # Update GENPARM.TBL for general parameters (REFKDT, SLOPE)
            if genparm_params:
                genparm_path = settings_dir / 'GENPARM.TBL'
                if genparm_path.exists():
                    if not self._update_genparm_tbl(
                        genparm_path, genparm_params
                    ):
                        success = False

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying WRF-Hydro parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_namelist_file(self, namelist_file: Path, params: Dict[str, float]) -> bool:
        """
        Update a Fortran namelist file with new parameter values.

        Args:
            namelist_file: Path to namelist file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            content = namelist_file.read_text(encoding='utf-8')

            for param_name, value in params.items():
                formatted = self._format_namelist_value(value)

                pattern = re.compile(
                    r'(\s*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/\n]+)',
                    re.IGNORECASE
                )
                match = pattern.search(content)

                if match:
                    content = pattern.sub(r'\g<1>' + formatted, content)
                    self.logger.debug(f"Updated {param_name} = {formatted}")
                else:
                    # Don't insert unknown params — Fortran namelists crash
                    # on variable names the compiled binary doesn't expect.
                    self.logger.debug(
                        f"Skipping {param_name}: not present in {namelist_file.name}"
                    )

            namelist_file.write_text(content, encoding='utf-8')
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating WRF-Hydro namelist: {e}")
            return False

    def _format_namelist_value(self, value: float) -> str:
        """Format a parameter value for Fortran namelist syntax."""
        abs_val = abs(value)
        if abs_val == 0.0:
            return '0.0'
        elif abs_val < 0.001 or abs_val >= 1e6:
            return f'{value:.6e}'
        elif abs_val == int(abs_val) and abs_val < 1e6:
            return f'{value:.1f}'
        else:
            return f'{value:.6f}'

    def _update_soilparm_tbl(
        self,
        tbl_path: Path,
        soilparm_params: Dict[str, tuple],
        config: Dict[str, Any]
    ) -> bool:
        """
        Update SOILPARM.TBL with calibrated soil parameter values.

        Modifies columns (BB, SATDK, MAXSMC) for all soil types uniformly
        by applying a multiplier relative to the default value, ensuring the
        TBL format stays valid for Fortran parsing.

        Args:
            tbl_path: Path to SOILPARM.TBL
            soilparm_params: Dict of param_name -> (value, target_info)
            config: Configuration dictionary

        Returns:
            True if successful
        """
        try:
            lines = tbl_path.read_text(encoding='utf-8').splitlines()
            new_lines = []

            # Column mapping: param_name -> (column_index_in_csv, format_fn)
            col_map = {}
            for param_name, (value, info) in soilparm_params.items():
                col_idx = info.get('column_index', None)
                if col_idx is not None:
                    col_map[param_name] = (col_idx, value)

            for line in lines:
                stripped = line.strip()
                # Data rows start with a digit (soil type index)
                if stripped and stripped[0].isdigit() and ',' in stripped:
                    parts = stripped.split(',')
                    soil_idx = parts[0].strip()
                    # Skip water (type 14) and special types
                    if soil_idx in ('14',):
                        new_lines.append(line)
                        continue
                    try:
                        for param_name, (col_idx, new_val) in col_map.items():
                            if col_idx < len(parts):
                                # Format to match existing style
                                old_str = parts[col_idx].strip()
                                # Detect if value uses E notation
                                if 'E' in old_str.upper():
                                    parts[col_idx] = f'  {new_val:.2E}'
                                else:
                                    parts[col_idx] = f'  {new_val:.3f}'
                        new_lines.append(','.join(parts))
                    except (ValueError, IndexError):
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            tbl_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
            for param_name, (col_idx, new_val) in col_map.items():
                self.logger.debug(
                    f"Updated SOILPARM.TBL column {col_idx} ({param_name}) = {new_val}"
                )
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating SOILPARM.TBL: {e}")
            return False

    def _update_genparm_tbl(
        self,
        tbl_path: Path,
        genparm_params: Dict[str, tuple]
    ) -> bool:
        """
        Update GENPARM.TBL with calibrated general parameter values.

        GENPARM.TBL uses a key-value format where parameter names appear on
        one line and their value(s) on subsequent lines. SLOPE_DATA has 9
        values (one per category), while scalar parameters have a single value.

        Args:
            tbl_path: Path to GENPARM.TBL
            genparm_params: Dict of param_name -> (value, target_info)

        Returns:
            True if successful
        """
        try:
            lines = tbl_path.read_text(encoding='utf-8').splitlines()
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                matched = False
                for param_name, (value, info) in genparm_params.items():
                    key = info.get('key', param_name + '_DATA')
                    if line.strip() == key:
                        new_lines.append(line)
                        i += 1
                        if key == 'SLOPE_DATA':
                            # Next line is count, then N values
                            count_line = lines[i]
                            n_values = int(count_line.strip())
                            new_lines.append(count_line)
                            i += 1
                            for _ in range(n_values):
                                new_lines.append(f'{value:.4f} ')
                                i += 1
                        else:
                            # Scalar: skip old value line, write new
                            i += 1  # skip old value
                            abs_val = abs(value)
                            if abs_val < 0.001 or abs_val >= 1e6:
                                new_lines.append(f'{value:.2E}')
                            else:
                                new_lines.append(f'{value:.4f}')
                        self.logger.debug(f"Updated GENPARM.TBL {key} = {value}")
                        matched = True
                        break
                if not matched:
                    new_lines.append(line)
                    i += 1

            tbl_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating GENPARM.TBL: {e}")
            return False

    def _update_namelist_path(self, namelist_file: Path, key: str, new_path: str) -> bool:
        """Update a quoted path value in a Fortran namelist file."""
        try:
            content = namelist_file.read_text(encoding='utf-8')
            pattern = re.compile(
                r"(\s*" + re.escape(key) + r"\s*=\s*)'[^']*'",
                re.IGNORECASE
            )
            if pattern.search(content):
                content = pattern.sub(r"\g<1>'" + new_path + "'", content)
            else:
                # Try unquoted match
                pattern2 = re.compile(
                    r"(\s*" + re.escape(key) + r"\s*=\s*)\S+",
                    re.IGNORECASE
                )
                if pattern2.search(content):
                    content = pattern2.sub(r"\g<1>'" + new_path + "'", content)
                else:
                    self.logger.warning(f"Could not find {key} in {namelist_file}")
                    return False
            namelist_file.write_text(content, encoding='utf-8')
            self.logger.debug(f"Updated {key} = '{new_path}'")
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating {key} in namelist: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run WRF-Hydro model.

        WRF-Hydro is executed from within the output directory where it
        reads namelist files (copied to cwd before execution).

        Args:
            config: Configuration dictionary
            settings_dir: WRF-Hydro settings directory (with namelists)
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        import shutil

        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            wrfhydro_output_dir = Path(kwargs.get('sim_dir', output_dir))
            wrfhydro_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean stale output
            self._cleanup_stale_output(wrfhydro_output_dir)

            # Get executable
            wrfhydro_exe = self._get_wrfhydro_executable(config, data_dir)
            if not wrfhydro_exe.exists():
                self.logger.error(f"WRF-Hydro executable not found: {wrfhydro_exe}")
                return False

            # Set up namelists, domain files, and TBL files in working directory.
            # Symlink static files; copy only files modified for this trial.
            calibrated_tbls = {'SOILPARM.TBL', 'GENPARM.TBL'}
            namelist_names = set()
            for pattern in ['*.namelist', 'namelist.*']:
                for f in settings_dir.glob(pattern):
                    namelist_names.add(f.name)

            for pattern in ['*.namelist', 'namelist.*', '*.nc', '*.TBL']:
                for f in settings_dir.glob(pattern):
                    dest = wrfhydro_output_dir / f.name
                    if dest.exists() or dest.is_symlink():
                        dest.unlink()
                    if f.name in namelist_names or f.name in calibrated_tbls:
                        shutil.copy2(f, dest)
                    else:
                        dest.symlink_to(f.resolve())

            # Fix OUTDIR in namelist.hrldas to point to worker directory
            # (prevents output files from being written to the original input dir)
            hrldas_file = config.get('WRFHYDRO_NAMELIST_FILE', 'namelist.hrldas')
            worker_hrldas = wrfhydro_output_dir / hrldas_file
            if worker_hrldas.exists():
                self._update_namelist_path(
                    worker_hrldas, 'OUTDIR', str(wrfhydro_output_dir)
                )

            # Symlink routing files (unchanged between trials)
            domain_name = config.get('DOMAIN_NAME', '')
            routing_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'WRFHYDRO' / 'routing'
            if routing_dir.exists():
                for f in routing_dir.glob('*.nc'):
                    dest = wrfhydro_output_dir / f.name
                    if not (dest.exists() or dest.is_symlink()):
                        dest.symlink_to(f.resolve())

            from symfluence.core.mpi_utils import find_mpirun
            mpirun = find_mpirun(wrfhydro_exe)
            if mpirun:
                cmd = [mpirun, '-np', '1', str(wrfhydro_exe)]
            else:
                cmd = [str(wrfhydro_exe)]

            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            timeout = config.get('WRFHYDRO_TIMEOUT', 7200)

            stdout_file = wrfhydro_output_dir / 'wrfhydro_stdout.log'
            stderr_file = wrfhydro_output_dir / 'wrfhydro_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = subprocess.run(
                        cmd,
                        cwd=str(wrfhydro_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"WRF-Hydro timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"WRF-Hydro failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output
            output_files = (
                list(wrfhydro_output_dir.glob('*CHRTOUT*')) +
                list(wrfhydro_output_dir.glob('*LDASOUT*'))
            )

            if not output_files:
                self._last_error = "No CHRTOUT or LDASOUT output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running WRF-Hydro: {e}")
            return False

    def _get_wrfhydro_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get WRF-Hydro executable path."""
        install_path = config.get('WRFHYDRO_INSTALL_PATH', 'default')
        exe_name = config.get('WRFHYDRO_EXE', 'wrf_hydro.exe')

        if install_path == 'default':
            return data_dir / "installs" / "wrfhydro" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale WRF-Hydro output files."""
        for pattern in ['*CHRTOUT*', '*LDASOUT*', '*CHANOBS*', '*.log']:
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
        Calculate metrics from WRF-Hydro output.

        Extracts streamflow from CHRTOUT NetCDF files, loads observations,
        aligns time series, and computes KGE/NSE.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            import xarray as xr

            sim_dir = Path(kwargs.get('sim_dir', output_dir))
            settings_dir = kwargs.get('settings_dir', None)

            # Find CHRTOUT output files
            output_files = sorted(sim_dir.glob('*CHRTOUT*'))

            if not output_files and settings_dir:
                output_files = sorted(Path(settings_dir).glob('*CHRTOUT*'))

            if not output_files:
                domain_name = config.get('DOMAIN_NAME', '')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                wrfhydro_output = data_dir / f'domain_{domain_name}' / 'simulations'
                if wrfhydro_output.exists():
                    output_files = sorted(wrfhydro_output.rglob('*CHRTOUT*'))

            if not output_files:
                # Fallback: try LDASOUT files (standalone/no-routing mode)
                output_files = sorted(sim_dir.glob('*LDASOUT*'))
                if not output_files and settings_dir:
                    output_files = sorted(Path(settings_dir).glob('*LDASOUT*'))
                if not output_files:
                    domain_name = config.get('DOMAIN_NAME', '')
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    wrfhydro_output = data_dir / f'domain_{domain_name}' / 'simulations'
                    if wrfhydro_output.exists():
                        output_files = sorted(wrfhydro_output.rglob('*LDASOUT*'))

                if not output_files:
                    self.logger.error(f"No WRF-Hydro CHRTOUT or LDASOUT files found in {sim_dir}")
                    return {'kge': self.penalty_score, 'error': 'No output files'}

                # Extract streamflow from LDASOUT (accumulated runoff)
                sim_series = self._extract_ldasout_streamflow(output_files, config)
                if sim_series is None:
                    return {'kge': self.penalty_score, 'error': 'LDASOUT extraction failed'}
            else:
                # Extract streamflow from CHRTOUT files
                dates = []
                flows = []
                for chrt_file in output_files:
                    try:
                        ds = xr.open_dataset(chrt_file)
                        flow_var = None
                        for var in ['streamflow', 'q_lateral', 'qSfcLatRunoff', 'qBucket']:
                            if var in ds:
                                flow_var = var
                                break

                        if flow_var is None:
                            ds.close()
                            continue

                        flow_data = ds[flow_var]

                        spatial_dims = [d for d in flow_data.dims if d not in ['time', 'reference_time']]
                        if spatial_dims:
                            flow_data = flow_data.isel({spatial_dims[0]: -1})

                        if 'time' in ds:
                            time_val = pd.to_datetime(ds['time'].values)
                            if hasattr(time_val, '__len__'):
                                for t, v in zip(time_val, flow_data.values.flatten()):
                                    dates.append(t)
                                    flows.append(float(v))
                            else:
                                dates.append(time_val)
                                flows.append(float(flow_data.values.flatten()[0]))
                        else:
                            fname = chrt_file.name
                            try:
                                time_str = fname.split('.')[0]
                                date = pd.Timestamp(time_str[:8])
                                dates.append(date)
                                flows.append(float(flow_data.values.flatten()[0]))
                            except (ValueError, IndexError):
                                pass

                        ds.close()

                    except Exception as e:  # noqa: BLE001 — calibration resilience
                        self.logger.debug(f"Error reading {chrt_file}: {e}")
                        continue

                if not dates:
                    return {'kge': self.penalty_score, 'error': 'No streamflow data extracted'}

                sim_series = pd.Series(flows, index=dates, name='WRFHYDRO_discharge_cms')
                sim_series = sim_series.resample('D').mean()

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
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating WRF-Hydro metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    def _extract_ldasout_streamflow(
        self,
        ldasout_files: list,
        config: Dict[str, Any]
    ) -> Optional[pd.Series]:
        """
        Derive streamflow from LDASOUT surface and subsurface runoff.

        SFCRNOFF and UGDRNOFF are accumulated mm from simulation start.
        Reads one file per day (last hour) for efficiency, then differences
        consecutive daily accumulated values to get daily runoff depth:
          delta_mm = total_runoff(day) - total_runoff(day-1)
          Q = delta_mm * area_m2 / (86400 * 1000)

        Args:
            ldasout_files: Sorted list of LDASOUT file paths
            config: Configuration dictionary

        Returns:
            Daily streamflow series in m3/s, or None on failure
        """
        import netCDF4

        catchment_area_km2 = float(config.get('CATCHMENT_AREA_KM2', 2210.0))
        area_m2 = catchment_area_km2 * 1e6

        # Parse timestamps from filenames and group by date
        file_by_date = {}
        for fpath in sorted(ldasout_files):
            fname = fpath.name
            ts_str = fname.split('.')[0]
            try:
                if len(ts_str) >= 10:
                    date_key = ts_str[:8]  # YYYYMMDD
                    file_by_date[date_key] = fpath  # keep last file per day
            except (ValueError, IndexError):
                continue

        if not file_by_date:
            self.logger.error("No valid LDASOUT files found")
            return None

        self.logger.debug(
            f"Reading {len(file_by_date)} daily LDASOUT files "
            f"(subsampled from {len(ldasout_files)} hourly)"
        )

        timestamps = []
        accum_values = []

        for date_key in sorted(file_by_date.keys()):
            fpath = file_by_date[date_key]
            try:
                nc = netCDF4.Dataset(str(fpath), 'r')
                sfcrnoff = float(nc['SFCRNOFF'][:].mean()) if 'SFCRNOFF' in nc.variables else 0.0
                ugdrnoff = float(nc['UGDRNOFF'][:].mean()) if 'UGDRNOFF' in nc.variables else 0.0
                nc.close()
            except Exception:  # noqa: BLE001 — calibration resilience
                continue

            if np.isnan(sfcrnoff) or sfcrnoff < -9000:
                sfcrnoff = 0.0
            if np.isnan(ugdrnoff) or ugdrnoff < -9000:
                ugdrnoff = 0.0

            ts = pd.Timestamp(
                year=int(date_key[:4]), month=int(date_key[4:6]),
                day=int(date_key[6:8])
            )
            timestamps.append(ts)
            accum_values.append(sfcrnoff + ugdrnoff)

        if not timestamps:
            self.logger.error("No valid LDASOUT runoff data found")
            return None

        # Build accumulated series and difference to get daily runoff
        accum = pd.Series(accum_values, index=pd.DatetimeIndex(timestamps)).sort_index()
        delta_mm = accum.diff()
        delta_mm.iloc[0] = accum.iloc[0]

        # Convert mm/day -> m3/s (86400 seconds per day)
        q_cms = delta_mm * area_m2 / (86400.0 * 1000.0)
        q_cms.name = 'WRFHYDRO_discharge_cms'
        return q_cms

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_wrfhydro_parameters_worker(task_data)


def _evaluate_wrfhydro_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
        worker = WRFHydroWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'WRF-Hydro worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
