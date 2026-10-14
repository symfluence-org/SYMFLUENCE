# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM Worker

Worker implementation for mHM model optimization.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.constants import ModelDefaults
from symfluence.core.exceptions import ConfigurationError
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

from ..parameters import FORCING_PARAMS

# mhm.nml entries that mHM writes to. Each worker points them at its own
# directory so parallel evaluations never share an output location.
_OUTPUT_KEYS = ('dirConfigOut', r'dir_Out\(1\)', r'dir_Total_Runoff\(1\)',
                r'dir_Bankfull_Runoff\(1\)')
_RESTART_KEYS = (r'mhm_file_RestartOut\(1\)', r'mrm_file_RestartOut\(1\)')


@R.workers.add('MHM')
class MHMWorker(BaseWorker):
    """
    Worker for mHM model calibration.

    Handles parameter application to Fortran namelists, mHM execution,
    and metric calculation from discharge output.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize mHM worker.

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
        Apply parameters to mHM namelist files.

        Uses regex-based parsing to update parameter values in the .nml files.

        Args:
            params: Parameter values to apply
            settings_dir: mHM settings directory containing .nml files
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying mHM parameters to {settings_dir}")

            # Always copy fresh namelists from the original settings location
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings_dir = data_dir / f'domain_{domain_name}' / 'settings' / 'MHM'

            if original_settings_dir.exists() and original_settings_dir.resolve() != settings_dir.resolve():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_settings_dir.glob('*.nml'):
                    shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(f"Copied mHM namelists from {original_settings_dir} to {settings_dir}")
            elif not settings_dir.exists():
                self.logger.error(f"mHM settings directory not found: {settings_dir} "
                                  f"(original also missing: {original_settings_dir})")
                return False

            # Find parameter namelist file -- mHM stores calibration
            # parameters in mhm_parameter.nml (with lower/upper/value/flag/scaling
            # columns).  Fall back to mhm.nml for legacy setups.
            param_namelist = settings_dir / 'mhm_parameter.nml'
            if not param_namelist.exists():
                param_namelist = settings_dir / 'mhm.nml'
            if not param_namelist.exists():
                nml_files = list(settings_dir.glob('*.nml'))
                if nml_files:
                    param_namelist = nml_files[0]
                else:
                    self.logger.error(f"mHM namelist file not found in {settings_dir}")
                    return False

            mhm_nml = settings_dir / 'mhm.nml'
            if mhm_nml.exists():
                out_dir = Path(kwargs['sim_dir']) if kwargs.get('sim_dir') else settings_dir / 'output'
                self._localise_outputs(mhm_nml, out_dir)

            forcing_params = {k: v for k, v in params.items() if k in FORCING_PARAMS}
            if (forcing_params or config.get('MHM_GLACIER_MB_TARGET') is not None) \
                    and not self._write_adjusted_precip(
                        settings_dir, data_dir / f'domain_{domain_name}', params, config):
                return False

            # Update parameter namelist with new parameter values
            nml_params = {k: v for k, v in params.items() if k not in FORCING_PARAMS}
            return self._update_namelist_file(param_namelist, nml_params)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying mHM parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def _set_nml_path(content: str, key: str, value: str) -> str:
        """Replace the quoted value of ``key = "..."`` in a namelist."""
        return re.sub(r'(^\s*' + key + r'\s*=\s*)"[^"]*"',
                      lambda m: f'{m.group(1)}"{value}"', content, flags=re.MULTILINE)

    def _localise_outputs(self, mhm_nml: Path, out_dir: Path) -> None:
        """Point every mHM output path in ``mhm_nml`` at ``out_dir`` (forward slashes on every OS)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        content = mhm_nml.read_text(encoding='utf-8')
        for key in _OUTPUT_KEYS:
            content = self._set_nml_path(content, key, f'{out_dir.as_posix()}/')
        for key in _RESTART_KEYS:
            match = re.search(r'^\s*' + key + r'\s*=\s*"([^"]*)"', content, flags=re.MULTILINE)
            if match:
                content = self._set_nml_path(content, key, (out_dir / Path(match.group(1)).name).as_posix())
        mhm_nml.write_text(content, encoding='utf-8')

    def _write_adjusted_precip(self, settings_dir: Path, project_dir: Path,
                               params: Dict[str, float],
                               config: Optional[Dict[str, Any]] = None) -> bool:
        """Write this worker's precipitation input with the forcing-side
        pseudo-parameters applied, and point its mhm.nml at it.

        ``precipCorrectionFactor`` scales precipitation everywhere.
        ``glacierIceDDF`` adds degree-day ice melt on the cells flagged in
        ``morph/glacier_mask.asc``: a per-cell snow cover on the ice is tracked
        with the corrected precipitation (snowfall below
        ``snowTreshholdTemperature``) and melted at half the ice factor, the
        usual snow/ice degree-day ratio; ice melts with whatever energy is left
        once that snow is gone. The melt enters mHM as liquid input on the
        sealed glacier cells. With MHM_GLACIER_MB_TARGET set, the ice factor is
        instead solved so the mean net balance over the calibration water years
        matches the target. With MHM_GLACIER_AREA_FROM_OUTLINE, melt is scaled
        by each cell's glacierised fraction. Annual ice melt and net balance
        are written to ``forcing_adjustment.json``.
        """
        import json
        import shutil

        import netCDF4 as nc4

        base_dir = project_dir / 'data' / 'forcing' / 'MHM_input'
        work_dir = settings_dir / 'forcing_adjusted'
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            for name in ('tavg.nc', 'pet.nc', 'header.txt'):
                target = work_dir / name
                if not target.exists():
                    target.symlink_to(base_dir / name)
            pre_path = work_dir / 'pre.nc'
            shutil.copy2(base_dir / 'pre.nc', pre_path)

            with nc4.Dataset(base_dir / 'pre.nc') as ds:
                pre = np.ma.filled(ds['pre'][:], np.nan)
                times = nc4.num2date(ds['time'][:], ds['time'].units,
                                     calendar=getattr(ds['time'], 'calendar', 'standard'),
                                     only_use_cftime_datetimes=False)
            pre = pre * float(params.get('precipCorrectionFactor', 1.0))

            config = config or {}
            summary: Dict[str, Any] = {k: float(params[k]) for k in FORCING_PARAMS if k in params}
            morph = project_dir / 'settings' / 'MHM' / 'morph'
            mb_target = config.get('MHM_GLACIER_MB_TARGET')
            if (morph / 'glacier_mask.asc').exists() and (
                    'glacierIceDDF' in params or mb_target is not None):
                glacier = np.loadtxt(morph / 'glacier_mask.asc', skiprows=6) == 1
                weight = np.ones(int(glacier.sum()))
                if config.get('MHM_GLACIER_AREA_FROM_OUTLINE'):
                    weight = np.loadtxt(morph / 'glacier_fraction.asc', skiprows=6)[glacier]
                with nc4.Dataset(base_dir / 'tavg.nc') as ds:
                    tavg = np.ma.filled(ds['tavg'][:], np.nan)
                index = pd.DatetimeIndex(times)
                pre_g, tavg_g = pre[:, glacier], tavg[:, glacier]
                t_snow = float(params.get('snowTreshholdTemperature', 1.0))
                if mb_target is not None:
                    ddf_ice = self._solve_ice_ddf(
                        pre_g, tavg_g, t_snow, weight, index,
                        self._calibration_water_years(config, index), float(mb_target))
                else:
                    ddf_ice = float(params['glacierIceDDF'])
                ice_melt, snow = self._glacier_ice_melt(pre_g, tavg_g, ddf_ice, t_snow,
                                                        return_snow=True)
                balance = self._glacier_net_balance(ice_melt, snow, weight, index)
                # Melt is per unit ice area; spread it over the whole cell.
                ice_melt *= weight
                pre[:, glacier] += ice_melt
                n_active = int(np.isfinite(pre[0]).sum())
                yearly = pd.Series(ice_melt.sum(axis=1) / max(n_active, 1),
                                   index=index).resample('YS-OCT').sum()
                summary.update(
                    glacierIceDDF=round(ddf_ice, 4),
                    glacier_cells=int(glacier.sum()),
                    glacier_cell_equivalents=round(float(weight.sum()), 2),
                    ice_melt_basin_mm_per_wy={str(t.year + 1): round(float(v), 1)
                                              for t, v in yearly.items()},
                    net_balance_mwe_per_wy={str(y): round(v, 3) for y, v in balance.items()})

            with nc4.Dataset(pre_path, 'a') as ds:
                ds['pre'][:] = np.where(np.isfinite(pre), pre, ds['pre']._FillValue)
            (work_dir / 'forcing_adjustment.json').write_text(json.dumps(summary, indent=1))

            mhm_nml = settings_dir / 'mhm.nml'
            content = mhm_nml.read_text(encoding='utf-8')
            for key in (r'dir_Precipitation\(1\)', r'dir_Temperature\(1\)', r'dir_ReferenceET\(1\)'):
                content = self._set_nml_path(content, key, f'{work_dir.as_posix()}/')
            mhm_nml.write_text(content, encoding='utf-8')
            return True
        except (ConfigurationError, OSError, RuntimeError, ValueError, KeyError, IndexError) as e:
            self.logger.error(f"Error writing adjusted mHM precipitation: {e}", exc_info=True)
            return False

    @staticmethod
    def _glacier_ice_melt(pre: np.ndarray, tavg: np.ndarray, ddf_ice: float,
                          t_snow: float, return_snow: bool = False):
        """Daily ice melt (mm) on glacier cells; arrays are (time, cell).

        With ``return_snow`` also returns the snow cover on the ice (mm) at
        the end of each day.
        """
        ddf_snow = 0.5 * ddf_ice
        snow = np.zeros(pre.shape[1])
        melt = np.zeros_like(pre)
        cover = np.zeros_like(pre) if return_snow else None
        for t in range(pre.shape[0]):
            temp = tavg[t]
            snow += np.where(temp <= t_snow, pre[t], 0.0)
            degree_days = np.maximum(temp, 0.0)
            potential = ddf_snow * degree_days
            snow_melt = np.minimum(snow, potential)
            snow -= snow_melt
            unused = np.where(potential > 0, 1.0 - snow_melt / np.where(potential > 0, potential, 1.0), 0.0)
            melt[t] = unused * ddf_ice * degree_days
            if cover is not None:
                cover[t] = snow
        return (melt, cover) if return_snow else melt

    @staticmethod
    def _water_year(index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.year + (index.month >= 10).astype(int))

    @classmethod
    def _calibration_water_years(cls, config: Dict[str, Any],
                                 index: pd.DatetimeIndex) -> list:
        """Water years lying entirely inside CALIBRATION_PERIOD."""
        period = str(config.get('CALIBRATION_PERIOD', ''))
        if ',' not in period:
            raise ConfigurationError('MHM_GLACIER_MB_TARGET needs CALIBRATION_PERIOD')
        start, end = (pd.Timestamp(p.strip()) for p in period.split(',')[:2])
        wy = cls._water_year(index)
        return sorted(int(y) for y in set(wy[(index >= start) & (index <= end)])
                      if pd.Timestamp(f'{y - 1}-10-01') >= start
                      and pd.Timestamp(f'{y}-09-30') <= end)

    @classmethod
    def _glacier_net_balance(cls, melt: np.ndarray, snow: np.ndarray, weight: np.ndarray,
                             index: pd.DatetimeIndex) -> Dict[int, float]:
        """Area-weighted glacier net balance (m w.e.) per complete water year:
        change in snow cover on the ice minus ice melt. ``melt`` is per unit
        glacier area."""
        w = weight / weight.sum()
        melt_s, snow_s = melt @ w, snow @ w
        wy = cls._water_year(index)
        out: Dict[int, float] = {}
        for y in np.unique(wy):
            sel = np.flatnonzero(wy == y)
            if index[sel[0]].month != 10 or index[sel[0]].day != 1 or \
                    index[sel[-1]].month != 9 or index[sel[-1]].day != 30:
                continue
            start = snow_s[sel[0] - 1] if sel[0] > 0 else 0.0
            out[int(y)] = float((snow_s[sel[-1]] - start - melt_s[sel].sum()) / 1000.0)
        return out

    def _solve_ice_ddf(self, pre: np.ndarray, tavg: np.ndarray, t_snow: float,
                       weight: np.ndarray, index: pd.DatetimeIndex, years: list,
                       target: float, lo: float = 1.0, hi: float = 30.0) -> float:
        """Ice degree-day factor whose mean net balance over ``years`` equals
        ``target`` (m w.e./yr). The balance falls monotonically as the factor
        rises, so bisect; clip to [lo, hi] when the target is out of reach."""
        if not years:
            raise ConfigurationError('no complete water years in CALIBRATION_PERIOD')
        n = int(np.flatnonzero(self._water_year(index) <= max(years))[-1]) + 1
        pre, tavg, index = pre[:n], tavg[:n], index[:n]

        def balance(ddf: float) -> float:
            melt, snow = self._glacier_ice_melt(pre, tavg, ddf, t_snow, return_snow=True)
            by_year = self._glacier_net_balance(melt, snow, weight, index)
            return float(np.mean([by_year[y] for y in years]))

        if balance(hi) > target:
            self.logger.warning(f'glacier net balance stays above {target} m w.e. at ice DDF {hi}')
            return hi
        if balance(lo) < target:
            self.logger.warning(f'glacier net balance is below {target} m w.e. even at ice DDF {lo}')
            return lo
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if balance(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _update_namelist_file(self, namelist_file: Path, params: Dict[str, float]) -> bool:
        """
        Update mHM Fortran namelist file with new parameter values.

        Supports two formats:

        1. **mhm_parameter.nml** (5-element tuple)::

               paramName = lower, upper, value, FLAG, SCALING

           In this case only the *value* (3rd element) is replaced.

        2. **Legacy / mhm.nml** (single value)::

               paramName = value

        Args:
            namelist_file: Path to namelist (.nml) file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            content = namelist_file.read_text(encoding='utf-8')

            # Detect whether this is a 5-column parameter file by checking
            # for the characteristic "lower, upper, value, flag, scaling" pattern
            is_param_nml = 'mhm_parameter' in namelist_file.name

            for param_name, value in params.items():
                formatted_value = self._format_namelist_value(value)

                if is_param_nml:
                    # 5-column format: paramName = lower, upper, VALUE, flag, scaling
                    # Capture: (prefix + lower, upper,) (value) (, flag, scaling)
                    pattern = re.compile(
                        r'(^[ \t]*' + re.escape(param_name)
                        + r'\s*=\s*'           # paramName =
                        + r'[^,]+,'             # lower,
                        + r'\s*[^,]+,'          # upper,
                        + r'\s*)'               # whitespace before value
                        + r'([^,]+)'            # VALUE  <-- group to replace
                        + r'(,\s*\d+,\s*\d+)',  # , flag, scaling
                        re.MULTILINE
                    )
                    match = pattern.search(content)
                    if match:
                        content = pattern.sub(
                            r'\g<1>' + formatted_value + r'\g<3>',
                            content
                        )
                        self.logger.debug(
                            f"Updated {param_name} value = {formatted_value} "
                            f"(5-column format)"
                        )
                    else:
                        self.logger.warning(
                            f"Parameter {param_name} not found in {namelist_file}"
                        )
                else:
                    # Legacy single-value format: paramName = value
                    pattern = re.compile(
                        r'(^[ \t]*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/]+)',
                        re.MULTILINE
                    )
                    match = pattern.search(content)
                    if match:
                        content = pattern.sub(
                            r'\g<1>' + formatted_value,
                            content
                        )
                        self.logger.debug(f"Updated {param_name} = {formatted_value}")
                    else:
                        self.logger.warning(
                            f"Parameter {param_name} not found in {namelist_file}"
                        )

            namelist_file.write_text(content, encoding='utf-8')
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating mHM namelist: {e}", exc_info=True)
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

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run mHM model.

        mHM is executed from within the settings directory where it reads
        the mhm.nml and mrm.nml namelist files.

        Args:
            config: Configuration dictionary
            settings_dir: mHM settings directory (with .nml files)
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            # Use sim_dir for output if provided
            mhm_output_dir = Path(kwargs.get('sim_dir', output_dir))
            mhm_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files
            self._cleanup_stale_output(mhm_output_dir)

            # Get executable
            mhm_exe = self._get_mhm_executable(config, data_dir)
            if not mhm_exe.exists():
                self.logger.error(f"mHM executable not found: {mhm_exe}")
                return False

            # Ensure we have namelist files in settings_dir
            if not (settings_dir / 'mhm.nml').exists():
                self.logger.error(f"mhm.nml not found in {settings_dir}")
                return False

            # Build command - mHM runs from within the settings directory
            cmd = [str(mhm_exe)]

            # Set environment
            env = os.environ.copy()
            env['MallocStackLogging'] = '0'

            # Run with timeout
            timeout = config.get('MHM_TIMEOUT', 300)

            stdout_file = mhm_output_dir / 'mhm_stdout.log'
            stderr_file = mhm_output_dir / 'mhm_stderr.log'

            try:
                with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                     open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                    result = run_subprocess(
                        cmd,
                        cwd=str(settings_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"mHM timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"mHM failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output - check multiple locations where mHM may write
            output_files = list(mhm_output_dir.glob('discharge*.nc'))
            if not output_files:
                # mHM may write to output subdirectory under settings
                for subdir in [settings_dir / 'output', settings_dir]:
                    output_files = list(subdir.glob('discharge*.nc'))
                    if output_files:
                        break

            if not output_files:
                # mHM namelist may point to settings/MHM/output directory
                domain_name = config.get('DOMAIN_NAME', '')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                mhm_input_output = data_dir / f'domain_{domain_name}' / 'settings' / 'MHM' / 'output'
                if mhm_input_output.exists():
                    output_files = list(mhm_input_output.glob('discharge*.nc'))

            if not output_files:
                self._last_error = "No discharge output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self._last_error = str(e)
            self.logger.error(f"Error running mHM: {e}", exc_info=True)
            return False

    def _get_mhm_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get mHM executable path."""
        install_path = config.get('MHM_INSTALL_PATH', 'default')
        exe_name = config.get('MHM_EXE', 'mhm')

        if install_path == 'default':
            exe = data_dir / "installs" / "mhm" / "bin" / exe_name
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

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale mHM output files."""
        for pattern in ['discharge*.nc', 'mHM_Fluxes_States_*.nc', '*.log']:
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
        Calculate metrics from mHM output.

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

            # Find discharge output file - search multiple locations
            output_files = list(sim_dir.glob('discharge*.nc'))

            if not output_files and settings_dir:
                for subdir in [Path(settings_dir) / 'output', Path(settings_dir)]:
                    output_files = list(subdir.glob('discharge*.nc'))
                    if output_files:
                        break

            if not output_files:
                # mHM namelist may point to settings/MHM/output directory
                domain_name = config.get('DOMAIN_NAME', '')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                mhm_input_output = data_dir / f'domain_{domain_name}' / 'settings' / 'MHM' / 'output'
                if mhm_input_output.exists():
                    output_files = list(mhm_input_output.glob('discharge*.nc'))

            if not output_files:
                output_files = list(sim_dir.glob('*.nc'))

            if not output_files:
                self.logger.error(f"No mHM discharge files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_file = output_files[0]

            # Extract discharge
            ds = xr.open_dataset(sim_file)

            # Find discharge variable (mHM v5.13+ uses Qsim_NNNNNNNNNN format)
            discharge = None
            for var in ['Qsim', 'Q', 'discharge', 'Qrouted']:
                if var in ds:
                    discharge = ds[var]
                    break

            if discharge is None:
                # Check for gauge-indexed variables like Qsim_0000000001
                qsim_vars = [v for v in ds.data_vars if v.startswith('Qsim')]
                if qsim_vars:
                    discharge = ds[qsim_vars[0]]

            if discharge is None:
                ds.close()
                return {'kge': self.penalty_score, 'error': 'No discharge variable'}

            # Handle spatial dimensions (take first gauge)
            spatial_dims = [d for d in discharge.dims if d not in ['time']]
            if spatial_dims:
                discharge = discharge.isel({spatial_dims[0]: 0})

            # mHM discharge is already in m3/s
            times = pd.to_datetime(ds['time'].values)
            streamflow_m3s = discharge.values.flatten()
            ds.close()

            sim_series = pd.Series(streamflow_m3s, index=times)

            # Skip warmup period (worker-level).
            # NOTE: mHM handles warmup internally via warming_Days in the
            # namelist, so output already excludes the warmup period.
            # MHM_WORKER_WARMUP_DAYS allows an optional extra skip if needed.
            warmup_days = int(config.get('MHM_WORKER_WARMUP_DAYS', 0))
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

            # Parse calibration period from config
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
            self.logger.error(f"Error calculating mHM metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_mhm_parameters_worker(task_data)


def _evaluate_mhm_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
        'MallocStackLogging': '0',
    })

    # Small random delay
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = MHMWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'mHM worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
