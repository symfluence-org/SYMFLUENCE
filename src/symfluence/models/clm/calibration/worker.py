# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Worker

Worker implementation for CLM5 model optimization.
Handles parameter application, CLM execution, and metric calculation.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.constants import ModelDefaults
from symfluence.core.logging_utils import log_once
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.registries import R


@R.workers.add('CLM')
class CLMWorker(BaseWorker):
    """
    Worker for CLM5 model calibration.

    Handles parameter application to 3 target files (namelist, params.nc,
    surfdata.nc), CLM execution via cesm.exe, and metric calculation
    from QRUNOFF output.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs,
    ) -> bool:
        """
        Apply parameters to CLM input files.

        Copies fresh clm5_params.nc and surfdata_clm.nc from originals,
        then applies parameter modifications.

        Args:
            params: Parameter values to apply
            settings_dir: CLM settings directory
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Applying CLM parameters to {settings_dir}")

            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir_raw = config.get('SYMFLUENCE_DATA_DIR', '.')
            self.logger.debug(
                f"CLM worker config: DOMAIN_NAME={domain_name}, "
                f"SYMFLUENCE_DATA_DIR={data_dir_raw}, "
                f"config type={type(config).__name__}, "
                f"config keys (sample)={list(config.keys())[:5] if hasattr(config, 'keys') else 'N/A'}"
            )
            data_dir = Path(data_dir_raw)
            original_params_dir = (
                data_dir / f'domain_{domain_name}' / 'settings' / 'CLM' / 'parameters'
            )
            self.logger.debug(f"CLM original_params_dir={original_params_dir}, exists={original_params_dir.exists()}")

            # Determine iteration-specific params directory
            # Prefer settings_dir/parameters; fall back to sibling
            params_dir = settings_dir / 'parameters'
            if not params_dir.exists():
                alt = settings_dir.parent / 'parameters'
                if alt.exists() and alt.resolve() != original_params_dir.resolve():
                    params_dir = alt

            # Copy fresh parameter files from original each iteration
            if original_params_dir.exists():
                if params_dir.resolve() == original_params_dir.resolve():
                    # Same directory — create a sibling to avoid self-copy
                    params_dir = settings_dir / 'parameters'
                params_dir.mkdir(parents=True, exist_ok=True)
                for f in original_params_dir.glob('*.nc'):
                    shutil.copy2(f, params_dir / f.name)
                self.logger.debug(
                    f"Copied CLM parameters from {original_params_dir}"
                )
            elif not params_dir.exists():
                self.logger.error(
                    f"CLM parameters dir not found: {params_dir}"
                )
                return False

            # Also copy namelists from settings/CLM to settings_dir
            original_settings_dir = (
                data_dir / f'domain_{domain_name}' / 'settings' / 'CLM'
            )
            if original_settings_dir.exists():
                settings_dir.mkdir(parents=True, exist_ok=True)
                for f in original_settings_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, settings_dir / f.name)
                self.logger.debug(
                    f"Copied CLM settings from {original_settings_dir}"
                )

            # Reject infeasible parameter combinations early (before file I/O)
            if not self._validate_params(params):
                log_once(self.logger, logging.WARNING, key='clm-infeasible-params',
                         message="Parameter combination rejected by feasibility check")
                return False

            # Apply parameters using parameter manager logic
            from .parameter_manager import CLM_PARAM_DEFS, check_removed_params
            check_removed_params(list(params))

            # Separate by target
            nl_params = {}
            params_nc = {}
            surfdata_params = {}

            for name, value in params.items():
                if name not in CLM_PARAM_DEFS:
                    log_once(self.logger, logging.WARNING, key=f'clm-unknown-param-{name}',
                             message=f"Unknown CLM param: {name}")
                    continue
                target = CLM_PARAM_DEFS[name][0]
                if target == 'namelist':
                    nl_params[name] = value
                elif target == 'params':
                    params_nc[name] = value
                elif target == 'surfdata':
                    surfdata_params[name] = value

            success = True

            if nl_params:
                success &= self._update_namelist(settings_dir, nl_params)
            if params_nc:
                success &= self._update_params_nc(params_dir, params_nc, config)
            if surfdata_params:
                success &= self._update_surfdata(params_dir, surfdata_params)

            # Update lnd_in to point to modified parameter files
            # and inject namelist calibration params (critical — CLM
            # reads lnd_in, NOT user_nl_clm when CIME is bypassed)
            success &= self._update_lnd_in(
                settings_dir, params_dir, nl_params
            )

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying CLM parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_namelist(
        self, settings_dir: Path, params: Dict[str, float]
    ) -> bool:
        """Update user_nl_clm with calibration parameters.

        Note: CLM reads lnd_in (not user_nl_clm) when CIME is bypassed.
        user_nl_clm is kept in sync for documentation / CIME compatibility.
        The actual parameter injection happens in _update_lnd_in().
        """
        nl_path = settings_dir / 'user_nl_clm'
        if not nl_path.exists():
            return True  # Non-fatal

        content = nl_path.read_text()

        for name, value in params.items():
            lines = content.split('\n')
            lines = [
                l for l in lines
                if not l.strip().startswith(f'{name} ')
                and not l.strip().startswith(f'{name}=')
            ]
            content = '\n'.join(lines)
            content += f"\n{name} = {value:.8g}"

        nl_path.write_text(content)
        return True

    def _update_lnd_in(
        self,
        settings_dir: Path,
        params_dir: Path,
        nl_params: Dict[str, float],
    ) -> bool:
        """Update lnd_in namelist with correct file paths and parameters.

        CLM reads lnd_in (not user_nl_clm) when CIME is bypassed.
        This method:
        1. Points paramfile and fsurdat to the iteration-specific copies
        2. Injects namelist calibration parameters into their correct
           &section in lnd_in (e.g. baseflow_scalar, int_snow_max)
        """
        import re

        from .parameter_manager import CLM_PARAM_DEFS

        lnd_in_path = settings_dir / 'lnd_in'
        if not lnd_in_path.exists():
            self.logger.warning(f"lnd_in not found: {lnd_in_path}")
            return True  # Non-fatal — lnd_in may be generated later

        content = lnd_in_path.read_text()
        abs_params_dir = params_dir.resolve()

        # Update paramfile path
        new_paramfile = abs_params_dir / 'clm5_params.nc'
        content = re.sub(
            r"paramfile\s*=\s*'[^']*'",
            f"paramfile = '{new_paramfile}'",
            content,
        )

        # Update fsurdat path
        new_fsurdat = abs_params_dir / 'surfdata_clm.nc'
        content = re.sub(
            r"fsurdat\s*=\s*'[^']*'",
            f"fsurdat = '{new_fsurdat}'",
            content,
        )

        # Inject namelist calibration params into their correct sections
        for name, value in nl_params.items():
            if name not in CLM_PARAM_DEFS:
                continue
            target, section, _ = CLM_PARAM_DEFS[name]
            if target != 'namelist' or section is None:
                continue

            # Format value as Fortran double
            val_str = f"{value:.8g}d00"

            # Try to update existing line first
            pattern = rf"({name}\s*=\s*)[\d.eEdD+-]+"
            if re.search(pattern, content):
                content = re.sub(pattern, rf"\g<1>{val_str}", content)
            else:
                # Insert into the correct &section before the closing /
                section_pattern = rf"(&{section}\s*\n)(.*?)(^/\s*$)"
                match = re.search(section_pattern, content, re.MULTILINE | re.DOTALL)
                if match:
                    insert = f" {name} = {val_str}\n"
                    content = (
                        content[:match.end(2)]
                        + insert
                        + content[match.start(3):]
                    )
                    self.logger.debug(
                        f"Inserted {name} = {val_str} into &{section}"
                    )

        lnd_in_path.write_text(content)
        self.logger.debug(
            f"Updated lnd_in: paramfile→{new_paramfile.name}, "
            f"fsurdat→{new_fsurdat.name}"
        )
        return True

    def _update_params_nc(
        self,
        params_dir: Path,
        params: Dict[str, float],
        config: Dict = None,
    ) -> bool:
        """Update clm5_params.nc with snow and PFT parameters.

        Uses netCDF4 directly (not xarray) to modify in-place, preserving
        the original NETCDF3_CLASSIC format that CLM/PIO requires.
        """
        import netCDF4

        from .parameter_manager import CLM_PARAM_DEFS

        params_file = params_dir / 'clm5_params.nc'
        if not params_file.exists():
            nc_files = list(params_dir.glob('clm5_params*.nc'))
            if nc_files:
                params_file = nc_files[0]
            else:
                self.logger.error(f"CLM params file not found in {params_dir}")
                return False

        active_pfts = self._get_active_pfts(params_dir)

        ds = netCDF4.Dataset(str(params_file), 'r+')

        for name, value in params.items():
            if name not in CLM_PARAM_DEFS:
                continue
            _, nc_var, _ = CLM_PARAM_DEFS[name]
            if nc_var is None or nc_var not in ds.variables:
                continue

            var = ds.variables[nc_var]
            is_pft_param = name in (
                'medlynslope', 'slatop', 'flnr', 'froot_leaf', 'stem_leaf'
            )

            if is_pft_param and 'pft' in var.dimensions and active_pfts:
                for pft_idx in active_pfts:
                    if pft_idx < var.shape[var.dimensions.index('pft')]:
                        var[pft_idx] = value
            else:
                var[:] = value

        # Constraint: SNOW_DENSITY_MIN < SNOW_DENSITY_MAX
        if 'SNOW_DENSITY_MIN' in ds.variables and 'SNOW_DENSITY_MAX' in ds.variables:
            dmin = float(ds.variables['SNOW_DENSITY_MIN'][:].flat[0])
            dmax = float(ds.variables['SNOW_DENSITY_MAX'][:].flat[0])
            if dmin >= dmax:
                ds.variables['SNOW_DENSITY_MIN'][:] = dmax * 0.5

        ds.close()
        return True

    def _update_surfdata(
        self, params_dir: Path, params: Dict[str, float]
    ) -> bool:
        """Update surfdata_clm.nc with soil multipliers.

        Uses netCDF4 directly (not xarray) to modify in-place, preserving
        the exact binary format that CLM/PIO requires.
        """
        import netCDF4

        from .parameter_manager import apply_texture_offsets

        surfdata_file = params_dir / 'surfdata_clm.nc'
        if not surfdata_file.exists():
            nc_files = list(params_dir.glob('surfdata*.nc'))
            if nc_files:
                surfdata_file = nc_files[0]
            else:
                self.logger.error(f"Surfdata not found in {params_dir}")
                return False

        ds = netCDF4.Dataset(str(surfdata_file), 'r+')

        if 'fmax' in params and 'FMAX' in ds.variables:
            ds.variables['FMAX'][:] = params['fmax']
        if 'organic_max' in params and 'ORGANIC' in ds.variables:
            org = ds.variables['ORGANIC'][:]
            ds.variables['ORGANIC'][:] = np.minimum(org, params['organic_max'])
        # Texture is applied to the fresh copy made in apply_parameters, so
        # offsets never compound across evaluations.
        if {'sand_offset', 'clay_offset'} & set(params) and \
                {'PCT_SAND', 'PCT_CLAY'} <= set(ds.variables):
            sand, clay = apply_texture_offsets(
                ds.variables['PCT_SAND'][:], ds.variables['PCT_CLAY'][:],
                params.get('sand_offset', 0.0), params.get('clay_offset', 0.0))
            ds.variables['PCT_SAND'][:] = sand
            ds.variables['PCT_CLAY'][:] = clay

        ds.close()
        return True

    def _get_active_pfts(self, params_dir: Path) -> list:
        """Get active PFT indices from surfdata."""
        surfdata_file = params_dir / 'surfdata_clm.nc'
        if not surfdata_file.exists():
            return [1, 12]

        try:
            ds = xr.open_dataset(surfdata_file)
            if 'PCT_NAT_PFT' in ds:
                pct = ds['PCT_NAT_PFT'].values.flatten()
                active = [i for i, p in enumerate(pct) if p > 0.0]
                ds.close()
                return active if active else [1, 12]
            ds.close()
        except Exception:  # noqa: BLE001 — calibration resilience
            pass
        return [1, 12]

    def _validate_params(self, params: Dict[str, float]) -> bool:
        """Validate parameter combinations for physical feasibility.

        Returns False for combinations known to crash CLM, saving ~3 min
        of wasted runtime per infeasible evaluation.
        """
        # Snow density ordering
        dmin = params.get('SNOW_DENSITY_MIN')
        dmax = params.get('SNOW_DENSITY_MAX')
        if dmin is not None and dmax is not None and dmin >= dmax:
            self.logger.debug(
                f"Infeasible: SNOW_DENSITY_MIN={dmin:.0f} >= "
                f"SNOW_DENSITY_MAX={dmax:.0f}"
            )
            return False

        return True

    def run_model(
        self,
        config: Dict,
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """
        Execute CLM5 for calibration.

        Args:
            config: Configuration dictionary
            settings_dir: CLM settings directory
            output_dir: Worker-specific output directory

        Returns:
            True if execution succeeded
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Clean stale output
            self._cleanup_stale_output(output_dir)

            # Get executable
            install_path = config.get('CLM_INSTALL_PATH', 'default')
            if install_path == 'default':
                # Installs are at SYMFLUENCE_CODE_DIR + _data/installs/clm
                code_dir = Path(config.get('SYMFLUENCE_CODE_DIR', '.'))
                install_path = str(code_dir.parent / (code_dir.name + '_data') / 'installs' / 'clm')
                if not Path(install_path).exists():
                    # Fallback: try standard data dir structure
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    install_path = str(data_dir.parent / 'installs' / 'clm')

            exe_name = config.get('CLM_EXE', 'cesm.exe')
            clm_exe = Path(install_path) / 'bin' / exe_name
            if not clm_exe.exists():
                clm_exe = Path(install_path) / exe_name
            if not clm_exe.exists():
                self.logger.error(f"CLM executable not found: {clm_exe}")
                return False

            # Copy all NUOPC runtime files to output_dir (CLM reads from cwd)
            runtime_files = [
                'nuopc.runconfig', 'nuopc.runseq', 'fd.yaml',
                'datm_in', 'datm.streams.xml', 'lnd_in',
                'drv_in', 'drv_flds_in', 'CASEROOT',
                'user_nl_clm',
            ]
            # Create timing dirs that cesm.exe expects
            (output_dir / 'timing' / 'checkpoints').mkdir(parents=True, exist_ok=True)
            # use_init_interp writes ./init_generated_files/finidat_interp_dest.nc
            # (fixed path in CTSM 5.3, not a namelist item)
            (output_dir / 'init_generated_files').mkdir(parents=True, exist_ok=True)

            for name in runtime_files:
                src = settings_dir / name
                if src.exists():
                    shutil.copy2(src, output_dir / name)

            # Execute — use a short timeout since single-point CLM
            # finishes in ~7 min but can hang in ESMF finalization
            timeout = int(config.get('CLM_TIMEOUT', 900))
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                # Disable macOS nano malloc zone — causes false heap
                # corruption detection in ESMF's JSON metadata handling
                'MallocNanoZone': '0',
            })

            # Run in a new session so MPI_ABORT cannot kill the parent
            # Python process (Open MPI sends SIGTERM to entire process group).
            # Write stdout/stderr to files instead of PIPE to avoid deadlock:
            # CLM+ESMF can write megabytes of output and a full pipe buffer
            # causes the process to block, appearing as a solver hang.
            stdout_file = open(output_dir / 'cesm_stdout.log', 'w')
            stderr_file = open(output_dir / 'cesm_stderr.log', 'w')
            proc = subprocess.Popen(
                [str(clm_exe)],
                cwd=str(output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )

            # Poll-based wait with smart hang detection.
            #
            # CLM single-point takes ~7-20 min for 7 years depending on
            # parameters.  Two hang modes exist:
            #   1. Simulation hang: solver stuck at 0% CPU mid-run
            #   2. Finalization hang: ESMF MPI cleanup hangs after output
            #
            # Strategy: monitor BOTH h0 file count AND CPU usage.
            #   - All h0 files written + short grace → finalization hang
            #   - 0% CPU for extended period → solver hang (kill early)
            import subprocess as _sp
            import time
            poll_interval = 10
            finalization_grace = 60   # seconds after all h0 files written
            cpu_idle_limit = 120      # seconds of 0% CPU before kill
            no_progress_limit = 600   # seconds with 0 h0 files → assume crash
            expected_h0 = 9           # 8-year run with hist_mfilt=365 + leap year overflow
            elapsed = 0
            all_done_time = None
            idle_since = None

            def _get_cpu_pct(pid):
                """Get CPU% for a process (macOS/Linux)."""
                try:
                    out = _sp.check_output(
                        ['ps', '-o', '%cpu=', '-p', str(pid)],
                        stderr=_sp.DEVNULL, timeout=5
                    )
                    return float(out.strip())
                except Exception:  # noqa: BLE001 — calibration resilience
                    return 100.0  # assume running if check fails

            while proc.poll() is None and elapsed < timeout:
                time.sleep(poll_interval)
                elapsed += poll_interval

                h0_count = len(list(output_dir.glob('*.clm2.h0.*.nc')))

                # Check 1: all output files written → finalization hang
                if h0_count >= expected_h0:
                    if all_done_time is None:
                        all_done_time = time.time()
                        self.logger.debug(
                            f"All {h0_count} h0 files written after "
                            f"{elapsed}s, waiting for clean exit..."
                        )
                    elif time.time() - all_done_time > finalization_grace:
                        self.logger.warning(
                            f"ESMF finalization hang "
                            f"({h0_count} h0, {finalization_grace}s) "
                            f"— killing"
                        )
                        proc.kill()
                        proc.wait()
                        break
                    continue  # skip CPU check when finalization is expected

                # Check 2: no output after extended time → SIGSEGV infinite loop
                if h0_count == 0 and elapsed > no_progress_limit:
                    self.logger.warning(
                        f"CLM no progress (0 h0 files after "
                        f"{elapsed}s) — likely SIGSEGV loop, killing"
                    )
                    proc.kill()
                    proc.wait()
                    break

                # Check 3: CPU idle → solver hang (skip first 60s startup)
                if elapsed > 60:
                    cpu = _get_cpu_pct(proc.pid)
                    if cpu < 0.5:
                        if idle_since is None:
                            idle_since = time.time()
                        elif time.time() - idle_since > cpu_idle_limit:
                            self.logger.warning(
                                f"CLM solver hung (0% CPU for "
                                f"{cpu_idle_limit}s, {h0_count} h0 files, "
                                f"{elapsed}s elapsed) — killing"
                            )
                            proc.kill()
                            proc.wait()
                            break
                    else:
                        idle_since = None

            if proc.poll() is None:
                proc.kill()
                proc.wait()
                self.logger.warning(
                    f"CLM execution timed out after {elapsed}s — killing"
                )

            # Close log files
            stdout_file.close()
            stderr_file.close()

            # CLM/ESMF often returns non-zero during MPI finalization
            # even when all output was written successfully. Check for
            # output files before declaring failure.
            hist_files = list(output_dir.glob('*.clm2.h0.*.nc'))
            # Need at least 7 history files (2003-2009 inclusive)
            # for warmup + calibration + evaluation periods
            min_hist_files = 7

            if proc.returncode != 0:
                if len(hist_files) >= min_hist_files:
                    log_once(
                        self.logger, logging.WARNING,
                        key='clm-esmf-finalization-rc',
                        message=(
                            f"CLM returned rc={proc.returncode} but "
                            f"{len(hist_files)} history files exist — "
                            f"treating as success (ESMF finalization issue)"
                        ),
                    )
                else:
                    # Read stderr from file for error reporting
                    stderr_path = output_dir / 'cesm_stderr.log'
                    stderr_text = ''
                    if stderr_path.exists():
                        stderr_text = stderr_path.read_text(errors='replace')
                    self.logger.error(
                        f"CLM failed (rc={proc.returncode}, "
                        f"{len(hist_files)} hist files): "
                        f"{stderr_text[-300:]}"
                    )
                    return False

            if not hist_files:
                self.logger.error("No CLM history files produced")
                return False

            self.logger.debug(
                f"CLM run complete: {len(hist_files)} history file(s)"
            )
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"CLM execution error: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict,
        **kwargs,
    ) -> Dict:
        """
        Calculate streamflow metrics from CLM output.

        Extracts QRUNOFF (mm/s), converts to m3/s, aligns with
        observations, and computes KGE/NSE.
        """
        try:
            output_dir = Path(output_dir)

            # Find and open history files
            hist_files = sorted(output_dir.glob('*.clm2.h0.*.nc'))
            if not hist_files:
                return {'kge': self.penalty_score, 'error': 'No CLM output'}

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                ds = xr.open_mfdataset(hist_files, combine='by_coords')

            # Extract QRUNOFF
            if 'QRUNOFF' in ds:
                qrunoff = ds['QRUNOFF']
            elif 'QOVER' in ds and 'QDRAI' in ds:
                qrunoff = ds['QOVER'] + ds['QDRAI']
            else:
                ds.close()
                return {
                    'kge': self.penalty_score,
                    'error': 'No runoff variable',
                }

            # Squeeze spatial dims
            for dim in list(qrunoff.dims):
                if dim != 'time' and qrunoff.sizes[dim] == 1:
                    qrunoff = qrunoff.squeeze(dim)

            total_runoff = qrunoff.values.flatten()
            times = pd.to_datetime(ds['time'].values)
            ds.close()

            # Convert mm/s → m3/s
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            area_km2 = self._streamflow_metrics.get_catchment_area(
                config, project_dir, domain_name, source='shapefile'
            )
            area_m2 = area_km2 * 1e6
            streamflow_m3s = total_runoff * area_m2 / 1000.0

            sim_series = pd.Series(streamflow_m3s, index=times)

            # Skip warmup
            warmup_days = int(config.get('CLM_WARMUP_DAYS', 365))
            if warmup_days > 0 and len(sim_series) > warmup_days:
                sim_series = sim_series.iloc[warmup_days:]
                self.logger.debug(
                    f"Skipped {warmup_days} warmup days"
                )

            # Load observations
            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Align over calibration period only (not evaluation)
            cal_period_str = config.get('CALIBRATION_PERIOD')
            calibration_period = None
            if cal_period_str:
                parts = [s.strip() for s in str(cal_period_str).split(',')]
                if len(parts) == 2:
                    calibration_period = (parts[0], parts[1])

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series,
                calibration_period=calibration_period,
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating CLM metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove ALL stale output/restart files before a new run.

        Must remove rpointer files, coupler/datm restarts, and restart
        history files — otherwise CMEPS treats the next run as a
        continuation instead of a cold start, suppressing h0 output.
        """
        for pattern in [
            '*.clm2.h0.*.nc',   # history files
            '*.clm2.r.*.nc',    # CLM restart files
            '*.clm2.rh0.*.nc',  # restart history buffer files
            '*.cpl.r.*.nc',     # coupler restart files
            '*.datm.r.*.nc',    # data atmosphere restart files
            'rpointer.*',       # restart pointer files
            '*.log',            # log files
        ]:
            for f in output_dir.glob(pattern):
                f.unlink()

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_clm_parameters_worker(task_data)


def _evaluate_clm_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
        worker = CLMWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'CLM worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1),
        }
