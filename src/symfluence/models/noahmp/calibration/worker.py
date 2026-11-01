# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Noah-MP Worker.

Worker implementation for Noah-MP model optimization.
Handles parameter application to namelist.input and SOILPARM.TBL,
model execution, and metric calculation from output.nc.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    BaseWorker,
    run_worker_task,
)
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('NOAHMP')
class NoahMPWorker(BaseWorker):
    """
    Worker for Noah-MP model calibration.

    Handles parameter application to namelist.input (scalar params) and
    SOILPARM.TBL (soil hydraulic params), noah-owp-modular execution,
    and metric calculation from SFCRNOFF + UGDRNOFF output.
    """

    # Parameters written directly into namelist.input
    NAMELIST_PARAMS: Set[str] = {'rain_snow_thresh', 'ZREF', 'refkdt'}

    # SOILPARM.TBL column indices (0-based) for soil hydraulic parameters
    SOILPARM_COLUMNS: Dict[str, int] = {
        'bexp': 1,
        'smcmax': 4,
        'smcref': 5,
        'psisat': 6,
        'dksat': 7,
        'smcwlt': 9,
    }

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
        Apply calibration parameters to Noah-MP configuration files.

        Namelist parameters (rain_snow_thresh, ZREF, refkdt) are updated
        via regex substitution in namelist.input.  Soil hydraulic parameters
        (bexp, smcmax, etc.) are applied to SOILPARM.TBL for the active
        soil type.

        Args:
            params: Parameter name -> value mapping.
            settings_dir: Noah-MP settings directory.

        Returns:
            True if all parameters applied successfully.
        """
        try:
            settings_dir = Path(settings_dir)

            # route_k is a post-model routing knob (Noah-MP is a 1-D column and
            # emits unrouted runoff). It does not go to the model; stash it for
            # calculate_metrics, which routes the runoff before scoring.
            if 'route_k' in params:
                try:
                    (settings_dir / 'route_k.txt').write_text(
                        f"{float(params['route_k']):.6f}\n")
                except OSError as e:
                    self.logger.warning(f"Could not write route_k.txt: {e}")

            # Split into namelist vs soil parameters
            nl_params = {k: v for k, v in params.items() if k in self.NAMELIST_PARAMS}
            soil_params = {k: v for k, v in params.items() if k in self.SOILPARM_COLUMNS}
            other_params = {
                k: v for k, v in params.items()
                if k not in self.NAMELIST_PARAMS and k not in self.SOILPARM_COLUMNS
                and k not in ('slope', 'noah_czil', 'route_k')
            }

            # slope and noah_czil go into namelist as well
            for extra in ('slope', 'noah_czil'):
                if extra in params:
                    nl_params[extra] = params[extra]

            if other_params:
                self.logger.warning(f"Unknown Noah-MP params ignored: {list(other_params)}")

            success = True

            if nl_params:
                success &= self._apply_namelist_params(nl_params, settings_dir)

            if soil_params:
                success &= self._update_soilparm_tbl(soil_params, settings_dir)

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying Noah-MP parameters: {e}", exc_info=True)
            return False

    def _apply_namelist_params(
        self,
        params: Dict[str, float],
        settings_dir: Path,
    ) -> bool:
        """Apply scalar parameters to namelist.input via regex substitution."""
        nl_path = settings_dir / 'namelist.input'
        if not nl_path.exists():
            self.logger.error(f"namelist.input not found: {nl_path}")
            return False

        content = nl_path.read_text()

        for name, value in params.items():
            # Try to update existing line
            pattern = rf"({re.escape(name)}\s*=\s*)[-+]?\d*\.?\d+([eEdD][-+]?\d+)?"
            if re.search(pattern, content):
                content = re.sub(pattern, rf"\g<1>{value:.8g}", content)
            else:
                # Insert into &forcing or &model_options section
                section = '&forcing' if name == 'ZREF' else '&model_options'
                sec_pattern = rf"({section}\s*\n)(.*?)(^/\s*$)"
                match = re.search(sec_pattern, content, re.MULTILINE | re.DOTALL)
                if match:
                    insert = f"  {name}               = {value:.8g}\n"
                    content = (
                        content[:match.end(2)]
                        + insert
                        + content[match.start(3):]
                    )
                else:
                    self.logger.warning(
                        f"Could not find section {section} for param {name}"
                    )

        nl_path.write_text(content)
        self.logger.debug(f"Updated namelist.input with {len(params)} params")
        return True

    def _update_soilparm_tbl(
        self,
        params: Dict[str, float],
        settings_dir: Path,
    ) -> bool:
        """Update SOILPARM.TBL for the active soil type.

        Reads ``isltyp`` from namelist.input to determine which soil-type
        row to modify, then rewrites the corresponding columns.
        """
        soilparm_path = settings_dir / 'SOILPARM.TBL'
        if not soilparm_path.exists():
            self.logger.warning(f"SOILPARM.TBL not found: {soilparm_path}")
            return False

        # Read isltyp from namelist
        nl_path = settings_dir / 'namelist.input'
        isltyp = 4  # default
        if nl_path.exists():
            content = nl_path.read_text()
            match = re.search(r'isltyp\s*=\s*(\d+)', content)
            if match:
                isltyp = int(match.group(1))

        lines = soilparm_path.read_text().splitlines()

        # SOILPARM.TBL format: header lines, then data lines with comma-separated
        # values. Find the data section for the active classification (STAS).
        # A data row starts with an integer soil-type index whose *second* field
        # is numeric (e.g. "1, 2.79, ..."). The dimension header
        # ("19,1  'BB  DRYSMC ...") also starts with a digit, but its second
        # field is the quoted column-name list, so it is not float-parseable.
        # NOTE: extended tables append a quoted soil-name label to every data row
        # (".. , 'SAND'"), so we must NOT skip lines merely because they contain a
        # quote — that would skip every data row.
        def _is_float(token: str) -> bool:
            try:
                float(token)
                return True
            except ValueError:
                return False

        data_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or not stripped[0].isdigit():
                continue
            parts = [p.strip() for p in stripped.split(',')]
            if (len(parts) >= 5
                    and parts[0].lstrip('-').isdigit()
                    and _is_float(parts[1])):
                data_start = i
                break

        if data_start is None:
            self.logger.warning("Could not find data section in SOILPARM.TBL")
            return False

        # The soil type row is at data_start + (isltyp - 1)
        target_line = data_start + isltyp - 1
        if target_line >= len(lines):
            self.logger.warning(f"Soil type {isltyp} exceeds SOILPARM.TBL rows")
            return False

        # Parse the row: columns separated by commas
        parts = [p.strip() for p in lines[target_line].split(',')]

        for param_name, col_idx in self.SOILPARM_COLUMNS.items():
            if param_name in params and col_idx < len(parts):
                parts[col_idx] = f"{params[param_name]:.6f}"

        lines[target_line] = ', '.join(parts)
        soilparm_path.write_text('\n'.join(lines) + '\n')
        self.logger.debug(
            f"Updated SOILPARM.TBL soil type {isltyp} with "
            f"{len(params)} params"
        )
        return True

    def run_model(
        self,
        config: Dict,
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """
        Execute noah-owp-modular for calibration.

        Args:
            config: Configuration dictionary.
            settings_dir: Noah-MP settings directory.
            output_dir: Worker-specific output directory.

        Returns:
            True if execution succeeded.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Resolve executable
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            install_path = resolve_config_value(config, lambda: config.model.noahmp.install_path, 'default', 'NOAHMP_INSTALL_PATH')
            if install_path == 'default':
                install_dir = data_dir / 'installs' / 'noah-owp-modular'
            else:
                install_dir = Path(install_path)

            exe_name = resolve_config_value(config, lambda: config.model.noahmp.exe, 'noah_owp_modular.exe', 'NOAHMP_EXE')
            # The noah-owp-modular build places the executable under run/; older
            # layouts used bin/ or the install root. Check all three.
            noahmp_exe = None
            for sub in ('run', 'bin', '.'):
                cand = install_dir / sub / exe_name
                if cand.exists():
                    noahmp_exe = cand
                    break
            if noahmp_exe is None:
                self.logger.error(
                    f"Noah-MP executable '{exe_name}' not found under "
                    f"{install_dir}/(run|bin|.)"
                )
                return False

            # noah-owp-modular reads namelist.input from cwd
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'

            timeout = int(resolve_config_value(config, lambda: config.model.noahmp.timeout, 600, 'NOAHMP_TIMEOUT'))

            with open(output_dir / 'noahmp_stdout.log', 'w') as out, \
                 open(output_dir / 'noahmp_stderr.log', 'w') as err:
                result = run_subprocess(
                    [str(noahmp_exe)],
                    cwd=str(settings_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    timeout=timeout,
                )

            if result.returncode != 0:
                stderr_path = output_dir / 'noahmp_stderr.log'
                stderr_text = stderr_path.read_text(errors='replace') if stderr_path.exists() else ''
                self.logger.error(
                    f"Noah-MP failed (rc={result.returncode}): "
                    f"{stderr_text[-300:]}"
                )
                return False

            # Check for output
            output_nc = settings_dir / 'output.nc'
            if output_nc.exists():
                # Move output to output_dir if not already there
                dest = output_dir / 'output.nc'
                if output_nc.resolve() != dest.resolve():
                    shutil.copy2(output_nc, dest)
                return True

            # Also check output_dir
            if (output_dir / 'output.nc').exists():
                return True

            self.logger.error("No Noah-MP output.nc produced")
            return False

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Noah-MP timed out after {timeout}s")
            return False
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Noah-MP execution error: {e}", exc_info=True)
            return False

    @staticmethod
    def _read_route_k(settings_dir) -> Optional[float]:
        """Read the calibrated Nash routing time constant (days), if present."""
        if not settings_dir:
            return None
        f = Path(settings_dir) / 'route_k.txt'
        if not f.exists():
            return None
        try:
            return float(f.read_text().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _route_nash(series: 'pd.Series', k_days: float, n: int = 2) -> 'pd.Series':
        """Route a flow series through an n-reservoir Nash cascade.

        Each linear reservoir is y[t] = a*y[t-1] + (1-a)*x[t] with
        a = exp(-1/k). Volume-preserving; attenuates and delays the response.
        """
        if k_days is None or k_days <= 0 or len(series) < 2:
            return series
        a = float(np.exp(-1.0 / k_days))
        y = series.to_numpy(dtype=float)
        for _ in range(max(1, int(n))):
            out = np.empty_like(y)
            out[0] = y[0]
            for i in range(1, len(y)):
                out[i] = a * out[i - 1] + (1.0 - a) * y[i]
            y = out
        return pd.Series(y, index=series.index)

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict,
        **kwargs,
    ) -> Dict:
        """
        Calculate streamflow metrics from Noah-MP output.

        Extracts SFCRNOFF (surface runoff) + UGDRNOFF (subsurface drainage),
        converts to m3/s, aligns with observations, and computes KGE/NSE.
        """
        try:
            output_dir = Path(output_dir)
            output_nc = output_dir / 'output.nc'
            if not output_nc.exists():
                return {'kge': self.penalty_score, 'error': 'No output.nc'}

            ds = xr.open_dataset(output_nc)

            # Total runoff = surface + subsurface (kg/m2 = mm)
            sfcrnoff = ds['SFCRNOFF'].values.flatten() if 'SFCRNOFF' in ds else np.zeros(ds.dims.get('time', 0))
            ugdrnoff = ds['UGDRNOFF'].values.flatten() if 'UGDRNOFF' in ds else np.zeros(ds.dims.get('time', 0))
            total_runoff_mm = sfcrnoff + ugdrnoff

            times = pd.to_datetime(ds['time'].values)
            ds.close()

            # Convert mm -> m3/s
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            project_dir = data_dir / f'domain_{domain_name}'

            area_km2 = self._streamflow_metrics.get_catchment_area(
                config, project_dir, domain_name, source='shapefile'
            )
            area_m2 = area_km2 * 1e6

            # mm/timestep -> m3/s (assuming hourly timestep)
            dt_hours = 1.0
            if len(times) > 1:
                dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
            streamflow_m3s = total_runoff_mm * area_m2 / (1000.0 * dt_hours * 3600.0)

            sim_series = pd.Series(streamflow_m3s, index=times).resample('D').mean()

            # Route the (unrouted column) runoff through a Nash cascade if a
            # routing time constant was calibrated (route_k, in days). This is
            # the dominant skill lever for Noah-MP: a 1-D column emits runoff
            # ~10x too flashy, and a 2-reservoir cascade attenuates it into a
            # realistic hydrograph (Bow at Banff: KGE -0.18 -> ~0.85).
            route_k = self._read_route_k(kwargs.get('settings_dir'))
            if route_k and route_k > 0:
                sim_series = self._route_nash(sim_series, route_k, n=2)

            # Load observations
            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Align
            cal_period_str = resolve_config_value(config, lambda: config.domain.calibration_period, None, 'CALIBRATION_PERIOD')
            calibration_period = None
            if cal_period_str:
                parts = [s.strip() for s in str(cal_period_str).split(',')]
                if len(parts) == 2:
                    calibration_period = (parts[0], parts[1])

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series,
                calibration_period=calibration_period,
            )

            return self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating Noah-MP metrics: {e}", exc_info=True)
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_noahmp_parameters_worker(task_data)


def _evaluate_noahmp_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        NoahMPWorker,
        task_data,
        model_name='Noah-MP',
        extra_env=MACOS_MALLOC_QUIET_ENV,
    )
