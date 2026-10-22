# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Worker

Worker implementation for PIHM model optimization.
Handles parameter application, PIHM execution, and metric calculation.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.constants import ModelDefaults
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('PIHM')
class PIHMWorker(BaseWorker):
    """
    Worker for PIHM model calibration.

    Handles parameter application to PIHM input files,
    PIHM execution, and streamflow metric calculation.
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
        """Apply parameters to PIHM input files."""
        try:
            self.logger.debug(f"Applying PIHM parameters to {settings_dir}")

            from .parameter_manager import PIHMParameterManager

            config = kwargs.get('config', self.config) or {}
            pm = PIHMParameterManager(config, self.logger, settings_dir)
            return pm.update_model_files(params)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying PIHM parameters: {e}", exc_info=True)
            return False

    def run_model(
        self,
        config: Dict,
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Execute PIHM for calibration.

        Sets up the correct MM-PIHM directory structure:
            output_dir/input/pihm_lumped/   -- input files
            output_dir/output/pihm_lumped/  -- model output
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            project_name = "pihm_lumped"

            # Set up MM-PIHM directory structure
            input_dir = output_dir / "input" / project_name
            pihm_output = output_dir / "output" / project_name
            input_dir.mkdir(parents=True, exist_ok=True)
            pihm_output.mkdir(parents=True, exist_ok=True)

            # Clean stale output
            for f in pihm_output.glob('*.txt'):
                f.unlink()
            for f in pihm_output.glob('*.dat'):
                f.unlink()

            # Copy input files to input/pihm_lumped/
            settings_dir = Path(settings_dir)
            for src in settings_dir.iterdir():
                if src.is_file() and src.name.startswith(project_name):
                    shutil.copy2(src, input_dir / src.name)

            # Get executable
            install_path = config.get('PIHM_INSTALL_PATH', 'default')
            if install_path == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                install_path = str(data_dir / 'installs' / 'pihm')

            exe_name = config.get('PIHM_EXE', 'flux-pihm')
            # Search order: bin/<exe>, <exe>, bin/pihm, pihm
            pihm_exe = None
            for candidate in [
                Path(install_path) / 'bin' / exe_name,
                Path(install_path) / exe_name,
                Path(install_path) / 'bin' / 'pihm',
                Path(install_path) / 'pihm',
            ]:
                if candidate.exists():
                    pihm_exe = candidate
                    break
            if pihm_exe is None:
                self.logger.error(f"PIHM executable not found in {install_path}")
                return False

            # Copy global lookup tables from PIHM install dir
            # The install dir is the MM-PIHM repo root containing input/
            pihm_install_dir = Path(install_path)
            pihm_install_input = pihm_install_dir / "input"
            if not pihm_install_input.exists():
                # Fallback: exe might be in bin/, so go up one level
                pihm_install_input = pihm_exe.parent.parent / "input"
            base_input = output_dir / "input"
            for gf in ["vegprmt.tbl", "co2.txt", "ndep.txt"]:
                src = pihm_install_input / gf
                dst = base_input / gf
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            epc_src = pihm_install_input / "epc"
            epc_dst = base_input / "epc"
            if epc_src.is_dir() and not epc_dst.exists():
                shutil.copytree(epc_src, epc_dst)

            timeout = int(config.get('PIHM_TIMEOUT', 3600))

            # Run PIHM: pihm -o <project_name> <project_name>
            # PIHM internally prepends "output/" to the -o argument
            env = os.environ.copy()
            result = run_subprocess(
                [str(pihm_exe), "-o", project_name, project_name],
                cwd=str(output_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                self.logger.error(
                    f"PIHM failed (rc={result.returncode}): "
                    f"{result.stderr[-300:] if result.stderr else 'no stderr'}"
                )
                return False

            # Verify output in output/pihm_lumped/
            flux_files = list(pihm_output.glob('*.river.flx*.txt'))
            gw_files = list(pihm_output.glob('*.gw.txt'))
            if not flux_files and not gw_files:
                self.logger.error("No PIHM output files produced")
                return False

            self.logger.debug(
                f"PIHM run complete: {len(flux_files)} flux, "
                f"{len(gw_files)} gw file(s)"
            )
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"PIHM execution error: {e}", exc_info=True)
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict,
        **kwargs,
    ) -> Dict:
        """Calculate streamflow metrics from PIHM output."""
        try:
            output_dir = Path(output_dir)

            # MM-PIHM output is in output/<project_name>/ subdirectory
            pihm_output = output_dir / "output" / "pihm_lumped"
            if not pihm_output.exists():
                pihm_output = output_dir  # fallback

            from symfluence.models.pihm.extractor import PIHMResultExtractor
            extractor = PIHMResultExtractor()

            river_flux = extractor.extract_variable(
                pihm_output, 'river_flux',
                outlet_segment=int(config.get('PIHM_OUTLET_SEGMENT', 1)),
            )

            if river_flux.empty:
                return {'kge': self.penalty_score, 'error': 'No PIHM output'}

            if (river_flux < 0).any():
                return {'kge': self.penalty_score, 'error': 'Negative PIHM outlet discharge'}
            interval = river_flux.index.to_series().diff().dropna().iloc[0]
            samples_per_day = pd.Timedelta(days=1) / interval
            if (samples_per_day != int(samples_per_day) or
                    not (river_flux.resample('D').count() == int(samples_per_day)).all()):
                return {'kge': self.penalty_score, 'error': 'Partial PIHM output day'}
            sim_series = river_flux.resample('D').mean()

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

            # Filter to calibration period if defined
            cal_period = config.get('CALIBRATION_PERIOD')
            if cal_period:
                parts = str(cal_period).split(',')
                if len(parts) == 2:
                    cal_start = pd.Timestamp(parts[0].strip())
                    cal_end = pd.Timestamp(parts[1].strip())
                    sim_series = sim_series[cal_start:cal_end]
                    obs_series = obs_series[cal_start:cal_end]

            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating PIHM metrics: {e}", exc_info=True)
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_pihm_parameters_worker(task_data)


def _evaluate_pihm_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    import traceback

    try:
        worker = PIHMWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'PIHM worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1),
        }
