# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Worker

Worker implementation for CLMParFlow model optimization.
Handles .pfidb parameter updates, CLMParFlow execution, and metric extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from symfluence.core.calibration.workers.base_worker import BaseWorker, run_worker_task
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.workers.add('CLMPARFLOW')
class CLMParFlowWorker(BaseWorker):
    """
    Parallel worker for CLMParFlow model calibration.

    Each worker instance:
    1. Copies .pfidb + CLM settings to an isolated directory
    2. Updates van Genuchten / hydraulic parameters in the .pfidb
    3. Executes CLMParFlow (ParFlow compiled with CLM)
    4. Extracts overland + subsurface flow and calculates metrics
    """

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs,
    ) -> bool:
        """Apply parameters by updating the .pfidb file in settings_dir."""
        import json

        from symfluence.models.parflow.calibration.parameter_manager import (
            PARAM_TO_PFIDB_KEYS,
            ROUTING_PARAM_NAMES,
            SNOW17_PARAM_NAMES,
            _read_pfidb,
            _write_pfidb,
        )

        pfidb_files = list(settings_dir.glob('*.pfidb'))
        if not pfidb_files:
            self.logger.error(f"No .pfidb file in {settings_dir}")
            return False

        try:
            pfidb_path = pfidb_files[0]
            entries = _read_pfidb(pfidb_path)

            # Separate routing, Snow-17, and subsurface params
            routing_params = {k: v for k, v in params.items() if k in ROUTING_PARAM_NAMES}
            snow_params = {k: v for k, v in params.items() if k in SNOW17_PARAM_NAMES}
            subsurface_params = {
                k: v for k, v in params.items()
                if k not in SNOW17_PARAM_NAMES and k not in ROUTING_PARAM_NAMES
            }

            # Write routing params to sidecar JSON for target extraction
            if routing_params:
                sidecar = settings_dir / 'routing_params.json'
                sidecar.write_text(json.dumps(routing_params))

            # Validate TOP > BOT
            top = subsurface_params.get('TOP')
            bot = subsurface_params.get('BOT')
            if top is not None and bot is not None and top <= bot:
                subsurface_params['BOT'] = top - 50.0

            # Update .pfidb keys for subsurface params
            for param_name, value in subsurface_params.items():
                pfidb_keys = PARAM_TO_PFIDB_KEYS.get(param_name, [])
                for key in pfidb_keys:
                    entries[key] = f'{value:g}'

            # If Snow-17 params changed, regenerate forcing
            if snow_params:
                from .parameter_manager import CLMParFlowParameterManager
                pm = CLMParFlowParameterManager(
                    kwargs.get('config', self.config or {}),
                    self.logger,
                    settings_dir,
                )
                pm._update_snow17_forcing(entries, snow_params)

            _write_pfidb(pfidb_path, entries)
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to apply CLMParFlow parameters: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs,
    ) -> bool:
        """Execute CLMParFlow with parameters from kwargs."""
        try:
            params = kwargs.get('params')

            # Apply parameters to .pfidb before running
            if params:
                if not self.apply_parameters(params, settings_dir):
                    return False

            from symfluence.models.clmparflow.runner import CLMParFlowRunner
            runner = CLMParFlowRunner(config, self.logger)

            # Override settings and output directories for worker isolation
            runner.settings_dir = settings_dir
            runner.output_dir = output_dir

            result = runner.run_clmparflow(sim_dir=output_dir)
            return result is not None

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error running CLMParFlow in worker: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Calculate performance metrics from CLMParFlow output."""
        try:
            from .targets import CLMParFlowStreamflowTarget

            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = config.get('DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"

            settings_dir = kwargs.get('settings_dir')
            target = CLMParFlowStreamflowTarget(
                config, project_dir, self.logger,
                settings_dir=settings_dir,
            )
            metrics = target.calculate_metrics(output_dir, calibration_only=True)

            if metrics:
                return metrics
            else:
                self.logger.warning("CLMParFlow target returned empty metrics")
                return {'kge': self.penalty_score}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating CLMParFlow metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for parallel execution."""
        return _evaluate_clmparflow_parameters_worker(task_data)


def _evaluate_clmparflow_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        CLMParFlowWorker,
        task_data,
        isolate=False,
        catch_exceptions=False,
    )
