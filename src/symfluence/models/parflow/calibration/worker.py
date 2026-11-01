# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Worker

Worker implementation for ParFlow model optimization.
Handles .pfidb parameter updates, ParFlow execution, and metric extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from symfluence.core.calibration.workers.base_worker import BaseWorker, run_worker_task
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.workers.add('PARFLOW')
class ParFlowWorker(BaseWorker):
    """
    Parallel worker for ParFlow model calibration.

    Each worker instance:
    1. Copies .pfidb settings to an isolated directory
    2. Updates van Genuchten / hydraulic parameters in the .pfidb
    3. Executes ParFlow
    4. Extracts overland + subsurface flow and calculates metrics
    """

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs,
    ) -> bool:
        """Apply parameters by updating the .pfidb file in settings_dir.

        Routing parameters (ROUTE_*) are written to a sidecar JSON file
        instead of the .pfidb, since they are post-processing parameters.
        Snow-17 parameters trigger forcing regeneration via the parameter manager.
        """
        import json

        from .parameter_manager import (
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
                from .parameter_manager import ParFlowParameterManager
                pm = ParFlowParameterManager(
                    kwargs.get('config', self.config or {}),
                    self.logger,
                    settings_dir,
                )
                pm._update_snow17_forcing(entries, snow_params)

            _write_pfidb(pfidb_path, entries)
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to apply ParFlow parameters: {e}", exc_info=True)
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
        """Execute ParFlow with parameters from kwargs.

        Args:
            config: Configuration dictionary
            settings_dir: Path to settings with .pfidb file
            output_dir: Path for model output
            **kwargs: Must include 'params' dict
        """
        try:
            params = kwargs.get('params')

            # Apply parameters to .pfidb before running
            if params:
                if not self.apply_parameters(params, settings_dir):
                    return False

            from symfluence.models.parflow.runner import ParFlowRunner
            runner = ParFlowRunner(config, self.logger)

            # Override settings and output directories for worker isolation
            runner.settings_dir = settings_dir
            runner.output_dir = output_dir

            result = runner.run_parflow(sim_dir=output_dir)
            return result is not None

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error running ParFlow in worker: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Calculate performance metrics from ParFlow output.

        Extracts overland flow + subsurface drainage and compares
        against observed streamflow using KGE, NSE, RMSE.
        """
        try:
            from .targets import ParFlowStreamflowTarget

            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"

            # Pass settings_dir so target can read routing_params.json
            settings_dir = kwargs.get('settings_dir')
            target = ParFlowStreamflowTarget(
                config, project_dir, self.logger,
                settings_dir=settings_dir,
            )
            metrics = target.calculate_metrics(output_dir, calibration_only=True)

            if metrics:
                return metrics
            else:
                self.logger.warning("ParFlow target returned empty metrics")
                return {'kge': self.penalty_score}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating ParFlow metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for parallel execution."""
        return _evaluate_parflow_parameters_worker(task_data)


def _evaluate_parflow_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI execution (delegates to ``run_worker_task``)."""
    return run_worker_task(
        ParFlowWorker,
        task_data,
        isolate=False,
        catch_exceptions=False,
    )
