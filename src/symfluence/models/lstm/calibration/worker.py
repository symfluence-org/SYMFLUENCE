# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Optimization Worker

Handles individual model evaluations for LSTM calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.calibration.targets import create_calibration_target
from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerResult, WorkerTask
from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.lstm import LSTMRunner


class LSTMWorker(BaseWorker):
    """
    Worker class for evaluating LSTM model during optimization.
    """

    def evaluate(self, task: WorkerTask) -> WorkerResult:
        """
        Evaluate a single parameter set.

        Args:
            task: Task containing parameters and paths

        Returns:
            WorkerResult with score and metrics
        """
        try:
            # 1. Update config with parameters
            # LSTM parameters might be learning rate, hidden size, etc.
            # or multipliers for physical inputs
            eval_config = task.config.copy()
            for key, val in task.params.items():
                eval_config[key] = val

            # Override directories for isolation
            eval_config['EXPERIMENT_ID'] = f"{resolve_config_value(task.config, lambda: task.config.domain.experiment_id, None, 'EXPERIMENT_ID')}_eval_{task.proc_id}"

            # 2. Run model
            runner = LSTMRunner(eval_config, self.logger)
            runner.run_lstm()

            # 3. Calculate score
            # Optimization results are in output_dir (or sim_dir)
            target = create_calibration_target(
                'streamflow', eval_config,
                Path(eval_config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{eval_config.get('DOMAIN_NAME')}",
                self.logger,
            )

            # Determine output directory (if routing was used, score comes from routed output)
            sim_dir = runner.output_dir
            if runner.requires_routing():
                # Check for mizuRoute or dRoute subdirs
                routing_model = eval_config.get('ROUTING_MODEL', 'none').lower()
                if 'droute' in routing_model:
                    sim_dir = sim_dir.parent / 'dRoute'
                else:
                    sim_dir = sim_dir.parent / 'mizuRoute'

            metrics = target.calculate_metrics(sim_dir)

            # Return primary metric as score
            metric_name = eval_config.get('OPTIMIZATION_METRIC', 'KGE')
            score = metrics.get(metric_name, self.penalty_score)

            return WorkerResult(
                individual_id=task.individual_id,
                score=score,
                metrics=metrics,
                params=task.params
            )

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in LSTM evaluation: {e}", exc_info=True)
            import traceback
            return WorkerResult(
                individual_id=task.individual_id,
                score=self.penalty_score,
                error=traceback.format_exc(),
                params=task.params
            )

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs: Any) -> bool:
        """Direct model execution hook."""
        runner = LSTMRunner(config, self.logger)
        runner.run_lstm()
        return True

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs: Any) -> bool:
        """Apply parameters to model configuration."""
        # For LSTM, we typically just pass parameters through config
        return True

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate objective metrics from LSTM model outputs.

        Args:
            output_dir: Path to model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments (sim_dir, proc_id, etc.)

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Determine the correct simulation directory
            sim_dir = output_dir
            routing_model = resolve_config_value(config, lambda: config.model.routing_model, 'none', 'ROUTING_MODEL').lower()
            if routing_model and routing_model != 'none':
                if 'droute' in routing_model:
                    potential_dir = output_dir.parent / 'dRoute'
                    if potential_dir.exists():
                        sim_dir = potential_dir
                else:
                    potential_dir = output_dir.parent / 'mizuRoute'
                    if potential_dir.exists():
                        sim_dir = potential_dir

            # Use StreamflowTarget for metrics calculation
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            domain_name = resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"

            target = create_calibration_target('streamflow', config, project_dir, self.logger)
            metrics = target.calculate_metrics(sim_dir)

            return metrics or {'KGE': self.penalty_score}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating LSTM metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'KGE': self.penalty_score}
