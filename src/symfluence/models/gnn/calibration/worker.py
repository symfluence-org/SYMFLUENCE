# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GNN Optimization Worker

Handles individual model evaluations for GNN calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.calibration.targets import create_calibration_target
from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerResult, WorkerTask
from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.gnn import GNNRunner


class GNNWorker(BaseWorker):
    """
    Worker class for evaluating GNN model during optimization.
    """

    def evaluate(self, task: WorkerTask) -> WorkerResult:
        try:
            eval_config = task.config.copy()
            for key, val in task.params.items():
                eval_config[key] = val

            eval_config['EXPERIMENT_ID'] = f"{resolve_config_value(task.config, lambda: task.config.domain.experiment_id, None, 'EXPERIMENT_ID')}_eval_{task.proc_id}"

            runner = GNNRunner(eval_config, self.logger)
            runner.run_gnn()

            target = create_calibration_target(
                'streamflow',
                eval_config,
                Path(eval_config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{eval_config.get('DOMAIN_NAME')}",
                self.logger
            )

            sim_dir = runner.output_dir
            metrics = target.calculate_metrics(sim_dir)
            metric_name = eval_config.get('OPTIMIZATION_METRIC', 'KGE')
            score = metrics.get(metric_name, self.penalty_score)

            return WorkerResult(
                individual_id=task.individual_id,
                score=score,
                metrics=metrics,
                params=task.params
            )

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in GNN evaluation: {e}", exc_info=True)
            import traceback
            return WorkerResult(
                individual_id=task.individual_id,
                score=self.penalty_score,
                error=traceback.format_exc(),
                params=task.params
            )

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs: Any) -> bool:
        runner = GNNRunner(config, self.logger)
        runner.run_gnn()
        return True

    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs: Any) -> bool:
        return True

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate metrics for GNN model."""
        target = create_calibration_target(
            'streamflow',
            config,
            Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR')) / f"domain_{resolve_config_value(config, lambda: config.domain.name, '', 'DOMAIN_NAME')}",
            self.logger
        )
        metrics = target.calculate_metrics(output_dir)
        return metrics or {'KGE': self.penalty_score}
