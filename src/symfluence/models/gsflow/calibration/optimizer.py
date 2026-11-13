# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Model Optimizer.

GSFLOW-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import GSFLOWWorker  # noqa: F401


@R.optimizers.add('GSFLOW')
class GSFLOWModelOptimizer(BaseModelOptimizer):
    """GSFLOW-specific optimizer using the unified BaseModelOptimizer framework."""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
        self.data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, None, 'SYMFLUENCE_DATA_DIR'))
        self.domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.gsflow_setup_dir = self.project_dir / 'settings' / 'GSFLOW'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("GSFLOWModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'GSFLOW'

    def _get_final_file_manager_path(self) -> Path:
        control_file = self._get_config_value(
            lambda: self.config.model.gsflow.control_file,
            dict_key='GSFLOW_CONTROL_FILE',
        )
        return self.gsflow_setup_dir / control_file

    def _create_parameter_manager(self):
        from .parameter_manager import GSFLOWParameterManager
        return GSFLOWParameterManager(
            self.config, self.logger, self.gsflow_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """GSFLOW has internal SFR routing."""
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        best_result = self.get_best_result()
        best_params = best_result.get('params')
        if not best_params:
            self.logger.warning("No best parameters found")
            return False

        self.worker.apply_parameters(best_params, self.gsflow_setup_dir)
        success = self.worker.run_model(
            self.config, self.gsflow_setup_dir, output_dir
        )

        if success:
            output_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ['statvar*', '*.csv']:
                for f in self.gsflow_setup_dir.glob(pattern):
                    if f.is_file():
                        shutil.copy2(f, output_dir / f.name)

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with split-sample support.

        Calculates calibration-period metrics using CALIBRATION_PERIOD and,
        if EVALUATION_PERIOD is configured, computes independent evaluation
        metrics on the held-out period.
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING GSFLOW FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                return None

            # Calibration period metrics (uses CALIBRATION_PERIOD from config)
            metrics = self.worker.calculate_metrics(
                final_output_dir, self.config, sim_dir=final_output_dir
            )

            if not metrics or metrics.get('kge', -999) <= -999:
                return None

            calib_metrics = {
                "KGE_Calib": metrics.get('kge', -999),
                "NSE_Calib": metrics.get('nse', -999),
            }
            self.logger.info(
                f"Calibration period: KGE={metrics.get('kge', 'N/A'):.4f}, "
                f"NSE={metrics.get('nse', 'N/A'):.4f}"
            )

            # Evaluation period metrics (if EVALUATION_PERIOD is configured)
            eval_metrics: Dict[str, float] = {}
            eval_period = self._get_config_value(lambda: self.config.domain.evaluation_period, default='', dict_key='EVALUATION_PERIOD')
            if eval_period and ',' in str(eval_period):
                eval_raw = self.worker.calculate_metrics(
                    final_output_dir, self.config,
                    sim_dir=final_output_dir, period=eval_period
                )
                if eval_raw and eval_raw.get('kge', -999) > -900:
                    eval_metrics = {
                        "KGE_Eval": float(eval_raw.get('kge', 0.0)),
                        "NSE_Eval": float(eval_raw.get('nse', 0.0)),
                    }
                    self.logger.info(
                        f"Evaluation period: KGE={eval_raw.get('kge', 'N/A'):.4f}, "
                        f"NSE={eval_raw.get('nse', 'N/A'):.4f}"
                    )
                else:
                    self.logger.warning(
                        f"Could not compute evaluation-period metrics "
                        f"for period '{eval_period}'"
                    )

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            self._save_final_evaluation_results(final_result, 'DDS')
            return final_result

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in final evaluation: {e}", exc_info=True)
            return None
