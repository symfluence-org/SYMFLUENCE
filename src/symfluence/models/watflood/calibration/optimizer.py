# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD Model Optimizer.

WATFLOOD-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import WATFLOODWorker  # noqa: F401


@R.optimizers.add('WATFLOOD')
class WATFLOODModelOptimizer(BaseModelOptimizer):
    """WATFLOOD-specific optimizer using the unified BaseModelOptimizer framework."""

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
        self.watflood_setup_dir = self.project_dir / 'WATFLOOD_input' / 'settings'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("WATFLOODModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'WATFLOOD'

    def _get_final_file_manager_path(self) -> Path:
        par_file = self._get_config_value(
            lambda: self.config.model.watflood.par_file,
            default='bow.par',
            dict_key='WATFLOOD_PAR_FILE'
        )
        # Check basin/ subdirectory first (standard WATFLOOD layout)
        basin_path = self.watflood_setup_dir / 'basin' / par_file
        if basin_path.exists():
            return basin_path
        return self.watflood_setup_dir / par_file

    def _create_parameter_manager(self):
        from .parameter_manager import WATFLOODParameterManager
        return WATFLOODParameterManager(
            self.config, self.logger, self.watflood_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """WATFLOOD has internal channel routing."""
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        best_result = self.get_best_result()
        best_params = best_result.get('params')
        if not best_params:
            self.logger.warning("No best parameters found")
            return False

        self.worker.apply_parameters(best_params, self.watflood_setup_dir)
        success = self.worker.run_model(
            self.config, self.watflood_setup_dir, output_dir
        )

        if success:
            output_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ['*.tb0', '*.csv']:
                for f in self.watflood_setup_dir.glob(pattern):
                    if f.is_file():
                        shutil.copy2(f, output_dir / f.name)

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        self.logger.info("=" * 60)
        self.logger.info("RUNNING WATFLOOD FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                return None

            # --- Calibration-period metrics ---
            cal_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
            cal_raw = self.worker.calculate_metrics(
                final_output_dir, cal_config, sim_dir=final_output_dir
            )

            if not cal_raw or cal_raw.get('kge', -999) <= -999:
                return None

            self.logger.info(f"Calibration period KGE: {cal_raw.get('kge', 'N/A'):.4f}")

            # --- Evaluation-period metrics ---
            eval_metrics: Dict[str, float] = {}
            eval_period = self._get_config_value(lambda: self.config.domain.evaluation_period, default='', dict_key='EVALUATION_PERIOD')
            if eval_period and ',' in str(eval_period):
                eval_cfg = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
                eval_cfg['CALIBRATION_PERIOD'] = eval_period  # worker filters on this key
                eval_raw = self.worker.calculate_metrics(
                    final_output_dir, eval_cfg, sim_dir=final_output_dir
                )
                if eval_raw and eval_raw.get('kge', -999) > -999:
                    eval_metrics = {"KGE_Eval": eval_raw.get('kge', -999)}
                    self.logger.info(f"Evaluation period KGE: {eval_raw.get('kge', 'N/A'):.4f}")
                else:
                    self.logger.warning("Evaluation-period metrics could not be computed")
            else:
                self.logger.info("No EVALUATION_PERIOD configured; skipping split-sample validation")

            return {
                'final_metrics': cal_raw,
                'calibration_metrics': {"KGE_Calib": cal_raw.get('kge', -999)},
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in final evaluation: {e}", exc_info=True)
            return None
