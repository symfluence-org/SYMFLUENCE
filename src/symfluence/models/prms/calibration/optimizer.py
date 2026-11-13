# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Model Optimizer

PRMS-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import PRMSWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('PRMS')
class PRMSModelOptimizer(BaseModelOptimizer):
    """
    PRMS-specific optimizer using the unified BaseModelOptimizer framework.

    Supports all standard optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    """

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

        self.prms_setup_dir = self.project_dir / 'settings' / 'PRMS'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("PRMSModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'PRMS'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to PRMS control file."""
        control_file = self._get_config_value(
            lambda: self.config.model.prms.control_file,
            dict_key='PRMS_CONTROL_FILE',
        )
        return self.prms_setup_dir / control_file

    def _create_parameter_manager(self):
        """Create PRMS parameter manager."""
        from .parameter_manager import PRMSParameterManager
        return PRMSParameterManager(
            self.config,
            self.logger,
            self.prms_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if external routing is needed for PRMS.

        PRMS has built-in segment routing, so external routing is typically
        not needed unless explicitly configured.
        """
        routing_model = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        return routing_model.lower() not in ('none', '', 'n/a')

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run PRMS for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.prms_setup_dir)

        success = self.worker.run_model(
            self.config,
            self.prms_setup_dir,
            output_dir
        )

        if success:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Copy statvar output to final_evaluation dir
            for pattern in ['statvar*', '*.csv']:
                for f in self.prms_setup_dir.glob(pattern):
                    if f.is_file():
                        shutil.copy2(f, output_dir / f.name)
            self.logger.info(f"Copied PRMS outputs to {output_dir}")

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation using PRMS worker metrics.

        Computes KGE separately for the calibration and evaluation periods
        so that split-sample validation is reported correctly.
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error("PRMS run failed during final evaluation")
                return None

            # --- Calibration-period metrics ---
            cal_period = self._get_config_value(lambda: self.config.domain.calibration_period, default='', dict_key='CALIBRATION_PERIOD')
            cal_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
            cal_config['CALIBRATION_PERIOD'] = cal_period
            cal_raw = self.worker.calculate_metrics(
                final_output_dir, cal_config,
                sim_dir=final_output_dir
            )

            if not cal_raw or cal_raw.get('kge', -999) <= -999:
                self.logger.error("Failed to calculate calibration-period metrics")
                return None

            calib_metrics = {"KGE_Calib": cal_raw.get('kge', -999)}
            self.logger.info(f"Calibration period KGE: {cal_raw.get('kge', 'N/A'):.4f}")

            # --- Evaluation-period metrics ---
            eval_metrics: Dict[str, float] = {}
            eval_period = self._get_config_value(lambda: self.config.domain.evaluation_period, default='', dict_key='EVALUATION_PERIOD')
            if eval_period and ',' in str(eval_period):
                eval_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
                eval_config['CALIBRATION_PERIOD'] = eval_period  # worker filters on this key
                eval_raw = self.worker.calculate_metrics(
                    final_output_dir, eval_config,
                    sim_dir=final_output_dir
                )
                if eval_raw and eval_raw.get('kge', -999) > -999:
                    eval_metrics = {"KGE_Eval": eval_raw.get('kge', -999)}
                    self.logger.info(f"Evaluation period KGE: {eval_raw.get('kge', 'N/A'):.4f}")
                else:
                    self.logger.warning("Evaluation-period metrics could not be computed")
            else:
                self.logger.info("No EVALUATION_PERIOD configured; skipping split-sample validation")

            final_result = {
                'final_metrics': cal_raw,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            return final_result

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in final evaluation: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None
