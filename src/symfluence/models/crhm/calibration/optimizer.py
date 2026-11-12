# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Model Optimizer

CRHM-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import CRHMWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('CRHM')
class CRHMModelOptimizer(BaseModelOptimizer):
    """
    CRHM-specific optimizer using the unified BaseModelOptimizer framework.

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

        self.crhm_setup_dir = self.project_dir / 'settings' / 'CRHM'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("CRHMModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'CRHM'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to CRHM project file."""
        prj_file = self._get_config_value(lambda: self.config.model.crhm.project_file, dict_key='CRHM_PROJECT_FILE')
        return self.crhm_setup_dir / prj_file

    def _create_parameter_manager(self):
        """Create CRHM parameter manager."""
        from .parameter_manager import CRHMParameterManager
        return CRHMParameterManager(
            self.config,
            self.logger,
            self.crhm_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed for CRHM.

        CRHM handles its own internal routing via the Netroute module,
        so external routing is typically not needed.
        """
        routing_integration = self._get_config_value(
            lambda: None,
            default='none',
            dict_key='CRHM_ROUTING_INTEGRATION'
        )
        global_routing = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        return (routing_integration.lower() != 'none' or
                global_routing.lower() not in ('none', ''))

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run CRHM for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.crhm_setup_dir)

        success = self.worker.run_model(
            self.config,
            self.crhm_setup_dir,
            output_dir
        )

        if success:
            # Copy CRHM output files to final_evaluation dir
            output_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ['*.csv', '*.obs', '*.txt']:
                for f in self.crhm_setup_dir.glob(pattern):
                    if f.is_file() and f.stat().st_size > 0 and 'output' in f.name.lower():
                        shutil.copy2(f, output_dir / f.name)
            self.logger.info(f"Copied CRHM outputs to {output_dir}")

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation using CRHM worker metrics."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error("CRHM run failed during final evaluation")
                return None

            # --- Calibration-period metrics ---
            cal_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
            cal_raw = self.worker.calculate_metrics(final_output_dir, cal_config)
            calib_kge = cal_raw.get('kge') if cal_raw else None

            if calib_kge is None or calib_kge <= -900:
                self.logger.error("Failed to calculate calibration-period metrics")
                return None

            # Sanity check against the optimization's DDS-verified best score:
            # a large gap means the standalone re-run did not reproduce the
            # model setup exactly and the reported skill is questionable.
            best_score = self.get_best_result().get('score')
            if best_score is not None and abs(calib_kge - best_score) > 0.05:
                self.logger.warning(
                    f"Final-eval calibration-period KGE {calib_kge:.3f} diverges from "
                    f"optimization best {best_score:.3f}; check the final re-run setup."
                )
            self.logger.info(f"Calibration period KGE: {calib_kge:.4f}")

            # --- Evaluation-period metrics ---
            eval_metrics: Dict[str, float] = {}
            eval_period = self._get_config_value(lambda: self.config.domain.evaluation_period, default='', dict_key='EVALUATION_PERIOD')
            if eval_period and ',' in str(eval_period):
                eval_cfg = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
                eval_cfg['CALIBRATION_PERIOD'] = eval_period  # worker filters on this key
                eval_raw = self.worker.calculate_metrics(final_output_dir, eval_cfg)
                eval_kge = eval_raw.get('kge') if eval_raw else None
                if eval_kge is not None and eval_kge > -900:
                    eval_metrics = {"KGE": eval_kge, "KGE_Eval": eval_kge}
                    self.logger.info(f"Evaluation period KGE: {eval_kge:.4f}")
                else:
                    self.logger.warning("Evaluation-period metrics could not be computed")
            else:
                self.logger.info("No EVALUATION_PERIOD configured; skipping split-sample validation")

            final_metrics = dict(cal_raw)
            if best_score is not None:
                final_metrics['optimization_best_kge'] = best_score
            # Use the canonical 'KGE' key so downstream aggregation finds it.
            calib_metrics = {"KGE": calib_kge, "KGE_Calib": calib_kge}

            final_result = {
                'final_metrics': final_metrics,
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
