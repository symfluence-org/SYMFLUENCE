# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH Model Optimizer

MESH-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with MESH.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.file_utils import safe_delete
from symfluence.core.registries import R

from .worker import MESHWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('MESH')
class MESHModelOptimizer(BaseModelOptimizer):
    """
    MESH-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = MESHModelOptimizer(config, logger)
        results_path = optimizer.run_dds()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize MESH optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("MESHModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'MESH'

    def _apply_best_parameters_for_final(self, best_params):
        """Apply best parameters for final evaluation, pointing to forcing directory."""
        forcing_dir = self.project_forcing_dir / 'MESH_input'
        return self.worker.apply_parameters(
            best_params,
            self.optimization_settings_dir,
            config=self.config,
            proc_forcing_dir=str(forcing_dir)
        )

    def run_final_evaluation(self, best_params):
        """Run final evaluation using MESH worker's metric calculation.

        Overrides the base implementation because the generic StreamflowTarget
        evaluator expects NetCDF files (SUMMA format), but MESH outputs CSV.
        The MESH worker's calculate_metrics already handles CSV output correctly.
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Running model with best parameters over full simulation period...")

        try:
            # Update file manager for full period
            self._update_file_manager_for_final_run()

            # Apply best parameters to forcing directory
            if not self._apply_best_parameters_for_final(best_params):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            # Setup output directory
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # Run MESH model
            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error(
                    "MESH run failed during final evaluation "
                    "(see the MESH output/coverage message above — often the forcing ends "
                    "before the configured simulation end date)"
                )
                return None

            # MESH writes output to forcing/MESH_input/results/
            # Copy results to final_evaluation directory and also use that path
            forcing_results = self.project_forcing_dir / 'MESH_input' / 'results'
            if forcing_results.exists():
                for f in forcing_results.iterdir():
                    if f.is_file():
                        shutil.copy2(f, final_output_dir / f.name)

            # Use MESH worker's calculate_metrics (handles CSV output)
            # Calibration period metrics (uses CALIBRATION_PERIOD from config)
            metrics = self.worker.calculate_metrics(
                final_output_dir, self.config
            )

            if not metrics or metrics.get('kge', -999) <= -900:
                self.logger.error("Failed to calculate final evaluation metrics")
                return None

            # Format calibration metrics
            calib_metrics = {"KGE_Calib": metrics.get('kge'), "NSE_Calib": metrics.get('nse')}

            # Evaluation period metrics (if EVALUATION_PERIOD is configured)
            eval_metrics: dict[str, float] = {}
            eval_period = self._get_config_value(lambda: self.config.domain.evaluation_period, default='', dict_key='EVALUATION_PERIOD')
            if eval_period and ',' in str(eval_period):
                eval_raw = self.worker.calculate_metrics(
                    final_output_dir, self.config, period=eval_period
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

            self.logger.info(f"Final evaluation: KGE={metrics.get('kge', 'N/A'):.4f}, "
                           f"NSE={metrics.get('nse', 'N/A'):.4f}")

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            self._save_final_evaluation_results(final_result, 'DDS')
            return final_result

        except (IOError, OSError, ValueError, RuntimeError) as e:
            self.logger.error(f"Error in final evaluation: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run MESH for final evaluation.

        Must pass proc_forcing_dir so MESH runs from forcing/MESH_input
        (where parameters were applied), not settings/MESH.
        """
        forcing_dir = self.project_forcing_dir / 'MESH_input'
        return self.worker.run_model(
            self.config,
            self.project_dir / 'settings' / 'MESH',
            output_dir,
            proc_forcing_dir=str(forcing_dir)
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to MESH input file (similar to file manager)."""
        mesh_input = self._get_config_value(lambda: self.config.model.mesh.input_file, dict_key='SETTINGS_MESH_INPUT')
        if mesh_input == 'default':
            mesh_input = 'MESH_input_run_options.ini'
        return self.project_dir / 'settings' / 'MESH' / mesh_input

    def _setup_parallel_dirs(self) -> None:
        """
        Setup MESH-specific parallel directories following SUMMA pattern.

        Creates:
        - simulations/run_{experiment_id}/process_N/
          - settings/MESH/
          - simulations/{experiment_id}/MESH/
          - forcing/MESH_input/  (MESH-specific)
          - output/
        """
        # Use algorithm name for base_dir, consistent with other models
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()
        base_dir = self._resolve_sim_base_dir(algorithm)

        # Create process directories using base class method
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'MESH',
            self.experiment_id
        )

        # Copy MESH settings to each process directory
        source_settings = self.project_dir / 'settings' / 'MESH'
        if source_settings.exists():
            self.copy_base_settings(source_settings, self.parallel_dirs, 'MESH')

        # MESH-SPECIFIC: Copy forcing directory to each process
        # MESH reads from forcing/MESH_input, but worker might look in settings/MESH
        source_forcing = self.project_forcing_dir / 'MESH_input'
        if source_forcing.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                # Create forcing directory structure: process_N/forcing/MESH_input/
                dest_forcing = dirs['root'] / 'forcing' / 'MESH_input'
                dest_forcing.parent.mkdir(parents=True, exist_ok=True)

                if dest_forcing.exists():
                    safe_delete(dest_forcing)

                try:
                    shutil.copytree(source_forcing, dest_forcing, symlinks=True)
                except OSError:
                    # Windows without admin/Developer Mode — fall back to resolving symlinks
                    shutil.copytree(source_forcing, dest_forcing, symlinks=False)
                self.logger.debug(f"Copied MESH forcing to {dest_forcing}")

                # ALSO copy to process_N/settings/MESH because WorkerTask sets settings_dir there
                dest_settings = dirs['root'] / 'settings' / 'MESH'
                dest_settings.mkdir(parents=True, exist_ok=True)
                for f in source_forcing.glob('*.ini'):
                    shutil.copy2(f, dest_settings / f.name)
                for f in source_forcing.glob('*.txt'):
                    shutil.copy2(f, dest_settings / f.name)

                # Update parallel_dirs to include forcing path
                dirs['forcing_dir'] = dest_forcing
                dirs['settings_dir'] = dest_settings
