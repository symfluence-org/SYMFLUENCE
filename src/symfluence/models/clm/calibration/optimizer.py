# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Model Optimizer

CLM-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import CLMWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('CLM')
class CLMModelOptimizer(BaseModelOptimizer):
    """
    CLM-specific optimizer using the unified BaseModelOptimizer framework.

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
        reporting_manager: Optional[Any] = None,
    ):
        self.experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
        self.data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, None, 'SYMFLUENCE_DATA_DIR'))
        self.domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.clm_setup_dir = self.project_dir / 'settings' / 'CLM'

        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

        self.logger.debug("CLMModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'CLM'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to CLM user_nl_clm namelist."""
        return self.clm_setup_dir / 'user_nl_clm'

    def _create_parameter_manager(self):
        """Create CLM parameter manager."""
        from .parameter_manager import CLMParameterManager
        return CLMParameterManager(
            self.config,
            self.logger,
            self.clm_setup_dir,
        )

    def _check_routing_needed(self) -> bool:
        """CLM does not use external routing."""
        return False

    def _apply_best_parameters_for_final(self, best_params) -> bool:
        """Skip — _run_model_for_final_evaluation handles param application."""
        return True

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run CLM for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        # Use a separate settings dir for final evaluation to avoid
        # SameFileError when source settings/CLM == settings_dir.
        import shutil
        final_settings = output_dir / 'settings' / 'CLM'
        final_settings.mkdir(parents=True, exist_ok=True)

        # Copy base settings
        for f in self.clm_setup_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, final_settings / f.name)

        self.worker.apply_parameters(
            best_params, final_settings, config=self.config
        )

        return self.worker.run_model(
            self.config,
            final_settings,
            output_dir,
        )
