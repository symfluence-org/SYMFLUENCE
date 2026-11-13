# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC Model Optimizer

VIC-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import VICWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('VIC')
class VICModelOptimizer(BaseModelOptimizer):
    """
    VIC-specific optimizer using the unified BaseModelOptimizer framework.

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
        # Extract config values before super().__init__ (which may reference them).
        # Works for both typed SymfluenceConfig and plain dict configs.
        self.experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
        self.data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
        self.domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.vic_setup_dir = self.project_dir / 'settings' / 'VIC'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("VICModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'VIC'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to VIC global parameter file."""
        global_file = self._get_config_value(
            lambda: self.config.model.vic.global_param_file,
            dict_key='VIC_GLOBAL_PARAM_FILE',
        )
        return self.vic_setup_dir / global_file

    def _create_parameter_manager(self):
        """Create VIC parameter manager."""
        from .parameter_manager import VICParameterManager
        return VICParameterManager(
            self.config,
            self.logger,
            self.vic_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed for VIC."""
        routing_integration = self._get_config_value(
            lambda: None,
            default='none',
            dict_key='VIC_ROUTING_INTEGRATION'
        )
        global_routing = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        return (routing_integration.lower() == 'mizuroute' or
                global_routing.lower() == 'mizuroute')

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run VIC for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.vic_setup_dir)

        return self.worker.run_model(
            self.config,
            self.vic_setup_dir,
            output_dir
        )
