# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow Model Optimizer."""
from __future__ import annotations

from pathlib import Path

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.registries import R

from .worker import WflowWorker  # noqa: F401


@R.optimizers.add('WFLOW')
class WflowModelOptimizer(BaseModelOptimizer):
    """Wflow-specific optimizer using the unified BaseModelOptimizer framework."""

    def __init__(self, config, logger, optimization_settings_dir=None, reporting_manager=None):
        if isinstance(config, dict):
            self.experiment_id = config.get('EXPERIMENT_ID')
            self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            self.domain_name = config.get('DOMAIN_NAME')
        else:
            self.experiment_id = config.domain.experiment_id
            self.data_dir = Path(config.system.data_dir)
            self.domain_name = config.domain.name
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.wflow_setup_dir = self.project_dir / 'settings' / 'WFLOW'
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

    def _get_model_name(self) -> str:
        return 'WFLOW'

    def _get_final_file_manager_path(self) -> Path:
        config_file = self._get_config_value(
            lambda: self.config.model.wflow.config_file,
            default='wflow_sbm.toml', dict_key='WFLOW_CONFIG_FILE'
        )
        return self.wflow_setup_dir / config_file

    def _create_parameter_manager(self):
        from .parameter_manager import WflowParameterManager
        return WflowParameterManager(self.config, self.logger, self.wflow_setup_dir)

    def _check_routing_needed(self) -> bool:
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        # Base optimizer already applied best params via _apply_best_parameters_for_final;
        # worker.run_model expects a flat dict, not a SymfluenceConfig object.
        return self.worker.run_model(self.config_dict, self.wflow_setup_dir, output_dir, final_evaluation=True)
