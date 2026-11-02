# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO Model Optimizer.

IGNACIO-specific optimizer inheriting from BaseModelOptimizer.
Uses spatial metrics (IoU/Dice) as objective function for fire
perimeter calibration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.registries import R

from .worker import IGNACIOWorker  # noqa: F401 - trigger worker registration


@R.optimizers.add('IGNACIO')
class IGNACIOModelOptimizer(BaseModelOptimizer):
    """
    IGNACIO-specific optimizer using the unified BaseModelOptimizer framework.

    Supports evolutionary algorithms (DDS, PSO, SCE, DE) for calibrating
    FBP parameters against observed fire perimeters. No gradient-based
    methods supported.

    Example:
        optimizer = IGNACIOModelOptimizer(config, logger)
        results_path = optimizer.run_dds()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.ignacio_input_dir = self.project_dir / 'IGNACIO_input'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager)

    def _get_model_name(self) -> str:
        return 'IGNACIO'

    def _create_parameter_manager(self):
        """Create IGNACIO parameter manager."""
        from .parameter_manager import IGNACIOParameterManager
        return IGNACIOParameterManager(
            self.config_dict, self.logger, self.ignacio_input_dir
        )

    def _check_routing_needed(self) -> bool:
        """IGNACIO does not use routing."""
        return False

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """Apply best FBP parameters for final evaluation."""
        if hasattr(self, '_worker') and self._worker is not None:
            self._worker._current_params = best_params
        return True

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run IGNACIO with best parameters for final evaluation."""
        if hasattr(self, '_worker') and self._worker is not None:
            return self._worker.run_model(
                self.config_dict,
                self.ignacio_input_dir,
                self.get_simulation_dir('IGNACIO', self._get_config_value(lambda: self.config.domain.experiment_id, default='default', dict_key='EXPERIMENT_ID')),
            )
        return False

    def _get_final_file_manager_path(self) -> Path:
        """Return path for final evaluation file manager."""
        return self.ignacio_input_dir / 'ignacio_config.yaml'

    def _setup_parallel_dirs(self) -> None:
        """Set up parallel directories for IGNACIO calibration."""
        self._resolve_sim_base_dir('calibration')
        n_processors = int(self._get_config_value(lambda: self.config.system.num_processes, default=1, dict_key='NUM_PROCESSES'))
        for i in range(n_processors):
            proc_dir = (
                self.project_dir / 'simulations' / 'calibration'
                / f'proc_{i}' / 'IGNACIO'
            )
            proc_dir.mkdir(parents=True, exist_ok=True)
