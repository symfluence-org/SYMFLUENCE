# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Model Optimizer

LSTM-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with LSTM.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value


# NOTE: LSTM is a self-training model — its weights are learned during the run
# step (run_lstm), not via an external DDS/PSO parameter search. It therefore
# registers no optimizer/worker; OptimizationManager skips it (see
# _SELF_TRAINING_MODELS). This class is retained for potential gradient-based
# (ADAM/LBFGS) experimentation but is intentionally not registered.
class LSTMModelOptimizer(BaseModelOptimizer):
    """
    LSTM-specific optimizer using the unified BaseModelOptimizer framework.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.config = config
        # Distributed mode detected from DOMAIN_DEFINITION_METHOD
        self._routing_needed = resolve_config_value(config, lambda: config.domain.definition_method, 'lumped', 'DOMAIN_DEFINITION_METHOD') == 'delineate'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self._routing_needed = self.worker.needs_routing(self.config, self.optimization_settings_dir)
        self.logger.debug(f"LSTMModelOptimizer initialized (routing needed: {self._routing_needed})")

    def _get_model_name(self) -> str:
        return 'LSTM'

    def _create_parameter_manager(self):
        """
        LSTM uses standard ParameterManager.
        Parameters could be LSTM hyperparameters or multipliers.
        """
        from symfluence.models.gnn.calibration.parameter_manager import MLParameterManager
        lstm_settings_dir = self.project_dir / 'settings' / 'LSTM'
        if not lstm_settings_dir.exists():
            lstm_settings_dir.mkdir(parents=True, exist_ok=True)

        return MLParameterManager(
            self.config,
            self.logger,
            lstm_settings_dir,
            params_key='LSTM_PARAMS_TO_CALIBRATE',
            bounds_key='LSTM_PARAMETER_BOUNDS'
        )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run LSTM for final evaluation."""
        from symfluence.models.lstm import LSTMRunner
        runner = LSTMRunner(self.config, self.logger)
        runner.run_lstm()
        return True

    def _get_final_file_manager_path(self) -> Path:
        # LSTM doesn't use a file manager text file like SUMMA
        return self.project_dir / 'settings' / 'LSTM' / 'dummy_fm.txt'

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel directories for LSTM."""
        algorithm = self._get_config_value(lambda: self.config.optimization.algorithm, default='optimization', dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM').lower()
        base_dir = self._resolve_sim_base_dir(algorithm)

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'LSTM',
            self.experiment_id
        )
