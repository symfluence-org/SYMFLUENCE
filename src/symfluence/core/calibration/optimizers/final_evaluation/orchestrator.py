# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Final Evaluation Orchestrator

Coordinates file manager updates, model decisions restoration,
and results persistence during the final evaluation phase.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from symfluence.core.mixins.config import resolve_config_value

from .file_manager_updater import FileManagerUpdater
from .model_decisions_updater import ModelDecisionsUpdater
from .results_saver import FinalResultsSaver

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class FinalEvaluationOrchestrator:
    """Coordinates the final evaluation workflow after optimization.

    Orchestrates file manager updates, settings restoration, parameter
    application, and results persistence.  Created lazily by the base
    optimizer and invoked from ``run_final_evaluation()``.

    Args:
        config: Typed SymfluenceConfig
        logger: Logger instance
        optimization_settings_dir: Path to optimization settings
        results_saver: FinalResultsSaver instance
    """

    def __init__(
        self,
        config: 'SymfluenceConfig',
        logger: logging.Logger,
        optimization_settings_dir: Path,
        results_saver: FinalResultsSaver,
    ):
        self._config = config
        self.logger = logger
        self.optimization_settings_dir = optimization_settings_dir
        self._results_saver = results_saver

    # Helper to access config values via ConfigMixin-style
    def _get_config_value(self, accessor, default=None, dict_key=None):
        """Access config value using accessor lambda or dict key."""
        return resolve_config_value(self._config, accessor, default, dict_key)

    # ------------------------------------------------------------------
    # File manager operations
    # ------------------------------------------------------------------

    def update_file_manager_for_final_run(self, file_manager_path: Path) -> None:
        """Update file manager to use full experiment period."""
        if not file_manager_path.exists() or not file_manager_path.is_file():
            return
        updater = FileManagerUpdater(file_manager_path, self._config, self.logger)
        updater.update_for_full_period()

    def update_file_manager_output_path(
        self, file_manager_path: Path, output_dir: Path
    ) -> None:
        """Update file manager with final evaluation output path."""
        if not file_manager_path.exists() or not file_manager_path.is_file():
            return
        updater = FileManagerUpdater(file_manager_path, self._config, self.logger)
        updater.update_output_path(output_dir)

    def restore_file_manager(self, file_manager_path: Path) -> None:
        """Restore file manager to calibration period settings."""
        if not file_manager_path.exists() or not file_manager_path.is_file():
            return
        updater = FileManagerUpdater(file_manager_path, self._config, self.logger)
        updater.restore_calibration_period()

    # ------------------------------------------------------------------
    # Model decisions
    # ------------------------------------------------------------------

    def restore_model_decisions(self) -> None:
        """Restore model decisions to optimization settings from backup."""
        updater = ModelDecisionsUpdater(self.optimization_settings_dir, self.logger)
        updater.restore_for_optimization()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @staticmethod
    def extract_period_metrics(all_metrics: Dict, prefix: str) -> Dict:
        """Extract metrics for a specific period (Calib or Eval)."""
        return FinalResultsSaver.extract_period_metrics(all_metrics, prefix)

    def log_results(
        self,
        calib_metrics: Dict[str, Any],
        eval_metrics: Dict[str, Any]
    ) -> None:
        """Log detailed final evaluation results."""
        self._results_saver.log_results(calib_metrics, eval_metrics)

    def save_results(
        self,
        final_result: Dict[str, Any],
        algorithm: str
    ) -> Optional[Path]:
        """Save final evaluation results to JSON."""
        return self._results_saver.save_results(final_result, algorithm)
