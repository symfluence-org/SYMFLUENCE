# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM Model Optimizer

Optimizer for PIHM calibration. Sets up parallel directories,
manages parameter perturbation, and delegates execution to PIHMWorker.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.file_utils import copy_file
from symfluence.core.registries import R

from .targets import PIHMStreamflowTarget  # noqa: F401 - triggers target registration
from .worker import PIHMWorker  # noqa: F401 - triggers worker registration


@R.optimizers.add('PIHM')
class PIHMModelOptimizer(BaseModelOptimizer):
    """Optimizer for PIHM calibration.

    Sets up parallel directories with PIHM settings and manages
    DDS-based parameter optimization.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None,
    ):
        self.config = config
        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

    def _get_model_name(self) -> str:
        return 'PIHM'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to PIHM settings for final evaluation."""
        return self.project_dir / 'settings' / 'PIHM'

    def _create_parameter_manager(self):
        """Create PIHM parameter manager."""
        from .parameter_manager import PIHMParameterManager
        settings_dir = self.project_dir / 'settings' / 'PIHM'
        return PIHMParameterManager(
            self.config, self.logger, settings_dir,
        )

    def _check_routing_needed(self) -> bool:
        """No external routing needed; PIHM provides river flux directly."""
        return False

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel directories with PIHM settings."""
        # Ensure PIHM settings exist
        source_pihm = self.project_dir / 'settings' / 'PIHM'
        if not source_pihm.exists() or not any(source_pihm.iterdir()):
            self.logger.info("PIHM settings not found, generating from config...")
            from symfluence.models.pihm.preprocessor import PIHMPreProcessor
            preprocessor = PIHMPreProcessor(self.config, self.logger)
            if isinstance(self.config, dict) and not preprocessor.config_dict:
                preprocessor.config_dict = self.config
            preprocessor.project_dir = self.project_dir
            preprocessor.run_preprocessing()

        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        base_dir = self._resolve_sim_base_dir(algorithm)

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir, 'PIHM', self.experiment_id,
        )

        # Copy PIHM settings to parallel directories
        if source_pihm.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest = dirs['settings_dir']
                dest.mkdir(parents=True, exist_ok=True)
                for item in source_pihm.iterdir():
                    if item.is_file():
                        copy_file(item, dest / item.name)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run PIHM for final evaluation."""
        settings_dir = self.project_dir / 'settings' / 'PIHM'
        return self.worker.run_model(
            self.config, settings_dir, output_dir,
        )

    def _update_file_manager_for_final_run(self) -> None:
        pass

    def _restore_file_manager_for_optimization(self) -> None:
        pass
