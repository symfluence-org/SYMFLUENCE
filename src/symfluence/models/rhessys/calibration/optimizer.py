# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Model Optimizer

RHESSys-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with RHESSys.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.registries import R

from .worker import RHESSysWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('RHESSys')
class RHESSysModelOptimizer(BaseModelOptimizer):
    """
    RHESSys-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = RHESSysModelOptimizer(config, logger)
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
        Initialize RHESSys optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("RHESSysModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'RHESSys'

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run RHESSys for final evaluation."""
        rhessys_settings_dir = self.project_dir / 'settings' / 'RHESSys'
        return self.worker.run_model(
            self.config,
            rhessys_settings_dir,
            output_dir
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to RHESSys world header file."""
        domain_name = self._get_config_value(lambda: self.config.domain.name, default='')
        return self.project_dir / 'settings' / 'RHESSys' / 'worldfiles' / f'{domain_name}.world.hdr'

    def _setup_parallel_dirs(self) -> None:
        """Setup RHESSys-specific parallel directories."""
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()
        base_dir = self._resolve_sim_base_dir(algorithm)

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir.absolute(),
            'RHESSys',
            self.experiment_id
        )

        # Copy RHESSys definition files to each parallel directory
        source_defs = self.project_dir / 'settings' / 'RHESSys' / 'defs'
        if not source_defs.exists():
            raise FileNotFoundError(
                f"RHESSys definition files not found at {source_defs}. "
                f"Ensure RHESSys model inputs are generated for this domain using 'configure_model'."
            )

        self._copy_defs_to_parallel_dirs(source_defs)

        # Set default_sim_dir to first proc's sim directory
        if self.parallel_dirs:
            first_proc = min(self.parallel_dirs.keys())
            self.default_sim_dir = self.parallel_dirs[first_proc]['sim_dir']

    def _copy_defs_to_parallel_dirs(self, source_defs: Path) -> None:
        """
        Copy RHESSys definition files to parallel directories.

        Each parallel worker needs its own copy of .def files for parameter modification.

        Args:
            source_defs: Path to source definition files
        """
        for proc_id, dirs in self.parallel_dirs.items():
            # Create defs directory in settings_dir
            proc_defs = dirs['settings_dir'] / 'defs'
            proc_defs.mkdir(parents=True, exist_ok=True)

            # Copy all .def files
            for def_file in source_defs.glob('*.def'):
                shutil.copy2(def_file, proc_defs / def_file.name)

            self.logger.debug(f"Copied RHESSys defs to proc_{proc_id}: {proc_defs}")

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        RHESSys-specific override: uses settings/RHESSys directory for defs.
        """
        try:
            # RHESSys defs are in settings/RHESSys/defs
            # The worker's apply_parameters method expects settings_dir TO CONTAIN a 'defs' subdirectory.
            rhessys_settings_dir = self.project_dir / 'settings' / 'RHESSys'
            return self.worker.apply_parameters(
                best_params,
                rhessys_settings_dir,  # This directory has the defs/ subdirectory
                config=self.config
            )
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying parameters for final evaluation: {e}", exc_info=True)
            return False
