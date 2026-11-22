# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NextGen Model Optimizer

NGEN-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with NextGen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.registries import R

from .worker import NgenWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('NGEN')
class NgenModelOptimizer(BaseModelOptimizer):
    """
    NextGen-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization

    Example:
        optimizer = NgenModelOptimizer(config, logger)
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
        Initialize NextGen optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        # Initialize NGEN-specific paths BEFORE super().__init__() so they're available in _setup_parallel_dirs
        # Set config early so _get_config_value works before super().__init__()
        self.config = config
        data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR'))
        domain_name = self._get_config_value(lambda: self.config.domain.name, default='default', dict_key='DOMAIN_NAME')
        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='optimization', dict_key='EXPERIMENT_ID')

        project_dir = data_dir / f"domain_{domain_name}"
        self.ngen_sim_dir = project_dir / 'simulations' / experiment_id / 'NGEN'
        self.ngen_setup_dir = project_dir / 'settings' / 'NGEN'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("NgenModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'NGEN'

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run NGEN for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.ngen_setup_dir,
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to NGEN realization file (similar to file manager)."""
        ngen_realization = self._get_config_value(
            lambda: self.config.model.ngen.realization,
            dict_key='SETTINGS_NGEN_REALIZATION',
        )
        if ngen_realization == 'default':
            ngen_realization = 'realization_config.json'
        return self.ngen_setup_dir / ngen_realization

    def _setup_parallel_dirs(self) -> None:
        """Setup NGEN-specific parallel directories."""
        # Use algorithm-specific directory (matching base_model_optimizer pattern)
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()
        base_dir = self._resolve_sim_base_dir(algorithm)

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'NGEN',
            self.experiment_id
        )

        # Copy NGEN settings to each parallel directory
        # ngen_setup_dir is set to a placeholder initially, so only proceed if it's a valid Path
        if self.ngen_setup_dir is not None and self.ngen_setup_dir.exists():
            self.copy_base_settings(self.ngen_setup_dir, self.parallel_dirs, 'NGEN')


# Backward compatibility alias
NgenOptimizer = NgenModelOptimizer
