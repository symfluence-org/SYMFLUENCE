# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Model Optimizer

SUMMA-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with SUMMA.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.file_utils import copy_file
from symfluence.core.registries import R

from .parameter_manager import SUMMAParameterManager  # noqa: F401 - trigger param manager registration
from .worker import SUMMAWorker  # noqa: F401 - trigger worker registration


@R.optimizers.add('SUMMA')
class SUMMAModelOptimizer(BaseModelOptimizer):
    """
    SUMMA-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization

    Example:
        optimizer = SUMMAModelOptimizer(config, logger)
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
        Initialize SUMMA optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        # Initialize properties required by _setup_parallel_dirs (called by base init)
        self.config = config
        self._routing_needed = self._check_routing_needed()

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        # SUMMA-specific paths
        self.summa_exe_path = self._get_summa_executable_path()
        self.mizuroute_exe_path = self._get_mizuroute_executable_path()

        self.logger.debug("SUMMAModelOptimizer initialized")
        self.logger.debug(f"Routing needed: {self._routing_needed}")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'SUMMA'

    def _create_parameter_manager(self):
        """Create SUMMA parameter manager."""
        # SUMMA ParameterManager expects to find localParamInfo.txt and attributes.nc
        # These are located in settings/SUMMA, not optimization/
        summa_settings_dir = self.project_dir / 'settings' / 'SUMMA'
        return SUMMAParameterManager(
            self.config,
            self.logger,
            summa_settings_dir
        )

    def _get_summa_executable_path(self) -> Path:
        """Get path to SUMMA executable."""
        summa_install = self._get_config_value(lambda: self.config.model.summa.install_path, default='default', dict_key='SUMMA_INSTALL_PATH')
        summa_exe_name = self._get_config_value(lambda: self.config.model.summa.exe, default='summa_sundials.exe', dict_key='SUMMA_EXE')

        if summa_install == 'default':
            return self.data_dir / 'installs' / 'summa' / 'bin' / summa_exe_name
        return Path(summa_install) / summa_exe_name if Path(summa_install).is_dir() else Path(summa_install)

    def _get_mizuroute_executable_path(self) -> Path:
        """Get path to mizuRoute executable."""
        mizu_install = self._get_config_value(lambda: self.config.model.mizuroute.install_path, default='default', dict_key='MIZUROUTE_INSTALL_PATH')
        if mizu_install == 'default':
            return self.data_dir / 'installs' / 'mizuRoute' / 'route' / 'bin' / 'mizuRoute.exe'
        return Path(mizu_install)

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed based on configuration."""
        calibration_var = self._get_config_value(lambda: self.config.optimization.calibration_variable, default='streamflow', dict_key='CALIBRATION_VARIABLE')

        if calibration_var != 'streamflow':
            return False

        domain_method = self._get_config_value(lambda: self.config.domain.definition_method, default='lumped', dict_key='DOMAIN_DEFINITION_METHOD')
        routing_delineation = self._get_config_value(lambda: self.config.domain.delineation.routing, default='lumped', dict_key='ROUTING_DELINEATION')

        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True

        return False

    @property
    def needs_routing(self) -> bool:
        """Check if routing is needed for this optimization."""
        return self._routing_needed

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run SUMMA for final evaluation."""
        return self.worker.run_model(
            self.config_dict,
            self.project_dir / 'settings' / 'SUMMA',
            output_dir,
            mode='run_def',
            mizuroute_dir=output_dir / 'mizuRoute',
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to SUMMA file manager."""
        summa_fm = self._get_config_value(lambda: self.config.model.summa.filemanager, default='fileManager.txt', dict_key='SETTINGS_SUMMA_FILEMANAGER')
        if summa_fm == 'default':
            summa_fm = 'fileManager.txt'
        return self.project_dir / 'settings' / 'SUMMA' / summa_fm

    def _setup_parallel_dirs(self) -> None:
        """Setup SUMMA-specific parallel directories."""
        algorithm = self._get_config_value(lambda: self.config.optimization.algorithm, default='optimization', dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM').lower()
        base_dir = self._resolve_sim_base_dir(algorithm)

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'SUMMA',
            self.experiment_id
        )

        # Add SUMMA-specific directory aliases expected by workers
        # DirectoryManager creates generic keys (sim_dir, settings_dir), but workers expect
        # model-specific keys (summa_dir, mizuroute_dir, etc.)
        for proc_id, dirs in self.parallel_dirs.items():
            # SUMMA directories (aliases for generic names)
            dirs['summa_dir'] = dirs['sim_dir']
            dirs['summa_settings_dir'] = dirs['settings_dir']

            # mizuRoute directories (sibling to SUMMA directories)
            dirs['mizuroute_dir'] = dirs['sim_dir'].parent / 'mizuRoute'
            dirs['mizuroute_settings_dir'] = dirs['root'] / 'settings' / 'mizuRoute'

            # Create mizuRoute output directory
            dirs['mizuroute_dir'].mkdir(parents=True, exist_ok=True)

        # Copy SUMMA settings to each parallel directory
        source_settings = self.project_dir / 'settings' / 'SUMMA'
        if source_settings.exists():
            self.copy_base_settings(source_settings, self.parallel_dirs, 'SUMMA')

        # Update SUMMA file managers with process-specific paths
        self.update_file_managers(
            self.parallel_dirs,
            'SUMMA',
            self.experiment_id,
            self._get_config_value(lambda: self.config.model.summa.filemanager, default='fileManager.txt', dict_key='SETTINGS_SUMMA_FILEMANAGER')
        )

        # If routing needed, also copy and configure mizuRoute settings
        if self._routing_needed:
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['mizuroute_settings_dir']
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            copy_file(item, mizu_dest / item.name)

                # Update mizuRoute control files with process-specific paths
                self.update_mizuroute_controls(
                    self.parallel_dirs,
                    'SUMMA',
                    self.experiment_id
                )
