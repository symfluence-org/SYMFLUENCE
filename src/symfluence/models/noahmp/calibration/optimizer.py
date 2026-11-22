# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Noah-MP Model Optimizer.

Noah-MP-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .parameter_manager import NoahMPParameterManager  # noqa: F401 — trigger registration
from .worker import NoahMPWorker  # noqa: F401 — Import to trigger worker registration


@R.optimizers.add('NOAHMP')
class NoahMPModelOptimizer(BaseModelOptimizer):
    """
    Noah-MP-specific optimizer using the unified BaseModelOptimizer framework.

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

        self.noahmp_sim_dir = self.project_dir / 'simulations' / 'NOAHMP'
        self.noahmp_setup_dir = self.project_dir / 'settings' / 'NOAHMP'

        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

        self.logger.debug("NoahMPModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'NOAHMP'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to Noah-MP namelist.input."""
        return self.noahmp_setup_dir / 'namelist.input'

    def _create_parameter_manager(self):
        return NoahMPParameterManager(
            self.config,
            self.logger,
            self.noahmp_setup_dir,
        )

    def _check_routing_needed(self) -> bool:
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        import re

        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        nl_path = self.noahmp_setup_dir / 'namelist.input'
        if nl_path.exists():
            content = nl_path.read_text()
            new_output = output_dir / 'output.nc'
            content = re.sub(
                r"output_filename\s*=\s*\"[^\"]*\"",
                f'output_filename    = "{new_output}"',
                content,
            )
            nl_path.write_text(content)

        self.worker.apply_parameters(
            best_params, self.noahmp_setup_dir, config=self.config
        )

        return self.worker.run_model(
            self.config,
            self.noahmp_setup_dir,
            output_dir,
        )

    def _setup_parallel_dirs(self) -> None:
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()
        base_dir = self._resolve_sim_base_dir(algorithm)
        self.parallel_dirs = self.setup_parallel_processing(base_dir, 'NOAHMP', self.experiment_id)
        if self.noahmp_setup_dir.exists():
            self.copy_base_settings(self.noahmp_setup_dir, self.parallel_dirs, 'NOAHMP')
