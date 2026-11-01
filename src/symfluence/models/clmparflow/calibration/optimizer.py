# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Model Optimizer

CLMParFlow-specific optimizer inheriting from BaseModelOptimizer.
Calibrates van Genuchten parameters, saturated hydraulic conductivity,
Manning's roughness, Snow-17 snow parameters, and routing parameters
via DDS or other algorithms.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .targets import CLMParFlowStreamflowTarget  # noqa: F401 - triggers target registration
from .worker import CLMParFlowWorker  # noqa: F401 - triggers worker registration


@R.optimizers.add('CLMPARFLOW')
class CLMParFlowModelOptimizer(BaseModelOptimizer):
    """
    CLMParFlow-specific optimizer using the unified BaseModelOptimizer framework.

    Calibrates integrated variably-saturated subsurface + overland flow parameters
    (with CLM land surface) by iteratively updating the .pfidb database file and
    re-running CLMParFlow.
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

        self.clmparflow_setup_dir = self.project_dir / 'settings' / 'CLMPARFLOW'

        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )

        self.logger.debug("CLMParFlowModelOptimizer initialized")

    def _get_model_name(self) -> str:
        return 'CLMPARFLOW'

    def _get_final_file_manager_path(self) -> Path:
        """CLMParFlow uses .pfidb files, not a file manager."""
        pfidb_files = list(self.clmparflow_setup_dir.glob('*.pfidb'))
        if pfidb_files:
            return pfidb_files[0]
        return self.clmparflow_setup_dir / 'clmparflow_run.pfidb'

    def _create_parameter_manager(self):
        """Create CLMParFlow parameter manager."""
        from .parameter_manager import CLMParFlowParameterManager
        return CLMParFlowParameterManager(
            self.config,
            self.logger,
            self.clmparflow_setup_dir,
        )

    def _check_routing_needed(self) -> bool:
        """CLMParFlow handles its own overland flow routing; no external routing needed."""
        return False

    def _setup_parallel_dirs(self) -> None:
        """Setup CLMParFlow-specific parallel directories."""
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        base_dir = self._resolve_sim_base_dir(algorithm)
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir, 'CLMPARFLOW', self.experiment_id
        )

        # Copy CLMParFlow settings (.pfidb, runname.txt, CLM files) to each parallel dir
        if self.clmparflow_setup_dir.exists():
            self.copy_base_settings(
                self.clmparflow_setup_dir, self.parallel_dirs, 'CLMPARFLOW'
            )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run CLMParFlow for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        return self.worker.run_model(
            self.config,
            self.clmparflow_setup_dir,
            output_dir,
            params=best_params,
        )

    def _update_file_manager_for_final_run(self) -> None:
        """CLMParFlow doesn't use a file manager."""
        pass

    def _restore_file_manager_for_optimization(self) -> None:
        """CLMParFlow doesn't use a file manager."""
        pass
