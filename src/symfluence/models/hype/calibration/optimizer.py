# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE Model Optimizer

HYPE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with HYPE.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.file_utils import copy_file
from symfluence.core.registries import R

from .worker import HYPEWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('HYPE')
class HYPEModelOptimizer(BaseModelOptimizer):
    """
    HYPE-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = HYPEModelOptimizer(config, logger)
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
        Initialize HYPE optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("HYPEModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'HYPE'

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run HYPE for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.project_dir / 'settings' / 'HYPE',
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to HYPE info file (similar to file manager)."""
        hype_info = self._get_config_value(lambda: self.config.model.hype.info_file, default='info.txt', dict_key='SETTINGS_HYPE_INFO')
        if hype_info == 'default':
            hype_info = 'info.txt'
        return self.project_dir / 'settings' / 'HYPE' / hype_info

    def _setup_parallel_dirs(self) -> None:
        """Setup HYPE-specific parallel directories."""
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()
        base_dir = self._resolve_sim_base_dir(algorithm)
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir.absolute(),
            'HYPE',
            self.experiment_id
        )

        # Copy HYPE settings to each parallel directory
        source_settings = self.project_dir / 'settings' / 'HYPE'
        if source_settings.exists():
            self.copy_base_settings(source_settings.absolute(), self.parallel_dirs, 'HYPE')

        # Update HYPE info files with process-specific paths
        self.update_file_managers(
            self.parallel_dirs,
            'HYPE',
            self.experiment_id,
            self._get_config_value(lambda: self.config.model.hype.info_file, default='info.txt', dict_key='SETTINGS_HYPE_INFO')
        )

        # If routing needed, also copy and configure mizuRoute settings
        routing_model = self._get_config_value(lambda: self.config.model.routing_model, default='none', dict_key='ROUTING_MODEL')
        if routing_model == 'mizuRoute':
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['root'] / 'settings' / 'mizuRoute'
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            copy_file(item, mizu_dest / item.name)

                # Update mizuRoute control files with process-specific paths
                self.update_mizuroute_controls(
                    self.parallel_dirs,
                    'HYPE',
                    self.experiment_id
                )

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """
        Update HYPE info.txt with final evaluation output directory.

        HYPE uses 'resultdir' in info.txt instead of SUMMA's 'outputPath'.
        """
        file_manager_path = self._get_final_file_manager_path()

        if not file_manager_path.exists():
            self.logger.warning(f"HYPE info file not found: {file_manager_path}")
            return

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # HYPE requires trailing slash on resultdir
        output_path_str = str(output_dir).rstrip('/') + '/'

        try:
            with open(file_manager_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            found_resultdir = False

            for line in lines:
                # Skip comments
                if line.strip().startswith('!!'):
                    updated_lines.append(line)
                    continue

                # Update resultdir for HYPE
                if line.strip().startswith('resultdir') and not line.strip().startswith('!!'):
                    updated_lines.append(f"resultdir\t{output_path_str}\n")
                    found_resultdir = True
                    self.logger.debug(f"Updated HYPE resultdir to: {output_path_str}")
                else:
                    updated_lines.append(line)

            if not found_resultdir:
                self.logger.warning("'resultdir' not found in HYPE info.txt - output may go to wrong location")

            with open(file_manager_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.info(f"Updated HYPE info.txt output path to: {output_dir}")

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to update HYPE info.txt output path: {e}", exc_info=True)

    def _update_file_manager_for_final_run(self) -> None:
        """
        Update HYPE info.txt dates for final evaluation run.

        HYPE uses 'bdate', 'cdate', 'edate' instead of SUMMA's 'simStartTime'/'simEndTime'.
        """
        file_manager_path = self._get_final_file_manager_path()

        if not file_manager_path.exists():
            self.logger.warning(f"HYPE info file not found: {file_manager_path}")
            return

        # Get full experiment period from config (same as base class)
        sim_start = self._get_config_value(lambda: self.config.domain.time_start)
        sim_end = self._get_config_value(lambda: self.config.domain.time_end)

        if not sim_start or not sim_end:
            self.logger.warning("Could not get simulation dates from config")
            return

        # Convert dates to HYPE format (YYYY-MM-DD)
        try:
            from datetime import datetime

            # Parse various date formats
            start_str = str(sim_start).split()[0]
            end_str = str(sim_end).split()[0]

            # Try to parse and reformat
            parsed = False
            for fmt in ['%Y-%m-%d', '%Y/%m/%d']:
                try:
                    start_dt = datetime.strptime(start_str, fmt)
                    end_dt = datetime.strptime(end_str, fmt)
                    hype_start = start_dt.strftime('%Y-%m-%d')
                    hype_end = end_dt.strftime('%Y-%m-%d')
                    parsed = True
                    break
                except ValueError:
                    continue

            if not parsed:
                # If parsing fails, use the strings as-is (they might already be in correct format)
                hype_start = start_str
                hype_end = end_str
                self.logger.debug(f"Using date strings as-is: {hype_start} to {hype_end}")

            with open(file_manager_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                # Skip comments
                if line.strip().startswith('!!'):
                    updated_lines.append(line)
                    continue

                # Update bdate (begin date)
                if line.strip().startswith('bdate') and not line.strip().startswith('!!'):
                    updated_lines.append(f"bdate\t{hype_start}\n")
                    self.logger.debug(f"Updated HYPE bdate to: {hype_start}")
                # Update edate (end date)
                elif line.strip().startswith('edate') and not line.strip().startswith('!!'):
                    updated_lines.append(f"edate\t{hype_end}\n")
                    self.logger.debug(f"Updated HYPE edate to: {hype_end}")
                else:
                    updated_lines.append(line)

            with open(file_manager_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.info(f"Updated HYPE info.txt dates: {hype_start} to {hype_end}")

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to update HYPE info.txt dates: {e}", exc_info=True)
