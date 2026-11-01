#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
GR Worker

Worker implementation for GR model optimization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from symfluence.core.calibration.workers.base_worker import BaseWorker, run_worker_task
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R
from symfluence.models.gr.runner import GRRunner

logger = logging.getLogger(__name__)


@R.workers.add('GR')
class GRWorker(BaseWorker):
    """
    Parallel worker for GR model calibration.

    Handles isolated model execution during optimization, where each worker
    instance can run GR4J with a different parameter set concurrently.
    Unlike file-based models (SUMMA, FUSE), GR parameters are passed directly
    to the R runtime, so no file modification is needed in apply_parameters().

    GR4J Parameters:
        - X1: Production store capacity (mm) [100-1500]
        - X2: Groundwater exchange coefficient (mm) [-5 to 5]
        - X3: Routing store capacity (mm) [1-500]
        - X4: Unit hydrograph time base (days) [0.5-10]

    Output:
        - Simulated streamflow as NetCDF or CSV
        - Performance metrics (KGE, NSE, RMSE, MAE)

    Note:
        GR models require R and rpy2. The worker delegates to GRRunner
        which manages the R/Python interface.
    """

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters for GR model run.

        Unlike file-based models (SUMMA, FUSE), GR parameters are passed directly
        to the R runtime via rpy2, so no file modification is needed. This method
        is a no-op for GR models.

        Args:
            params: Dictionary of parameter names to values
                   (e.g., {'X1': 350.0, 'X2': 0.5, 'X3': 100.0, 'X4': 2.0})
            settings_dir: Path to model settings directory (unused for GR)
            **kwargs: Additional arguments (unused)

        Returns:
            Always returns True since no file operations are performed

        Note:
            Parameters are applied in run_model() method when calling GRRunner.
            This design allows thread-safe parallel execution without file conflicts.
        """
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Execute GR model with specified parameters.

        Creates a GRRunner instance and executes the GR4J model with parameters
        provided in kwargs. The runner handles the R/Python interface via rpy2
        and executes the airGR package functions.

        Args:
            config: Configuration dictionary containing model settings
            settings_dir: Path to settings directory for this worker instance
            output_dir: Path where model output should be saved
            **kwargs: Additional arguments including:
                     - params: Dictionary of GR parameters (X1, X2, X3, X4)

        Returns:
            True if model execution successful, False otherwise

        Raises:
            Exception: If R execution fails or output files not generated

        Note:
            - Requires R with airGR package installed
            - Parameters are passed directly to R runtime (no file modification)
            - Output directory is overridden for worker isolation
            - Each worker gets its own output directory to avoid conflicts
        """
        try:
            # Get parameters from kwargs (passed by _evaluate_once)
            # WorkerTask passes params as a direct argument to run_model in newer BaseWorker
            # but let's be safe and check both kwargs and params if it were passed explicitly
            params = kwargs.get('params')

            # If not in kwargs, it might be in task (if we're calling it from evaluate)
            # Actually, BaseWorker._evaluate_once calls run_model(task.config, task.settings_dir, task.output_dir, **task.additional_data)
            # So params is NOT passed by default in BaseWorker unless it's in additional_data.
            # We need to ensure params are passed.
            if params:
                self.logger.debug(f"Worker received params: {params}")
            else:
                self.logger.warning("Worker run_model received NO params!")

            # Create a runner instance
            # We use the config provided in the task and pass settings_dir for isolation
            runner = GRRunner(config, self.logger, settings_dir=settings_dir)

            # Override output directory to the one provided for this worker
            runner.output_dir = output_dir
            runner.output_path = output_dir

            # Execute GR
            success_path = runner.run_gr(params=params)

            return success_path is not None
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error running GR model in worker: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from GR model output.

        Uses GRStreamflowTarget to extract simulated streamflow, align with
        observations, and calculate performance metrics (KGE, NSE, RMSE, MAE).

        Args:
            output_dir: Path to directory containing GR model output
            config: Configuration dictionary with observation settings
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metric names to values, e.g.:
            {'kge': 0.75, 'nse': 0.72, 'rmse': 10.5, 'mae': 8.2}
            Returns {'kge': penalty_score} if calculation fails

        Note:
            - Automatically filters to calibration period specified in config
            - Handles unit conversions (mm/day to cms if needed)
            - Returns penalty score (-9999) if metrics cannot be calculated
            - Primary metric (KGE) used for optimization objective
        """
        try:
            from symfluence.models.gr.calibration.targets import GRStreamflowTarget

            # Resolve project directory
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
            domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"

            # Initialize target
            target = GRStreamflowTarget(config, project_dir, self.logger)

            # Calculate metrics (handles period filtering and unit conversion internally)
            metrics = target.calculate_metrics(
                output_dir,
                calibration_only=True
            )

            if metrics:
                return metrics
            else:
                self.logger.warning("GR calibration target returned empty metrics")
                return {'kge': self.penalty_score}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating GR metrics via target: {e}", exc_info=True)
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for parallel execution."""
        return _evaluate_gr_parameters_worker(task_data)


def _evaluate_gr_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI execution.

    Naming matches convention for dynamic resolution in BaseModelOptimizer.
    """
    return run_worker_task(
        GRWorker,
        task_data,
        isolate=False,
        catch_exceptions=False,
    )
