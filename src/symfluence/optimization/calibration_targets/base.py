#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
SYMFLUENCE Calibration Targets - Base Classes

This module provides base calibration target classes that alias the centralized
evaluators in symfluence.evaluation.evaluators. Model-specific targets have been
moved to their own files (e.g., summa_calibration_targets.py, gr_calibration_targets.py).

The classes here provide backward compatibility by aliasing evaluators as "Targets":
- CalibrationTarget (alias for ModelEvaluator)
- StreamflowTarget (alias for StreamflowEvaluator)
- ETTarget (alias for ETEvaluator)
- SnowTarget (alias for SnowEvaluator)
- etc.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.core.modeling.evaluation_registry import EvaluationRegistry
from symfluence.evaluation.evaluators import BasinETEvaluator as BasinETTarget
from symfluence.evaluation.evaluators import BasinSCAEvaluator as BasinSCATarget
from symfluence.evaluation.evaluators import BasinSWEEvaluator as BasinSWETarget
from symfluence.evaluation.evaluators import ETEvaluator as ETTarget
from symfluence.evaluation.evaluators import GlacierMassBalanceEvaluator as GlacierMBTarget
from symfluence.evaluation.evaluators import GroundwaterEvaluator as GroundwaterTarget
from symfluence.evaluation.evaluators import ModelEvaluator as CalibrationTarget
from symfluence.evaluation.evaluators import SnowEvaluator as SnowTarget
from symfluence.evaluation.evaluators import SoilMoistureEvaluator as SoilMoistureTarget
from symfluence.evaluation.evaluators import StreamflowEvaluator as StreamflowTarget
from symfluence.evaluation.evaluators import TWSEvaluator as TWSTarget


class MultivariateTarget(CalibrationTarget):
    """
    Multivariate calibration target that combines multiple variables.
    Delegates scoring to the AnalysisManager and MultivariateObjective.
    """

    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        from symfluence.core.config.coercion import coerce_config
        from symfluence.core.registries import R
        from symfluence.evaluation.analysis_manager import AnalysisManager

        # Convert config dict to SymfluenceConfig if needed (required by AnalysisManager)
        config = coerce_config(config, warn=False)
        self._config = config  # Update internal config as well

        self.analysis_manager = AnalysisManager(config, logger)
        obj_cls = R.objectives.get('MULTIVARIATE')
        self.objective_handler = obj_cls(config, logger) if obj_cls else None

        # Get requested variables from weights/metrics config
        try:
            if hasattr(config, '__getitem__'):  # Dict-like interface
                weights = config.get('OBJECTIVE_WEIGHTS', config.get('objective_weights', {'STREAMFLOW': 1.0}))
            else:
                # Try attribute access for SymfluenceConfig
                weights = getattr(config, 'OBJECTIVE_WEIGHTS', getattr(config, 'objective_weights', {'STREAMFLOW': 1.0}))
        except (KeyError, ValueError, AttributeError):
            weights = {'STREAMFLOW': 1.0}

        self.variables = list(weights.keys()) if weights else ['STREAMFLOW']
        self.logger.debug(f"MultivariateTarget initialized with variables: {self.variables}")

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get all NetCDF simulation output files for multivariate evaluation.

        Returns all .nc files since multivariate targets may need both daily
        and hourly outputs for different variable evaluations.

        Args:
            sim_dir: Path to simulation output directory.

        Returns:
            List of all NetCDF file paths in the simulation directory.
        """
        return list(sim_dir.glob("*.nc"))

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Not directly used by MultivariateTarget as it delegates to component evaluators."""
        raise NotImplementedError("MultivariateTarget delegates extraction to individual evaluators.")

    def get_observed_data_path(self) -> Path:
        """Not directly used by MultivariateTarget."""
        return self.project_observations_dir

    def _load_observed_data(self) -> Optional[pd.Series]:
        """MultivariateTarget does not load a single observation series."""
        return None

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Not directly used by MultivariateTarget."""
        return None

    def needs_routing(self) -> bool:
        """Multivariate target might need routing if streamflow is one of the components."""
        return 'STREAMFLOW' in [v.upper() for v in self.variables]

    def evaluate_to_scalar(self, sim_dir: Path, **kwargs) -> float:
        """
        Evaluate multiple variables and return a composite scalar score.
        """
        # 1. Extract simulated data for all requested variables
        sim_results = {}
        for var in self.variables:
            evaluator = EvaluationRegistry.get_evaluator(
                var, self.config, self.logger, self.project_dir, target=var
            )
            if evaluator:
                sim_files = evaluator.get_simulation_files(sim_dir)
                if sim_files:
                    sim_results[var] = evaluator.extract_simulated_data(sim_files)

        # 2. Run multivariate evaluation
        eval_results = self.analysis_manager.run_multivariate_evaluation(sim_results)

        # 3. Calculate scalar objective
        return self.objective_handler.calculate(eval_results)

    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None,
                         mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """
        Calculate multivariate metrics.
        Overrides base implementation to delegate to AnalysisManager.
        """
        from pathlib import Path
        sim_dir = Path(sim) if isinstance(sim, (str, Path)) else None
        if not sim_dir:
            self.logger.error("MultivariateTarget requires a simulation directory path")
            return None

        # 1. Extract simulated data for all requested variables
        sim_results = {}
        for var in self.variables:
            evaluator = EvaluationRegistry.get_evaluator(
                var, self.config, self.logger, self.project_dir, target=var
            )
            if evaluator:
                # Pass mizuroute_dir if needed
                files_dir = mizuroute_dir if evaluator.needs_routing() and mizuroute_dir else sim_dir
                sim_files = evaluator.get_simulation_files(files_dir)

                if sim_files:
                    try:
                        sim_results[var] = evaluator.extract_simulated_data(sim_files)
                    except (KeyError, ValueError, AttributeError) as e:
                        self.logger.warning(f"Failed to extract data for {var}: {e}")

        # 2. Run multivariate evaluation (returns dict of metrics per variable)
        eval_results = self.analysis_manager.run_multivariate_evaluation(sim_results)

        # Ensure no None values in results
        for var in eval_results:
            if eval_results[var] is None:
                eval_results[var] = {}

        # 3. Calculate scalar objective (KGE, etc for the composite)
        scalar_score = self.objective_handler.calculate(eval_results)

        # Flatten results into a single dict for return
        metrics = {'KGE': scalar_score, 'Calib_KGE': scalar_score, 'Obj': scalar_score}

        for var, var_metrics in eval_results.items():
            for metric, value in var_metrics.items():
                metrics[f"{var}_{metric}"] = value
                metrics[f"Calib_{var}_{metric}"] = value

        return metrics


# Re-export for backward compatibility
__all__ = [
    'CalibrationTarget',
    'ETTarget',
    'StreamflowTarget',
    'SoilMoistureTarget',
    'SnowTarget',
    'GroundwaterTarget',
    'TWSTarget',
    'MultivariateTarget',
    'GlacierMBTarget',
    'BasinSWETarget',
    'BasinSCATarget',
    'BasinETTarget',
]
