#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
SUMMA Structure Analyzer

This module implements the Structure Ensemble Analysis for the SUMMA model,
often coupled with mizuRoute for routing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.evaluation.metrics import kge, kge_prime, mae, nse, rmse
from symfluence.evaluation.structure_ensemble import BaseStructureEnsembleAnalyzer

# Import MizuRouteRunner at module level for test mocking,
# but still use lazy loading in the property to avoid circular imports
from symfluence.models.mizuroute.runner import MizuRouteRunner
from symfluence.models.summa.runner import SummaRunner


class SummaStructureAnalyzer(BaseStructureEnsembleAnalyzer):
    """
    Structure Ensemble Analyzer for SUMMA and mizuRoute.

    Coordinates multiple runs of SUMMA with different model decisions,
    optionally followed by mizuRoute routing, and evaluates the performance
    of each structural configuration.

    Note: MizuRoute runner is lazily loaded only when routing is needed,
    preventing unnecessary dependencies and circular import risks.
    """

    def __init__(self, config: Any, logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the SUMMA structure analyzer.
        """
        super().__init__(config, logger, reporting_manager)

        # Initialize SUMMA runner
        self.summa_runner = SummaRunner(config, logger)

        # MizuRoute runner lazily loaded via property (see below)
        self._mizuroute_runner = None
        self._routing_needed = None

        self.model_decisions_path = self.project_dir / "settings" / "SUMMA" / "modelDecisions.txt"

    def _needs_routing(self) -> bool:
        """
        Determine if routing (mizuRoute) is needed for this analysis.

        Uses RoutingDecider to check configuration and spatial setup.
        Result is cached for performance.

        Returns:
            True if routing is needed, False otherwise
        """
        if self._routing_needed is None:
            from symfluence.models.utilities.routing_decider import RoutingDecider
            decider = RoutingDecider()
            settings_dir = self.project_dir / "settings" / "SUMMA"
            self._routing_needed = decider.needs_routing(
                self._config_dict_cache,
                'SUMMA',
                settings_dir
            )

            if self._routing_needed:
                self.logger.info("Routing (mizuRoute) is enabled for SUMMA structure analysis")
            else:
                self.logger.info("Routing (mizuRoute) is disabled for SUMMA structure analysis")

        assert self._routing_needed is not None
        return self._routing_needed

    @property
    def mizuroute_runner(self):
        """
        Lazy-load MizuRoute runner only when routing is needed.

        This prevents circular dependencies and unnecessary imports when
        routing is disabled via configuration.

        Returns:
            MizuRouteRunner instance if routing is needed, None otherwise

        Raises:
            RuntimeError: If routing is not configured but mizuroute_runner is accessed
        """
        if not self._needs_routing():
            raise RuntimeError(
                "MizuRoute runner requested but routing is not configured. "
                "Check ROUTING_MODEL, DOMAIN_DEFINITION_METHOD, and ROUTING_DELINEATION settings."
            )

        if self._mizuroute_runner is None:
            # Lazy instantiation (import is at module level for test mocking)
            self._mizuroute_runner = MizuRouteRunner(self.config, self.logger)
            self.logger.debug("MizuRoute runner initialized (lazy loading)")

        return self._mizuroute_runner

    def _initialize_decision_options(self) -> Dict[str, List[str]]:
        """Initialize SUMMA decision options from configuration."""
        return self._resolve(
            lambda: self.config.model.summa.decision_options,
            'SUMMA_DECISION_OPTIONS', {}
        )

    def _initialize_output_folder(self) -> Path:
        """Initialize the output folder for SUMMA analysis results."""
        return self.project_dir / "reporting" / "decision_analysis"

    def _initialize_master_file(self) -> Path:
        """Initialize the master results file path for SUMMA."""
        return self.project_dir / 'optimization' / f"{self.experiment_id}_model_decisions_comparison.csv"

    def update_model_decisions(self, combination: Tuple[str, ...]):
        """
        Update the SUMMA modelDecisions.txt file with a new combination.

        Args:
            combination (Tuple[str, ...]): Tuple of decision values to use.
        """
        if not self.model_decisions_path.exists():
            self.logger.error(f"SUMMA model decisions file not found: {self.model_decisions_path}")
            raise FileNotFoundError(f"Could not find {self.model_decisions_path}")

        decision_keys = list(self.decision_options.keys())
        with open(self.model_decisions_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        option_map = dict(zip(decision_keys, combination))

        for i, line in enumerate(lines):
            for option, value in option_map.items():
                if line.strip().startswith(option):
                    # Maintain format: DecisionName  Value  ! Comment
                    lines[i] = f"{option.ljust(30)} {value.ljust(15)} ! {line.split('!')[-1].strip()}\n"

        with open(self.model_decisions_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def run_model(self):
        """
        Execute SUMMA, optionally followed by mizuRoute routing.

        Routing is only executed if configured via ROUTING_MODEL or spatial settings.
        """
        self.logger.info("Executing SUMMA model run")
        self.summa_runner.run_summa()

        if self._needs_routing():
            self.logger.info("Executing mizuRoute routing")
            self.mizuroute_runner.run_mizuroute()
        else:
            self.logger.info("Skipping mizuRoute routing (not configured)")

    def _get_optimization_target(self) -> str:
        return (self._resolve(
            lambda: self.config.optimization.target,
            'OPTIMIZATION_TARGET', 'streamflow'
        ) or 'streamflow').lower()

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the current model run.

        Dispatches to target-specific logic: streamflow uses mizuRoute routed
        runoff; non-streamflow targets (SWE, SCA, ET, …) delegate to the
        matching evaluator from the EvaluationRegistry.

        Returns:
            Dict: Dictionary containing KGE, NSE, MAE, and RMSE.
        """
        opt_target = self._get_optimization_target()
        if opt_target not in ('streamflow', 'flow', 'discharge'):
            return self._calculate_evaluator_metrics(opt_target)
        return self._calculate_streamflow_metrics()

    def _calculate_evaluator_metrics(self, opt_target: str) -> Dict[str, float]:
        """Delegate metric calculation to the appropriate evaluator."""
        from symfluence.evaluation.registry import EvaluationRegistry

        target_to_family = {
            'swe': 'SNOW', 'sca': 'SNOW', 'snow_depth': 'SNOW', 'snow': 'SNOW',
            'et': 'ET', 'latent_heat': 'ET', 'evapotranspiration': 'ET',
            'sm_point': 'SOIL_MOISTURE', 'sm_smap': 'SOIL_MOISTURE',
            'sm_esa': 'SOIL_MOISTURE', 'sm_ismn': 'SOIL_MOISTURE',
            'soil_moisture': 'SOIL_MOISTURE', 'sm': 'SOIL_MOISTURE',
        }
        family = target_to_family.get(opt_target, opt_target.upper())

        evaluator = EvaluationRegistry.get_evaluator(
            family, self.config, self.logger, self.project_dir,
            target=opt_target,
        )
        if evaluator is None:
            self.logger.error(
                "No evaluator registered for '%s' (family '%s')", opt_target, family,
            )
            return {'kge': np.nan, 'kgep': np.nan, 'nse': np.nan, 'mae': np.nan, 'rmse': np.nan}

        sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'SUMMA'
        result = evaluator.calculate_metrics(
            sim=sim_dir, calibration_only=False,
        )
        if not result:
            return {'kge': np.nan, 'kgep': np.nan, 'nse': np.nan, 'mae': np.nan, 'rmse': np.nan}

        return {
            'kge': float(result.get('KGE', result.get('Calib_KGE', np.nan))),
            'kgep': float(result.get('KGEp', result.get('Calib_KGEp', np.nan))),
            'nse': float(result.get('NSE', result.get('Calib_NSE', np.nan))),
            'mae': float(result.get('MAE', result.get('Calib_MAE', np.nan))),
            'rmse': float(result.get('RMSE', result.get('Calib_RMSE', np.nan))),
        }

    def _calculate_streamflow_metrics(self) -> Dict[str, float]:
        """Original streamflow-specific metric calculation using mizuRoute output."""
        # Load observations
        obs_file_path = self._resolve(
            lambda: self.config.paths.observations_path,
            'OBSERVATIONS_PATH', 'default'
        )
        if obs_file_path == 'default' or not obs_file_path:
            obs_file_path = self.project_observations_dir / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
        else:
            obs_file_path = Path(obs_file_path)

        if not obs_file_path.exists():
            self.logger.error(f"Observation file not found: {obs_file_path}")
            raise FileNotFoundError(f"Missing observation file: {obs_file_path}")

        # Load simulation results (mizuRoute output)
        sim_reach_ID = self._resolve(
            lambda: self.config.evaluation.sim_reach_id,
            'SIM_REACH_ID', None
        )
        sim_path_config = self._resolve(
            lambda: self.config.paths.simulations_path,
            'SIMULATIONS_PATH', 'default'
        )

        if sim_path_config == 'default' or not sim_path_config:
            start_year = (self.time_start or '1990').split('-')[0]
            sim_file_path = (
                self.project_dir / 'simulations' / self.experiment_id /
                'mizuRoute' / f"{self.experiment_id}.h.{start_year}-01-01-03600.nc"
            )
        else:
            sim_file_path = Path(sim_path_config)

        if not sim_file_path.exists():
            self.logger.error(f"Simulation output file not found: {sim_file_path}")
            raise FileNotFoundError(f"Missing simulation output: {sim_file_path}")

        # Process observations
        dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
        if 'discharge_cms' in dfObs.columns:
            obs_series = dfObs['discharge_cms'].resample('h').mean()
        else:
            data_col = [c for c in dfObs.columns if c.lower() not in ['datetime', 'date']][0]
            obs_series = dfObs[data_col].resample('h').mean()

        # Process simulations
        with xr.open_dataset(sim_file_path, engine='netcdf4') as ds:
            if 'reachID' in ds.variables:
                segment_index = ds['reachID'].values == int(sim_reach_ID)
                ds_sel = ds.sel(seg=segment_index)
            else:
                ds_sel = ds.isel(seg=0)

            var_name = 'IRFroutedRunoff' if 'IRFroutedRunoff' in ds_sel.variables else 'KWTroutedRunoff'
            if var_name not in ds_sel.variables:
                var_name = [v for v in ds_sel.variables if 'routedRunoff' in v][0]

            sim_df = ds_sel[var_name].to_dataframe().reset_index()
            sim_df.set_index('time', inplace=True)
            sim_df.index = sim_df.index.round(freq='h')
            sim_series = sim_df[var_name]

        # Align series
        obs_aligned = obs_series.reindex(sim_series.index).dropna()
        sim_aligned = sim_series.reindex(obs_aligned.index).dropna()

        obs_vals = obs_aligned.values
        sim_vals = sim_aligned.values

        if len(obs_vals) == 0:
            self.logger.warning("No overlapping data between observations and simulations")
            return {'kge': np.nan, 'kgep': np.nan, 'nse': np.nan, 'mae': np.nan, 'rmse': np.nan}

        return {
            'kge': float(kge(obs_vals, sim_vals, transfo=1)),
            'kgep': float(kge_prime(obs_vals, sim_vals, transfo=1)),
            'nse': float(nse(obs_vals, sim_vals, transfo=1)),
            'mae': float(mae(obs_vals, sim_vals, transfo=1)),
            'rmse': float(rmse(obs_vals, sim_vals, transfo=1))
        }
