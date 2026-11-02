#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
FUSE Structure Analyzer

This module implements the Structure Ensemble Analysis for the FUSE model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.core.metrics import kge, kge_prime, mae, nse, rmse
from symfluence.core.modeling.structure_ensemble import BaseStructureEnsembleAnalyzer
from symfluence.models.fuse.runner import FUSERunner

# Known bad decision combinations that cause 100% failure in FUSE.
# Each entry maps a decision key to {value: {other_key: set_of_incompatible_values}}.
# These were identified empirically in Section 4.6 decision ensemble analysis.
KNOWN_BAD_COMBINATIONS = [
    # perc_f2sat + intflwsome: percolation from field-capacity-to-saturation
    # is incompatible with interflow — causes water balance errors and crashes.
    {'QPERC': 'perc_f2sat', 'QINTF': 'intflwsome'},
]


class FuseStructureAnalyzer(BaseStructureEnsembleAnalyzer):
    """
    Structure Ensemble Analyzer for FUSE.

    Coordinates multiple runs of FUSE with different model structure decisions
    and evaluates their performance against observations.
    """

    def __init__(self, config: Any, logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the FUSE structure analyzer.
        """
        super().__init__(config, logger, reporting_manager)

        # Decision ensemble uses FUSE's built-in SCE calibration (calib_sce)
        # with para_def, not external DDS optimization.
        if isinstance(config, dict):
            config.setdefault('FUSE_RUN_INTERNAL_CALIBRATION', True)
        else:
            self.config_dict.setdefault('FUSE_RUN_INTERNAL_CALIBRATION', True)

        # Initialize FuseRunner
        self.fuse_runner = FUSERunner(config, logger)

        # Use short FUSE file ID (Fortran CHARACTER(LEN=6) limit in selectmodl.f90)
        self.fuse_file_id = self._get_config_value(
            lambda: self.config.model.fuse.file_id,
            default=self.experiment_id,
            dict_key='FUSE_FILE_ID'
        )
        self.model_decisions_path = (self.project_dir / "settings" / "FUSE" /
                                   f"fuse_zDecisions_{self.fuse_file_id}.txt")

        # Storage for simulation results (used for visualization)
        self.simulation_results: dict[str, Any] = {}
        self.observed_streamflow = None
        self.area_km2 = None

    def _initialize_decision_options(self) -> Dict[str, List[str]]:
        """Initialize FUSE decision options from config or use defaults."""
        # Default decision options as fallback
        default_options = {
            'RFERR': ['additive_e', 'multiplc_e'],
            'ARCH1': ['tension1_1', 'tension2_1', 'onestate_1'],
            'ARCH2': ['tens2pll_2', 'unlimfrc_2', 'unlimpow_2', 'fixedsiz_2'],
            'QSURF': ['arno_x_vic', 'prms_varnt', 'tmdl_param'],
            'QPERC': ['perc_f2sat', 'perc_w2sat', 'perc_lower'],
            'ESOIL': ['sequential', 'rootweight'],
            'QINTF': ['intflwnone', 'intflwsome'],
            'Q_TDH': ['rout_gamma', 'no_routing'],
            'SNOWM': ['temp_index', 'no_snowmod']
        }

        config_options = self._get_config_value(
            lambda: self.config.model.fuse.decision_options,
            default=None,
            dict_key='FUSE_DECISION_OPTIONS'
        )

        if config_options:
            self.logger.info("Using decision options from configuration")
            validated_options = {}
            for decision, options in default_options.items():
                if decision in config_options:
                    config_vals = config_options[decision]
                    validated_options[decision] = config_vals if isinstance(config_vals, list) else options
                else:
                    validated_options[decision] = options
            return validated_options
        else:
            self.logger.info("Using default FUSE decision options")
            return default_options

    def _initialize_output_folder(self) -> Path:
        """Initialize the output folder for FUSE analysis results."""
        return self.project_dir / "reporting" / "FUSE_decision_analysis"

    def _initialize_master_file(self) -> Path:
        """Initialize the master results file path for FUSE."""
        return self.project_dir / 'optimization' / f"{self.experiment_id}_fuse_decisions_comparison.csv"

    def generate_combinations(self) -> List[Tuple[str, ...]]:
        """Generate combinations, filtering out known bad decision pairs."""
        all_combos = super().generate_combinations()
        decision_keys = list(self.decision_options.keys())

        filtered = []
        for combo in all_combos:
            combo_dict = dict(zip(decision_keys, combo))
            if self._is_valid_combination(combo_dict):
                filtered.append(combo)

        n_removed = len(all_combos) - len(filtered)
        if n_removed > 0:
            self.logger.info(
                f"Filtered {n_removed} known-bad combinations "
                f"({len(filtered)} remaining from {len(all_combos)} total)"
            )
        return filtered

    def _is_valid_combination(self, combo_dict: Dict[str, str]) -> bool:
        """Check if a decision combination is valid (not in known-bad list)."""
        for bad_combo in KNOWN_BAD_COMBINATIONS:
            if all(combo_dict.get(key) == value for key, value in bad_combo.items()):
                return False
        return True

    def update_model_decisions(self, combination: Tuple[str, ...]):
        """
        Update the FUSE model decisions file with a new combination.

        Args:
            combination (Tuple[str, ...]): Tuple of decision values to use.
        """
        try:
            with open(self.model_decisions_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # The decisions are in lines 2-10 (1-based indexing)
            decision_lines = range(1, 10)
            decision_keys = list(self.decision_options.keys())
            option_map = dict(zip(decision_keys, combination))

            for line_idx in decision_lines:
                line_parts = lines[line_idx].split()
                if len(line_parts) >= 2:
                    decision_key = line_parts[1]
                    if decision_key in option_map:
                        new_value = option_map[decision_key]
                        rest_of_line = ' '.join(line_parts[1:])
                        lines[line_idx] = f"{new_value:<10} {rest_of_line}\n"

            with open(self.model_decisions_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error updating FUSE model decisions: {str(e)}")
            raise

    def run_model(self):
        """Execute the FUSE model."""
        self.fuse_runner.run_fuse()

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics comparing simulated FUSE streamflow to observations.

        Returns:
            Dict: Dictionary containing KGE, NSE, MAE, and RMSE.
        """
        obs_file_path = self._get_config_value(
            lambda: self.config.paths.observations_path,
            default=None,
            dict_key='OBSERVATIONS_PATH'
        )
        if obs_file_path == 'default' or not obs_file_path:
            obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
        else:
            obs_file_path = Path(obs_file_path)

        # FUSE names output files using FUSE_FILE_ID (not experiment_id)
        # due to Fortran CHARACTER(LEN=6) limit in selectmodl.f90
        sim_file_path = self.get_simulation_dir('FUSE') / f"{self.domain_name}_{self.fuse_file_id}_runs_best.nc"

        if not sim_file_path.exists():
            # Try runs_def as fallback
            sim_file_path = self.get_simulation_dir('FUSE') / f"{self.domain_name}_{self.fuse_file_id}_runs_def.nc"

        # Load observations if not already cached
        if self.observed_streamflow is None:
            dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            if 'discharge_cms' in dfObs.columns:
                self.observed_streamflow = dfObs['discharge_cms'].resample('d').mean()
            else:
                data_col = [c for c in dfObs.columns if c.lower() not in ['datetime', 'date']][0]
                self.observed_streamflow = dfObs[data_col].resample('d').mean()

        # Load simulations
        with xr.open_dataset(sim_file_path, decode_timedelta=True) as dsSim:
            # Extract streamflow variable (q_routed preferred, q_instnt fallback)
            if 'q_routed' in dsSim:
                daSim = dsSim['q_routed']
            elif 'q_instnt' in dsSim:
                daSim = dsSim['q_instnt']
            else:
                self.logger.error(f"No streamflow variable found in {sim_file_path}")
                return {'kge': np.nan, 'kgep': np.nan, 'nse': np.nan, 'mae': np.nan, 'rmse': np.nan}

            for dim in ['param_set', 'latitude', 'longitude']:
                if dim in daSim.dims:
                    daSim = daSim.isel({dim: 0})
            dfSim = daSim.to_pandas()

        # Calculate catchment area for unit conversion (mm/d to cms)
        if self.area_km2 is None:
            self.area_km2 = self._calculate_catchment_area()

        # Convert units
        dfSim_cms = dfSim * self.area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Cache for visualization
        current_combo = tuple(self.get_current_decisions())
        self.simulation_results[str(current_combo)] = dfSim_cms

        # Align series
        obs_aligned = self.observed_streamflow.reindex(dfSim_cms.index).dropna()
        sim_aligned = dfSim_cms.reindex(obs_aligned.index).dropna()

        obs_vals = obs_aligned.values
        sim_vals = sim_aligned.values

        if len(obs_vals) == 0:
            return {'kge': np.nan, 'kgep': np.nan, 'nse': np.nan, 'mae': np.nan, 'rmse': np.nan}

        return {
            'kge': float(kge(obs_vals, sim_vals, transfo=1)),
            'kgep': float(kge_prime(obs_vals, sim_vals, transfo=1)),
            'nse': float(nse(obs_vals, sim_vals, transfo=1)),
            'mae': float(mae(obs_vals, sim_vals, transfo=1)),
            'rmse': float(rmse(obs_vals, sim_vals, transfo=1))
        }

    def get_current_decisions(self) -> List[str]:
        """Read current decisions from the FUSE decisions file."""
        if not self.model_decisions_path.exists():
            return []

        with open(self.model_decisions_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        decisions = []
        for line in lines[1:10]:
            parts = line.strip().split()
            if parts:
                decisions.append(parts[0])
        return decisions

    def _calculate_catchment_area(self) -> float:
        """Calculate total catchment area in km2."""
        basin_name = self._get_config_value(
            lambda: self.config.paths.river_basins_name,
            default='default',
            dict_key='RIVER_BASINS_NAME'
        )
        if basin_name == 'default':
            definition_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped',
                dict_key='DOMAIN_DEFINITION_METHOD'
            )
            basin_name = f"{self.domain_name}_riverBasins_{definition_method}.shp"

        basin_path = self._get_config_value(
            lambda: self.config.paths.river_basins_path,
            default=None,
            dict_key='RIVER_BASINS_PATH'
        )
        if not basin_path or basin_path == 'default':
            basin_path = self.project_dir / 'shapefiles' / 'river_basins' / basin_name
        else:
            basin_path = Path(basin_path)

        if basin_path.exists():
            try:
                basin_gdf = gpd.read_file(basin_path)
                if 'GRU_area' in basin_gdf.columns:
                    return basin_gdf['GRU_area'].sum() / 1e6
                else:
                    # Try to calculate from geometry if area column missing
                    if basin_gdf.crs and basin_gdf.crs.is_geographic:
                        basin_gdf = basin_gdf.to_crs(epsg=3857) # Simple projection for area
                    return basin_gdf.geometry.area.sum() / 1e6
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Failed to calculate area from shapefile: {e}", exc_info=True)

        return 1.0 # Default fallback

    def visualize_results(self, results_file: Path):
        """Perform visualization of FUSE analysis results."""
        super().visualize_results(results_file)

        if self.reporting_manager:
            self.logger.info("Generating FUSE hydrograph highlights")
            for metric in ['kge', 'nse', 'kgep']:
                try:
                    self.reporting_manager.visualize_hydrographs_with_highlight(
                        results_file,
                        self.simulation_results,
                        self.observed_streamflow,
                        self.decision_options,
                        self.output_folder,
                        metric
                    )
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Highlight visualization for {metric} failed: {e}", exc_info=True)
