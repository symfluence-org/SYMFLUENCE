#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Groundwater Evaluator.

Evaluates simulated groundwater from SUMMA against well observations (depth) or
GRACE satellite water storage anomalies (total water storage).

Groundwater Targets:
    - gw_depth: Well water table depth (positive values, meters below surface)
    - gw_grace: GRACE total water storage anomalies (for aquifer storage changes)

Model Output (SUMMA):
    - scalarTotalSoilWat: Total soil water (kg/m² → mm, converted to meters)
    - scalarAquiferStorage: Aquifer storage (m directly)
    - Water storage components: SWE, soil water, aquifer, canopy

Observations:
    - gw_depth: Well observations (depth below surface, meters)
    - gw_grace: GRACE monthly anomalies (mm water thickness)

Well Observations Characteristics:
    - Variable frequency (daily, weekly, monthly, quarterly)
    - Can have gaps and inconsistent measurements
    - May require datum correction/offset
    - Often need auto-alignment to match simulated mean

Configuration:
    GW_BASE_DEPTH: Reference depth for groundwater (default: 50.0 m)
    GW_AUTO_ALIGN: Auto-align simulated mean to observed (default: True)
    GRACE_PROCESSING_CENTER: GRACE center ('csr', 'jpl', 'gsfc', default: 'csr')
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import xarray as xr

from symfluence.core.modeling.observation_paths import first_existing_path, groundwater_observation_candidates
from symfluence.core.modeling.output_file_locator import OutputFileLocator
from symfluence.core.registries import R

from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@R.evaluators.add('GROUNDWATER')
class GroundwaterEvaluator(ModelEvaluator):
    """Groundwater evaluator comparing SUMMA to well or GRACE observations.

    Supports two evaluation modes for groundwater:
    1. Well observations: Comparison of simulated vs observed water table depth
    2. GRACE observations: Comparison of storage anomalies

    Well Depth Evaluation (gw_depth):
        - Converts simulated storage to depth below surface
        - Formula: gw_depth = (base_depth - storage_m).abs()
        - Handles two storage variables:
          * scalarTotalSoilWat: Soil water (kg/m² → mm → m)
          * scalarAquiferStorage: Aquifer storage (m directly)
        - Auto-alignment: Shifts simulated to match observed mean (useful for datum offsets)

    GRACE Evaluation (gw_grace):
        - Sums water storage components (SWE, soil, aquifer, canopy)
        - Compares with GRACE satellite anomalies
        - Auto unit conversion based on data range

    Configuration:
        GW_BASE_DEPTH: Reference depth for depth calculation (default: 50.0 m)
        GW_AUTO_ALIGN: Auto-align simulated mean to observed (default: True)
        GRACE_PROCESSING_CENTER: GRACE center to use (default: 'csr')

    Attributes:
        optimization_target: 'gw_depth' or 'gw_grace'
        variable_name: Same as optimization_target
        grace_center: GRACE processing center ('csr', 'jpl', 'gsfc')
    """

    def __init__(
        self,
        config: 'SymfluenceConfig',
        project_dir: Path,
        logger: logging.Logger,
        target: Optional[str] = None,
    ):
        """Initialize groundwater evaluator with target determination.

        Determines evaluation target (well depth vs GRACE) from configuration
        and initializes GRACE processing center selection.

        Args:
            config: Typed configuration object
            project_dir: Project root directory
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger, target=target)

        self.optimization_target = self._target_override_if_valid(('gw_depth', 'gw_grace')) or self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow',
            dict_key='OPTIMIZATION_TARGET'
        )
        if self.optimization_target not in ['gw_depth', 'gw_grace']:
            eval_var = self._get_config_value(
                lambda: None, default='', dict_key='EVALUATION_VARIABLE'
            )
            if 'gw_' in eval_var:
                self.optimization_target = eval_var

        self.variable_name = self.optimization_target
        self.grace_center = self._get_config_value(
            lambda: None,
            default='csr',
            dict_key='GRACE_PROCESSING_CENTER'
        )

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate SUMMA output files containing groundwater storage variables.

        Searches for NetCDF files with scalarTotalSoilWat, scalarAquiferStorage,
        or other water storage components needed for groundwater evaluation.

        Args:
            sim_dir: Directory containing SUMMA simulation output

        Returns:
            List[Path]: Paths to groundwater output files (NetCDF)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_groundwater_files(sim_dir)

    def calculate_metrics(
        self,
        sim: Any,
        obs: Optional[pd.Series] = None,
        mizuroute_dir: Optional[Path] = None,
        calibration_only: bool = True
    ) -> Optional[Dict[str, float]]:
        """Calculate groundwater metrics with optional auto-alignment.

        Overrides base class to apply auto-alignment AFTER both simulated
        and observed data are loaded, avoiding the side effect of loading
        observations during extraction.

        Auto-alignment (GW_AUTO_ALIGN=True):
            Shifts simulated groundwater depth to match observed mean.
            Useful for datum offsets in well observations where absolute
            depth values may not be directly comparable.

        Args:
            sim: Either a Path to simulation directory or pre-loaded pd.Series
            obs: Optional pre-loaded observations (if None, loads from file)
            mizuroute_dir: mizuRoute directory (unused for groundwater)
            calibration_only: If True, only calculate calibration period metrics

        Returns:
            Dictionary of metrics or None if calculation fails
        """
        try:
            # 1. Prepare simulated data (standard extraction)
            if isinstance(sim, (str, Path)):
                sim_dir = Path(sim)
                sim_files = self.get_simulation_files(sim_dir)
                if not sim_files:
                    self.logger.error(f"No simulation files found in {sim_dir}")
                    return None
                sim_data = self.extract_simulated_data(sim_files)
            else:
                sim_data = sim

            if sim_data is None:
                self.logger.error("Failed to extract simulated groundwater data")
                return None

            # Validate simulated data
            is_valid, error_msg = self._validate_data(sim_data, 'simulated')
            if not is_valid:
                self.logger.error(error_msg)
                return None

            # 2. Prepare observed data
            if obs is None:
                obs_data = self._load_observed_data()
            else:
                obs_data = obs

            if obs_data is None or len(obs_data) == 0:
                self.logger.error("Failed to load observed groundwater data")
                return None

            # Validate observed data
            is_valid, error_msg = self._validate_data(obs_data, 'observed')
            if not is_valid:
                self.logger.error(error_msg)
                return None

            # 3. Apply auto-alignment AFTER both datasets loaded
            auto_align = self._get_config_value(
                lambda: self.config.evaluation.usgs_gw.auto_align,
                default=True,
                dict_key='GW_AUTO_ALIGN'
            )
            if auto_align and self.optimization_target == 'gw_depth':
                # Find common time indices for alignment
                common_idx = sim_data.index.intersection(obs_data.index)
                if len(common_idx) > 0:
                    obs_mean = obs_data.loc[common_idx].mean()
                    sim_mean = sim_data.loc[common_idx].mean()
                    offset = obs_mean - sim_mean
                    sim_data = sim_data + offset
                    self.logger.info(
                        f"Auto-aligned groundwater: offset={offset:.3f}m "
                        f"(obs_mean={obs_mean:.3f}, sim_mean={sim_mean:.3f})"
                    )
                else:
                    self.logger.warning(
                        "Cannot auto-align groundwater: no overlapping time indices"
                    )

            # 4. Delegate to base class for metric calculation
            return super().calculate_metrics(
                sim=sim_data,
                obs=obs_data,
                mizuroute_dir=mizuroute_dir,
                calibration_only=calibration_only
            )

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error calculating groundwater metrics: {str(e)}")
            return None

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        # Sort files to try daily first
        sim_files.sort(key=lambda x: "day" in x.name, reverse=True)

        for sim_file in sim_files:
            try:
                self.logger.debug(f"Trying to extract groundwater from {sim_file.name}")
                with xr.open_dataset(sim_file) as ds:
                    data = None
                    if self.optimization_target == 'gw_depth':
                        data = self._extract_groundwater_depth(ds)
                    elif self.optimization_target == 'gw_grace':
                        data = self._extract_total_water_storage(ds)
                    else:
                        data = self._extract_groundwater_depth(ds)

                    if data is not None and not data.empty:
                        self.logger.info(f"Successfully extracted {len(data)} points from {sim_file.name}")
                        return data
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"Failed to extract from {sim_file.name}: {e}")

        raise ValueError(f"Could not extract groundwater data from any of {sim_files}")

    def _extract_groundwater_depth(self, ds: xr.Dataset) -> pd.Series:
        """Extract groundwater depth from SUMMA output.

        Converts storage variables to depth-below-surface representation.
        Note: Auto-alignment is applied in calculate_metrics() after both
        simulated and observed data are loaded, not here.

        Args:
            ds: xarray Dataset with groundwater storage variables

        Returns:
            pd.Series: Groundwater depth time series (meters below surface)
        """
        if 'scalarTotalSoilWat' in ds.variables:
            # Use base class method for spatial dimension collapse
            sim_data = self._collapse_spatial_dims(ds['scalarTotalSoilWat'], aggregate='mean')

            # Convert storage to depth-below-surface if comparing to GGMN
            # TotalSoilWat is in kg/m2 (mm). Convert to meters.
            sim_data_m = sim_data / 1000.0

            base_depth = float(self._get_config_value(
                lambda: self.config.evaluation.usgs_gw.base_depth,
                default=50.0,
                dict_key='GW_BASE_DEPTH'
            ))
            gw_depth_sim = (base_depth - sim_data_m).abs()

            return gw_depth_sim
        elif 'scalarAquiferStorage' in ds.variables:
            # Use base class method for spatial dimension collapse
            sim_data = self._collapse_spatial_dims(ds['scalarAquiferStorage'], aggregate='mean')
            base_depth = float(self._get_config_value(
                lambda: self.config.evaluation.usgs_gw.base_depth,
                default=50.0,
                dict_key='GW_BASE_DEPTH'
            ))
            gw_depth_sim = (base_depth - sim_data).abs()

            return gw_depth_sim
        else:
            return pd.Series()

    def _extract_total_water_storage(self, ds: xr.Dataset) -> pd.Series:
        """Extract total water storage for GRACE comparison."""
        try:
            storage_components = {}
            if 'scalarSWE' in ds.variables:
                storage_components['swe'] = ds['scalarSWE']
            if 'scalarTotalSoilWat' in ds.variables:
                storage_components['soil'] = ds['scalarTotalSoilWat']
            if 'scalarAquiferStorage' in ds.variables:
                storage_components['aquifer'] = ds['scalarAquiferStorage']
            if 'scalarCanopyWat' in ds.variables:
                storage_components['canopy'] = ds['scalarCanopyWat']

            if not storage_components:
                raise ValueError("No water storage components found")

            total_storage = None
            for component_name, component_data in storage_components.items():
                # Use base class method for spatial dimension collapse
                sim_data = self._collapse_spatial_dims(component_data, aggregate='mean')

                if total_storage is None:
                    total_storage = sim_data
                else:
                    total_storage = total_storage + sim_data

            return self._convert_tws_units(total_storage)
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error calculating TWS: {str(e)}")
            raise

    def _convert_tws_units(self, tws_data: pd.Series) -> pd.Series:
        """Convert TWS units based on data range heuristics.

        SUMMA outputs water storage components in kg/m² (equivalent to mm).
        GRACE observations are typically in cm or mm of equivalent water thickness.
        This method applies automatic unit conversion based on data magnitude.

        Unit Detection Heuristics:
            The method infers units from the data range (max - min):

            1. data_range > 1000 mm → Likely in 0.1 mm units (decamillimeters)
               - Divide by 10 to convert to mm
               - Common in some SUMMA configurations or legacy outputs
               - Example: range of 5000 → actually 500 mm seasonal variation

            2. 10 < data_range ≤ 1000 mm → Likely in cm
               - Multiply by 100 to convert cm → mm (GRACE convention)
               - GRACE anomalies typically range 5-50 cm in most basins
               - Example: range of 25 cm → 2500 mm after conversion

            3. data_range ≤ 10 → Likely already in meters or needs no conversion
               - Return unchanged
               - Example: range of 0.5 m = 500 mm (reasonable for small basins)

        Note:
            These thresholds are empirically derived from typical SUMMA/GRACE
            data ranges. For unusual basins (e.g., very large seasonal storage
            in monsoon regions), consider using explicit unit configuration
            via TWS_UNITS config parameter.

        Args:
            tws_data: Total water storage time series (unknown units)

        Returns:
            pd.Series: TWS data converted to consistent units (mm)
        """
        data_range = tws_data.max() - tws_data.min()
        if data_range > 1000:
            # Data range > 1000 suggests units are 0.1 mm (decamillimeters)
            self.logger.debug(f"TWS range={data_range:.1f} > 1000: dividing by 10 (assumed 0.1mm units)")
            return tws_data / 10.0
        elif data_range > 10:
            # Data range 10-1000 suggests units are cm, convert to mm
            self.logger.debug(f"TWS range={data_range:.1f} in [10, 1000]: multiplying by 100 (assumed cm units)")
            return tws_data * 100.0
        # Data range ≤ 10 suggests meters or already mm - return unchanged
        self.logger.debug(f"TWS range={data_range:.1f} ≤ 10: no conversion applied")
        return tws_data

    def get_observed_data_path(self) -> Path:
        return first_existing_path(
            groundwater_observation_candidates(
                self.project_dir,
                self.domain_name,
                self.optimization_target,
            )
        )

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'gw_depth':
            for col in columns:
                if any(term in col.lower() for term in ['depth', 'depth_m', 'water_level']):
                    return col
            if 'Depth_m' in columns:
                return 'Depth_m'
        elif self.optimization_target == 'gw_grace':
            grace_columns = {
                'jpl': ['grace_jpl_tws'],
                'csr': ['grace_csr_tws'],
                'gsfc': ['grace_gsfc_tws']
            }
            preferred_cols = grace_columns.get(self.grace_center, ['grace_csr_tws'])
            for col in preferred_cols:
                if col in columns:
                    return col
            for col in columns:
                if 'grace' in col.lower() and 'tws' in col.lower():
                    return col
        return None

    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                return None

            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)

            if not date_col or not data_col:
                return None

            if self.optimization_target == 'gw_depth':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            else:
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%m/%d/%Y', errors='coerce')

            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)

            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')
            obs_series = obs_series.dropna()

            if obs_series.index.tz is not None:
                obs_series.index = obs_series.index.tz_convert('UTC').tz_localize(None)

            return obs_series
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error loading observed groundwater data: {str(e)}")
            return None

    def needs_routing(self) -> bool:
        """Determine if groundwater evaluation requires streamflow routing.

        Groundwater is measured at point-scale (wells) and stored at basin scale
        (GRACE) without requiring streamflow routing models. Storage is evaluated
        directly without downstream propagation.

        Returns:
            bool: False (groundwater evaluator never requires routing)
        """
        return False
