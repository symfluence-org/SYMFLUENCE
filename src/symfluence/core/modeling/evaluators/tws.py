#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Total Water Storage (TWS) Evaluator.

Evaluates simulated water storage from SUMMA using either total water storage change
and comparing against GRACE satellite data or using glacier only mass balance and
comparing against observed glacier mass balance data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.modeling.observation_paths import (
    first_existing_path,
    tws_default_observation_path,
    tws_observation_candidates,
)
from symfluence.core.modeling.output_file_locator import OutputFileLocator
from symfluence.core.registries import R

from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@R.evaluators.add('TWS')
class TWSEvaluator(ModelEvaluator):
    """Total Water Storage evaluator comparing SUMMA to GRACE satellites.

    Evaluates simulated water storage from SUMMA using one of two metrics
    1) total water storage change against GRACE satellite observations of water storage anomalies.
    2) glacier annual maximum and minimum mass balance against observations

    Key Features:
        - Multi-component storage summation: SWE + canopy + soil + aquifer
        - Flexible storage component selection via configuration
        - GRACE data support from multiple processing centers (JPL, CSR, GSFC)
        - Anomaly computation with configurable baselines
        - Advanced signal processing: detrending, variability scaling
        - Diagnostic exports for visualization and analysis

    Output Variables:
        - basin__Storage: sum of scalarSWE+calarCanopyWat+scalarTotalSoilWat
            +scalarAquiferStorage+scalarGlceWE or cumsum of basin__StorageChange
        - basin__MassBalanceMax: glacier mass balance units of mm of water, maximum per year
        - basin__MassBalanceMin: glacier mass balance units of mm of water, minimum per year

    Configuration:
        TWS_GRACE_COLUMN: GRACE data column name (default: 'grace_jpl_anomaly')
        TWS_ANOMALY_BASELINE: Baseline period for anomaly calc ('overlap' or year range)
        TWS_UNIT_CONVERSION: Scaling factor for model storage (default: 1.0)
        TWS_STORAGE_COMPONENTS: Comma-separated storage variables to sum
        TWS_DETREND: Remove linear trend
        TWS_SCALE_TO_OBS: Scale model variability to match observations
        TWS_OBS_PATH: Direct path override for GRACE data

    Physical Basis:
        GRACE measures vertically-integrated water column changes via:
        - Temporal gravity variations → water storage changes
        - Signal includes: surface water, soil moisture, groundwater, snow/ice
        - Monthly resolution averages diurnal/seasonal noise
        - Spatial footprint ~300 km (cannot resolve sub-grid variability)

    Attributes:
        optimization_target: 'stor_grace' or 'stor_mb'
        grace_column: GRACE data column name
        anomaly_baseline: Baseline period specification
        unit_conversion: Scaling factor for model storage values
        detrend: Whether to remove linear trends
        scale_to_obs: Whether to scale model variability to observations
        storage_vars: List of SUMMA storage variable names to sum
    """

    DEFAULT_STORAGE_VARS = [
        'scalarSWE', 'scalarCanopyWat', 'scalarTotalSoilWat', 'scalarAquiferStorage'
    ]

    def __init__(
        self,
        config: 'SymfluenceConfig',
        project_dir: Path,
        logger: logging.Logger,
        target: Optional[str] = None,
    ):
        """Initialize TWS evaluator with storage component and signal processing options.

        Configures water storage components to sum (SWE, canopy, soil, aquifer, glacier),
        GRACE data column selection, and optional signal processing (detrending,
        variability scaling) to handle model-observation mismatches.

        Configuration Parameters:
            TWS_GRACE_COLUMN: GRACE variable column name (default: 'grace_jpl_anomaly')
                - Supports multiple GRACE processing centers:
                  'grace_jpl_anomaly' (JPL, most widely used)
                  'grace_csr_anomaly' (CSR, U.Texas)
                  'grace_gsfc_anomaly' (GSFC, NASA)
            TWS_ANOMALY_BASELINE: Baseline period for anomaly ('overlap' by default)
            TWS_UNIT_CONVERSION: Scaling factor for model storage (default: 1.0)
            TWS_STORAGE_COMPONENTS: Comma-separated SUMMA storage variables
                - Default: 'scalarSWE, scalarCanopyWat, scalarTotalSoilWat, scalarAquiferStorage'
                - Can include glacier mass change: 'scalarGlceWE'
                - Or use cumulative basin__StorageChange
            TWS_DETREND: Remove linear trend from both sim and obs (default: False)
            TWS_SCALE_TO_OBS: Scale model variability to match observations (default: False)
                - Centers both series on zero-mean, rescales model std dev to match obs
                - Useful when pattern is correct but amplitude is wrong

        Args:
            config: Typed configuration object (SymfluenceConfig)
            project_dir: Project root directory
            logger: Logger instance

        Raises:
            None (uses defaults for missing config values)
        """
        super().__init__(config, project_dir, logger, target=target)

        self.optimization_target = self._target_override_if_valid(('stor_grace', 'stor_mb')) or self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow',
            dict_key='OPTIMIZATION_TARGET'
        )
        if self.optimization_target not in ['stor_grace', 'stor_mb']:
            eval_var = self._get_config_value(
                lambda: self.config.evaluation.evaluation_variable,
                default='',
                dict_key='EVALUATION_VARIABLE'
            )
            if eval_var in ['stor_grace', 'stor_mb']:
                self.optimization_target = eval_var
            else:
                self.optimization_target = 'stor_grace'

        self.grace_column = self._get_config_value(
            lambda: self.config.evaluation.tws.grace_column,
            default='grace_jpl_anomaly',
            dict_key='TWS_GRACE_COLUMN'
        )
        self.anomaly_baseline = self._get_config_value(
            lambda: self.config.evaluation.tws.anomaly_baseline,
            default='overlap',
            dict_key='TWS_ANOMALY_BASELINE'
        )
        self.unit_conversion = self._get_config_value(
            lambda: self.config.evaluation.tws.unit_conversion,
            default=1.0,
            dict_key='TWS_UNIT_CONVERSION'
        )

        # Detrending option: removes linear trend from both series before comparison
        self.detrend = self._get_config_value(
            lambda: self.config.evaluation.tws.detrend,
            default=False,
            dict_key='TWS_DETREND'
        )

        # Scaling option: scale model variability to match observed
        # Useful when model has correct pattern but wrong amplitude
        self.scale_to_obs = self._get_config_value(
            lambda: self.config.evaluation.tws.scale_to_obs,
            default=False,
            dict_key='TWS_SCALE_TO_OBS'
        )

        storage_str = self._get_config_value(
            lambda: self.config.evaluation.tws.storage_components,
            default='',
            dict_key='TWS_STORAGE_COMPONENTS'
        )
        if storage_str:
            self.storage_vars = [v.strip() for v in storage_str.split(',') if v.strip()]
        else:
            self.storage_vars = self.DEFAULT_STORAGE_VARS.copy()

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate SUMMA output files containing water storage variables.

        Searches for NetCDF files containing any of the configured storage
        components (SWE, canopy water, soil water, aquifer storage, glacier ice mass change).

        Args:
            sim_dir: Directory containing SUMMA simulation output files

        Returns:
            List[Path]: Paths to storage variable files (typically most recent daily output)
        """
        locator = OutputFileLocator(self.logger)
        # TWS needs the most recent file with storage components
        most_recent = locator.get_most_recent(sim_dir, 'tws')
        return [most_recent] if most_recent else []

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract and sum water storage components from SUMMA output files.

        Implements intelligent file selection to find all storage variables across
        multiple SUMMA output files (daily, timestep, etc.), then sums components
        to compute total water storage.

        File Search Strategy:
            1. Start with provided sim_files[0] (typically daily output)
            2. Search output directory for alternative files:
               - Daily files (*_day.nc): Preferred for most storage variables
               - Timestep files (*_timestep.nc): Look for more storage variables
            3. Test each file for availability of storage variables
            4. Select file(s) with maximum matching variables
            5. If no single file has all vars, try combining vars from multiple files

        Storage Component Summation:
            Iterates through self.storage_vars (e.g., SWE, canopy, soil, aquifer):
            1. Check variable availability in selected file
            2. Extract values as float array
            3. Replace fill values (-9999 from SUMMA) with NaN
            4. Unit conversions:
               - Aquifer storage: m → mm (multiply by 1000)
               - Others: Already in mm or compatible
            5. Collapse spatial dimensions (HRU/GRU) via nanmean
            6. Accumulate: total_tws += component (handles NaN values properly)

        Spatial Aggregation (HRU/GRU):
            Single HRU/GRU: Mean reduces single value (identity operation)
            Multiple HRU/GRU: Averages across dimension via nanmean()
            Result: Scalar time series (1D array of length = n_timesteps)

        Storage Component Units (SUMMA) by HRU or DOM:
            - scalarSWE: kg/m² → mm (multiply by 1.0, density=1000 kg/m³)
            - scalarCanopyWat: mm (water depth)
            - scalarTotalSoilWat: mm (water depth)
            - scalarAquiferStorage: m → mm (multiply by 1000.0)
            - scalarGlceWE: kg/m² → mm (multiply by 1.0, density=1000 kg/m³)
            OR could use cumulative basin__StorageChange by GRU, sums these (kg/m²/s → mm/s summed)
        OR
        Glacier Only Storage Components (SUMMA) by GRU
            - Glacier mass balance: (change in basin__GlacierStorage)/basin__GlacierArea
            - Units of Gt/m² → mm via x10⁻¹² and subtract initial value
             - Calibrate to seasonal maximum and minimum

        Args:
            sim_files: List of SUMMA output files (typically daily NetCDF)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Time series of total water storage (mm) with datetime index

        Raises:
            ValueError: If no storage variables found in any candidate file
            Exception: If time coordinate cannot be extracted

        Notes:
            - Handles NaN intelligently: NaN + value = value (not NaN)
            - Fill values (-9999) replaced with NaN to avoid data corruption
            - Logs which file used and how many variables found for debugging
        """
        output_file = sim_files[0]

        # Try to find a file with storage variables
        output_dir = output_file.parent
        potential_files = [
            output_file,  # Original file
        ]
        # Add daily file alternatives (storage vars are often in daily output)
        for f in output_dir.glob('*_day.nc'):
            if f not in potential_files:
                potential_files.insert(0, f)  # Prefer daily files
        # Also check timestep files
        for f in output_dir.glob('*timestep.nc'):
            if f not in potential_files:
                potential_files.append(f)

        # Find a file that has ALL requested storage variables (if possible)
        # or at least the most critical ones including glacier mass
        best_file = None
        best_var_count = 0
        for candidate_file in potential_files:
            try:
                with xr.open_dataset(candidate_file) as test_ds:
                    matching_vars = [v for v in self.storage_vars if v in test_ds.data_vars]
                    if len(matching_vars) > best_var_count:
                        best_var_count = len(matching_vars)
                        best_file = candidate_file
                        self.logger.debug(f"Found {len(matching_vars)}/{len(self.storage_vars)} vars in {candidate_file.name}")
                    # If we find all variables, use this file
                    if len(matching_vars) == len(self.storage_vars):
                        break
            except (OSError, IOError, KeyError):
                continue

        if best_file:
            output_file = best_file
            self.logger.debug(f"Using {output_file.name} for TWS extraction ({best_var_count} storage vars)")

        try:
            ds = xr.open_dataset(output_file)
            total_tws = None

            # Find time coordinate
            time_coord = None
            for var in ['time', 'Time', 'datetime']:
                if var in ds.coords or var in ds.dims:
                    time_coord = pd.to_datetime(ds[var].values)
                    break

            if time_coord is None:
                raise ValueError("Could not extract time coordinate")

            if self.optimization_target == 'stor_mb':
                if 'basin__GlacierArea' in ds.variables and 'basin__GlacierStorage' in ds.variables:
                    area = ds['basin__GlacierArea']
                    area = np.where(area <= 0, np.nan, area)
                    total_tws = ds['basin__GlacierStorage']/area
                    # convert from Gt/m² (km³ of water/m² to mm
                    total_tws = total_tws * 1e9 * 1000.0
                else:
                    raise ValueError("Glacier mass balance variables (basin__GlacierArea, basin__GlacierStorage) not found in SUMMA output")
            else:
                if 'basin__StorageChange' in ds.data_vars:
                    total_tws = ds['basin__StorageChange']
                    total_tws = np.where(total_tws < -999, np.nan, total_tws)
                    # integrate: mm/s * seconds -> mm per timestep, then cumulative sum over time
                    dt = self._get_config_value(
                        lambda: self.config.forcing.time_step_size,
                        default=1,
                        dict_key='FORCING_TIME_STEP_SIZE'
                    )
                    if hasattr(total_tws, 'sel') or hasattr(total_tws, 'dims'):
                        total_tws = (total_tws * dt).cumsum(dim=time_coord)
                    else:
                        total_tws = np.cumsum(total_tws * dt, axis=0)
                else:
                    for var in self.storage_vars:
                        if var in ds.data_vars:
                            data = ds[var].values.astype(float)

                            # Replace fill values (-9999) with NaN
                            # SUMMA uses -9999 for missing/invalid data in some domains
                            data = np.where(data < -999, np.nan, data)

                            # Unit conversion: aquifer storage is in meters, others usually in mm
                            if 'aquifer' in var.lower():
                                self.logger.debug(f"Converting {var} from meters to mm")
                                data = data * 1000.0

                            if data.ndim > 1:
                                axes_to_sum = tuple(range(1, data.ndim))
                                data = np.nanmean(data, axis=axes_to_sum)

                            if total_tws is None:
                                total_tws = data.copy()
                            else:
                                # Use nansum behavior: NaN + value = value
                                total_tws = np.where(np.isnan(total_tws), data,
                                                    np.where(np.isnan(data), total_tws, total_tws + data))
                        else:
                            self.logger.warning(f"Storage variable {var} not found in {output_file}")

            if total_tws is None:
                raise ValueError(f"No storage variables {self.storage_vars} found in {output_file}")

            tws_series = pd.Series(total_tws.flatten(), index=time_coord, name='simulated_tws')
            tws_series = tws_series * self.unit_conversion
            ds.close()

            return tws_series
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error loading SUMMA output: {e}")
            raise

    def get_observed_data_path(self) -> Path:
        """Resolve path to GRACE water storage anomaly observations.

        Implements flexible path resolution supporting multiple GRACE data organization
        patterns and locations. Checks multiple possible locations and file naming
        conventions commonly used in glacier and hydrology projects.

        Path Search Priority:
            1. TWS_OBS_PATH config override (if specified, return immediately)
            2. observations/storage/grace/ (glacier domain organization)
            3. observations/grace/ (standard organization)

        File Naming Conventions Searched:
            - Domain-specific HRU format: {domain}_HRUs_GRUs_grace_tws_anomaly.csv
            - Elevation-based HRU format: {domain}_HRUs_elevation_grace_tws_anomaly_by_hru.csv
            - Processed format: {domain}_grace_tws_processed.csv
            - Generic format: {domain}_grace_tws_anomaly.csv
            - Generic without domain: grace_tws_anomaly.csv, tws_anomaly.csv

        GRACE Satellites:
            - GRACE (Gravity Recovery and Climate Experiment): Twin satellites measuring
              Earth's gravity field to infer water storage changes
            - GRACE Follow-On: Continuation mission (2018-present)
            - Temporal resolution: Monthly anomalies
            - Spatial resolution: ~300 km × 300 km footprint
            - Sensitivity: 1-2 cm equivalent water height
        GRACE Data Products:
            - Multiple processing centers: JPL (Jet Propulsion Lab), CSR (U.Texas), GSFC (NASA)
            - Released as time-variable gravity fields → converted to water storage anomalies
            - Units: mm equivalent water thickness (relative to 2004-2009 baseline)
            OR
        Glacier Mass Balance Data:
            - Annually reported maximum in mm.w.e. and minimum mm.w.e.
            - These are usually in April and October in the Northern Hemisphere

        Directory Hierarchy:
            - Preprocessed subdirectory (preferred for processed data)
            - Root directory (fallback for raw/alternative formats)

        Returns Path even if file doesn't exist (may trigger acquisition later).

        Args:
            None (uses self.project_dir, self.domain_name, self.grace_column)

        Returns:
            Path: Absolute path to GRACE observation file
        """
        tws_obs_path = self._get_config_value(
            lambda: self.config.evaluation.tws.obs_path,
            default=None,
            dict_key='TWS_OBS_PATH'
        )
        if tws_obs_path:
            return Path(tws_obs_path)
        return first_existing_path(
            tws_observation_candidates(
                self.project_dir,
                self.domain_name,
                self.optimization_target,
            ),
            default=tws_default_observation_path(self.project_dir, self.domain_name),
        )

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify GRACE data column from multiple processing center options.

        GRACE water storage anomalies are provided by multiple processing centers
        (JPL, CSR, GSFC) which produce similar but slightly different results due
        to different processing algorithms and filtering approaches.

        Column Priority:
            1. Configured column (TWS_GRACE_COLUMN from config, default: 'grace_jpl_anomaly')
            2. Fallback GRACE centers: 'grace_csr_anomaly', 'grace_gsfc_anomaly'
            3. Generic pattern matching: any column containing 'grace' and 'anomaly'

        GRACE Processing Centers:
            - JPL (Jet Propulsion Lab): Most widely used, published since 2002
            - CSR (Center for Space Research, U.Texas): Alternative processing
            - GSFC (NASA Goddard Space Flight Center): Alternative processing

        Args:
            columns: List of column names from GRACE CSV file

        Returns:
            Optional[str]: Name of GRACE anomaly column, or None if not found
        """
        if self.optimization_target == 'stor_mb':
            # For independent storage anomaly, expect a specific column
            for col in columns:
                if 'mass_balance' in col.lower() or 'stor_mb' in col.lower():
                    return col
            # Fallback to exact match
            if 'Mass_Balance' in columns:
                return 'Mass_Balance'
            return None
        else:
            if self.grace_column in columns:
                return self.grace_column
            # Fallback to known columns
            for fallback in ['grace_jpl_anomaly', 'grace_csr_anomaly', 'grace_gsfc_anomaly']:
                if fallback in columns:
                    return fallback
            return next((c for c in columns if 'grace' in c.lower() and 'anomaly' in c.lower()), None)

    def needs_routing(self) -> bool:
        """Determine if water storage evaluation requires routing model.

        Water storage (measured by GRACE or simulated by SUMMA) is integrated
        over the basin and does NOT require streamflow routing. Storage changes
        are evaluated directly without propagation.

        Returns:
            bool: False (TWS evaluator never requires routing)
        """
        return False

    def _detrend_series(self, series: pd.Series) -> pd.Series:
        """Remove linear trend from a time series via least-squares regression.

        Detrending isolates anomalies and seasonal patterns by removing the
        underlying long-term trend. Critical for glacierized basins where
        glacier mass loss creates a dominant trend (>80% of variance) that
        obscures seasonal signals.

        Mathematical Approach:
            1. Fit linear model: y = a*t + b using valid (non-NaN) data points
            2. Compute fitted trend: trend = a*t + b for all time points
            3. Detrended series: y_detrended = y - trend
            4. Result: seasonal/anomaly signal with trend removed

        NaN Handling:
            - Ignores NaN values during linear regression
            - Returns original series if < 2 valid data points
            - Preserves NaN positions in detrended output

        Impact on Correlation:
            For glacierized basins: r ≈ 0.30 (with trend) → r ≈ 0.78 (detrended)
            The long-term trend completely dominates model-obs mismatch, preventing
            proper calibration of seasonal storage changes.

        Args:
            series: Time series (pd.Series with datetime index and float values)

        Returns:
            pd.Series: Detrended series (same index and length as input)

        Notes:
            Uses numpy.polyfit (degree 1) for robust linear regression with NaN handling
        """
        # Convert datetime index to numeric for linear regression
        x = np.arange(len(series))
        y = series.values

        # Handle NaN values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return series  # Not enough data to detrend

        # Fit linear trend using valid points only
        coeffs = np.polyfit(x[valid_mask], y[valid_mask], 1)
        trend = np.polyval(coeffs, x)

        # Remove trend
        detrended = y - trend
        return pd.Series(detrended, index=series.index, name=series.name)

    def _scale_to_obs_variability(self, sim_series: pd.Series, obs_series: pd.Series) -> pd.Series:
        """Scale model variability to match observed variability amplitude.

        Addresses systematic amplitude differences where model captures the correct
        temporal pattern (seasonal cycle, event timing) but with wrong magnitude.
        Commonly occurs when:
        - SWE accumulation too high or too low
        - Soil water variability underestimated
        - Parameter uncertainty affects storage amplitude

        Rescaling Strategy:
            1. Remove means from both series: center on zero
            2. Compute standard deviations: σ_sim, σ_obs
            3. Scale simulated: sim_scaled = (sim_centered / σ_sim) × σ_obs
            4. Result: both series have same mean and std dev

        Advantage vs Standard Normalization:
            - Preserves zero-mean anomalies (natural time series origin)
            - Corrects amplitude while maintaining temporal structure
            - Allows model with correct pattern but wrong magnitude to achieve high KGE

        Use Cases:
            - Model SWE pattern correct, but 20% too high
            - Soil water storage correct seasonal shape, wrong amplitude
            - When detrending is insufficient for matching observations

        Args:
            sim_series: Simulated water storage time series (zero-mean anomalies)
            obs_series: Observed storage time series (zero-mean anomalies)

        Returns:
            pd.Series: Rescaled simulated series with same std dev as observations

        Notes:
            - Handles zero variability (returns original if sim_std = 0)
            - Both input series should be anomalies (zero-mean) before calling
        """
        obs_std = obs_series.std()
        sim_std = sim_series.std()

        if sim_std == 0 or np.isnan(sim_std):
            return sim_series

        # Scale sim to have same std as obs, keeping zero-mean
        sim_centered = sim_series - sim_series.mean()
        scaled = (sim_centered / sim_std) * obs_std

        return pd.Series(scaled.values, index=sim_series.index, name=sim_series.name)

    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None,
                         mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """Calculate TWS performance metrics comparing SUMMA storage to GRACE anomalies.

        Overrides base class to implement TWS-specific metric calculation pipeline:
        1. Load absolute storage from SUMMA daily/timestep output
        2. Aggregate to monthly means (match GRACE temporal resolution)
        3. Compute anomalies relative to overlap period baseline
        4. Apply optional signal processing: detrending, variability scaling
        5. Calculate performance metrics (KGE, RMSE, NSE, correlation, bias)

        Args:
            sim: Path to simulation directory or pre-loaded pd.Series
            obs: Optional pre-loaded pd.Series of observations. If None, loads from file.
            mizuroute_dir: mizuRoute simulation directory (unused for TWS)
            calibration_only: Whether to calculate only calibration metrics
        """
        sim_dir = Path(sim) if isinstance(sim, (str, Path)) else None
        if sim_dir is None:
            raise ValueError("TWS evaluation requires simulation directory Path")
        try:
            self.logger.debug(f"[TWS] Starting metrics calculation for {sim_dir}")

            # Load simulated data (absolute storage)
            sim_files = self.get_simulation_files(sim_dir)
            if not sim_files:
                self.logger.error(f"[TWS] No simulation files found in {sim_dir}")
                return None

            sim_tws = self.extract_simulated_data(sim_files)
            self.logger.debug(f"[TWS] Extracted simulated TWS: {len(sim_tws)} points")

            # Load observed data (anomalies)
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"[TWS] Observed data path does not exist: {obs_path}")
                return None

            obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
            col = self._get_observed_data_column(obs_df.columns)
            if not col:
                self.logger.error(f"[TWS] Could not find GRACE/MB column in {obs_path}. Available: {list(obs_df.columns)}")
                return None

            obs_tws = obs_df[col].dropna()

            if self.optimization_target == 'stor_mb':
                # make a timeseries of maximum and minimum values and repeat value on month
                # compute annual maxima/minima from simulated TWS and their timestamps
                annual_max = sim_tws.resample('A').max()
                annual_min = sim_tws.resample('A').min()
                annual_max_idx = sim_tws.resample('A').idxmax()
                annual_min_idx = sim_tws.resample('A').idxmin()

                # For each annual max, find the most recent annual min occurring <= that max
                max_minus_prev_min = []
                for max_ts in annual_max_idx.values:
                    if pd.isna(max_ts):
                        max_minus_prev_min.append(np.nan)
                        continue
                    # candidate minima up to and including the max timestamp
                    candidates = annual_min_idx[annual_min_idx <= max_ts]
                    if len(candidates) == 0:
                        max_minus_prev_min.append(np.nan)
                    else:
                        prev_min_ts = candidates.iloc[-1]
                        try:
                            max_val = float(sim_tws.loc[max_ts])
                            prev_min_val = float(sim_tws.loc[prev_min_ts])
                            max_minus_prev_min.append(max_val - prev_min_val)
                        except Exception:  # noqa: BLE001 — must-not-raise contract
                            max_minus_prev_min.append(np.nan)

                annual_max_mb = pd.Series(max_minus_prev_min, index=annual_max.index)

                # For each annual min, find the most recent annual max occurring <= that min
                min_minus_prev_max = []
                for min_ts in annual_min_idx.values:
                    if pd.isna(min_ts):
                        min_minus_prev_max.append(np.nan)
                        continue
                    candidates = annual_max_idx[annual_max_idx <= min_ts]
                    if len(candidates) == 0:
                        min_minus_prev_max.append(np.nan)
                    else:
                        prev_max_ts = candidates.iloc[-1]
                        try:
                            min_val = float(sim_tws.loc[min_ts])
                            prev_max_val = float(sim_tws.loc[prev_max_ts])
                            min_minus_prev_max.append(min_val - prev_max_val)
                        except Exception:  # noqa: BLE001 — must-not-raise contract
                            min_minus_prev_max.append(np.nan)

                annual_min_mb = pd.Series(min_minus_prev_max, index=annual_min.index)

                # Map to observation frequency:
                obs_years = pd.DatetimeIndex(obs_tws.index).year
                annual_years = pd.DatetimeIndex(annual_max_mb.index).year

                # count observations per year to detect annual vs bi-annual
                obs_counts = obs_tws.groupby(obs_tws.index.year).size()
                is_annual = obs_counts.max() == 1 and obs_counts.min() == 1
                is_biannual = obs_counts.max() == 2 and obs_counts.min() == 2

                if is_annual:
                    # annual observations: sum of the year's max and min-derived metrics
                    mapped = []
                    for y in obs_years:
                        if y in annual_years:
                            try:
                                v_max = annual_max_mb.loc[annual_max_mb.index.year == y].values
                                v_min = annual_min_mb.loc[annual_min_mb.index.year == y].values
                                vm = (float(v_max[0]) if len(v_max) > 0 and not pd.isna(v_max[0]) else 0.0)
                                vn = (float(v_min[0]) if len(v_min) > 0 and not pd.isna(v_min[0]) else 0.0)
                                mapped.append(vm + vn)
                            except Exception:  # noqa: BLE001 — must-not-raise contract
                                mapped.append(np.nan)
                        else:
                            mapped.append(np.nan)
                    sim_anomaly = pd.Series(mapped, index=obs_tws.index, name='sim_tws')
                    obs_anomaly = obs_tws
                elif is_biannual:
                    # bi-annual observations: assign +value at the month matching annual max, -value at month matching annual min
                    mapped = []
                    # build dicts for quick lookup by year
                    max_month_by_year = {iy: ts.month for iy, ts in zip(annual_max_mb.index.year, annual_max_idx.values) if not pd.isna(ts)}
                    min_month_by_year = {iy: ts.month for iy, ts in zip(annual_min_mb.index.year, annual_min_idx.values) if not pd.isna(ts)}
                    max_val_by_year = {iy: (float(v) if not pd.isna(v) else np.nan) for iy, v in zip(annual_max_mb.index.year, annual_max_mb.values)}
                    min_val_by_year = {iy: (float(v) if not pd.isna(v) else np.nan) for iy, v in zip(annual_min_mb.index.year, annual_min_mb.values)}

                    for ts in obs_tws.index:
                        y = ts.year
                        m = ts.month
                        if y not in annual_years:
                            mapped.append(np.nan)
                            continue
                        if max_month_by_year.get(y) == m:
                            mapped.append(abs(max_val_by_year.get(y, np.nan)))
                        elif min_month_by_year.get(y) == m:
                            mapped.append(-abs(min_val_by_year.get(y, np.nan)))
                        else:
                            mapped.append(np.nan)
                    sim_anomaly = pd.Series(mapped, index=obs_tws.index, name='sim_tws')
                    obs_anomaly = obs_tws
                else:
                    self.logger.error("Mass balance observation data not annual or biannual")
                    return None

            else: # GRACE type data
                # Aggregate simulated to monthly means to match GRACE
                sim_monthly = sim_tws.resample('MS').mean()

                # Check if we have any simulated data
                if len(sim_monthly) == 0:
                    self.logger.error("[TWS] No simulated TWS data available for metrics calculation")
                    return None

                # Find common period
                common_idx = sim_monthly.index.intersection(obs_tws.index)
                if len(common_idx) == 0:
                    sim_start, sim_end = sim_monthly.index[0], sim_monthly.index[-1]
                    obs_start, obs_end = obs_tws.index[0], obs_tws.index[-1]
                    self.logger.error(f"[TWS] No overlapping period between simulation ({sim_start} to {sim_end}) and observations ({obs_start} to {obs_end})")
                    return None

                self.logger.debug(f"[TWS] Found common period with {len(common_idx)} points")
                sim_matched = sim_monthly.loc[common_idx]
                obs_matched = obs_tws.loc[common_idx]

                # Calculate anomalies
                if self.anomaly_baseline == 'overlap':
                    baseline_mean_sim = sim_matched.mean()
                    baseline_mean_obs = obs_matched.mean()
                else: # year range XXXX-XXXX
                    # Parse the year-range baseline and compute mean over that period
                    try:
                        ys = int(self.anomaly_baseline[:4])
                        ye = int(self.anomaly_baseline[-4:])
                        # Ensure DatetimeIndex for year selection
                        idx_years = pd.DatetimeIndex(sim_matched.index).year
                        mask = (idx_years >= ys) & (idx_years <= ye)
                        if mask.any():
                            baseline_mean_sim = sim_matched.loc[mask].mean()
                            baseline_mean_obs = obs_matched.loc[mask].mean()
                        else:
                            self.logger.warning(f"[TWS] Baseline years {ys}-{ye} not in simulation overlap; using full-overlap mean")
                            baseline_mean_sim = sim_matched.mean()
                            baseline_mean_obs = obs_matched.mean()
                    except Exception as exc:  # noqa: BLE001 — must-not-raise contract
                        self.logger.warning(f"[TWS] Invalid TWS_ANOMALY_BASELINE '{self.anomaly_baseline}': {exc}; using overlap mean")
                        baseline_mean_sim = sim_matched.mean()
                        baseline_mean_obs = obs_matched.mean()
                sim_anomaly = sim_matched - baseline_mean_sim
                obs_anomaly = obs_matched - baseline_mean_obs

                # Apply detrending if configured
                if self.detrend:
                    self.logger.info("[TWS] Applying linear detrending to both series")
                    sim_anomaly = self._detrend_series(sim_anomaly)
                    obs_anomaly = self._detrend_series(obs_anomaly)

                # Apply variability scaling if configured
                # This forces model variability to match observed (removes alpha penalty in KGE)
                if self.scale_to_obs:
                    self.logger.info("[TWS] Scaling model variability to match observed")
                    sim_anomaly = self._scale_to_obs_variability(sim_anomaly, obs_anomaly)

            # Calculate metrics. GRACE comparisons are on centred anomalies,
            # whose means are ~0 by construction, so the KGE bias term is
            # undefined there and is left out (see _calculate_anomaly_metrics).
            if self.optimization_target == 'stor_mb':
                metrics = self._calculate_performance_metrics(obs_anomaly, sim_anomaly)
            else:
                metrics = self._calculate_anomaly_metrics(obs_anomaly, sim_anomaly)
            self.logger.debug(f"[TWS] Calculated metrics: {metrics}")
            return metrics

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"[TWS] Error calculating TWS metrics: {str(e)}")
            import traceback
            self.logger.debug(f"[TWS] Traceback: {traceback.format_exc()}")
            return None

    def _calculate_anomaly_metrics(
        self, obs_anomaly: pd.Series, sim_anomaly: pd.Series
    ) -> Dict[str, float]:
        """Metrics for zero-mean anomaly series (GRACE TWS).

        Both series are centred on a baseline mean before comparison, so
        ``beta = mean(sim) / mean(obs)`` is a ratio of two near-zero residuals:
        it swings arbitrarily (and can flip sign) between otherwise identical
        runs, which made the standard KGE effectively random. Here KGE keeps only
        the correlation and variability terms::

            KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2)

        ``beta`` and ``PBIAS`` (also normalised by the ~0 observed mean) are
        reported as NaN. NSE, RMSE, MAE and r are unchanged.
        """
        result = self._calculate_performance_metrics(obs_anomaly, sim_anomaly)
        r = result.get('r', np.nan)
        alpha = result.get('alpha', np.nan)
        if r is None or alpha is None or not (np.isfinite(r) and np.isfinite(alpha)):
            kge_value = np.nan
        else:
            kge_value = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2))
        result['KGE'] = kge_value
        result['beta'] = np.nan
        result['PBIAS'] = np.nan
        return result

    def get_diagnostic_data(self, sim_dir: Path) -> Dict[str, Any]:
        """Export diagnostic data for visualization and detailed analysis.

        Extracts matched time series and intermediate results from TWS evaluation
        for use in plotting, trend analysis, component breakdown, and debugging.

        Diagnostic Data Exported:
            time: Common time index (monthly, for matched data)
            sim_tws: Simulated total storage (absolute, mm)
            sim_anomaly: Storage anomalies (detrended/processed version)
            sim_anomaly_raw: Storage anomalies (before signal processing)
            obs_anomaly: GRACE water storage anomalies (detrended/processed version)
            obs_anomaly_raw: GRACE anomalies (before signal processing)
            grace_column: Selected GRACE data column name
            grace_all_columns: All GRACE columns in overlap period (for comparison)
            detrend_applied: Boolean indicating if detrending was applied
            scale_applied: Boolean indicating if variability scaling was applied

        Use Cases:
            1. Visualization: Plot sim vs obs with/without signal processing
            2. Component analysis: Compare raw vs processed anomalies
            3. Trend analysis: Examine detrended vs raw series
            4. Data QC: Verify time alignment and overlap period
            5. Comparison: Compare multiple GRACE processing centers

        Processing Applied:
            - Temporal aggregation: Daily SUMMA to monthly means
            - Anomaly calculation: Relative to overlap period mean
            - Optional detrending: Remove linear trends
            - Optional scaling: Match model variability to observations

        Args:
            sim_dir: Directory containing SUMMA simulation output

        Returns:
            Dict[str, Any]: Diagnostic dictionary with keys:
            - time: np.datetime64 array of monthly times
            - sim_tws, sim_anomaly, etc.: np.ndarray of values
            - grace_all_columns: pd.DataFrame of GRACE data for overlap
            - detrend_applied, scale_applied: bool flags

        Notes:
            - Returns empty dict if simulation files or observations not found
            - Handles NaN values by creating valid_mask for clean overlap period
            - Preserves raw anomalies before processing for comparison
        """
        sim_files = self.get_simulation_files(sim_dir)
        if not sim_files:
            return {}

        sim_tws = self.extract_simulated_data(sim_files)

        obs_path = self.get_observed_data_path()
        if not obs_path.exists():
            return {}

        obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
        col = self._get_observed_data_column(obs_df.columns)
        if not col:
            return {}

        sim_monthly = sim_tws.resample('MS').mean()
        obs_monthly = obs_df[col].copy()

        common_index = sim_monthly.index.intersection(obs_monthly.index)
        valid_mask = ~(sim_monthly.loc[common_index].isna() | obs_monthly.loc[common_index].isna())

        sim_matched = sim_monthly.loc[common_index][valid_mask]
        obs_matched = obs_monthly.loc[common_index][valid_mask]

        # Calculate anomalies
        if self.anomaly_baseline == 'overlap':
            baseline_mean_sim = sim_matched.mean()
            baseline_mean_obs = obs_matched.mean()
        else: # year range XXXX-XXXX
            # Parse the year-range baseline and compute mean over that period
            try:
                ys = int(self.anomaly_baseline[:4])
                ye = int(self.anomaly_baseline[-4:])
                # Ensure DatetimeIndex for year selection
                idx_years = pd.DatetimeIndex(sim_matched.index).year
                mask = (idx_years >= ys) & (idx_years <= ye)
                if mask.any():
                    baseline_mean_sim = sim_matched.loc[mask].mean()
                    baseline_mean_obs = obs_matched.loc[mask].mean()
                else:
                    self.logger.warning(f"[TWS] Baseline years {ys}-{ye} not in simulation overlap; using full-overlap mean")
                    baseline_mean_sim = sim_matched.mean()
                    baseline_mean_obs = obs_matched.mean()
            except Exception as exc:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"[TWS] Invalid TWS_ANOMALY_BASELINE '{self.anomaly_baseline}': {exc}; using overlap mean")
                baseline_mean_sim = sim_matched.mean()
                baseline_mean_obs = obs_matched.mean()
        sim_anomaly = sim_matched - baseline_mean_sim
        obs_anomaly = obs_matched - baseline_mean_obs

        # Store raw values for diagnostics
        sim_anomaly_raw = sim_anomaly.copy()
        obs_anomaly_raw = obs_anomaly.copy()

        # Apply detrending if configured
        if self.detrend:
            sim_anomaly = self._detrend_series(sim_anomaly)
            obs_anomaly = self._detrend_series(obs_anomaly)

        # Apply scaling if configured
        if self.scale_to_obs:
            sim_anomaly = self._scale_to_obs_variability(sim_anomaly, obs_anomaly)

        return {
            'time': sim_matched.index,
            'sim_tws': sim_matched.values,
            'sim_anomaly': sim_anomaly.values,
            'sim_anomaly_raw': sim_anomaly_raw.values,
            'obs_anomaly': obs_anomaly.values,
            'obs_anomaly_raw': obs_anomaly_raw.values,
            'grace_column': self.grace_column,
            'grace_all_columns': obs_df.loc[common_index][valid_mask],
            'detrend_applied': self.detrend,
            'scale_applied': self.scale_to_obs
        }
