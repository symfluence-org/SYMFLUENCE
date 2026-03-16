#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Base Model Evaluator

This module provides the abstract base class for different evaluation variables.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.mixins import ConfigurableMixin
from symfluence.evaluation import metrics

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class ModelEvaluator(ConfigurableMixin, ABC):
    """
    Abstract base class for hydrological model evaluation.

    Provides standardized infrastructure for comparing simulated and observed
    data across different hydrological variables (streamflow, snow, ET, etc.).
    Handles time series alignment, period-based evaluation (calibration/validation),
    and multi-metric calculation using the centralized metrics module.

    Subclasses must implement:
        - get_simulation_files(): Locate model output files
        - extract_simulated_data(): Parse simulation results
        - get_observed_data_path(): Locate observation files
        - needs_routing(): Whether mizuRoute output is required
        - _get_observed_data_column(): Identify data column in obs files

    Attributes:
        config: SymfluenceConfig instance with typed access
        calibration_period: Tuple of (start, end) timestamps for calibration
        evaluation_period: Tuple of (start, end) timestamps for validation
        eval_timestep: Target timestep for comparison ('native', 'hourly', 'daily')
    """

    def __init__(
        self,
        config: 'SymfluenceConfig',
        project_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self._project_dir = project_dir or Path(".")
        self._logger = logger

        # Parse time periods from typed config (with dict_key fallback for worker dicts)
        calibration_period_str = self._get_config_value(
            lambda: self.config.domain.calibration_period,
            default='',
            dict_key='CALIBRATION_PERIOD'
        )
        evaluation_period_str = self._get_config_value(
            lambda: self.config.domain.evaluation_period,
            default='',
            dict_key='EVALUATION_PERIOD'
        )
        self.calibration_period: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = self._parse_date_range(calibration_period_str)
        self.evaluation_period: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = self._parse_date_range(evaluation_period_str)

        # Parse calibration/evaluation timestep (with dict_key fallback for worker dicts)
        self.eval_timestep = self._get_config_value(
            lambda: self.config.optimization.calibration_timestep,
            default='native',
            dict_key='CALIBRATION_TIMESTEP'
        ).lower()
        if self.eval_timestep not in ['native', 'hourly', 'daily']:
            self.logger.warning(
                f"Invalid calibration_timestep '{self.eval_timestep}'. "
                "Using 'native'. Valid options: 'native', 'hourly', 'daily'"
            )
            self.eval_timestep = 'native'

        if self.eval_timestep != 'native':
            self.logger.debug(f"Evaluation will use {self.eval_timestep} timestep")

    @property
    def variable_type(self) -> str:
        """Return the variable type for resampling behavior.

        Override in subclasses for flux variables (precipitation, ET) that
        should use sum aggregation instead of mean.

        Returns:
            'state' (default) for state variables - use mean aggregation
            'flux' for flux/accumulation variables - use sum aggregation
        """
        return 'state'

    def evaluate(self, sim: Any, obs: Optional[pd.Series] = None,
                 mizuroute_dir: Optional[Path] = None,
                 calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """Alias for calculate_metrics for consistency with other parts of the system"""
        return self.calculate_metrics(sim, obs, mizuroute_dir, calibration_only)

    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None,
                         mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """
        Calculate performance metrics for this target.

        Args:
            sim: Either a Path to simulation directory or a pre-loaded pd.Series
            obs: Optional pre-loaded pd.Series of observations. If None, loads from file.
            mizuroute_dir: mizuRoute simulation directory (if needed and sim is Path)
            calibration_only: If True, only calculate calibration period metrics
        """
        try:
            # 1. Prepare simulated data
            if isinstance(sim, (str, Path)):
                sim_dir = Path(sim)
                # Determine which simulation directory to use
                if self.needs_routing() and mizuroute_dir:
                    output_dir = mizuroute_dir
                else:
                    output_dir = sim_dir

                # Get simulation files
                sim_files = self.get_simulation_files(output_dir)
                if not sim_files:
                    self.logger.error(f"No simulation files found in {output_dir}")
                    return None

                # Extract simulated data
                sim_data = self.extract_simulated_data(sim_files)
                self.logger.debug(f"Extracted {len(sim_data)} simulated data points from {len(sim_files)} file(s)")
            else:
                sim_data = sim

            if sim_data is None:
                self.logger.error("Failed to extract simulated data")
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
                self.logger.error("Failed to load observed data (check path and column names)")
                return None

            # Validate observed data
            is_valid, error_msg = self._validate_data(obs_data, 'observed')
            if not is_valid:
                self.logger.error(error_msg)
                return None

            self.logger.debug(f"Loaded {len(obs_data)} observed data points")

            # 3. Align time series and calculate metrics
            metrics_dict = {}

            # Always calculate metrics for calibration period if available
            if self.calibration_period[0] and self.calibration_period[1]:
                calib_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.calibration_period, "Calib"
                )
                metrics_dict.update(calib_metrics)

                # Also add unprefixed versions for the primary (calibration) period
                # to support model runners/loggers expecting simple names
                for k, v in calib_metrics.items():
                    unprefixed_key = k.replace("Calib_", "")
                    if unprefixed_key not in metrics_dict:
                        metrics_dict[unprefixed_key] = v

            # Only calculate evaluation period metrics if requested (final evaluation)
            if not calibration_only and self.evaluation_period[0] and self.evaluation_period[1]:
                eval_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.evaluation_period, "Eval"
                )
                metrics_dict.update(eval_metrics)

            # If no specific periods, calculate for full overlap (fallback)
            if not metrics_dict:
                full_metrics = self._calculate_period_metrics(obs_data, sim_data, (None, None), "")
                metrics_dict.update(full_metrics)

            return metrics_dict

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error calculating metrics for {self.__class__.__name__}: {str(e)}")
            return None

    @abstractmethod
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get relevant simulation output files for this target"""
        pass

    @abstractmethod
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract simulated data from output files"""
        pass

    @abstractmethod
    def get_observed_data_path(self) -> Path:
        """Get path to observed data file"""
        pass

    @abstractmethod
    def needs_routing(self) -> bool:
        """Whether this target requires mizuRoute routing"""
        pass

    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed data from file"""
        try:
            obs_path = self.get_observed_data_path()
            return self._load_observed_data_from_path(obs_path)

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error loading observed data: {str(e)}")
            return None

    def _load_observed_data_from_path(self, obs_path: Path) -> Optional[pd.Series]:
        """Load observed data from a specific path."""
        if not obs_path.exists():
            self.logger.error(f"Observed data file not found: {obs_path}")
            return None

        # Model-ready NetCDF store
        if obs_path.suffix == '.nc':
            return self._load_observed_data_from_netcdf(obs_path)

        # Try to read with index_col=0 first (handles GRACE/TWS files where date is first column)
        try:
            obs_df = pd.read_csv(obs_path, index_col=0)
            obs_df.index = pd.to_datetime(obs_df.index, format='mixed', dayfirst=True)

            # Check if index looks like dates
            if isinstance(obs_df.index, pd.DatetimeIndex):
                # Index is already a datetime, use it directly
                data_col = self._get_observed_data_column(obs_df.columns)
                if data_col:
                    self.logger.debug(f"Loaded {obs_path} with date index, data column: {data_col}")
                    return obs_df[data_col]

            # Check for "Unnamed: 0" or numeric index that might be dates
            if obs_df.index.name == 'Unnamed: 0' or obs_df.index.name is None:
                try:
                    obs_df.index = pd.to_datetime(obs_df.index)
                    if isinstance(obs_df.index, pd.DatetimeIndex):
                        data_col = self._get_observed_data_column(obs_df.columns)
                        if data_col:
                            self.logger.debug(f"Loaded {obs_path} with parsed date index, data column: {data_col}")
                            return obs_df[data_col]
                except (ValueError, TypeError):
                    pass  # Index is not a date, fall through to standard parsing

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.debug(f"Could not parse {obs_path} with index_col=0: {e}")

        # Fallback: Standard CSV read with explicit date column search
        obs_df = pd.read_csv(obs_path)

        # Find date and data columns
        date_col = next((col for col in obs_df.columns
                         if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)

        data_col = self._get_observed_data_column(obs_df.columns)

        if not date_col or not data_col:
            self.logger.error(f"Could not identify date/data columns in {obs_path}. "
                            f"Columns: {list(obs_df.columns)}")
            return None

        # Process data
        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
        obs_df.set_index('DateTime', inplace=True)

        return obs_df[data_col]

    def _load_observed_data_from_netcdf(self, nc_path: Path) -> Optional[pd.Series]:
        """Load observed data from the model-ready grouped NetCDF store."""
        try:
            import xarray as xr

            group = self._get_observation_group()
            ds = xr.open_dataset(nc_path, group=group)

            # Take the first non-coordinate data variable
            data_vars = [v for v in ds.data_vars if v not in ('gauge_id', 'hru_id', 'station_id', 'basin_id')]
            if not data_vars:
                self.logger.error(f"No data variables in {nc_path} group {group}")
                ds.close()
                return None

            var_name = data_vars[0]
            da = ds[var_name]

            # Collapse spatial dim if present (take first or squeeze)
            spatial_dims = [d for d in da.dims if d != 'time']
            if spatial_dims:
                da = da.isel({spatial_dims[0]: 0})

            series = da.to_series().dropna()
            series.name = var_name
            ds.close()
            return series

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error reading NetCDF observations from {nc_path}: {e}")
            return None

    def _get_observation_group(self) -> str:
        """Return the NetCDF group name for this evaluator type.

        Subclasses can override this to point to their observation group.
        Default mapping is based on the class name.
        """
        class_name = self.__class__.__name__.lower()
        if 'streamflow' in class_name:
            return 'streamflow'
        elif 'snow' in class_name:
            return 'snow'
        elif 'et' in class_name or 'evapotranspiration' in class_name:
            return 'et'
        elif 'soil' in class_name:
            return 'soil_moisture'
        elif 'tws' in class_name or 'groundwater' in class_name:
            return 'terrestrial_water_storage'
        return 'streamflow'

    def _validate_data(
        self,
        data: pd.Series,
        data_name: str,
        min_valid_points: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """Validate data series for quality issues.

        Performs comprehensive validation of input data before metric calculation
        to catch common data quality issues early with clear error messages.

        Validation Checks:
            1. None check: Data must not be None
            2. Empty check: Data must have at least one element
            3. All-NaN check: Data must have at least one valid (non-NaN) value
            4. Minimum points: Must have at least min_valid_points valid values
            5. Constant warning: Logs warning if all values are identical

        Args:
            data: pandas Series to validate
            data_name: Human-readable name for error messages (e.g., 'simulated', 'observed')
            min_valid_points: Minimum number of non-NaN points required (default: 10)

        Returns:
            Tuple of (is_valid, error_message):
                - (True, None) if data passes all checks
                - (False, "error description") if validation fails

        Example:
            is_valid, error_msg = self._validate_data(sim_data, 'simulated')
            if not is_valid:
                self.logger.error(error_msg)
                return None
        """
        if data is None:
            return False, f"{data_name} data is None"

        if len(data) == 0:
            return False, f"{data_name} data is empty"

        valid_count = data.notna().sum()
        if valid_count == 0:
            return False, f"{data_name} data contains only NaN values"

        if valid_count < min_valid_points:
            return False, (
                f"{data_name} data has insufficient valid points: "
                f"{valid_count} < {min_valid_points}"
            )

        # Warn about constant data (some metrics undefined)
        if data.dropna().nunique() == 1:
            self.logger.warning(
                f"{data_name} data is constant (all values = {data.dropna().iloc[0]:.4g}). "
                "Some metrics (e.g., correlation, NSE) may be undefined."
            )

        return True, None

    @abstractmethod
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the data column in observed data file"""
        pass

    def _calculate_period_metrics(self, obs_data: pd.Series, sim_data: pd.Series,
                                period: Tuple, prefix: str) -> Dict[str, float]:
        """Calculate metrics for a specific time period with explicit filtering"""
        try:
            # Ensure indices are DatetimeIndex
            if not isinstance(obs_data.index, pd.DatetimeIndex):
                obs_data.index = pd.to_datetime(obs_data.index)
            if not isinstance(sim_data.index, pd.DatetimeIndex):
                sim_data.index = pd.to_datetime(sim_data.index)

            # EXPLICIT filtering for both datasets (consistent with parallel worker)
            # Round BOTH indices to ensure alignment (fixes misalignment with daily data)
            if period[0] and period[1]:
                # Round both observed and simulated indices consistently
                obs_data_rounded = obs_data.copy()
                obs_data_rounded.index = obs_data_rounded.index.round('h')
                sim_data_rounded = sim_data.copy()
                sim_data_rounded.index = sim_data_rounded.index.round('h')

                # Filter observed data to period
                obs_period_mask = (obs_data_rounded.index >= period[0]) & (obs_data_rounded.index <= period[1])
                obs_period = obs_data_rounded[obs_period_mask].copy()

                # Explicitly filter simulated data to same period (like parallel worker)
                sim_period_mask = (sim_data_rounded.index >= period[0]) & (sim_data_rounded.index <= period[1])
                sim_period = sim_data_rounded[sim_period_mask].copy()

                # Apply spinup removal if configured
                spinup_years = self._get_config_value(
                    lambda: self.config.evaluation.spinup_years,
                    default=0,
                    dict_key='EVALUATION_SPINUP_YEARS'
                )
                try:
                    spinup_years = int(float(spinup_years))
                except (TypeError, ValueError):
                    spinup_years = 0

                if spinup_years > 0 and not obs_period.empty:
                    cutoff = obs_period.index.min() + pd.DateOffset(years=spinup_years)
                    obs_period = obs_period[obs_period.index >= cutoff]
                    sim_period = sim_period[sim_period.index >= cutoff]
                    self.logger.debug(f"Applied {spinup_years} year spinup removal, cutoff: {cutoff}")

                # Log filtering results for debugging
                self.logger.debug(f"{prefix} period filtering: {period[0]} to {period[1]}")
                self.logger.debug(f"{prefix} observed points in period: {len(obs_period)}")
                self.logger.debug(f"{prefix} simulated points in period: {len(sim_period)}")
            else:
                # Round BOTH indices consistently for alignment
                obs_period = obs_data.copy()
                obs_period.index = obs_period.index.round('h')
                sim_period = sim_data.copy()
                sim_period.index = sim_period.index.round('h')

            # ENHANCED: Normalize timezones before intersection to ensure match
            # Some datasets come from NetCDF (UTC) while others from CSV (naive)
            if obs_period.index.tz is not None:
                obs_period.index = obs_period.index.tz_localize(None)
            if sim_period.index.tz is not None:
                sim_period.index = sim_period.index.tz_localize(None)

            # Resample to evaluation timestep if specified in config
            if self.eval_timestep != 'native':
                self.logger.debug(f"Resampling data to {self.eval_timestep} timestep")
                # DEBUG: Log before resampling
                if len(obs_period) > 0 and len(sim_period) > 0:
                    self.logger.debug(f"BEFORE resample - obs: {obs_period.min():.3f} to {obs_period.max():.3f}, sim: {sim_period.min():.3f} to {sim_period.max():.3f}")

                obs_period = self._resample_to_timestep(obs_period, self.eval_timestep)
                sim_period = self._resample_to_timestep(sim_period, self.eval_timestep)

                self.logger.debug(f"After resampling - obs points: {len(obs_period)}, sim points: {len(sim_period)}")
                # DEBUG: Log after resampling
                if len(obs_period) > 0 and len(sim_period) > 0:
                    self.logger.debug(f"AFTER resample - obs: {obs_period.min():.3f} to {obs_period.max():.3f}, sim: {sim_period.min():.3f} to {sim_period.max():.3f}")

            # Final check: ensure both are midnight-aligned if daily
            if self.eval_timestep == 'daily':
                obs_period.index = obs_period.index.normalize()
                sim_period.index = sim_period.index.normalize()

            # Find common time indices
            common_idx = obs_period.index.intersection(sim_period.index)

            if len(common_idx) == 0:
                self.logger.debug(f"No common time indices for {prefix} period")
                if len(obs_period) > 0 and len(sim_period) > 0:
                    self.logger.debug(f"Obs index sample: {obs_period.index[0]} to {obs_period.index[-1]} (type: {obs_period.index.dtype})")
                    self.logger.debug(f"Sim index sample: {sim_period.index[0]} to {sim_period.index[-1]} (type: {sim_period.index.dtype})")
                return {}

            obs_common = obs_period.loc[common_idx]
            sim_common = sim_period.loc[common_idx]

            # Log final aligned data for debugging
            self.logger.debug(f"{prefix} aligned data points: {len(common_idx)}")
            self.logger.debug(f"{prefix} obs mean: {obs_common.mean():.4f}, range: {obs_common.min():.4f} to {obs_common.max():.4f}")
            self.logger.debug(f"{prefix} sim mean: {sim_common.mean():.4f}, range: {sim_common.min():.4f} to {sim_common.max():.4f}")

            # Calculate metrics
            base_metrics = self._calculate_performance_metrics(obs_common, sim_common)

            # Optionally compute log-likelihood if observation uncertainties are available
            likelihood_metrics = self._calculate_likelihood_metrics(obs_common, sim_common)
            if likelihood_metrics:
                base_metrics.update(likelihood_metrics)

            # Add prefix if specified
            if prefix:
                return {f"{prefix}_{k}": v for k, v in base_metrics.items()}
            else:
                return base_metrics

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error calculating period metrics: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}

    def _resample_to_timestep(self, data: pd.Series, target_timestep: str) -> pd.Series:
        """
        Resample time series data to target timestep.

        Aggregation (fine → coarse) uses mean for state variables (temperature,
        storage) and sum for flux variables (precipitation, ET).

        Upsampling (coarse → fine) is rejected as it creates synthetic data
        through interpolation, which is inappropriate for observations.

        Args:
            data: Time series data with DatetimeIndex
            target_timestep: Target timestep ('hourly' or 'daily')

        Returns:
            Resampled time series

        Raises:
            ValueError: If upsampling is attempted (coarse to fine resolution)
        """
        if target_timestep == 'native' or data is None or len(data) == 0:
            return data

        try:
            # Infer current frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq is None:
                # Try to infer from first few differences
                if len(data) > 1:
                    time_diff = data.index[1] - data.index[0]
                    self.logger.debug(f"Inferred time difference: {time_diff}")
                else:
                    self.logger.warning("Cannot infer frequency from single data point")
                    return data
            else:
                self.logger.debug(f"Inferred frequency: {inferred_freq}")

            # Determine current timestep
            time_diff = data.index[1] - data.index[0] if len(data) > 1 else pd.Timedelta(hours=1)

            # Check if already at target timestep
            if target_timestep == 'hourly' and pd.Timedelta(minutes=45) <= time_diff <= pd.Timedelta(minutes=75):
                self.logger.debug("Data already at hourly timestep")
                return data
            elif target_timestep == 'daily' and pd.Timedelta(hours=20) <= time_diff <= pd.Timedelta(hours=28):
                self.logger.debug("Data already at daily timestep")
                return data

            # Determine aggregation function based on variable type
            agg_func = 'sum' if self.variable_type == 'flux' else 'mean'

            # Perform resampling
            if target_timestep == 'hourly':
                if time_diff < pd.Timedelta(hours=1):
                    # Aggregation: sub-hourly to hourly
                    self.logger.debug(f"Aggregating {time_diff} data to hourly using {agg_func}")
                    resampled = pd.Series(data.resample('h').agg(agg_func))
                elif time_diff > pd.Timedelta(hours=1):
                    # Upsampling: daily/coarser to hourly - REJECT
                    raise ValueError(
                        f"Cannot upsample {time_diff} data to hourly: "
                        f"interpolation creates synthetic observations. "
                        f"Use native timestep or aggregated (coarser) timestep instead."
                    )
                else:
                    resampled = data

            elif target_timestep == 'daily':
                if time_diff < pd.Timedelta(days=1):
                    # Aggregation: hourly/sub-daily to daily
                    self.logger.debug(f"Aggregating {time_diff} data to daily using {agg_func}")
                    resampled = pd.Series(data.resample('D').agg(agg_func))
                elif time_diff > pd.Timedelta(days=1):
                    # Upsampling: weekly/monthly to daily - REJECT
                    raise ValueError(
                        f"Cannot upsample {time_diff} data to daily: "
                        f"interpolation creates synthetic observations."
                    )
                else:
                    resampled = data
            else:
                resampled = data

            # Remove any NaN values introduced by resampling at edges
            resampled = resampled.dropna()

            self.logger.debug(
                f"Resampled from {len(data)} to {len(resampled)} points "
                f"(target: {target_timestep}, agg: {agg_func})"
            )

            return resampled

        except ValueError:
            # Re-raise ValueError for upsampling rejection
            raise
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error resampling to {target_timestep}: {str(e)}")
            self.logger.warning("Returning original data without resampling")
            return data

    def _calculate_likelihood_metrics(
        self,
        obs_common: pd.Series,
        sim_common: pd.Series,
    ) -> Dict[str, float]:
        """
        Optionally compute Gaussian log-likelihood using observation uncertainties.

        This is activated when the config specifies LIKELIHOOD_FUNCTION (e.g., 'gaussian').
        Observation uncertainties are loaded from the flux data file (_uc columns)
        and combined with model error to form the total error variance.

        Returns empty dict if likelihood mode is not enabled or uncertainties
        are unavailable, preserving backward compatibility.
        """
        likelihood_function = self._get_config_value(
            lambda: self.config.optimization.likelihood_function,
            default='',
            dict_key='LIKELIHOOD_FUNCTION'
        )
        if not likelihood_function:
            return {}

        try:
            from symfluence.evaluation.likelihood import gaussian_log_likelihood

            # Load observation uncertainty aligned to the common index
            obs_uc = self._load_observation_uncertainty(obs_common.index)

            # Get model error configuration
            model_error_fraction = self._get_config_value(
                lambda: self.config.optimization.model_error_fraction,
                default=0.0,
                dict_key='MODEL_ERROR_FRACTION'
            )
            try:
                model_error_fraction = float(model_error_fraction)
            except (TypeError, ValueError):
                model_error_fraction = 0.0

            model_error_base = self._get_config_value(
                lambda: self.config.optimization.model_error_base,
                default=0.0,
                dict_key='MODEL_ERROR_BASE'
            )
            try:
                model_error_base = float(model_error_base)
            except (TypeError, ValueError):
                model_error_base = 0.0

            obs_arr = obs_common.values.astype(np.float64)
            sim_arr = sim_common.values.astype(np.float64)

            # Build model error: sigma_model = base + fraction * |sim|
            sigma_model = None
            if model_error_base > 0 or model_error_fraction > 0:
                sigma_model = model_error_base + model_error_fraction * np.abs(sim_arr)

            # Observation uncertainty array (or None)
            obs_uc_arr = None
            if obs_uc is not None and len(obs_uc) == len(obs_arr):
                obs_uc_arr = obs_uc.values.astype(np.float64)

            log_lik = gaussian_log_likelihood(
                obs_arr, sim_arr,
                obs_uncertainty=obs_uc_arr,
                model_error=sigma_model,
            )

            return {'log_likelihood': log_lik}

        except Exception as e:  # noqa: BLE001 — optional feature, must not break metrics
            self.logger.debug(f"Likelihood computation skipped: {e}")
            return {}

    def _load_observation_uncertainty(
        self, time_index: pd.DatetimeIndex
    ) -> Optional[pd.Series]:
        """
        Load observation uncertainty data aligned to the given time index.

        Override in subclasses that know how to locate uncertainty data
        (e.g., FLUXNET _uc columns). Default returns None (no uncertainty).
        """
        return None

    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics between observed and simulated data"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')

            # Use centralized metrics module for all calculations
            result = metrics.calculate_all_metrics(observed, simulated)

            # Return subset of metrics for compatibility
            return {
                'KGE': result['KGE'],
                'NSE': result['NSE'],
                'RMSE': result['RMSE'],
                'PBIAS': result['PBIAS'],
                'MAE': result['MAE'],
                'correlation': result['correlation'],
                'r': result['r'],
                'alpha': result['alpha'],
                'beta': result['beta']
            }

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'KGE': np.nan,
                'NSE': np.nan,
                'RMSE': np.nan,
                'PBIAS': np.nan,
                'MAE': np.nan,
                'correlation': np.nan,
            }

    def _parse_date_range(self, date_range_str: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse date range string from config"""
        if not date_range_str:
            return None, None

        try:
            dates = [d.strip() for d in date_range_str.split(',')]
            if len(dates) >= 2:
                return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.warning(f"Could not parse date range '{date_range_str}': {str(e)}")

        return None, None

    def align_series(self, sim: pd.Series, obs: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align simulation and observation series after dropping spinup years."""
        spinup_years = self._get_config_value(
            lambda: self.config.evaluation.spinup_years,
            default=0,
            dict_key='EVALUATION_SPINUP_YEARS'
        )
        try:
            spinup_years = int(float(spinup_years))
        except (TypeError, ValueError):
            spinup_years = 0
        spinup_years = max(0, spinup_years)

        if sim.empty or obs.empty:
            return sim, obs

        common_start = max(sim.index.min(), obs.index.min())
        cutoff = common_start + pd.DateOffset(years=spinup_years) if spinup_years else common_start

        sim_trimmed = sim[sim.index >= cutoff]
        obs_trimmed = obs[obs.index >= cutoff]

        common_idx = sim_trimmed.index.intersection(obs_trimmed.index)
        if not common_idx.empty:
            sim_trimmed = sim_trimmed.loc[common_idx]
            obs_trimmed = obs_trimmed.loc[common_idx]
        else:
            self.logger.warning("No overlapping indices after alignment; returning trimmed series")

        return sim_trimmed, obs_trimmed

    def _collapse_spatial_dims(self, data_array: xr.DataArray, aggregate: str = 'mean') -> pd.Series:
        """
        Collapse spatial dimensions from xarray DataArray to pandas Series.

        Handles common spatial dimension patterns in SUMMA/FUSE/NGEN output:
        - Single HRU/GRU: select index 0
        - Multiple HRU/GRU: aggregate (mean by default)
        - Other spatial dims: select first or aggregate

        Args:
            data_array: xarray DataArray with time and possibly spatial dimensions
            aggregate: Aggregation method for multiple spatial units ('mean', 'sum', 'first')

        Returns:
            pandas Series with time index
        """
        spatial_dims = ['hru', 'gru', 'param_set', 'latitude', 'longitude', 'seg', 'reachID']

        result = data_array

        for dim in spatial_dims:
            if dim in result.dims:
                dim_size = result.shape[result.dims.index(dim)]
                if dim_size == 1:
                    result = result.isel({dim: 0})
                elif aggregate == 'mean':
                    result = result.mean(dim=dim)
                elif aggregate == 'sum':
                    result = result.sum(dim=dim)
                elif aggregate == 'first':
                    result = result.isel({dim: 0})

        # Handle any remaining non-time dimensions
        non_time_dims = [dim for dim in result.dims if dim != 'time']
        for dim in non_time_dims:
            dim_size = result.shape[result.dims.index(dim)]
            if dim_size == 1:
                result = result.isel({dim: 0})
            elif aggregate == 'mean':
                result = result.mean(dim=dim)
            elif aggregate == 'sum':
                result = result.sum(dim=dim)
            else:
                result = result.isel({dim: 0})

        return cast(pd.Series, result.to_pandas())

    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """
        Find timestamp/date column in a DataFrame.

        Searches for common date column names used across different data sources.

        Args:
            columns: List of column names from DataFrame

        Returns:
            Name of date column, or None if not found
        """
        # Priority order for timestamp column candidates
        timestamp_candidates = [
            'timestamp', 'TIMESTAMP_START', 'TIMESTAMP_END',
            'datetime', 'DateTime', 'time', 'Time',
            'date', 'Date', 'DATE'
        ]

        # First check exact matches
        for candidate in timestamp_candidates:
            if candidate in columns:
                return candidate

        # Then check partial matches
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['timestamp', 'datetime', 'date', 'time']):
                return col

        return None
