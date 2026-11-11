# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcing Data Processor

Shared utility for loading, subsetting, resampling, and transforming forcing data
across model preprocessors. Consolidates logic previously duplicated in FUSE,
NGEN, GR, and SUMMA preprocessors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.mixins.logging import _class_logger_name
from symfluence.core.modeling.forcing_naming import resolve_forcing_files

logger = logging.getLogger(__name__)


class ForcingDataProcessor:
    """
    Processes forcing data for hydrological model preprocessors.

    Handles:
    - Loading forcing files (single or multiple NetCDF)
    - Time subsetting to simulation windows
    - Temporal resampling (hourly to daily, etc.)
    - Variable mapping and renaming
    - Unit conversions
    """

    # Standard variable mappings (ERA5 style to common names)
    DEFAULT_VARIABLE_MAPPING = {
        'air_temperature': 'temp',
        'precipitation_flux': 'pr',
        'precipitation_rate': 'pr',
        'specific_humidity': 'specific_humidity',
        'wind_speed': 'wind_speed',
        'shortwave_radiation': 'surface_downwelling_shortwave_flux',
        'longwave_radiation': 'surface_downwelling_longwave_flux',
        'surface_pressure': 'surface_air_pressure',
    }

    # Standard unit conversions
    DEFAULT_UNIT_CONVERSIONS = {
        # Temperature: Kelvin to Celsius
        'temp_k_to_c': lambda x: x - 273.15,
        # Precipitation: kg/m2/s to mm/day
        'precip_rate_to_mm_day': lambda x: x * 86400,
        # Precipitation: mm/s to mm/day
        'precip_mm_s_to_mm_day': lambda x: x * 86400,
        # Wind: m/s (no conversion)
        'wind_identity': lambda x: x,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the forcing data processor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(
            _class_logger_name(self.__class__))

    def load_forcing_data(
        self,
        forcing_path: Path,
        pattern: str = '*.nc',
        concat_dim: str = 'time',
        data_vars: str = 'all',
        catchment_path: Optional[Path] = None,
    ) -> xr.Dataset:
        """
        Load forcing data from NetCDF files.

        Args:
            forcing_path: Path to directory containing forcing files
            pattern: Glob pattern for files
            concat_dim: Dimension to concatenate along
            data_vars: How to handle data variables ('all', 'minimal', 'different')
            catchment_path: The run's catchment shapefile; when it exists, files
                are selected by its HRU geometry as well as the discretization

        Returns:
            xr.Dataset with forcing data
        """
        forcing_files = sorted(forcing_path.glob(pattern))
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files matching '{pattern}' in {forcing_path}")

        # Scope to THIS run's spatial discretization so a store holding more than
        # one discretization of the domain (a lumped hru=1 forcing beside an
        # elevation hru=12 forcing) does not collapse into a conflicting-'hru'
        # merge below (issue #339). A no-op when the discretization is unknown or
        # the store's names predate namespacing, so single-discretization and
        # legacy stores never regress.
        forcing_files = resolve_forcing_files(
            forcing_files, self._get_discretization(), catchment_path,
            self._get_hru_id_field())

        self.logger.info(f"Loading {len(forcing_files)} forcing files from {forcing_path}")

        try:
            # Use open_mfdataset for efficiency with multiple files
            ds = xr.open_mfdataset(
                forcing_files,
                concat_dim=concat_dim,
                combine='nested',
                data_vars=data_vars,
                coords='minimal',
                compat='override'
            )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"open_mfdataset failed ({e}), falling back to manual concat", exc_info=True)
            datasets = [xr.open_dataset(f) for f in forcing_files]
            ds = xr.concat(datasets, dim=concat_dim, data_vars=data_vars)
            for d in datasets:
                d.close()

        return ds

    def subset_to_time_window(
        self,
        ds: xr.Dataset,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        time_var: str = 'time'
    ) -> xr.Dataset:
        """
        Subset dataset to a time window.

        Args:
            ds: Dataset to subset
            start_time: Start of time window
            end_time: End of time window
            time_var: Name of time coordinate

        Returns:
            Subsetted dataset
        """
        if time_var not in ds.coords and time_var not in ds.dims:
            self.logger.warning(f"Dataset has no '{time_var}' coordinate, skipping subset")
            return ds

        # Ensure unique time index if we have overlaps in loaded datasets
        if time_var in ds.dims and len(ds[time_var]) > 0:
            ds = ds.drop_duplicates(dim=time_var)

        original_len = len(ds[time_var])
        ds = ds.sel({time_var: slice(start_time, end_time)})
        new_len = len(ds[time_var])

        self.logger.debug(f"Subset forcing from {original_len} to {new_len} timesteps")

        if new_len == 0:
            raise ValueError(
                f"No forcing data in time window {start_time} to {end_time}"
            )

        return ds

    def validate_temporal_completeness(
        self,
        ds: xr.Dataset,
        expected_freq_seconds: Optional[int] = None,
        strategy: str = 'error',
        max_gap_hours: int = 24,
        time_var: str = 'time'
    ) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """
        Check forcing data for missing timesteps and act based on strategy.

        Args:
            ds: Dataset to validate
            expected_freq_seconds: Expected timestep in seconds.
                If None, inferred from data.
            strategy: How to handle gaps:
                - 'error': Raise ValueError on any gap
                - 'warn': Log warnings but continue
                - 'interpolate': Fill gaps with linear interpolation
                - 'fill_forward': Fill gaps with last known value
            max_gap_hours: Gaps exceeding this escalate to error regardless
                of strategy (safety limit)
            time_var: Name of time coordinate

        Returns:
            Tuple of (dataset, report) where report contains gap details

        Raises:
            ValueError: When gaps are found and strategy is 'error', or when
                any gap exceeds max_gap_hours
        """
        if time_var not in ds.coords and time_var not in ds.dims:
            return ds, {'valid': True, 'message': 'No time coordinate found, skipping validation'}

        times = pd.to_datetime(ds[time_var].values)
        if len(times) < 2:
            return ds, {'valid': True, 'message': 'Fewer than 2 timesteps, skipping validation'}

        times_sorted = times.sort_values()

        if expected_freq_seconds is None:
            deltas = np.diff(times_sorted).astype('timedelta64[s]').astype(int)
            expected_freq_seconds = int(np.median(deltas))

        expected_freq = pd.Timedelta(seconds=expected_freq_seconds)
        expected_index = pd.date_range(
            start=times_sorted[0], end=times_sorted[-1], freq=expected_freq
        )
        actual_set = set(times_sorted)
        missing = sorted(expected_index.difference(actual_set))

        report: Dict[str, Any] = {
            'valid': len(missing) == 0,
            'expected_timesteps': len(expected_index),
            'actual_timesteps': len(times_sorted),
            'missing_timesteps': len(missing),
            'expected_freq_seconds': expected_freq_seconds,
            'coverage_pct': len(times_sorted) / len(expected_index) * 100 if len(expected_index) > 0 else 100.0,
        }

        if not missing:
            self.logger.info(
                f"Forcing temporal completeness OK: {len(times_sorted)} timesteps, "
                f"{expected_freq_seconds}s interval, 100% coverage"
            )
            return ds, report

        gap_runs = self._identify_gap_runs(missing, expected_freq)
        report['gaps'] = gap_runs
        max_gap_found_hours = max(g['duration_hours'] for g in gap_runs) if gap_runs else 0
        report['max_gap_hours'] = max_gap_found_hours

        gap_summary = "; ".join(
            f"{g['start']} to {g['end']} ({g['missing_count']} steps, {g['duration_hours']:.1f}h)"
            for g in gap_runs[:5]
        )
        if len(gap_runs) > 5:
            gap_summary += f"; ... and {len(gap_runs) - 5} more gaps"

        msg = (
            f"Forcing data has {len(missing)} missing timesteps across {len(gap_runs)} gap(s) "
            f"(expected {expected_freq_seconds}s interval, {report['coverage_pct']:.1f}% coverage). "
            f"Gaps: {gap_summary}"
        )

        if max_gap_found_hours > max_gap_hours:
            raise ValueError(
                f"{msg}\nLargest gap ({max_gap_found_hours:.1f}h) exceeds FORCING_MAX_GAP_HOURS "
                f"({max_gap_hours}h). This gap is too large to safely fill — please fix the source data."
            )

        if strategy == 'error':
            raise ValueError(
                f"{msg}\nSet FORCING_MISSING_DATA_STRATEGY to 'warn', 'interpolate', or "
                f"'fill_forward' to handle gaps, or fix the source data."
            )
        elif strategy == 'warn':
            self.logger.warning(msg)
        elif strategy in ('interpolate', 'fill_forward'):
            self.logger.warning(f"{msg} — filling with strategy='{strategy}'")
            ds = self._fill_temporal_gaps(
                ds, expected_freq, times_sorted[0], times_sorted[-1],
                method=strategy, time_var=time_var
            )
            report['filled'] = True
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")

        return ds, report

    def _identify_gap_runs(
        self, missing: List[pd.Timestamp], expected_freq: pd.Timedelta
    ) -> List[Dict[str, Any]]:
        """Group consecutive missing timesteps into gap runs."""
        if not missing:
            return []

        runs: List[Dict[str, Any]] = []
        run_start = missing[0]
        prev = missing[0]

        for ts in missing[1:]:
            if ts - prev > expected_freq * 1.5:
                runs.append({
                    'start': str(run_start),
                    'end': str(prev),
                    'missing_count': int((prev - run_start) / expected_freq) + 1,
                    'duration_hours': (prev - run_start + expected_freq).total_seconds() / 3600,
                })
                run_start = ts
            prev = ts

        runs.append({
            'start': str(run_start),
            'end': str(prev),
            'missing_count': int((prev - run_start) / expected_freq) + 1,
            'duration_hours': (prev - run_start + expected_freq).total_seconds() / 3600,
        })
        return runs

    def _fill_temporal_gaps(
        self,
        ds: xr.Dataset,
        freq: pd.Timedelta,
        start: pd.Timestamp,
        end: pd.Timestamp,
        method: str = 'interpolate',
        time_var: str = 'time'
    ) -> xr.Dataset:
        """Reindex to complete time axis and fill gaps."""
        complete_index = pd.date_range(start=start, end=end, freq=freq)
        ds = ds.reindex({time_var: complete_index})
        if method == 'interpolate':
            ds = ds.interpolate_na(dim=time_var, method='linear')
        elif method == 'fill_forward':
            ds = ds.ffill(dim=time_var)
        return ds

    def resample_to_frequency(
        self,
        ds: xr.Dataset,
        target_freq: str = 'D',
        method: str = 'mean',
        time_var: str = 'time'
    ) -> xr.Dataset:
        """
        Resample dataset to target frequency.

        Reductions run with flox/numbagg/bottleneck disabled so the output is
        bit-identical regardless of which optional accelerators are installed.

        Args:
            ds: Dataset to resample
            target_freq: Target frequency ('D' for daily, 'h' for hourly, etc.)
            method: Aggregation method ('mean', 'sum', 'max', 'min')
            time_var: Name of time coordinate

        Returns:
            Resampled dataset
        """
        if time_var not in ds.coords and time_var not in ds.dims:
            self.logger.warning(f"Dataset has no '{time_var}' coordinate, skipping resample")
            return ds

        # Detect current frequency
        times = pd.to_datetime(ds[time_var].values)
        if len(times) > 1:
            current_freq = pd.infer_freq(times[:min(len(times), 100)])
            if current_freq and current_freq == target_freq:
                self.logger.debug(f"Data already at {target_freq} frequency, skipping resample")
                return ds

        self.logger.debug(f"Resampling forcing data to {target_freq} frequency using {method}")

        # Deliberately pin the plain-numpy reduction path: flox/numbagg/bottleneck
        # change summation order, so the resampled forcing would differ at the
        # floating-point level depending on which optional accelerators happen to
        # be installed (conda envs often pull flox in with xarray). Disabling them
        # keeps forcing bit-identical across pip/uv/conda/HPC environments. This
        # runs in one-time preprocessing, not the calibration loop, so the cost is
        # small. (RTI architecture review item 20: investigated, kept deliberately.)
        with xr.set_options(use_flox=False, use_numbagg=False, use_bottleneck=False):
            resampler = ds.resample({time_var: target_freq})
            if method == 'mean':
                ds = resampler.mean()
            elif method == 'sum':
                ds = resampler.sum()
            elif method == 'max':
                ds = resampler.max()
            elif method == 'min':
                ds = resampler.min()
            else:
                raise ValueError(f"Unknown resample method: {method}")

        return ds

    def apply_variable_mapping(
        self,
        ds: xr.Dataset,
        mapping: Optional[Dict[str, str]] = None,
        copy_data: bool = True
    ) -> xr.Dataset:
        """
        Apply variable name mapping to dataset.

        Args:
            ds: Dataset to modify
            mapping: Dict mapping source names to target names.
                     If None, uses DEFAULT_VARIABLE_MAPPING.
            copy_data: If True, creates new variables. If False, renames.

        Returns:
            Dataset with mapped variables
        """
        if mapping is None:
            mapping = self.DEFAULT_VARIABLE_MAPPING

        for source, target in mapping.items():
            if source in ds.data_vars and target not in ds.data_vars:
                if copy_data:
                    ds[target] = ds[source].copy()
                else:
                    ds = ds.rename({source: target})
                self.logger.debug(f"Mapped variable '{source}' -> '{target}'")

        return ds

    def apply_unit_conversion(
        self,
        ds: xr.Dataset,
        variable: str,
        conversion: Union[str, Callable],
        output_name: Optional[str] = None
    ) -> xr.Dataset:
        """
        Apply unit conversion to a variable.

        Args:
            ds: Dataset to modify
            variable: Variable name to convert
            conversion: Either a key from DEFAULT_UNIT_CONVERSIONS or a callable
            output_name: Name for converted variable (default: same as input)

        Returns:
            Dataset with converted variable
        """
        if variable not in ds.data_vars:
            self.logger.warning(f"Variable '{variable}' not found for unit conversion")
            return ds

        if isinstance(conversion, str):
            if conversion not in self.DEFAULT_UNIT_CONVERSIONS:
                raise ValueError(f"Unknown conversion: {conversion}")
            conversion_func = self.DEFAULT_UNIT_CONVERSIONS[conversion]
        else:
            conversion_func = conversion

        output_name = output_name or variable
        ds[output_name] = conversion_func(ds[variable])

        self.logger.debug(f"Applied unit conversion to '{variable}'")

        return ds

    def prepare_forcing_for_model(
        self,
        forcing_path: Path,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        target_freq: str = 'D',
        variable_mapping: Optional[Dict[str, str]] = None,
        unit_conversions: Optional[Dict[str, str]] = None,
        missing_data_strategy: Optional[str] = None,
        max_gap_hours: Optional[int] = None,
        expected_freq_seconds: Optional[int] = None,
        catchment_path: Optional[Path] = None,
    ) -> xr.Dataset:
        """
        Complete forcing preparation pipeline.

        Combines loading, subsetting, temporal validation, resampling,
        and variable mapping.

        Args:
            forcing_path: Path to forcing files
            start_time: Start of simulation window
            end_time: End of simulation window
            target_freq: Target temporal frequency
            variable_mapping: Variable name mapping dict
            unit_conversions: Dict mapping variable names to conversion keys
            missing_data_strategy: Gap handling ('error', 'warn',
                'interpolate', 'fill_forward'). Falls back to config then 'error'.
            max_gap_hours: Maximum gap before forced error. Falls back to
                config then 24.
            expected_freq_seconds: Expected timestep. If None, inferred.
            catchment_path: The run's catchment shapefile (see load_forcing_data).

        Returns:
            Prepared forcing dataset
        """
        # Load
        ds = self.load_forcing_data(forcing_path, catchment_path=catchment_path)

        # Subset
        ds = self.subset_to_time_window(ds, start_time, end_time)

        # Validate temporal completeness before resampling
        strategy = missing_data_strategy or self._get_config_str(
            'FORCING_MISSING_DATA_STRATEGY', 'error'
        )
        gap_limit = max_gap_hours or self._get_config_int(
            'FORCING_MAX_GAP_HOURS', 24
        )
        ds, _report = self.validate_temporal_completeness(
            ds,
            expected_freq_seconds=expected_freq_seconds,
            strategy=strategy,
            max_gap_hours=gap_limit,
        )

        # Resample
        ds = self.resample_to_frequency(ds, target_freq)

        # Map variables
        ds = self.apply_variable_mapping(ds, variable_mapping)

        # Apply unit conversions
        if unit_conversions:
            for var, conversion in unit_conversions.items():
                ds = self.apply_unit_conversion(ds, var, conversion)

        return ds

    def _get_discretization(self) -> Optional[str]:
        """Resolve the run's spatial discretization from dict or typed config.

        Reads ``SUB_GRID_DISCRETIZATION`` (canonical) with the legacy
        ``DOMAIN_DISCRETIZATION`` alias, or ``config.domain.discretization`` on a
        typed config. Returns ``None`` when unset, which makes forcing selection a
        no-op.
        """
        cfg = self.config
        if isinstance(cfg, dict):
            return cfg.get('SUB_GRID_DISCRETIZATION') or cfg.get('DOMAIN_DISCRETIZATION')
        domain = getattr(cfg, 'domain', None)
        disc = getattr(domain, 'discretization', None) if domain is not None else None
        if disc:
            return disc
        getter = getattr(cfg, 'get', None)
        if callable(getter):
            return cfg.get('SUB_GRID_DISCRETIZATION') or cfg.get('DOMAIN_DISCRETIZATION')
        return None

    def _get_hru_id_field(self) -> str:
        """The catchment HRU id column (``CATCHMENT_SHP_HRUID``), default ``HRU_ID``."""
        from symfluence.core.mixins.config import resolve_config_value

        cfg = self.config
        return str(resolve_config_value(
            cfg, lambda: cfg.paths.catchment_hruid, 'HRU_ID', 'CATCHMENT_SHP_HRUID') or 'HRU_ID')

    def _get_config_str(self, key: str, default: str) -> str:
        """Extract a string config value from config dict or typed config."""
        if isinstance(self.config, dict):
            return str(self.config.get(key, default))
        # Typed config: try forcing subsection
        forcing = getattr(self.config, 'forcing', None)
        if forcing is not None:
            field_map = {
                'FORCING_MISSING_DATA_STRATEGY': 'missing_data_strategy',
                'FORCING_MAX_GAP_HOURS': 'max_gap_hours',
            }
            attr = field_map.get(key)
            if attr and hasattr(forcing, attr):
                return str(getattr(forcing, attr))
        return default

    def _get_config_int(self, key: str, default: int) -> int:
        """Extract an int config value from config dict or typed config."""
        val = self._get_config_str(key, str(default))
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_spatial_mean(
        self,
        ds: xr.Dataset,
        variable: str,
        spatial_dims: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Calculate spatial mean of a variable.

        Args:
            ds: Dataset containing variable
            variable: Variable name
            spatial_dims: Spatial dimensions to average over.
                         Auto-detected if not provided.

        Returns:
            DataArray with spatial mean
        """
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")

        if spatial_dims is None:
            # Auto-detect spatial dimensions
            possible_spatial = ['latitude', 'longitude', 'lat', 'lon', 'x', 'y', 'hru']
            spatial_dims = [str(d) for d in ds[variable].dims if str(d) in possible_spatial]

        if not spatial_dims:
            return ds[variable]

        return ds[variable].mean(dim=spatial_dims)

    def align_forcing_to_observations(
        self,
        forcing_ds: xr.Dataset,
        obs_times: pd.DatetimeIndex,
        time_var: str = 'time'
    ) -> xr.Dataset:
        """
        Align forcing data to observation timestamps.

        Args:
            forcing_ds: Forcing dataset
            obs_times: Observation time index
            time_var: Name of time coordinate

        Returns:
            Forcing dataset aligned to observation times
        """
        # Find common time period
        forcing_times = pd.to_datetime(forcing_ds[time_var].values)
        common_start = max(forcing_times.min(), obs_times.min())
        common_end = min(forcing_times.max(), obs_times.max())

        self.logger.debug(f"Common period: {common_start} to {common_end}")

        return self.subset_to_time_window(
            forcing_ds,
            pd.Timestamp(common_start),
            pd.Timestamp(common_end),
            time_var
        )

    def create_encoding_dict(
        self,
        ds: xr.Dataset,
        fill_value: float = -9999.0,
        dtype: str = 'float32',
        compression: bool = False,
        complevel: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create encoding dictionary for NetCDF output.

        Args:
            ds: Dataset to create encoding for
            fill_value: Fill value for missing data
            dtype: Data type for variables
            compression: Whether to enable zlib compression
            complevel: Compression level (1-9)

        Returns:
            Encoding dict for ds.to_netcdf()
        """
        encoding = {}
        for var in ds.data_vars:
            var_name = str(var)
            var_encoding: Dict[str, Any] = {
                '_FillValue': fill_value,
                'dtype': dtype
            }
            if compression:
                var_encoding['zlib'] = True
                var_encoding['complevel'] = complevel
            encoding[var_name] = var_encoding

        return encoding

    # =========================================================================
    # Convenience Methods for Variable Extraction
    # =========================================================================

    # Standard variable name fallback sequences
    VARIABLE_FALLBACKS = {
        'precip': ['pr', 'precipitation_flux', 'precipitation', 'precip', 'rainfall', 'prcp', 'PRCP', 'PPT'],
        'temp': ['temp', 'air_temperature', 'air_temperature', 'tas', 'T2', 'tmean', 'TAVE'],
        'pet': ['pet', 'potevap', 'evspsblpot', 'PET', 'eto', 'potential_evapotranspiration'],
        'shortwave': ['surface_downwelling_shortwave_flux', 'sw_radiation', 'rsds', 'SWDOWN', 'ssrd', 'shortwave_radiation'],
        'longwave': ['surface_downwelling_longwave_flux', 'lw_radiation', 'rlds', 'LWDOWN', 'strd', 'longwave_radiation'],
        'humidity': ['specific_humidity', 'specific_humidity', 'q', 'QVAPOR', 'hus'],
        'wind': ['wind_speed', 'wind_speed', 'sfcWind', 'WIND', 'wspd', 'u10'],
        'pressure': ['surface_air_pressure', 'surface_pressure', 'ps', 'PSFC', 'sp'],
    }

    def extract_variable_with_fallback(
        self,
        ds: xr.Dataset,
        variable_type: str,
        fallback_names: Optional[List[str]] = None
    ) -> Optional[xr.DataArray]:
        """
        Extract variable from dataset using standard fallback sequences.

        Tries multiple common variable names in order until one is found.
        This consolidates the pattern of checking multiple variable names
        that was previously duplicated across model preprocessors.

        Args:
            ds: Dataset to extract variable from
            variable_type: Type of variable ('precip', 'temp', 'pet', 'shortwave',
                          'longwave', 'humidity', 'wind', 'pressure')
            fallback_names: Custom list of names to try. If None, uses
                           VARIABLE_FALLBACKS[variable_type]

        Returns:
            xr.DataArray if variable found, None otherwise

        Example:
            >>> processor.extract_variable_with_fallback(ds, 'temp')
            # Tries: temp, airtemp, air_temperature, tas, T2, tmean, TAVE
        """
        if fallback_names is None:
            if variable_type not in self.VARIABLE_FALLBACKS:
                self.logger.warning(
                    f"Unknown variable type '{variable_type}'. "
                    f"Known types: {list(self.VARIABLE_FALLBACKS.keys())}"
                )
                return None
            fallback_names = self.VARIABLE_FALLBACKS[variable_type]

        for var_name in fallback_names:
            if var_name in ds.data_vars:
                self.logger.debug(f"Found '{variable_type}' variable as '{var_name}'")
                return ds[var_name]

        self.logger.warning(
            f"Could not find '{variable_type}' variable. Tried: {fallback_names}. "
            f"Available: {list(ds.data_vars)}"
        )
        return None

    def standardize_temperature_units(
        self,
        temp_data: xr.DataArray,
        target_unit: str = 'celsius'
    ) -> xr.DataArray:
        """
        Convert temperature to Celsius if needed.

        Automatically detects if temperature is in Kelvin (values > 200)
        and converts to Celsius. This is a common pattern in hydrological
        model preprocessing.

        Args:
            temp_data: Temperature DataArray
            target_unit: Target unit ('celsius' or 'kelvin')

        Returns:
            Temperature DataArray in target units

        Example:
            >>> temp_c = processor.standardize_temperature_units(ds['temp'])
        """
        import numpy as np

        # Get a sample of values to detect units
        sample = temp_data.values.flatten()
        sample = sample[~np.isnan(sample)]

        if len(sample) == 0:
            self.logger.warning("Temperature data is all NaN, cannot detect units")
            return temp_data

        mean_temp = np.nanmean(sample[:1000])  # Sample first 1000 values

        # Heuristic: If mean > 200, likely Kelvin
        is_kelvin = mean_temp > 200

        if target_unit == 'celsius':
            if is_kelvin:
                self.logger.debug(f"Converting temperature from Kelvin to Celsius (mean={mean_temp:.1f}K)")
                return temp_data - 273.15
            else:
                self.logger.debug(f"Temperature already in Celsius (mean={mean_temp:.1f}°C)")
                return temp_data
        elif target_unit == 'kelvin':
            if not is_kelvin:
                self.logger.debug(f"Converting temperature from Celsius to Kelvin (mean={mean_temp:.1f}°C)")
                return temp_data + 273.15
            else:
                self.logger.debug(f"Temperature already in Kelvin (mean={mean_temp:.1f}K)")
                return temp_data
        else:
            raise ValueError(f"Unknown target unit: {target_unit}. Use 'celsius' or 'kelvin'")

    def extract_forcing_variables(
        self,
        ds: xr.Dataset,
        required_vars: List[str],
        optional_vars: Optional[List[str]] = None,
        standardize_temp: bool = True
    ) -> Dict[str, xr.DataArray]:
        """
        Extract multiple forcing variables with fallbacks and standardization.

        Convenience method that combines variable extraction and temperature
        standardization for common forcing preparation workflows.

        Args:
            ds: Source dataset
            required_vars: List of required variable types (e.g., ['precip', 'temp'])
            optional_vars: List of optional variable types
            standardize_temp: If True, convert temperature to Celsius

        Returns:
            Dict mapping variable types to DataArrays

        Raises:
            ValueError: If a required variable cannot be found

        Example:
            >>> vars = processor.extract_forcing_variables(
            ...     ds, required_vars=['precip', 'temp', 'pet']
            ... )
            >>> precip = vars['precip']
            >>> temp = vars['temp']  # Already in Celsius
        """
        result = {}

        # Extract required variables
        for var_type in required_vars:
            data = self.extract_variable_with_fallback(ds, var_type)
            if data is None:
                raise ValueError(
                    f"Required variable '{var_type}' not found in dataset. "
                    f"Tried: {self.VARIABLE_FALLBACKS.get(var_type, [var_type])}"
                )

            # Standardize temperature
            if standardize_temp and var_type == 'temp':
                data = self.standardize_temperature_units(data)

            result[var_type] = data

        # Extract optional variables
        if optional_vars:
            for var_type in optional_vars:
                data = self.extract_variable_with_fallback(ds, var_type)
                if data is not None:
                    if standardize_temp and var_type == 'temp':
                        data = self.standardize_temperature_units(data)
                    result[var_type] = data

        return result
