# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data Quality Handler

Shared utility for handling data quality issues across model preprocessors.
Consolidates NaN handling, fill value logic, and data validation that was
previously duplicated across FUSE, GR, and other preprocessors.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class DataQualityHandler:
    """
    Handles data quality issues for model preprocessors.

    Provides:
    - NaN detection and replacement
    - Fill value management
    - Data range validation
    - Outlier detection
    """

    # Standard fill values by data type
    DEFAULT_FILL_VALUES = {
        'float32': -9999.0,
        'float64': -9999.0,
        'int32': -9999,
        'int64': -9999,
        'default': -9999.0
    }

    # Variable-specific valid ranges
    VALID_RANGES = {
        'temp': (-100, 60),           # Temperature in Celsius
        'air_temperature': (150, 350),        # Temperature in Kelvin
        'pr': (0, 1000),              # Precipitation mm/day
        'precipitation_flux': (0, 0.01),         # Precipitation rate kg/m2/s
        'specific_humidity': (0, 0.05),         # Specific humidity kg/kg
        'wind_speed': (0, 100),          # Wind speed m/s
        'surface_downwelling_shortwave_flux': (0, 1500),        # Shortwave radiation W/m2
        'surface_downwelling_longwave_flux': (0, 700),         # Longwave radiation W/m2
        'surface_air_pressure': (50000, 110000),   # Pressure Pa
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data quality handler.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def handle_nan_values(
        self,
        data: Union[np.ndarray, xr.DataArray],
        fill_value: Optional[float] = None,
        method: str = 'fill'
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Handle NaN values in data.

        Args:
            data: Input data array
            fill_value: Value to use for NaN replacement
            method: Handling method:
                   - 'fill': Replace with fill_value
                   - 'interpolate': Linear interpolation (time series)
                   - 'forward_fill': Forward fill
                   - 'drop': Remove NaN values (returns mask)

        Returns:
            Data with NaN values handled
        """
        if fill_value is None:
            dtype = str(data.dtype) if hasattr(data, 'dtype') else 'default'
            fill_value = self.DEFAULT_FILL_VALUES.get(dtype, self.DEFAULT_FILL_VALUES['default'])

        if isinstance(data, xr.DataArray):
            return self._handle_nan_xarray(data, fill_value, method)
        else:
            return self._handle_nan_numpy(np.asarray(data), fill_value, method)

    def _handle_nan_numpy(
        self,
        data: np.ndarray,
        fill_value: float,
        method: str
    ) -> np.ndarray:
        """Handle NaN values in numpy array."""
        if not np.any(np.isnan(data)):
            return data

        nan_count = np.sum(np.isnan(data))
        self.logger.debug(f"Found {nan_count} NaN values in data")

        if method == 'fill':
            return np.nan_to_num(data, nan=fill_value)
        elif method == 'interpolate':
            # Linear interpolation along first axis
            result = data.copy()
            if data.ndim == 1:
                mask = ~np.isnan(data)
                if np.any(mask):
                    result = np.interp(
                        np.arange(len(data)),
                        np.where(mask)[0],
                        data[mask]
                    )
            return result
        elif method == 'forward_fill':
            result = data.copy()
            mask = np.isnan(result)
            idx = np.where(~mask, np.arange(mask.shape[-1]), 0)
            np.maximum.accumulate(idx, axis=-1, out=idx)
            return result[..., idx[..., :]]
        else:
            return np.nan_to_num(data, nan=fill_value)

    def _handle_nan_xarray(
        self,
        data: xr.DataArray,
        fill_value: float,
        method: str
    ) -> xr.DataArray:
        """Handle NaN values in xarray DataArray."""
        if not data.isnull().any():
            return data

        nan_count = data.isnull().sum().values
        self.logger.debug(f"Found {nan_count} NaN values in DataArray")

        if method == 'fill':
            return data.fillna(fill_value)
        elif method == 'interpolate':
            return data.interpolate_na(dim='time', method='linear')
        elif method == 'forward_fill':
            return data.ffill(dim='time')
        else:
            return data.fillna(fill_value)

    def validate_data_range(
        self,
        data: Union[np.ndarray, xr.DataArray],
        variable: Optional[str] = None,
        valid_range: Optional[Tuple[float, float]] = None,
        clip: bool = False
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Dict[str, Any]]:
        """
        Validate data is within expected range.

        Args:
            data: Data to validate
            variable: Variable name (for auto range lookup)
            valid_range: (min, max) tuple, or auto-detected from variable
            clip: If True, clip values to range instead of flagging

        Returns:
            Tuple of (validated_data, validation_report)
        """
        if valid_range is None and variable:
            valid_range = self.VALID_RANGES.get(variable)

        if valid_range is None:
            return data, {'valid': True, 'message': 'No range validation performed'}

        min_val, max_val = valid_range

        if isinstance(data, xr.DataArray):
            data_min = float(data.min())
            data_max = float(data.max())
            below_count = int((data < min_val).sum())
            above_count = int((data > max_val).sum())
        else:
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            below_count = np.sum(data < min_val)
            above_count = np.sum(data > max_val)

        report = {
            'valid': below_count == 0 and above_count == 0,
            'data_min': data_min,
            'data_max': data_max,
            'valid_range': valid_range,
            'below_range_count': below_count,
            'above_range_count': above_count
        }

        if not report['valid']:
            self.logger.warning(
                f"Data range validation failed: {data_min:.2f} to {data_max:.2f}, "
                f"expected {min_val} to {max_val}"
            )
            if clip:
                if isinstance(data, xr.DataArray):
                    data = data.clip(min=min_val, max=max_val)
                else:
                    data = np.clip(data, min_val, max_val)
                report['clipped'] = True

        return data, report

    def detect_outliers(
        self,
        data: Union[np.ndarray, xr.DataArray],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect outliers in data.

        Args:
            data: Input data
            method: Detection method:
                   - 'iqr': Interquartile range method
                   - 'zscore': Z-score method
                   - 'mad': Median absolute deviation
            threshold: Threshold for outlier detection

        Returns:
            Boolean mask where True indicates outlier
        """
        if isinstance(data, xr.DataArray):
            values = data.values.flatten()
        else:
            values = np.asarray(data).flatten()

        # Remove NaN for statistics
        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            return np.zeros(data.shape, dtype=bool)

        if method == 'iqr':
            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (values < lower) | (values > upper)

        elif method == 'zscore':
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            if std == 0:
                return np.zeros(data.shape, dtype=bool)
            zscore = np.abs((values - mean) / std)
            outliers = zscore > threshold

        elif method == 'mad':
            median = np.median(valid_values)
            mad = np.median(np.abs(valid_values - median))
            if mad == 0:
                return np.zeros(data.shape, dtype=bool)
            modified_zscore = 0.6745 * (values - median) / mad
            outliers = np.abs(modified_zscore) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        outlier_count = np.sum(outliers & ~np.isnan(values))
        if outlier_count > 0:
            self.logger.debug(f"Detected {outlier_count} outliers using {method} method")

        return outliers.reshape(data.shape) if hasattr(data, 'shape') else outliers

    def get_fill_value(
        self,
        dtype: Optional[str] = None,
        variable: Optional[str] = None
    ) -> float:
        """
        Get recommended fill value for a data type or variable.

        Args:
            dtype: Data type string
            variable: Variable name

        Returns:
            Recommended fill value
        """
        if dtype:
            return self.DEFAULT_FILL_VALUES.get(dtype, self.DEFAULT_FILL_VALUES['default'])
        return self.DEFAULT_FILL_VALUES['default']

    def prepare_for_netcdf(
        self,
        data: Union[np.ndarray, xr.DataArray],
        fill_value: float = -9999.0,
        dtype: str = 'float32'
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Prepare data for NetCDF output.

        Handles NaN values and ensures correct dtype.

        Args:
            data: Data to prepare
            fill_value: Fill value for NaN
            dtype: Target data type

        Returns:
            Data ready for NetCDF output
        """
        # Handle NaN values
        data = self.handle_nan_values(data, fill_value)

        # Convert dtype
        if isinstance(data, xr.DataArray):
            data = data.astype(dtype)
        else:
            data = np.asarray(data).astype(dtype)

        return data

    def generate_quality_report(
        self,
        ds: xr.Dataset,
        variables: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate data quality report for a dataset.

        Args:
            ds: Dataset to analyze
            variables: Variables to check (default: all)

        Returns:
            Dict of variable names to quality reports
        """
        if variables is None:
            variables = list(ds.data_vars)

        report = {}
        for var in variables:
            if var not in ds.data_vars:
                continue

            data = ds[var]
            var_report = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'nan_count': int(data.isnull().sum()),
                'nan_percent': float(data.isnull().mean() * 100),
                'min': float(data.min()) if not data.isnull().all() else None,
                'max': float(data.max()) if not data.isnull().all() else None,
                'mean': float(data.mean()) if not data.isnull().all() else None,
            }

            # Check range if known
            if var in self.VALID_RANGES:
                _, range_report = self.validate_data_range(data, variable=var)
                var_report['range_validation'] = range_report

            report[var] = var_report

        return report
