# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parameter Regionalization Strategies for Distributed Hydrological Models.

Provides multiple strategies for handling spatially distributed parameters:

- lumped: Single parameter set applied uniformly across all units
- transfer_function: Parameters derived from unit attributes via linear functions
- zones: Group units into zones with shared parameters
- distributed: Independent parameters for each unit (requires regularization)

This module is model-agnostic.  Each model supplies its own
param_config mapping parameters to physical attributes.

Configuration key:
    PARAMETER_REGIONALIZATION: lumped | transfer_function | zones | distributed
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.mixins.config import resolve_config_value


class ParameterRegionalization(ABC):
    """Abstract base class for parameter regionalization strategies."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_units: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.param_bounds = param_bounds
        self.n_units = n_units
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Return the parameters/coefficients to be calibrated."""

    @abstractmethod
    def to_distributed(
        self,
        calibration_params: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert calibration parameters to distributed parameter values.

        Returns:
            Tuple of (param_array [n_units, n_params], param_names).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""

    def get_coefficient_transforms(self) -> Dict[str, str]:
        """Return transform type per calibration coefficient.

        Returns:
            {coeff_name: 'log' | 'linear'}.  Default returns 'linear'.
        """
        return {k: 'linear' for k in self.get_calibration_parameters()}


class LumpedRegionalization(ParameterRegionalization):
    """All units share the same parameter values."""

    @property
    def name(self) -> str:
        return "lumped"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        return self.param_bounds.copy()

    def to_distributed(
        self, calibration_params: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        param_names = list(calibration_params.keys())
        n_params = len(param_names)
        param_array = np.zeros((self.n_units, n_params))
        for i, pname in enumerate(param_names):
            param_array[:, i] = calibration_params[pname]
        return param_array, param_names


class TransferFunctionRegionalization(ParameterRegionalization):
    """Transfer-function parameter regionalization (MPR-style).

    Linear form: param = a + b * attr_norm

    Each model must supply a param_config dict mapping parameter names
    to their driving attribute and whether b is calibrated.
    """

    LOG_TRANSFORM_ATTRS = {
        'precip_mm_yr', 'aridity', 'climate.prec_annual_mean',
        'soil.ksat', 'soil.regolith_thickness_mean',
    }

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_units: int,
        attributes: pd.DataFrame,
        param_config: Dict[str, Dict],
        b_bounds: Tuple[float, float] = (-1.5, 1.5),
        transfer_function_type: str = 'linear',
        log_transform_attrs: Optional[set] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(param_bounds, n_units, logger)
        self.attributes = attributes.copy()
        self.param_config = param_config
        self.b_bounds = b_bounds
        self.transfer_function_type = transfer_function_type
        self.log_transform_attrs = log_transform_attrs or self.LOG_TRANSFORM_ATTRS
        self._normalize_attributes()
        self._build_coefficient_map()

    def _normalize_attributes(self):
        """Normalize numeric attributes to [0, 1] with optional log-transform."""
        self.attr_stats: Dict[str, Dict] = {}
        referenced_attrs: set[str] = {
            cfg['attribute']
            for cfg in self.param_config.values()
            if cfg.get('attribute')
        }
        all_attrs = referenced_attrs | set(self.log_transform_attrs)

        for col in all_attrs:
            if col not in self.attributes.columns:
                continue
            values = self.attributes[col].values.copy()
            if col in self.log_transform_attrs:
                values = np.log1p(np.maximum(values, 0))
                transform = 'log1p'
            else:
                transform = 'none'
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            self.attr_stats[col] = {'min': min_val, 'max': max_val, 'transform': transform}
            if range_val > 0:
                self.attributes[f'{col}_norm'] = (values - min_val) / range_val
            else:
                self.attributes[f'{col}_norm'] = 0.5

    def _build_coefficient_map(self):
        self.coeff_to_param: Dict[str, Tuple[str, bool]] = {}
        self.param_to_coeffs: Dict[str, List[str]] = {}
        self.param_to_attr: Dict[str, str] = {}

        for param_name, config in self.param_config.items():
            if param_name not in self.param_bounds:
                continue
            attr = config.get('attribute', 'precip_mm_yr')
            calibrate_b = config.get('calibrate_b', False)
            attr_norm = f'{attr}_norm'
            if attr_norm not in self.attributes.columns:
                fallback = config.get('fallback')
                if fallback:
                    fallback_norm = f'{fallback}_norm'
                    if fallback_norm in self.attributes.columns:
                        self.logger.info(
                            f"{param_name}: '{attr}' not available, "
                            f"using fallback '{fallback}'"
                        )
                        attr = fallback
                        attr_norm = fallback_norm
            if attr_norm in self.attributes.columns:
                self.param_to_attr[param_name] = attr_norm
            else:
                self.param_to_attr[param_name] = attr
            coeff_names = [f'{param_name}_a']
            self.coeff_to_param[f'{param_name}_a'] = (param_name, False)
            if calibrate_b:
                coeff_names.append(f'{param_name}_b')
                self.coeff_to_param[f'{param_name}_b'] = (param_name, True)
            self.param_to_coeffs[param_name] = coeff_names

    @property
    def name(self) -> str:
        return "transfer_function"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for param_name, coeff_names in self.param_to_coeffs.items():
            p_min, p_max = self.param_bounds[param_name]
            p_range = p_max - p_min
            cfg = self.param_config.get(param_name, {})
            for coeff_name in coeff_names:
                if coeff_name.endswith('_a'):
                    bounds[coeff_name] = (p_min, p_max)
                elif coeff_name.endswith('_b'):
                    b_min = self.b_bounds[0] * p_range
                    b_max = self.b_bounds[1] * p_range
                    b_sign = cfg.get('b_sign')
                    if b_sign == 'positive':
                        b_min = max(b_min, 0.0)
                    elif b_sign == 'negative':
                        b_max = min(b_max, 0.0)
                    bounds[coeff_name] = (b_min, b_max)
        return bounds

    def get_coefficient_transforms(self) -> Dict[str, str]:
        transforms: Dict[str, str] = {}
        for param_name, coeff_names in self.param_to_coeffs.items():
            cfg = self.param_config.get(param_name, {})
            t = cfg.get('transform', 'linear')
            for cn in coeff_names:
                transforms[cn] = t if cn.endswith('_a') else 'linear'
        return transforms

    def to_distributed(
        self, calibration_params: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        param_names = list(self.param_to_coeffs.keys())
        n_params = len(param_names)
        param_array = np.zeros((self.n_units, n_params))
        for i, param_name in enumerate(param_names):
            attr_name = self.param_to_attr[param_name]
            a = calibration_params.get(f'{param_name}_a', 1.0)
            b = calibration_params.get(f'{param_name}_b', 0.0)
            if attr_name in self.attributes.columns:
                attr_vals = self.attributes[attr_name].values
            else:
                attr_vals = np.full(self.n_units, 0.5)
            values = a + b * attr_vals
            p_min, p_max = self.param_bounds[param_name]
            values = np.clip(values, p_min, p_max)
            param_array[:, i] = values
        return param_array, param_names


class ZoneRegionalization(ParameterRegionalization):
    """Units grouped into zones with shared parameters."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_units: int,
        zone_assignments: np.ndarray,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(param_bounds, n_units, logger)
        self.zone_assignments = zone_assignments
        self.n_zones = len(np.unique(zone_assignments))
        self.logger.info(f"Zone regionalization: {self.n_zones} zones")

    @property
    def name(self) -> str:
        return "zones"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for param_name, (p_min, p_max) in self.param_bounds.items():
            for zone in range(self.n_zones):
                bounds[f'{param_name}_z{zone}'] = (p_min, p_max)
        return bounds

    def to_distributed(
        self, calibration_params: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)
        param_array = np.zeros((self.n_units, n_params))
        for i, param_name in enumerate(param_names):
            for zone in range(self.n_zones):
                coeff_name = f'{param_name}_z{zone}'
                value = calibration_params.get(coeff_name, 0.0)
                mask = self.zone_assignments == zone
                param_array[mask, i] = value
        return param_array, param_names


class DistributedRegionalization(ParameterRegionalization):
    """Independent parameters per unit.  Requires strong regularization."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_units: int,
        regularization: str = 'spatial_smoothing',
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(param_bounds, n_units, logger)
        self.regularization = regularization
        n_params = len(param_bounds) * n_units
        self.logger.warning(
            f"Distributed regionalization: {n_params} parameters! "
            f"Consider using transfer_function or zones instead."
        )

    @property
    def name(self) -> str:
        return "distributed"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for param_name, (p_min, p_max) in self.param_bounds.items():
            for unit in range(self.n_units):
                bounds[f'{param_name}_s{unit}'] = (p_min, p_max)
        return bounds

    def to_distributed(
        self, calibration_params: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)
        param_array = np.zeros((self.n_units, n_params))
        for i, param_name in enumerate(param_names):
            for unit in range(self.n_units):
                coeff_name = f'{param_name}_s{unit}'
                param_array[unit, i] = calibration_params.get(coeff_name, 0.0)
        return param_array, param_names


class RegionalizationFactory:
    """Factory for creating parameter regionalization strategies."""

    @staticmethod
    def create(
        method: str,
        param_bounds: Dict[str, Tuple[float, float]],
        n_units: int,
        config: Optional[Dict[str, Any]] = None,
        attributes: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None,
    ) -> ParameterRegionalization:
        config = config or {}
        logger = logger or logging.getLogger(__name__)
        method = method.lower().replace('-', '_')

        if method == 'lumped':
            return LumpedRegionalization(param_bounds=param_bounds, n_units=n_units, logger=logger)

        elif method == 'transfer_function':
            if attributes is None:
                raise ValueError("transfer_function regionalization requires 'attributes' DataFrame")
            param_config = config.get('TRANSFER_FUNCTION_PARAM_CONFIG')
            if param_config is None:
                raise ValueError(
                    "transfer_function regionalization requires "
                    "'TRANSFER_FUNCTION_PARAM_CONFIG' in config"
                )
            return TransferFunctionRegionalization(
                param_bounds=param_bounds, n_units=n_units, attributes=attributes,
                param_config=param_config,
                b_bounds=resolve_config_value(config, lambda: config.optimization.transfer_function_b_bounds, (-1.0, 1.0), 'TRANSFER_FUNCTION_B_BOUNDS'),
                transfer_function_type=resolve_config_value(
                    config,
                    lambda: config.optimization.transfer_function_type,
                    dict_key='TRANSFER_FUNCTION_TYPE',
                ),
                log_transform_attrs=resolve_config_value(config, lambda: config.optimization.transfer_function_log_attrs, None, 'TRANSFER_FUNCTION_LOG_ATTRS'),
                logger=logger,
            )

        elif method == 'zones':
            zone_assignments = config.get('zone_assignments')
            if zone_assignments is None:
                raise ValueError("zones regionalization requires 'zone_assignments' in config")
            return ZoneRegionalization(
                param_bounds=param_bounds, n_units=n_units,
                zone_assignments=zone_assignments, logger=logger,
            )

        elif method == 'distributed':
            return DistributedRegionalization(
                param_bounds=param_bounds, n_units=n_units,
                regularization=config.get('regularization', 'spatial_smoothing'),
                logger=logger,
            )

        else:
            raise ValueError(
                f"Unknown regionalization method: {method}. "
                f"Choose from: lumped, transfer_function, zones, distributed"
            )


def get_regionalization_info() -> Dict[str, str]:
    return {
        'lumped': "Single parameter set for all units.",
        'transfer_function': "MPR-style: param = a + b * attr_norm.",
        'zones': "Units grouped into zones with shared parameters.",
        'distributed': "Independent parameters for each unit.",
    }
