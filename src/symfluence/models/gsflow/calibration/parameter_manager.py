# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Parameter Manager.

Manages parameters for both PRMS (####-delimited params.dat) and
MODFLOW-NWT (UPW package) components of GSFLOW.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.exceptions import ConfigurationError
from symfluence.core.registries import R


@R.parameter_managers.add('GSFLOW')
class GSFLOWParameterManager(BaseParameterManager):
    """Parameter manager for GSFLOW (PRMS + MODFLOW-NWT)."""

    # Parameters are written by parameter_files.apply_gsflow_parameters.
    # Note: soil_rechr_max, gwflow_coef, gw_seep_coef and tmax_allrain have no
    # effect in COUPLED mode (GSFLOW v2.4.0+ ignores them); do not calibrate them.
    # The all-rain threshold is tmax_allrain_offset (above tmax_allsnow).

    def __init__(
        self,
        config: Any,
        logger: logging.Logger,
        settings_dir: Path
    ):
        super().__init__(config, logger, settings_dir)

        # Get calibration parameter list
        params_str = (
            self._get_config_value(lambda: None, default='', dict_key='GSFLOW_PARAMS_TO_CALIBRATE') or
            'soil_moist_max,ssr2gw_rate,K,SY,slowcoef_lin,carea_max,smidx_coef,jh_coef,tmax_allsnow,rain_adj,snow_adj'
        )
        self.gsflow_params = [p.strip() for p in params_str.split(',') if p.strip()]

        # File references
        self.param_file = self._get_config_value(lambda: None, default='params.dat', dict_key='GSFLOW_PARAMETER_FILE')

    def _get_parameter_names(self) -> List[str]:
        """Get ordered list of parameter names to calibrate."""
        return list(self.gsflow_params)

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Load parameter bounds from GSFLOW parameter definitions."""
        from ..parameters import PARAM_BOUNDS
        bounds = {p: dict(PARAM_BOUNDS.get(p, {'min': 0.001, 'max': 10.0}))
                  for p in self.gsflow_params}
        overrides = self._get_config_value(lambda: None, default={}, dict_key='GSFLOW_PARAM_BOUNDS') or {}
        for name, values in overrides.items():
            if name in bounds:
                bounds[name].update(values)
        if 'SY' in bounds:
            thts = self._get_config_value(lambda: None, default=0.30,
                                         dict_key='GSFLOW_UZF_SATURATED_WATER_CONTENT')
            bounds['SY']['max'] = min(bounds['SY']['max'], float(thts) - 0.01)
            if bounds['SY']['max'] <= bounds['SY']['min']:
                raise ConfigurationError('GSFLOW UZF porosity leaves no valid specific-yield range')
        return bounds

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        """Write denormalized parameters to GSFLOW model files."""
        return self.apply_parameters(params)

    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for all calibration parameters."""
        return {p: self.param_bounds.get(p, {'min': 0.001, 'max': 10.0})
                for p in self.gsflow_params}

    def get_default_parameters(self) -> Dict[str, float]:
        """Get default parameter values."""
        from ..parameters import DEFAULT_PARAMS
        return {p: DEFAULT_PARAMS.get(p, 1.0) for p in self.gsflow_params}

    def apply_parameters(self, params: Dict[str, float], target_dir: Optional[Path] = None) -> bool:
        """Apply parameters to PRMS params.dat and MODFLOW UPW package."""
        target = target_dir or self.settings_dir

        try:
            # Same code path as the calibration worker (PRMS blocks of any
            # dimension, derived parameters, UPW K/SY, SFR streambed K).
            from ..parameter_files import apply_gsflow_parameters
            apply_gsflow_parameters(target, params, self.param_file)
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying GSFLOW parameters: {e}", exc_info=True)
            return False

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values."""
        from ..parameters import DEFAULT_PARAMS
        return {p: DEFAULT_PARAMS.get(p, 1.0) for p in self.gsflow_params}
