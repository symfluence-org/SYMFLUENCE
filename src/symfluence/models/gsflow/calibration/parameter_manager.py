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
from symfluence.core.registries import R


@R.parameter_managers.add('GSFLOW')
class GSFLOWParameterManager(BaseParameterManager):
    """Parameter manager for GSFLOW (PRMS + MODFLOW-NWT)."""

    # PRMS parameters updated in ####-delimited params.dat
    # Note: soil_rechr_max, gwflow_coef, gw_seep_coef, and tmax_allrain are
    # excluded — GSFLOW v2.4.0+ in COUPLED mode silently ignores them.
    PRMS_PARAMS = [
        'soil_moist_max', 'ssr2gw_rate', 'slowcoef_lin',
        'carea_max', 'smidx_coef',
        'jh_coef', 'tmax_allsnow',
        'rain_adj', 'snow_adj',
    ]

    # MODFLOW-NWT parameters updated in UPW package
    MODFLOW_PARAMS = ['K', 'SY']

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
                raise ValueError('GSFLOW UZF porosity leaves no valid specific-yield range')
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
            # Split into PRMS and MODFLOW parameters
            prms_params = {k: v for k, v in params.items() if k in self.PRMS_PARAMS}
            modflow_params = {k: v for k, v in params.items() if k in self.MODFLOW_PARAMS}

            # Update PRMS parameter file
            if prms_params:
                self._update_prms_params(target, prms_params)

            # Update MODFLOW UPW package
            if modflow_params:
                from ..coupling import GSFLOWCouplingManager
                coupler = GSFLOWCouplingManager(self.config_dict, self.logger)
                coupler.update_modflow_parameters(target, modflow_params)

            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying GSFLOW parameters: {e}", exc_info=True)
            return False

    def _update_prms_params(self, settings_dir: Path, params: Dict[str, float]) -> bool:
        """Update PRMS ####-delimited parameter file."""
        param_path = settings_dir / self.param_file
        if not param_path.exists():
            self.logger.warning(f"PRMS parameter file not found: {param_path}")
            return False

        content = param_path.read_text(encoding='utf-8')
        blocks = content.split('####\n')

        updated_blocks = []
        for block in blocks:
            updated_block = block
            for param_name, value in params.items():
                lines = block.strip().split('\n')
                for line in lines:
                    if line.strip() == param_name:
                        updated_block = self._replace_block_values(
                            block, param_name, value
                        )
                        break
            updated_blocks.append(updated_block)

        param_path.write_text('####\n'.join(updated_blocks), encoding='utf-8')
        return True

    def _block_contains_param(self, block: str, param_name: str) -> bool:
        lines = block.strip().split('\n')
        return any(line.strip() == param_name for line in lines)

    def _replace_block_values(self, block: str, param_name: str, value: float) -> str:
        """Replace value lines in a PRMS parameter block."""
        stripped = block.strip()
        lines = stripped.split('\n')

        param_idx = None
        for i, line in enumerate(lines):
            if line.strip() == param_name:
                param_idx = i
                break

        if param_idx is None:
            return block

        try:
            dim_size = int(lines[param_idx + 3].strip())
        except (IndexError, ValueError):
            dim_size = 1

        value_start = param_idx + 5
        formatted = f"{value:.6f}"

        new_lines = lines[:value_start]
        for _ in range(dim_size):
            new_lines.append(formatted)

        remaining = value_start + dim_size
        if remaining < len(lines):
            new_lines.extend(lines[remaining:])

        result = '\n'.join(new_lines)
        if block.endswith('\n'):
            result += '\n'
        return result

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values."""
        from ..parameters import DEFAULT_PARAMS
        return {p: DEFAULT_PARAMS.get(p, 1.0) for p in self.gsflow_params}
