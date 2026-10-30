# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coupled Groundwater Parameter Manager

Manages a joint parameter space for any land surface model coupled with
MODFLOW groundwater. The land surface model is determined by the
LAND_SURFACE_MODEL config key; its parameter manager is loaded dynamically
from the OptimizerRegistry. MODFLOW parameters (K, SY, DRAIN_CONDUCTANCE)
are managed locally.

Config keys:
    LAND_SURFACE_MODEL: Land surface model name (SUMMA, CLM, MESH, etc.)
    GROUNDWATER_MODEL: Must be MODFLOW
    MODFLOW_PARAMS_TO_CALIBRATE: Comma-separated MODFLOW params (default: K,SY,DRAIN_CONDUCTANCE)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R

# MODFLOW groundwater parameter bounds
MODFLOW_DEFAULT_BOUNDS = {
    'K': {
        'min': 0.01, 'max': 100.0,
        'transform': 'log',
        'description': 'Hydraulic conductivity (m/d)',
    },
    'SY': {
        'min': 0.005, 'max': 0.35,
        'transform': 'linear',
        'description': 'Specific yield (-)',
    },
    'DRAIN_CONDUCTANCE': {
        'min': 1e4, 'max': 1e8,
        'transform': 'log',
        'description': 'Drain conductance (m2/d) — lumped cell total',
    },
    # Basin grid mode (MODFLOW_GRID_MODE: basin)
    'AQUIFER_THICKNESS': {
        'min': 10.0, 'max': 500.0,
        'transform': 'log',
        'description': 'Aquifer thickness below land surface / streambed (m)',
    },
    'RIVER_LEAKANCE': {
        'min': 0.01, 'max': 1000.0,
        'transform': 'log',
        'description': 'River drain conductance per reach length (m2/d per m)',
    },
    'SEEPAGE_LEAKANCE': {
        'min': 1e-4, 'max': 10.0,
        'transform': 'log',
        'description': 'Land-surface seepage drain conductance per area (1/d)',
    },
    'LAKE_LEAKANCE': {
        'min': 1e-3, 'max': 100.0,
        'transform': 'log',
        'description': 'Lake-bed conductance per lake area (1/d)',
    },
    'K_ANISOTROPY': {
        'min': 0.01, 'max': 1.0,
        'transform': 'log',
        'description': 'Horizontal anisotropy K22/K (-), K22 across the K_ANGLE direction',
    },
    'K_ANGLE': {
        'min': 0.0, 'max': 90.0,
        'transform': 'linear',
        'description': 'Direction of K, degrees counter-clockwise from the grid x axis',
    },
    'ZONE_RECHARGE_FACTOR': {
        'min': 0.0, 'max': 1.0,
        'transform': 'linear',
        'description': 'Share of recharge kept in MODFLOW_REDUCED_RECHARGE_ZONES (e.g. glacier beds)',
    },
}


@R.parameter_managers.add('COUPLED_GW')
class CoupledGWParameterManager(BaseParameterManager):
    """Manages joint land-surface + MODFLOW parameter space.

    Dynamically loads the land surface model's parameter manager based on
    LAND_SURFACE_MODEL config and combines its parameter space with MODFLOW
    groundwater parameters.
    """

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path):
        """Initialize coupled parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            settings_dir: Base settings directory (parent of model-specific dirs)
        """
        super().__init__(config, logger, settings_dir)

        # Determine the land surface model
        from ..land_adapters import resolve_land_model
        self.land_model_name = resolve_land_model(self.config_dict or config)

        # Resolve model-specific settings directories
        self.land_settings_dir = settings_dir / self.land_model_name
        if not self.land_settings_dir.exists():
            # Fall back: settings_dir might already be the land model dir
            self.land_settings_dir = settings_dir
        self.modflow_settings_dir = settings_dir / 'MODFLOW'
        if not self.modflow_settings_dir.exists():
            self.modflow_settings_dir = settings_dir.parent / 'MODFLOW'

        # Parse MODFLOW parameter names from config
        modflow_params_str = self._get_config_value(
            lambda: None,
            default='K,SY,DRAIN_CONDUCTANCE',
            dict_key='MODFLOW_PARAMS_TO_CALIBRATE',
        )
        self.modflow_params = [
            p.strip() for p in str(modflow_params_str).split(',') if p.strip()
        ]

        # Initialize land surface parameter manager via registry
        self._land_pm = self._create_land_parameter_manager()

    def _create_land_parameter_manager(self) -> BaseParameterManager:
        """Dynamically instantiate the land surface model's parameter manager."""
        pm_cls = R.parameter_managers.get(self.land_model_name)
        if pm_cls is None:
            raise ValueError(
                f"No parameter manager registered for land surface model "
                f"'{self.land_model_name}'. Available: "
                f"{R.optimizers.keys()}"
            )
        return pm_cls(self.config_dict, self.logger, self.land_settings_dir)

    def _get_parameter_names(self) -> List[str]:
        """Return combined land-surface + MODFLOW parameter names."""
        return self._land_pm.all_param_names + self.modflow_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Load bounds from land surface param manager + MODFLOW defaults."""
        bounds: Dict[str, Dict[str, Any]] = dict(self._land_pm.param_bounds)

        for param in self.modflow_params:
            if param in MODFLOW_DEFAULT_BOUNDS:
                bounds[param] = {
                    'min': float(MODFLOW_DEFAULT_BOUNDS[param]['min']),
                    'max': float(MODFLOW_DEFAULT_BOUNDS[param]['max']),
                    'transform': str(MODFLOW_DEFAULT_BOUNDS[param].get('transform', 'linear')),
                }

        # Config-level overrides (highest priority)
        config_bounds = self._get_config_value(
            lambda: self.config.optimization.parameter_bounds,
            default={}
        )
        if config_bounds:
            for param_name, limit_list in config_bounds.items():
                if isinstance(limit_list, (list, tuple)) and len(limit_list) >= 2:
                    bounds[param_name] = {
                        'min': float(limit_list[0]),
                        'max': float(limit_list[1]),
                    }

        return bounds

    def _format_parameter_value(self, param_name: str, value: float) -> Any:
        """Format parameter value — delegate land params, float for MODFLOW."""
        if param_name in self.modflow_params:
            return float(value)
        return self._land_pm._format_parameter_value(param_name, value)

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        """Denormalize and enforce land-surface-model-specific constraints."""
        params = super().denormalize_parameters(normalized_array)

        # Apply land surface model constraints if available
        land_params = {
            k: v for k, v in params.items() if k not in self.modflow_params
        }
        if land_params and hasattr(self._land_pm, '_enforce_parameter_constraints'):
            constrained = self._land_pm._enforce_parameter_constraints(land_params)
            params.update(constrained)

        return params

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        """Update both land surface and MODFLOW model files."""
        land_params = {
            k: v for k, v in params.items() if k not in self.modflow_params
        }
        modflow_params = {
            k: v for k, v in params.items() if k in self.modflow_params
        }

        success = True

        if land_params:
            success = success and self._land_pm.update_model_files(land_params)

        if modflow_params:
            success = success and self._update_modflow_files(modflow_params)

        return success

    def _update_modflow_files(self, params: Dict[str, float]) -> bool:
        """Rewrite MODFLOW text input files with new parameter values."""
        try:
            d = self.modflow_settings_dir

            from ..basin_model import apply_basin_parameters, params_from_config
            mf_cfg = self.config.model.modflow if getattr(self.config, 'model', None) else None
            if apply_basin_parameters(d, params, params_from_config(mf_cfg)):
                return True

            if 'K' in params:
                self._write_npf(d, params['K'])

            if 'SY' in params:
                ss = float(self._get_config_value(
                    lambda: self.config.model.modflow.ss if self.config.model and self.config.model.modflow else None,
                    default=1e-5
                ))
                self._write_sto(d, params['SY'], ss)

            if 'DRAIN_CONDUCTANCE' in params:
                drain_elev = self._get_config_value(
                    lambda: self.config.model.modflow.drain_elevation if self.config.model and self.config.model.modflow else None,
                    default=None
                )
                if drain_elev is None:
                    top = float(self._get_config_value(
                        lambda: self.config.model.modflow.top if self.config.model and self.config.model.modflow else None,
                        default=1500.0
                    ))
                    bot = float(self._get_config_value(
                        lambda: self.config.model.modflow.bot if self.config.model and self.config.model.modflow else None,
                        default=1400.0
                    ))
                    drain_elev = (top + bot) / 2.0
                else:
                    drain_elev = float(drain_elev)
                self._write_drn(d, drain_elev, params['DRAIN_CONDUCTANCE'])

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to update MODFLOW files: {e}", exc_info=True)
            return False

    def _write_npf(self, d: Path, k: float) -> None:
        """Write node property flow (hydraulic conductivity) file."""
        (d / "gwf.npf").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_SPECIFIC_DISCHARGE\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  ICELLTYPE\n"
            "    CONSTANT 1\n"
            "  K\n"
            f"    CONSTANT {k}\n"
            "END GRIDDATA\n"
        )

    def _write_sto(self, d: Path, sy: float, ss: float) -> None:
        """Write storage file."""
        (d / "gwf.sto").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  ICONVERT\n"
            "    CONSTANT 1\n"
            "  SS\n"
            f"    CONSTANT {ss}\n"
            "  SY\n"
            f"    CONSTANT {sy}\n"
            "END GRIDDATA\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  TRANSIENT\n"
            "END PERIOD 1\n"
        )

    def _write_drn(self, d: Path, drain_elev: float, drain_cond: float) -> None:
        """Write drain package file."""
        (d / "gwf.drn").write_text(
            "BEGIN OPTIONS\n"
            "  PRINT_INPUT\n"
            "  PRINT_FLOWS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN DIMENSIONS\n"
            "  MAXBOUND 1\n"
            "END DIMENSIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            f"  1 1 1 {drain_elev} {drain_cond}\n"
            "END PERIOD 1\n"
        )

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        """Get initial parameter values from land surface files and MODFLOW defaults."""
        params = self._land_pm.get_initial_parameters() or {}

        modflow_defaults = {
            'K': float(self._get_config_value(
                lambda: self.config.model.modflow.k if self.config.model and self.config.model.modflow else None,
                default=5.0
            )),
            'SY': float(self._get_config_value(
                lambda: self.config.model.modflow.sy if self.config.model and self.config.model.modflow else None,
                default=0.05
            )),
            'DRAIN_CONDUCTANCE': float(self._get_config_value(
                lambda: self.config.model.modflow.drain_conductance if self.config.model and self.config.model.modflow else None,
                default=1e4
            )),
        }
        mf = self.config.model.modflow if getattr(self.config, 'model', None) else None
        if mf is not None:
            modflow_defaults.update(AQUIFER_THICKNESS=float(mf.aquifer_thickness),
                                    RIVER_LEAKANCE=float(mf.river_leakance),
                                    SEEPAGE_LEAKANCE=float(mf.seepage_leakance),
                                    LAKE_LEAKANCE=float(mf.lake_leakance),
                                    K_ANISOTROPY=float(mf.k_anisotropy), K_ANGLE=float(mf.k_angle),
                                    ZONE_RECHARGE_FACTOR=float(mf.zone_recharge_factor))
        for p in self.modflow_params:
            if p in modflow_defaults:
                params[p] = modflow_defaults[p]

        return params
