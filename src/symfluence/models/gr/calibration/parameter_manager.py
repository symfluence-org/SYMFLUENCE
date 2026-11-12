#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
GR Parameter Manager

Handles GR parameter bounds, normalization, and configuration updates.
Since GR doesn't use parameter files but receives them via config/runner,
this manager simply prepares the parameters for the GRRunner.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.parameters.parameter_bounds_registry import get_gr_bounds
from symfluence.core.registries import R


@R.parameter_managers.add('GR')
class GRParameterManager(BaseParameterManager):
    """Handles GR parameter bounds, normalization, and configuration updates."""

    def __init__(self, config: Dict, logger: logging.Logger, gr_settings_dir: Path):
        super().__init__(config, logger, gr_settings_dir)

        # GR-specific setup
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name, dict_key='DOMAIN_NAME'
        )
        self.experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID'
        )

        # Parse GR parameters to calibrate from config
        gr_params_str = self._get_config_value(
            lambda: self.config.model.gr.params_to_calibrate,
            dict_key='GR_PARAMS_TO_CALIBRATE',
        )

        self.gr_params = [p.strip() for p in str(gr_params_str).split(',') if p.strip()]

        # Glacier mode is lumped-only: refuse at setup to calibrate a glacier
        # parameter the distributed runner never reads.
        from ..glacier import require_lumped_for_glacier
        require_lumped_for_glacier(
            not self._runs_distributed(),
            bool(self._get_config_value(lambda: self.config.model.gr.glacier_enabled, dict_key='GR_GLACIER_ENABLED')),
            self.gr_params,
        )

    def _runs_distributed(self) -> bool:
        """Whether the GR runner will take its distributed path (anything but lumped)."""
        from symfluence.core.modeling.spatial_modes import SpatialMode
        mode = self._get_config_value(lambda: self.config.model.gr.spatial_mode, dict_key='GR_SPATIAL_MODE')
        if mode in (None, 'auto', 'default'):
            method = self._get_config_value(
                lambda: self.config.domain.definition_method, default='lumped',
                dict_key='DOMAIN_DEFINITION_METHOD'
            )
            return str(method).lower() in (
                'delineate', 'distributed', 'subset', 'semi_distributed', 'semidistributed'
            )
        return SpatialMode.from_string(str(mode)) != SpatialMode.LUMPED

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return GR parameter names from config."""
        return self.gr_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return GR parameter bounds from central registry, with config overrides."""
        bounds = get_gr_bounds()
        config_bounds = self._get_config_value(
            lambda: self.config.model.gr.gr4j_param_bounds,
            default=None
        ) or self._get_config_value(
            lambda: self.config.model.gr.param_bounds,
            default=None
        )
        if config_bounds:
            self.logger.info("Using config-specified GR parameter bounds")
            self._apply_config_bounds_override(bounds, config_bounds)
        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        GR doesn't have a parameter file to update.
        Parameters are passed via GR_EXTERNAL_PARAMS in config.
        We return True as 'applying' parameters happens in the worker/runner.
        """
        return True

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from config or defaults."""
        # Check for explicit initial params in config
        initial_params = self._get_config_value(
            lambda: self.config.model.gr.initial_params,
            dict_key='GR_INITIAL_PARAMS',
        )

        if initial_params == 'midpoint':
            self.logger.info("Using midpoint of parameter bounds as initial guess")
            return self._get_default_initial_values()

        if initial_params == 'default':
            # Try to load from previous internal calibration if it exists
            params = self._load_params_from_rdata()
            if params:
                self.logger.info(f"Loaded {len(params)} initial parameters from previous GR calibration")
                return {k: v for k, v in params.items() if k in self.gr_params}

            # Fallback to standard airGR defaults (instead of bounds midpoints)
            self.logger.debug("Using standard airGR defaults for initial parameters")
            defaults = {
                'X1': 350.0, 'X2': 0.0, 'X3': 100.0, 'X4': 1.7,
                'CTG': 0.5, 'Kf': 4.0, 'Gratio': 0.1, 'Albedo_diff': 0.1
            }
            return {k: v for k, v in defaults.items() if k in self.gr_params}

        if initial_params and isinstance(initial_params, dict):
            # Filter to only include parameters we are calibrating
            return {k: float(v) for k, v in initial_params.items() if k in self.gr_params}

        return self._get_default_initial_values()

    def _load_params_from_rdata(self) -> Optional[Dict[str, float]]:
        """Attempt to load parameters from GR_calib.Rdata in the simulation directory."""
        try:
            data_dir = self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')
            domain = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
            exp_id = self._get_config_value(lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID')

            if not all([data_dir, domain, exp_id]):
                return None

            rdata_path = Path(data_dir) / f"domain_{domain}" / "simulations" / exp_id / "GR" / "GR_calib.Rdata"

            if not rdata_path.exists():
                return None

            from ..r_environment import configure_r_dll_search
            configure_r_dll_search()

            import rpy2.robjects as robjects
            robjects.r['load'](str(rdata_path))

            if 'OutputsCalib' not in robjects.globalenv:
                return None

            outputs_calib = robjects.globalenv['OutputsCalib']
            param_final = list(outputs_calib.rx2('ParamFinalR'))

            # Map based on parameter count
            # 4: GR4J
            # 6: GR4J + CemaNeige (CTG, Kf)
            # 8: GR4J + CemaNeige + Hysteresis (Gratio, Albedo_diff)
            if len(param_final) == 4:
                param_names = ['X1', 'X2', 'X3', 'X4']
            elif len(param_final) == 6:
                param_names = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf']
            elif len(param_final) == 8:
                param_names = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf', 'Gratio', 'Albedo_diff']
            else:
                self.logger.warning(f"Unexpected number of parameters in Rdata: {len(param_final)}")
                # Generic names
                param_names = [f"P{i+1}" for i in range(len(param_final))]

            self.logger.info(f"Loaded {len(param_final)} parameters from Rdata: {param_names}")
            return {name: val for name, val in zip(param_names, param_final)}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.debug(f"Could not load initial parameters from Rdata: {e}", exc_info=True)
            return None

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds)."""
        params = {}
        for param_name in self.gr_params:
            bounds = self.param_bounds.get(param_name)
            if bounds:
                params[param_name] = (bounds['min'] + bounds['max']) / 2
            else:
                # Default values for GR parameters if not in bounds registry
                defaults = {
                    'X1': 350.0, 'X2': 0.0, 'X3': 100.0, 'X4': 1.7,
                    'CTG': 0.0, 'Kf': 3.69
                }
                params[param_name] = defaults.get(param_name, 1.0)
        return params
