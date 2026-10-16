# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow Parameter Manager."""
from __future__ import annotations

import logging
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R


@R.parameter_managers.add('WFLOW')
class WflowParameterManager(BaseParameterManager):
    """Handles Wflow parameter bounds, normalization, and file updates."""

    PARAM_VAR_MAP = {
        'G_Cfmax': 'G_Cfmax', 'G_TT': 'G_TT', 'G_SIfrac': 'G_SIfrac',
        'KsatVer': 'KsatVer', 'f': 'f', 'SoilThickness': 'SoilThickness',
        'InfiltCapPath': 'InfiltCapPath', 'RootingDepth': 'RootingDepth',
        'KsatHorFrac': 'KsatHorFrac', 'n_river': 'N_River',
        'PathFrac': 'PathFrac', 'thetaS': 'thetaS', 'thetaR': 'thetaR',
        'Cfmax': 'Cfmax', 'TT': 'TT', 'TTI': 'TTI', 'TTM': 'TTM', 'WHC': 'WHC',
        'cf_soil': 'cf_soil',
    }

    DEFAULT_BOUNDS = {
        'G_Cfmax': {'min': 2.0, 'max': 12.0},
        'G_TT': {'min': -2.0, 'max': 2.0},
        'G_SIfrac': {'min': 0.001, 'max': 0.005},
        'KsatVer': {'min': 10.0, 'max': 3000.0},
        'f': {'min': 0.0001, 'max': 0.01},
        'SoilThickness': {'min': 100.0, 'max': 5000.0},
        'InfiltCapPath': {'min': 10.0, 'max': 500.0},
        'RootingDepth': {'min': 50.0, 'max': 2000.0},
        'KsatHorFrac': {'min': 0.0, 'max': 100.0},
        'n_river': {'min': 0.01, 'max': 0.1},
        'PathFrac': {'min': 0.0, 'max': 1.0},
        'thetaS': {'min': 0.3, 'max': 0.6},
        'thetaR': {'min': 0.01, 'max': 0.15},
        'Cfmax': {'min': 1.0, 'max': 10.0},
        'TT': {'min': -3.0, 'max': 3.0},
        'TTI': {'min': 0.1, 'max': 5.0},
        'TTM': {'min': -3.0, 'max': 3.0},
        'WHC': {'min': 0.0, 'max': 0.8},
        # Native Wflow forcing scale on precipitation (TOML input entry, not staticmaps)
        'PCORR': {'min': 0.8, 'max': 1.6},
        # Frozen-soil infiltration fraction (active only with soil_infiltration_reduction__flag)
        'cf_soil': {'min': 0.001, 'max': 0.2},
        # Post-hoc routing (smooths instantaneous net_runoff_volume_flux)
        'ROUTE_ALPHA': {'min': 0.0, 'max': 0.95},    # fast reservoir retention
        'ROUTE_BETA': {'min': 0.9, 'max': 0.9999},   # slow reservoir retention (half-life up to ~416 days)
        'ROUTE_SPLIT': {'min': 0.1, 'max': 0.9},     # fast store fraction
        'ROUTE_BASEFLOW': {'min': 0.0, 'max': 15.0},  # constant baseflow offset (m³/s)
    }

    # Post-hoc routing params (not in staticmaps, applied by worker after Wflow run)
    ROUTING_PARAMS = {'ROUTE_ALPHA', 'ROUTE_BETA', 'ROUTE_SPLIT', 'ROUTE_BASEFLOW'}

    # Forcing multipliers applied through Wflow's native TOML `scale` on the input entry
    FORCING_SCALE_PARAMS = {'PCORR': 'atmosphere_water__precipitation_volume_flux'}

    def __init__(self, config: Dict, logger: logging.Logger, wflow_settings_dir: Path):
        super().__init__(config, logger, wflow_settings_dir)
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        wflow_params_str = self._get_config_value(
            lambda: self.config.model.wflow.params_to_calibrate,
            default=None, dict_key='WFLOW_PARAMS_TO_CALIBRATE'
        )
        if wflow_params_str is None:
            wflow_params_str = ','.join(list(self.PARAM_VAR_MAP.keys()) + sorted(self.ROUTING_PARAMS))
        params_list = [p.strip() for p in wflow_params_str.split(',') if p.strip()]
        # Always include routing params for lumped mode (post-hoc smoothing of
        # instantaneous net_runoff_volume_flux); they are no-ops for distributed.
        # Honor the configured list; no implicit added-water offset.
        self.params_to_calibrate = params_list

    def _get_parameter_names(self) -> List[str]:
        return list(self.params_to_calibrate)

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        bounds = {p: dict(self.DEFAULT_BOUNDS[p]) for p in self.params_to_calibrate if p in self.DEFAULT_BOUNDS}
        overrides = self._get_config_value(
            lambda: self.config.model.wflow.parameter_bounds,
            default={}, dict_key='WFLOW_PARAMETER_BOUNDS') or {}
        for name, limits in overrides.items():
            if name not in bounds or set(limits) != {'min', 'max'}:
                raise ValueError(f'Invalid Wflow bound override: {name}')
            if not np.isfinite(list(limits.values())).all() or limits['min'] >= limits['max']:
                raise ValueError(f'Invalid Wflow bounds for {name}')
            bounds[name] = dict(limits)
        return bounds

    # Physically-based defaults matching preprocessor output (baseline KGE ~0.04)
    # Much better starting point than midpoints of bounds for DDS
    PREPROCESSOR_DEFAULTS = {
        "G_Cfmax": 5.0, "G_TT": 0.0, "G_SIfrac": 0.001,
        'KsatVer': 250.0, 'f': 0.003, 'SoilThickness': 2000.0,
        'InfiltCapPath': 50.0, 'RootingDepth': 500.0, 'KsatHorFrac': 50.0,
        'n_river': 0.036, 'PathFrac': 0.01, 'thetaS': 0.45, 'thetaR': 0.05,
        'Cfmax': 3.75, 'TT': 0.0, 'TTI': 1.0, 'TTM': 0.0, 'WHC': 0.1, 'PCORR': 1.0, 'cf_soil': 0.038,
        'ROUTE_ALPHA': 0.5, 'ROUTE_BETA': 0.98, 'ROUTE_SPLIT': 0.5, 'ROUTE_BASEFLOW': 0.0,
    }

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        bounds = self._load_parameter_bounds()
        specified = self._get_config_value(
            lambda: self.config.model.wflow.initial_parameters,
            default={}, dict_key='WFLOW_INITIAL_PARAMETERS') or {}
        for name, value in specified.items():
            if name not in bounds or not np.isfinite(value) or not bounds[name]['min'] <= value <= bounds[name]['max']:
                raise ValueError(f'Invalid Wflow initial parameter: {name}')
        initial = {}
        for p, b in bounds.items():
            default = specified.get(p, self.PREPROCESSOR_DEFAULTS.get(p))
            if default is not None and b['min'] <= default <= b['max']:
                initial[p] = default
            else:
                initial[p] = (b['min'] + b['max']) / 2.0
        return initial

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        if 'thetaR' in params and 'thetaS' in params:
            if params['thetaR'] >= params['thetaS']:
                params['thetaR'] = params['thetaS'] * 0.15
        bounds = self._load_parameter_bounds()
        for param, value in params.items():
            if param in bounds:
                params[param] = float(np.clip(value, bounds[param]['min'], bounds[param]['max']))
        return True

    @classmethod
    def set_forcing_scales(cls, toml_path: Path, scales: Dict[str, float]) -> None:
        """Rewrite [input.forcing] entries with a native Wflow scale (idempotent)."""
        content = toml_path.read_text()
        forcing = tomllib.loads(content).get('input', {}).get('forcing', {})
        for param, value in scales.items():
            name = cls.FORCING_SCALE_PARAMS[param]
            entry = forcing.get(name)
            variable = entry.get('netcdf_variable_name') if isinstance(entry, dict) else entry
            if not isinstance(variable, str):
                raise ValueError(f'Wflow forcing entry {name} not found in {toml_path}')
            line = f'{name} = {{ netcdf_variable_name = "{variable}", scale = {float(value)!r} }}'
            content, count = re.subn(rf'^{name}\s*=.*$', line, content, flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f'Expected one {name} entry in {toml_path}, found {count}')
        toml_path.write_text(content)

    def update_model_files(self, params: Dict[str, float], settings_dir: Optional[Path] = None, **kwargs) -> bool:
        try:
            self.validate_parameters(params)
            target_dir = settings_dir or self.settings_dir
            staticmaps_name = self._get_config_value(
                lambda: self.config.model.wflow.staticmaps_file,
                default='wflow_staticmaps.nc', dict_key='WFLOW_STATICMAPS_FILE'
            )
            path = target_dir / staticmaps_name
            if not path.exists():
                return False
            ds = xr.open_dataset(path)
            ds_new = ds.load()
            ds.close()
            for param_name, value in params.items():
                nc_var = self.PARAM_VAR_MAP.get(param_name)
                if nc_var and nc_var in ds_new:
                    mask = ~np.isnan(ds_new[nc_var].values)
                    ds_new[nc_var].values[mask] = value
            # Write to temp file then rename (avoids macOS HDF5 file lock)
            tmp_path = path.parent / f'{path.stem}_tmp{path.suffix}'
            ds_new.to_netcdf(tmp_path)
            shutil.move(str(tmp_path), str(path))
            forcing_scales = {p: v for p, v in params.items() if p in self.FORCING_SCALE_PARAMS}
            if forcing_scales:
                toml_name = self._get_config_value(
                    lambda: self.config.model.wflow.config_file,
                    default='wflow_sbm.toml', dict_key='WFLOW_CONFIG_FILE')
                self.set_forcing_scales(target_dir / toml_name, forcing_scales)
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to update Wflow parameters: {e}")
            return False
