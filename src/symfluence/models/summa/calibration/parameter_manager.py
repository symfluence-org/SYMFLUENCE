# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Parameter Manager

Handles parameter bounds, normalization, file generation, and soil depth
calculations for SUMMA local and basin parameters.

This module was refactored from core/parameter_manager.py to follow the
registry pattern used by all other model parameter managers.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
import xarray as xr

from symfluence.core.profiling import ProfilerContext
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_depth_bounds, get_mizuroute_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('SUMMA')
class SUMMAParameterManager(BaseParameterManager):
    """
    Parameter manager for SUMMA model calibration.

    Handles:
    - Local parameters (HRU-level): e.g., k_soil, theta_sat, critSoilWilting
    - Basin parameters (GRU-level): e.g., routingGammaShape, routingGammaScale
    - Depth parameters: total_mult, shape_factor for soil depth calibration
    - mizuRoute parameters: velo, diff for routing calibration

    File Operations:
    - Reads bounds from localParamInfo.txt, basinParamInfo.txt
    - Writes trialParams.nc with calibrated parameter values
    - Updates coldState.nc for soil depth calibration
    - Updates mizuRoute param.nml.default for routing parameters

    Configuration Keys:
    - PARAMS_TO_CALIBRATE: Comma-separated list of local parameters
    - BASIN_PARAMS_TO_CALIBRATE: Comma-separated list of basin parameters
    - CALIBRATE_DEPTH: Boolean to enable depth calibration
    - CALIBRATE_MIZUROUTE: Boolean to enable routing calibration
    - MIZUROUTE_PARAMS_TO_CALIBRATE: Comma-separated list of routing parameters
    - SETTINGS_SUMMA_ATTRIBUTES: Path to attributes.nc file
    - SETTINGS_SUMMA_COLDSTATE: Path to coldState.nc file
    - SETTINGS_SUMMA_TRIALPARAMS: Path to trialParams.nc file
    """

    def __init__(self, config: Dict, logger: logging.Logger, optimization_settings_dir: Path):
        """
        Initialize SUMMA parameter manager.

        Args:
            config: Configuration dictionary with SUMMA-specific settings
            logger: Logger instance
            optimization_settings_dir: Path to SUMMA settings directory
        """
        # Initialize base class
        super().__init__(config, logger, optimization_settings_dir)

        # Setup profiling context
        self._profiler_ctx = ProfilerContext("summa_parameter_manager")

        # Parse parameter lists from config
        local_params_raw = self._get_config_value(lambda: self.config.model.summa.params_to_calibrate, default='', dict_key='PARAMS_TO_CALIBRATE') or ''
        basin_params_raw = self._get_config_value(lambda: None, default='', dict_key='BASIN_PARAMS_TO_CALIBRATE') or ''
        self.local_params = [p.strip() for p in str(local_params_raw).split(',') if p.strip()]
        self.basin_params = [p.strip() for p in str(basin_params_raw).split(',') if p.strip()]

        # Identify depth parameters
        self.depth_params = []
        if self._get_config_value(lambda: None, default=False, dict_key='CALIBRATE_DEPTH'):
            self.depth_params = ['total_mult', 'shape_factor']

        # Handle special multiplier parameter
        if 'total_soil_depth_multiplier' in self.local_params:
            self.depth_params.append('total_soil_depth_multiplier')
            self.local_params.remove('total_soil_depth_multiplier')

        # Parse mizuRoute parameters
        self.mizuroute_params = []
        if self._get_config_value(lambda: None, default=False, dict_key='CALIBRATE_MIZUROUTE'):
            mizuroute_params_str = self._get_config_value(lambda: None, default=None, dict_key='MIZUROUTE_PARAMS_TO_CALIBRATE')
            if mizuroute_params_str is None:
                mizuroute_params_str = 'velo,diff'
            self.mizuroute_params = [p.strip() for p in str(mizuroute_params_str).split(',') if p.strip()]

        # Load original soil depths if depth calibration enabled
        self.original_depths = None
        if self.depth_params:
            self.original_depths = self._load_original_depths()

        # Get attribute file path
        self.attr_file_path = self.settings_dir / self._get_config_value(lambda: self.config.model.summa.attributes, default='attributes.nc', dict_key='SETTINGS_SUMMA_ATTRIBUTES')

        # Regionalization setup
        self._regionalization_method = self._get_config_value(
            lambda: self.config.model.summa.parameter_regionalization,
            default='lumped', dict_key='PARAMETER_REGIONALIZATION'
        )
        self._regionalization = None  # Lazy-initialized in _init_regionalization()
        self._regionalization_initialized = False

    # ========================================================================
    # REGIONALIZATION
    # ========================================================================

    def _init_regionalization(self):
        """Lazily initialize the regionalization strategy.

        Called on first access to ensure parameter bounds are already loaded.
        """
        if self._regionalization_initialized:
            return
        self._regionalization_initialized = True

        if self._regionalization_method == 'lumped':
            return  # No regionalization object needed for lumped mode

        from symfluence.models.summa.calibration.summa_regionalization import (
            create_summa_regionalization,
        )

        # Convert bounds from {name: {min, max}} to {name: (min, max)} tuples
        raw_bounds = self._parse_all_bounds()
        tuple_bounds: Dict[str, Tuple[float, float]] = {}
        for p in self.local_params:
            if p in raw_bounds:
                tuple_bounds[p] = (raw_bounds[p]['min'], raw_bounds[p]['max'])

        # Get HRU count
        try:
            with xr.open_dataset(self.attr_file_path) as ds:
                n_hrus = ds.sizes.get('hru', 1)
        except Exception:  # noqa: BLE001
            n_hrus = 1

        # Optional CSV path for supplementary attributes
        csv_path = self._get_config_value(
            lambda: self.config.model.summa.transfer_function_attributes_path,
            default=None, dict_key='TRANSFER_FUNCTION_ATTRIBUTES'
        )
        csv_path = Path(csv_path) if csv_path else None

        # Optional per-parameter config override
        param_config = self._get_config_value(
            lambda: self.config.model.summa.transfer_function_param_config,
            default=None, dict_key='TRANSFER_FUNCTION_PARAM_CONFIG'
        )

        self._regionalization = create_summa_regionalization(
            method=self._regionalization_method,
            param_bounds=tuple_bounds,
            n_hrus=n_hrus,
            attributes_nc_path=self.attr_file_path,
            csv_path=csv_path,
            param_config=param_config,
            logger=self.logger,
        )
        self.logger.info(
            f"SUMMA regionalization: {self._regionalization.name} — "
            f"{len(self._regionalization.get_calibration_parameters())} calibration coefficients "
            f"for {len(tuple_bounds)} local params across {n_hrus} HRUs"
        )

    @property
    def _use_regionalization(self) -> bool:
        """Whether regionalization is active (non-lumped)."""
        return self._regionalization_method != 'lumped'

    def expand_coefficients_to_distributed(self, coeff_params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Convert regionalization coefficients to per-HRU parameter arrays.

        Args:
            coeff_params: Coefficient values from the optimizer
                          (e.g. ``{'k_soil_a': 0.001, 'k_soil_b': -0.0005}``).

        Returns:
            Dictionary of ``{param_name: np.ndarray[n_hrus]}``.
        """
        self._init_regionalization()
        param_array, param_names = self._regionalization.to_distributed(coeff_params)
        return {name: param_array[:, i] for i, name in enumerate(param_names)}

    # ========================================================================
    # PARAMETER CONSTRAINTS (Override Base Implementation)
    # ========================================================================

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        """
        Denormalize parameters and enforce SUMMA-specific constraints.

        Overrides base implementation to apply physical constraints (e.g., theta_sat > theta_res)
        that prevent model crashes.
        """
        # Call base implementation first to get raw denormalized values
        params = super().denormalize_parameters(normalized_array)

        # Enforce constraints
        params = self._enforce_parameter_constraints(params)

        return params

    def _enforce_parameter_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce physical constraints between parameters to prevent SUMMA crashes.

        Constraints enforced (soil moisture ordering):
        1. theta_sat > theta_res (min gap 0.05)
        2. theta_sat > fieldCapacity (if known)
        3. fieldCapacity > theta_res (if known)
        4. critSoilTranspire > critSoilWilting
        5. fieldCapacity > critSoilWilting (min gap 0.01)
        5b. critSoilWilting > theta_res (min gap 0.01)
        6. albedoMax >= albedoMinWinter
        """
        validated = params.copy()

        # Helper to extract scalar for comparison
        def get_scalar(name, p_dict):
            if name not in p_dict: return None
            val = p_dict[name]
            if isinstance(val, np.ndarray):
                # Handle scalar wrapped in array or full HRU array (assume homogeneous for check)
                return float(val.flatten()[0])
            return float(val)

        # Load defaults for context if not cached
        if not hasattr(self, '_cached_defaults'):
             self._cached_defaults = self._extract_default_parameters()

        # Create a view of full parameter set (defaults + calibrated)
        full_params = self._cached_defaults.copy()
        full_params.update(validated)

        # 1. Check theta_sat vs theta_res
        theta_sat = get_scalar('theta_sat', full_params)
        theta_res = get_scalar('theta_res', full_params)

        if theta_sat is not None and theta_res is not None:
            min_gap = 0.05
            if theta_sat < (theta_res + min_gap):
                # Violation!
                if 'theta_sat' in validated:
                    # Bump theta_sat up
                    new_val = theta_res + min_gap
                    validated['theta_sat'] = self._format_parameter_value('theta_sat', new_val)
                elif 'theta_res' in validated:
                    # Push theta_res down
                    new_val = theta_sat - min_gap
                    if new_val < 0.001: new_val = 0.001
                    validated['theta_res'] = self._format_parameter_value('theta_res', new_val)

        # 2. Check field capacity constraints
        fc = get_scalar('fieldCapacity', full_params)
        if fc is not None:
            # theta_sat > fc
            if theta_sat is not None and theta_sat < (fc + 0.01):
                 if 'theta_sat' in validated:
                     new_val = fc + 0.01
                     validated['theta_sat'] = self._format_parameter_value('theta_sat', new_val)

            # fc > theta_res
            if theta_res is not None and fc < (theta_res + 0.01):
                 if 'theta_res' in validated:
                     new_val = fc - 0.01
                     if new_val < 0.001: new_val = 0.001
                     validated['theta_res'] = self._format_parameter_value('theta_res', new_val)

        # 3. Check soil stress parameters
        crit_trans = get_scalar('critSoilTranspire', full_params)
        crit_wilt = get_scalar('critSoilWilting', full_params)

        if crit_trans is not None and crit_wilt is not None:
            if crit_trans < (crit_wilt + 0.01):
                if 'critSoilTranspire' in validated:
                    new_val = crit_wilt + 0.01
                    validated['critSoilTranspire'] = self._format_parameter_value('critSoilTranspire', new_val)
                elif 'critSoilWilting' in validated:
                    new_val = crit_trans - 0.01
                    if new_val < 0: new_val = 0.0
                    validated['critSoilWilting'] = self._format_parameter_value('critSoilWilting', new_val)

        # 4. fieldCapacity > critSoilWilting — wilting point must be below
        #    field capacity or the soil column becomes numerically unstable
        fc = get_scalar('fieldCapacity', {**full_params, **validated})
        crit_wilt = get_scalar('critSoilWilting', {**full_params, **validated})
        if fc is not None and crit_wilt is not None:
            if fc < (crit_wilt + 0.01):
                if 'critSoilWilting' in validated:
                    new_val = fc - 0.01
                    if new_val < 0: new_val = 0.0
                    validated['critSoilWilting'] = self._format_parameter_value('critSoilWilting', new_val)
                elif 'fieldCapacity' in validated:
                    new_val = crit_wilt + 0.01
                    validated['fieldCapacity'] = self._format_parameter_value('fieldCapacity', new_val)

        # 4b. critSoilWilting > theta_res — residual moisture must be below
        #     wilting point for consistent soil moisture ordering
        crit_wilt = get_scalar('critSoilWilting', {**full_params, **validated})
        theta_res = get_scalar('theta_res', {**full_params, **validated})
        if crit_wilt is not None and theta_res is not None:
            if crit_wilt < (theta_res + 0.01):
                if 'theta_res' in validated:
                    new_val = crit_wilt - 0.01
                    if new_val < 0.001: new_val = 0.001
                    validated['theta_res'] = self._format_parameter_value('theta_res', new_val)
                elif 'critSoilWilting' in validated:
                    new_val = theta_res + 0.01
                    validated['critSoilWilting'] = self._format_parameter_value('critSoilWilting', new_val)

        # 5. albedoMax >= albedoMinWinter — minimum winter albedo cannot
        #    exceed the maximum albedo or SUMMA snow calculations fail
        albedo_max = get_scalar('albedoMax', full_params)
        albedo_min_w = get_scalar('albedoMinWinter', full_params)

        if albedo_max is not None and albedo_min_w is not None:
            if albedo_min_w > albedo_max:
                if 'albedoMinWinter' in validated:
                    validated['albedoMinWinter'] = self._format_parameter_value('albedoMinWinter', albedo_max)
                elif 'albedoMax' in validated:
                    validated['albedoMax'] = self._format_parameter_value('albedoMax', albedo_min_w)

        return validated

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS FROM BASE CLASS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return all SUMMA parameter names in calibration order.

        When regionalization is active, local parameter names are replaced by
        the coefficient names (e.g. ``k_soil_a``, ``k_soil_b``).
        """
        if self._use_regionalization:
            self._init_regionalization()
            coeff_names = list(self._regionalization.get_calibration_parameters().keys())
            return coeff_names + self.basin_params + self.depth_params + self.mizuroute_params
        return self.local_params + self.basin_params + self.depth_params + self.mizuroute_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Parse SUMMA parameter bounds from localParamInfo.txt, etc.

        When regionalization is active, local parameter bounds are replaced by
        coefficient bounds from the regionalization strategy.
        """
        if self._use_regionalization:
            self._init_regionalization()
            # Coefficient bounds from regionalization (tuples → dicts)
            bounds: Dict[str, Dict[str, float]] = {}
            for name, (lo, hi) in self._regionalization.get_calibration_parameters().items():
                bounds[name] = {'min': lo, 'max': hi}

            # Add non-local bounds normally
            bounds.update(self._parse_non_local_bounds())
            return bounds

        return self._parse_all_bounds()

    def update_model_files(self, params: Dict[str, np.ndarray]) -> bool:
        """
        Update SUMMA model files with new parameter values.

        When regionalization is active, coefficient values are first expanded
        to spatially distributed per-HRU arrays, then written as usual.

        Updates:
        - trialParams.nc: Main parameter file
        - coldState.nc: Soil depths (if depth calibration enabled)
        - param.nml.default: mizuRoute parameters (if routing calibration enabled)

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if all updates successful
        """
        if self._use_regionalization:
            params = self._expand_regionalized_params(params)

        success = True
        success = success and self._generate_trial_params_file(params)
        if self.depth_params:
            success = success and self._update_soil_depths(params)
        if self.mizuroute_params:
            success = success and self._update_mizuroute_parameters(params)
        return success

    def get_initial_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Get initial parameter values from existing files or defaults."""
        # Try to load existing optimized parameters
        existing_params = self._load_existing_optimized_parameters()
        if existing_params:
            self.logger.info("Loaded existing optimized parameters")
            return existing_params

        # Extract parameters from model files
        return self._extract_default_parameters()

    # ========================================================================
    # OVERRIDE FORMATTING HOOK FOR SUMMA'S ARRAY-BASED PARAMETERS
    # ========================================================================

    def _format_parameter_value(self, param_name: str, value: float) -> Any:
        """
        Format parameter value for SUMMA.

        SUMMA uses arrays for most parameters:
        - Regionalization coefficients stay as scalars
        - Local parameters are expanded to HRU count
        - Basin parameters use single-element arrays
        - Depth parameters use single-element arrays
        - mizuRoute parameters use scalar values
        """
        # Regionalization coefficients are scalar — don't expand
        if self._use_regionalization and (param_name.endswith('_a') or param_name.endswith('_b')):
            return value
        if param_name in self.depth_params:
            return np.array([value])
        elif param_name in self.mizuroute_params:
            return value  # mizuRoute uses scalars
        elif param_name in self.basin_params:
            return np.array([value])
        else:
            # Local parameters - expand to HRU count
            return self._expand_to_hru_count(value)

    # ========================================================================
    # REGIONALIZATION HELPERS
    # ========================================================================

    def _parse_non_local_bounds(self) -> Dict[str, Dict[str, float]]:
        """Parse bounds for basin, depth, and mizuRoute params (non-local)."""
        bounds: Dict[str, Dict[str, float]] = {}

        if self.basin_params:
            basin_param_file = self.settings_dir / 'basinParamInfo.txt'
            bounds.update(self._parse_param_info_file(basin_param_file, self.basin_params))

        if self.depth_params:
            depth_bounds = get_depth_bounds()
            for param in self.depth_params:
                if param in depth_bounds:
                    bounds[param] = depth_bounds[param]

        if self.mizuroute_params:
            mizuroute_bounds = get_mizuroute_bounds()
            for param in self.mizuroute_params:
                if param in mizuroute_bounds:
                    bounds[param] = mizuroute_bounds[param]

        return bounds

    def _expand_regionalized_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Expand regionalization coefficients into per-HRU arrays.

        Separates coefficient params from non-coefficient params, runs the
        transfer function, applies physical constraints, and merges back.
        """
        self._init_regionalization()

        # Split coefficients from non-coefficient params (basin, depth, mizuroute)
        coeff_params: Dict[str, float] = {}
        other_params: Dict[str, Any] = {}

        coeff_names = set(self._regionalization.get_calibration_parameters().keys())
        for name, val in params.items():
            if name in coeff_names:
                coeff_params[name] = float(val) if not isinstance(val, (int, float)) else val
            else:
                other_params[name] = val

        # Expand coefficients → distributed per-HRU arrays
        distributed = self.expand_coefficients_to_distributed(coeff_params)

        # Merge: distributed local params + original non-local params
        merged = {}
        merged.update(distributed)
        merged.update(other_params)

        # Apply physical constraints on the expanded parameters
        merged = self._enforce_parameter_constraints(merged)

        return merged

    # ========================================================================
    # BOUNDS PARSING
    # ========================================================================

    def _parse_all_bounds(self) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from all parameter info files and allow config overrides."""
        bounds = {}

        # Parse local parameter bounds
        if self.local_params:
            local_param_file = self.settings_dir / 'localParamInfo.txt'
            local_bounds = self._parse_param_info_file(local_param_file, self.local_params)
            bounds.update(local_bounds)

        # Parse basin parameter bounds
        if self.basin_params:
            basin_param_file = self.settings_dir / 'basinParamInfo.txt'
            basin_bounds = self._parse_param_info_file(basin_param_file, self.basin_params)
            bounds.update(basin_bounds)

        # Add depth parameter bounds from central registry
        if self.depth_params:
            depth_bounds = get_depth_bounds()
            for param in self.depth_params:
                if param in depth_bounds:
                    bounds[param] = depth_bounds[param]

        # Add mizuRoute parameter bounds from central registry
        if self.mizuroute_params:
            mizuroute_bounds = get_mizuroute_bounds()
            for param in self.mizuroute_params:
                if param in mizuroute_bounds:
                    bounds[param] = mizuroute_bounds[param]
                else:
                    self.logger.warning(f"Unknown mizuRoute parameter: {param}")

        # Config-level overrides (highest priority)
        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='PARAMETER_BOUNDS')
        if config_bounds:
            self.logger.info(f"Applying {len(config_bounds)} parameter bound overrides from configuration")
            for param_name, limit_list in config_bounds.items():
                if len(limit_list) >= 2:
                    bounds[param_name] = {'min': float(limit_list[0]), 'max': float(limit_list[1])}
                    self.logger.debug(f"Overrode bounds for {param_name}: {bounds[param_name]}")

        return bounds

    def _parse_param_info_file(self, file_path: Path, param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from a SUMMA parameter info file."""
        bounds: Dict[str, Dict[str, float]] = {}

        if not file_path.exists():
            self.logger.error(f"Parameter file not found: {file_path}")
            return bounds

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue

                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 4:
                        continue

                    param_name = parts[0]
                    if param_name in param_names:
                        try:
                            min_val = float(parts[2].replace('d', 'e').replace('D', 'e'))
                            max_val = float(parts[3].replace('d', 'e').replace('D', 'e'))

                            if min_val > max_val:
                                min_val, max_val = max_val, min_val

                            if min_val == max_val:
                                range_val = abs(min_val) * 0.1 if min_val != 0 else 0.1
                                min_val -= range_val
                                max_val += range_val

                            bounds[param_name] = {'min': min_val, 'max': max_val}

                        except ValueError as e:
                            self.logger.error(f"Could not parse bounds for {param_name}: {str(e)}")

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading parameter file {file_path}: {str(e)}")

        return bounds

    # ========================================================================
    # INITIAL PARAMETER EXTRACTION
    # ========================================================================

    def _load_existing_optimized_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Load existing optimized parameters from default settings."""
        trial_params_path = self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')
        if trial_params_path == 'default':
            return None

        # Implementation would check for existing trialParams.nc file
        return None

    def _extract_default_parameters(self) -> Dict[str, np.ndarray]:
        """Extract default parameter values from parameter info files."""
        defaults = {}

        # Parse local parameters
        if self.local_params:
            local_defaults = self._parse_defaults_from_file(
                self.settings_dir / 'localParamInfo.txt',
                self.local_params
            )
            defaults.update(local_defaults)

        # Parse basin parameters
        if self.basin_params:
            basin_defaults = self._parse_defaults_from_file(
                self.settings_dir / 'basinParamInfo.txt',
                self.basin_params
            )
            defaults.update(basin_defaults)

        # Add depth parameters
        if self.depth_params:
            defaults['total_mult'] = np.array([1.0])
            defaults['shape_factor'] = np.array([1.0])

        # Add mizuRoute parameters
        if self.mizuroute_params:
            for param in self.mizuroute_params:
                defaults[param] = self._get_default_mizuroute_value(param)

        # Expand to HRU count
        return self._expand_defaults_to_hru_count(defaults)

    def _parse_defaults_from_file(self, file_path: Path, param_names: List[str]) -> Dict[str, np.ndarray]:
        """Parse default values from parameter info file."""
        defaults: Dict[str, np.ndarray] = {}

        if not file_path.exists():
            return defaults

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue

                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        param_name = parts[0]
                        if param_name in param_names:
                            try:
                                default_val = float(parts[1].replace('d', 'e').replace('D', 'e'))
                                defaults[param_name] = np.array([default_val])
                            except ValueError:
                                continue
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error parsing defaults from {file_path}: {str(e)}")

        return defaults

    def _get_default_mizuroute_value(self, param_name: str) -> float:
        """Get default value for mizuRoute parameter."""
        defaults = {
            'velo': 1.0,
            'diff': 1000.0,
            'mann_n': 0.025,
            'wscale': 0.001,
            'fshape': 2.5,
            'tscale': 86400
        }
        return defaults.get(param_name, 1.0)

    def _expand_defaults_to_hru_count(self, defaults: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Expand parameter defaults to match HRU count."""
        try:
            # Get HRU count from attributes file
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)

            expanded_defaults = {}
            routing_params = ['routingGammaShape', 'routingGammaScale']

            for param_name, values in defaults.items():
                if param_name in self.basin_params or param_name in routing_params:
                    expanded_defaults[param_name] = values
                elif param_name in self.depth_params or param_name in self.mizuroute_params:
                    expanded_defaults[param_name] = values
                else:
                    expanded_defaults[param_name] = np.full(num_hrus, values[0])

            return expanded_defaults

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error expanding defaults: {str(e)}")
            return defaults

    def _expand_to_hru_count(self, value: float) -> np.ndarray:
        """Expand single value to HRU count."""
        try:
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            return np.full(num_hrus, value)
        except (OSError, IOError, KeyError) as e:
            self.logger.debug(f"Could not read HRU count from attributes, using single value: {e}")
            return np.array([value])

    # ========================================================================
    # SOIL DEPTH CALIBRATION
    # ========================================================================

    def _load_original_depths(self) -> Optional[np.ndarray]:
        """Load original soil depths from coldState.nc."""
        try:
            coldstate_path = self.settings_dir / self._get_config_value(lambda: self.config.model.summa.coldstate, default='coldState.nc', dict_key='SETTINGS_SUMMA_COLDSTATE')

            if not coldstate_path.exists():
                return None

            with nc.Dataset(coldstate_path, 'r') as ds:
                if 'mLayerDepth' in ds.variables:
                    return ds.variables['mLayerDepth'][:, 0].copy()

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error loading original depths: {str(e)}")

        return None

    def _update_soil_depths(self, params: Dict[str, np.ndarray]) -> bool:
        """Update soil depths in coldState.nc."""
        if self.original_depths is None:
            return False

        try:
            total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
            shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']

            # Calculate new depths using shape method
            new_depths = self._calculate_new_depths(total_mult, shape_factor)
            if new_depths is None:
                return False

            # Calculate layer heights
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]

            # Update coldState.nc
            coldstate_path = self.settings_dir / self._get_config_value(lambda: self.config.model.summa.coldstate, default='coldState.nc', dict_key='SETTINGS_SUMMA_COLDSTATE')

            # Track NetCDF read-modify-write operation (secondary IOPS bottleneck)
            with self._profiler_ctx.track_netcdf_write(str(coldstate_path)):
                with nc.Dataset(coldstate_path, 'r+') as ds:
                    if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                        return False

                    num_hrus = ds.dimensions['hru'].size
                    for h in range(num_hrus):
                        ds.variables['mLayerDepth'][:, h] = new_depths
                        ds.variables['iLayerHeight'][:, h] = heights

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating soil depths: {str(e)}")
            return False

    def _calculate_new_depths(self, total_mult: float, shape_factor: float) -> Optional[np.ndarray]:
        """Calculate new soil depths using shape method."""
        if self.original_depths is None:
            return None

        arr = self.original_depths.copy()
        n = len(arr)
        idx = np.arange(n)

        # Calculate shape weights
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)

        # Normalize weights
        w /= w.mean()

        # Apply multipliers
        new_depths = arr * w * total_mult

        return new_depths

    # ========================================================================
    # MIZUROUTE PARAMETER HANDLING
    # ========================================================================

    def _update_mizuroute_parameters(self, params: Dict) -> bool:
        """Update mizuRoute parameters in param.nml.default."""
        try:
            mizuroute_settings_dir = self.settings_dir.parent / "mizuRoute"
            param_file = mizuroute_settings_dir / "param.nml.default"

            if not param_file.exists():
                return True  # Skip if file doesn't exist

            # Track file read operation
            with self._profiler_ctx.track_file_read(str(param_file)):
                with open(param_file, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Update parameters
            updated_content = content
            for param_name in self.mizuroute_params:
                if param_name in params:
                    param_value = params[param_name]
                    pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'

                    if param_name in ['tscale']:
                        replacement = rf'\g<1>{int(param_value)}'
                    else:
                        replacement = rf'\g<1>{param_value:.6f}'

                    updated_content = re.sub(pattern, replacement, updated_content)

            # Track file write operation
            with self._profiler_ctx.track_file_write(str(param_file), size_bytes=len(updated_content)):
                with open(param_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating mizuRoute parameters: {str(e)}")
            return False

    # ========================================================================
    # TRIAL PARAMETERS FILE GENERATION
    # ========================================================================

    def _generate_trial_params_file(self, params: Dict[str, np.ndarray]) -> bool:
        """Generate trialParams.nc file with proper dimensions."""
        try:
            trial_params_path = self.settings_dir / self._get_config_value(lambda: self.config.model.summa.trialparams, default='trialParams.nc', dict_key='SETTINGS_SUMMA_TRIALPARAMS')

            # Get HRU and GRU counts from attributes
            with self._profiler_ctx.track_file_read(str(self.attr_file_path)):
                with xr.open_dataset(self.attr_file_path) as ds:
                    num_hrus = ds.sizes.get('hru', 1)
                    num_grus = ds.sizes.get('gru', 1)

                    # Get original hruId values
                    if 'hruId' in ds.variables:
                        original_hru_ids = ds.variables['hruId'][:].copy()
                    else:
                        original_hru_ids = np.arange(1, num_hrus + 1)
                        self.logger.warning(f"hruId not found in attributes.nc, using sequential IDs 1 to {num_hrus}")

                    # Get original gruId values
                    if 'gruId' in ds.variables:
                        original_gru_ids = ds.variables['gruId'][:].copy()
                    else:
                        original_gru_ids = np.arange(1, num_grus + 1)
                        self.logger.warning(f"gruId not found in attributes.nc, using sequential IDs 1 to {num_grus}")

            # Define parameter levels
            routing_params = ['routingGammaShape', 'routingGammaScale']

            # Track NetCDF write operation (primary IOPS bottleneck)
            with self._profiler_ctx.track_netcdf_write(str(trial_params_path)):
                with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                    # Create dimensions
                    output_ds.createDimension('hru', num_hrus)
                    output_ds.createDimension('gru', num_grus)

                    # Create coordinate variables with ORIGINAL ID values
                    hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                    hru_var[:] = original_hru_ids

                    gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                    gru_var[:] = original_gru_ids

                    # Add parameters
                    for param_name, param_values in params.items():
                        param_values_array = np.asarray(param_values)

                        if param_name in routing_params or param_name in self.basin_params:
                            # GRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"

                            if len(param_values_array) == 1:
                                param_var[:] = param_values_array[0]
                            else:
                                param_var[:] = param_values_array[:num_grus]
                        else:
                            # HRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"

                            if len(param_values_array) == num_hrus:
                                param_var[:] = param_values_array
                            elif len(param_values_array) == 1:
                                param_var[:] = param_values_array[0]
                            else:
                                param_var[:] = param_values_array[:num_hrus]

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error generating trial params file: {str(e)}")
            return False


# Backward compatibility alias
ParameterManager = SUMMAParameterManager
