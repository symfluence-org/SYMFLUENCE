#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
FUSE Parameter Manager.

Handles FUSE parameter bounds, normalization, and NetCDF parameter-file
structure (including correct ``par`` dimension indexing).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.parameters.parameter_bounds_registry import get_fuse_bounds
from symfluence.core.logging_utils import log_once
from symfluence.core.registries import R

# Mapping of FUSE decisions to the parameters they require for meaningful calibration.
# If a decision is active but its required params are not being calibrated, the model
# structure effectively has uncalibrated degrees of freedom (poisoned pathways).
DECISION_REQUIRED_PARAMS = {
    # TOPMODEL surface runoff needs LOGLAMB (log of topographic index) and TISHAPE
    'tmdl_param': {'LOGLAMB', 'TISHAPE'},
    # Interflow requires IFLWRTE (interflow rate)
    'intflwsome': {'IFLWRTE'},
    # Gamma routing needs TIMEDELAY
    'rout_gamma': {'TIMEDELAY'},
    # Temperature index snow model needs MBASE, MFMAX, MFMIN, PXTEMP
    'temp_index': {'MBASE', 'MFMAX', 'MFMIN', 'PXTEMP'},
    # Percolation from field capacity to saturation needs PERCRTE, PERCEXP
    'perc_f2sat': {'PERCRTE', 'PERCEXP'},
    # Percolation from wilting point to saturation needs PERCRTE, PERCEXP
    'perc_w2sat': {'PERCRTE', 'PERCEXP'},
    # SAC-style lower-layer percolation needs PERCRTE
    'perc_lower': {'PERCRTE'},
}


@R.parameter_managers.add('FUSE')
class FUSEParameterManager(BaseParameterManager):
    """Handles FUSE parameter bounds, normalization, and file updates."""

    def __init__(self, config: Dict, logger: logging.Logger, fuse_settings_dir: Path):
        # Initialize base class
        super().__init__(config, logger, fuse_settings_dir)

        # FUSE-specific setup
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default=None,
            dict_key='DOMAIN_NAME'
        )

        # Parse FUSE parameters to calibrate
        fuse_params_str = self._get_config_value(
            lambda: self.config.model.fuse.params_to_calibrate,
            default=None,
            dict_key='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
        )
        # Handle None, empty string, or 'default' as signal to use default parameter list
        if fuse_params_str is None or fuse_params_str == '' or fuse_params_str == 'default':
            # Provide sensible defaults if not specified
            self.logger.info("Using default FUSE calibration parameters.")
            fuse_params_str = 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE'

        self.fuse_params = [p.strip() for p in fuse_params_str.split(',') if p.strip()]

        # Path to FUSE parameter files
        self.data_dir = Path(self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.',
            dict_key='SYMFLUENCE_DATA_DIR'
        ))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.fuse_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'FUSE'
        self.fuse_setup_dir = self.project_dir / 'settings' / 'FUSE'
        from symfluence.models.fuse.calibration.file_manager import resolve_fuse_id
        raw_fuse_id = self._get_config_value(lambda: self.config.model.fuse.file_id, default=self.experiment_id, dict_key='FUSE_FILE_ID')
        self.fuse_id = resolve_fuse_id({'FUSE_FILE_ID': raw_fuse_id, 'EXPERIMENT_ID': self.experiment_id})

        # Parameter file path. Calibration writes to para_def.nc (read by FUSE
        # run_pre mode); para_sce/para_best are produced by FUSE's own SCE runs.
        self.para_def_path = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"
        self.param_file_path = self.para_def_path

        # Regionalization setup (lazy-initialized)
        self._regionalization_method = self._get_config_value(
            lambda: self.config.model.fuse.parameter_regionalization,
            default='lumped',
            dict_key='PARAMETER_REGIONALIZATION'
        )
        use_tf = self._get_config_value(
            lambda: self.config.model.fuse.use_transfer_functions,
            default=False,
            dict_key='USE_TRANSFER_FUNCTIONS'
        )
        if use_tf and self._regionalization_method == 'lumped':
            self._regionalization_method = 'transfer_function'
        self._regionalization = None
        self._regionalization_initialized = False

    # ========================================================================
    # REGIONALIZATION
    # ========================================================================

    def _init_regionalization(self):
        """Lazily initialize the regionalization strategy."""
        if self._regionalization_initialized:
            return
        self._regionalization_initialized = True
        if self._regionalization_method == 'lumped':
            return

        import pandas as pd

        from symfluence.core.calibration.regionalization.strategies import RegionalizationFactory
        from symfluence.models.fuse.calibration.parameter_regionalization import FUSE_DEFAULT_PARAM_CONFIG

        raw_bounds = self._get_raw_fuse_bounds()
        tuple_bounds: Dict[str, Tuple[float, float]] = {}
        for p in self.fuse_params:
            if p in raw_bounds:
                tuple_bounds[p] = (raw_bounds[p]['min'], raw_bounds[p]['max'])

        attrs_path = self._get_config_value(
            lambda: None,
            default=None, dict_key='TRANSFER_FUNCTION_ATTRIBUTES'
        )
        attributes = None
        n_units = 1
        if attrs_path and attrs_path != 'default' and Path(attrs_path).exists():
            attributes = pd.read_csv(attrs_path)
            n_units = len(attributes)
        else:
            para_def = self.fuse_setup_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"
            if para_def.exists():
                with xr.open_dataset(para_def) as ds:
                    n_units = ds.sizes.get('par', 1)

        param_config = self._get_config_value(
            lambda: None,
            default=None, dict_key='TRANSFER_FUNCTION_PARAM_CONFIG'
        ) or FUSE_DEFAULT_PARAM_CONFIG

        config: Dict[str, Any] = {'TRANSFER_FUNCTION_PARAM_CONFIG': param_config}
        b_bounds = self._get_config_value(
            lambda: self.config.optimization.transfer_function_b_bounds,
            default=None, dict_key='TRANSFER_FUNCTION_B_BOUNDS'
        )
        if b_bounds:
            config['TRANSFER_FUNCTION_B_BOUNDS'] = tuple(b_bounds)

        self._regionalization = RegionalizationFactory.create(
            method=self._regionalization_method, param_bounds=tuple_bounds,
            n_units=n_units, config=config, attributes=attributes, logger=self.logger,
        )
        self.logger.info(
            f"FUSE regionalization: {self._regionalization.name} — "
            f"{len(self._regionalization.get_calibration_parameters())} coefficients "
            f"for {len(tuple_bounds)} params across {n_units} units"
        )

    @property
    def _use_regionalization(self) -> bool:
        """Whether regionalization is active (non-lumped)."""
        return self._regionalization_method != 'lumped'

    def _get_raw_fuse_bounds(self) -> Dict[str, Dict[str, float]]:
        """Load raw FUSE parameter bounds with config overrides (no regionalization)."""
        bounds = get_fuse_bounds()
        config_bounds = self._get_config_value(
            lambda: self.config.model.fuse.param_bounds,
            default=None,
            dict_key='FUSE_PARAM_BOUNDS'
        )
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)
        return bounds

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS FROM BASE CLASS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return parameter/coefficient names depending on regionalization mode."""
        if self._use_regionalization:
            self._init_regionalization()
            if self._regionalization:
                return list(self._regionalization.get_calibration_parameters().keys())
        return self.fuse_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return FUSE parameter bounds or regionalization coefficient bounds.

        Priority:
        1. Regionalization coefficient bounds (dynamic, from RegionalizationFactory)
        2. FUSE_PARAM_BOUNDS from config (user-specified)
        3. Registry defaults for any parameters not in config
        """
        if self._use_regionalization:
            self._init_regionalization()
            if self._regionalization:
                bounds: Dict[str, Dict[str, float]] = {}
                transforms = self._regionalization.get_coefficient_transforms()
                for name, (lo, hi) in self._regionalization.get_calibration_parameters().items():
                    entry: Dict[str, Any] = {'min': lo, 'max': hi}
                    t = transforms.get(name, 'linear')
                    if t != 'linear':
                        entry['transform'] = t
                    bounds[name] = entry
                return bounds
        return self._get_raw_fuse_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        Update FUSE constraints file with new parameter values.

        FUSE's run_def mode regenerates para_def.nc from the constraints file,
        so we must modify the constraints file to change parameter values.

        When using parameter regionalization (transfer_function, zones, distributed),
        the worker handles applying the coefficients via the regionalization system.
        """
        if self._use_regionalization:
            # In regionalization mode, the worker handles parameter application
            # via _apply_regionalization(). Skip constraints file update here.
            self.logger.debug(
                f"Regionalization mode ({self._regionalization_method}): "
                f"skipping constraints file update (handled by worker)"
            )
            return True

        return self._update_constraints_file(params)

    def _update_constraints_file(self, params: Dict[str, float]) -> bool:
        """Update the fuse_zConstraints_snow.txt file with new default values.

        FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
        The default value column starts at position 4 and is exactly 9 characters.
        """
        try:
            constraints_file = self.fuse_setup_dir / 'fuse_zConstraints_snow.txt'

            if not constraints_file.exists():
                self.logger.error(f"FUSE constraints file not found: {constraints_file}")
                return False

            # Read the constraints file with encoding fallback
            try:
                with open(constraints_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(
                    f"UTF-8 decode error reading {constraints_file}, falling back to latin-1"
                )
                with open(constraints_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Fortran format: (L1,1X,I1,1X,3(F9.3,1X),...)
            # Default value column: position 4-12 (9 chars, F9.3 format)
            DEFAULT_VALUE_START = 4
            DEFAULT_VALUE_WIDTH = 9

            updated_lines = []
            params_updated = set()

            for line in lines:
                # Skip header line (starts with '(') and comment lines
                stripped = line.strip()
                if stripped.startswith('(') or stripped.startswith('*') or stripped.startswith('!'):
                    updated_lines.append(line)
                    continue

                # Check if this line contains any of our parameters
                updated = False
                for param_name, value in params.items():
                    # Match exact parameter name (avoid partial matches)
                    parts = line.split()
                    if len(parts) >= 13 and param_name in parts:
                        # Format value to exactly 9 characters (F9.3 format)
                        new_value = f"{value:9.3f}"

                        # Replace the fixed-width column in the line
                        # Position 4-12 is the default value (9 characters)
                        if len(line) > DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:
                            new_line = (
                                line[:DEFAULT_VALUE_START] +
                                new_value +
                                line[DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:]
                            )
                            updated_lines.append(new_line)
                            params_updated.add(param_name)
                            updated = True
                            break

                if not updated:
                    updated_lines.append(line)

            # Write updated constraints file
            with open(constraints_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            if params_updated:
                self.logger.debug(f"Updated FUSE constraints: {params_updated}")

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating FUSE constraints file: {e}", exc_info=True)
            return False

    # Note: get_initial_parameters() is already defined below and matches the signature
    # Note: Parameter bounds are now provided by the central ParameterBoundsRegistry
    # Note: all_param_names property and get_parameter_bounds() are inherited from BaseParameterManager

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from existing FUSE parameter file or coefficient bounds."""
        try:
            if self._use_regionalization:
                # In regionalization mode, return initial guesses for coefficients
                # (not raw parameters from file, since we're calibrating coefficients)
                return self._get_default_initial_values()

            if not self.param_file_path.exists():
                log_once(self.logger, logging.WARNING, key=f'fuse-param-file-missing-{self.param_file_path}',
                        message=f"FUSE parameter file not found: {self.param_file_path}")
                return self._get_default_initial_values()

            with xr.open_dataset(self.param_file_path) as ds:
                params = {}
                for param_name in self.fuse_params:
                    if param_name in ds.variables:
                        # Get the parameter value (assuming parameter set 0)
                        params[param_name] = float(ds[param_name].isel(par=0).values)
                    else:
                        log_once(self.logger, logging.WARNING, key=f'fuse-param-missing-{param_name}',
                                 message=f"Parameter {param_name} not found in file")
                        # Use default value from bounds
                        bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                        params[param_name] = (bounds['min'] + bounds['max']) / 2

                return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {str(e)}", exc_info=True)
            return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (or coefficient values in regionalization mode)."""
        config_initial = self._get_config_value(lambda: None, default=None, dict_key='INITIAL_PARAMETERS')
        if config_initial and isinstance(config_initial, dict):
            params = {}
            for name in self.all_param_names:
                if name in config_initial:
                    params[name] = float(config_initial[name])
                else:
                    b = self.param_bounds.get(name, {'min': 0, 'max': 1})
                    params[name] = (b['min'] + b['max']) / 2
            self.logger.info(f"Using INITIAL_PARAMETERS from config ({sum(1 for n in params if n in config_initial)}/{len(params)} specified)")
            return params
        params = {}
        for param_name, param_bounds_dict in self.param_bounds.items():
            params[param_name] = (param_bounds_dict['min'] + param_bounds_dict['max']) / 2
        return params

    def validate_params_for_decisions(self, decisions_path: Path) -> List[str]:
        """
        Validate that calibrated parameters cover the active model decisions.

        Reads the FUSE decisions file, checks each active decision against
        DECISION_REQUIRED_PARAMS, and warns if required parameters are missing
        from the calibration set.

        Args:
            decisions_path: Path to the fuse_zDecisions_*.txt file

        Returns:
            List of warning messages (empty if all OK)
        """
        warnings_list: List[str] = []

        if not decisions_path.exists():
            self.logger.debug(f"Decisions file not found for validation: {decisions_path}")
            return warnings_list

        # Parse active decisions from the file
        active_decisions = {}
        try:
            with open(decisions_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Lines 2-10 (1-indexed) contain decisions: "value KEY ! comment"
            for line in lines[1:10]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    decision_value = parts[0]
                    decision_key = parts[1]
                    active_decisions[decision_key] = decision_value
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.debug(f"Could not parse decisions file: {e}", exc_info=True)
            return warnings_list

        calibrated_params = set(self.fuse_params)

        for decision_key, decision_value in active_decisions.items():
            required = DECISION_REQUIRED_PARAMS.get(decision_value)
            if required:
                missing = required - calibrated_params
                if missing:
                    msg = (
                        f"Decision {decision_key}={decision_value} requires parameters "
                        f"{missing} but they are not being calibrated. "
                        f"This may cause poor model performance."
                    )
                    warnings_list.append(msg)
                    self.logger.warning(f"FUSE decision-param mismatch: {msg}")

            # Warn about no_snowmod for catchments with snow params configured
            if decision_value == 'no_snowmod':
                snow_params_in_calibration = calibrated_params & {
                    'MBASE', 'MFMAX', 'MFMIN', 'PXTEMP', 'LAPSE'
                }
                if snow_params_in_calibration:
                    msg = (
                        f"SNOWM=no_snowmod but snow parameters {snow_params_in_calibration} "
                        f"are being calibrated. This is likely wrong for a catchment with "
                        f"snow processes. Consider using SNOWM=temp_index."
                    )
                    warnings_list.append(msg)
                    self.logger.warning(f"FUSE snow mismatch: {msg}")

        return warnings_list
