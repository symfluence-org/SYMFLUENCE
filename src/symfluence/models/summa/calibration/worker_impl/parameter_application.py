#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Parameter Application for SUMMA Workers

This module contains functions for applying optimization parameters
to SUMMA model configuration files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.mixins.config import resolve_config_value

from ..forcing_scaling import PRECIP_MULT_PARAM, apply_precip_multiplier
from ..parameter_files import (
    has_depth_params,
    resolve_settings_file_names,
    select_trial_params,
    update_mizuroute_params,
    update_soil_depths,
    write_trial_params,
)


def _configured_list(value) -> List[str]:
    return [p.strip() for p in str(value or '').split(',') if p.strip()]


def _mizuroute_params(config) -> List[str]:
    """mizuRoute parameters under calibration (empty unless CALIBRATE_MIZUROUTE)."""
    if not resolve_config_value(config, lambda: config.model.mizuroute.calibrate, False, 'CALIBRATE_MIZUROUTE'):
        return []
    return _configured_list(resolve_config_value(
        config, lambda: config.model.mizuroute.params_to_calibrate, 'velo,diff', 'MIZUROUTE_PARAMS_TO_CALIBRATE'))


def _apply_parameters_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Apply a trial parameter set to the SUMMA (and mizuRoute) settings files.

    Uses the writers in ``calibration.parameter_files`` that
    ``SUMMAParameterManager.update_model_files`` uses too, with the settings
    file names taken from config.
    """
    try:
        config = task_data['config']
        logger.debug(f"Applying parameters: {list(params.keys())} (consistent method)")

        mizuroute_params = _mizuroute_params(config)

        # 1. Handle soil depth parameters
        if has_depth_params(params):
            logger.debug("Updating soil depths (consistent)")
            if not _update_soil_depths_worker(params, task_data, settings_dir, logger, debug_info):
                return False

        # 2. Handle mizuRoute parameters
        if mizuroute_params and any(p in params for p in mizuroute_params):
            logger.debug("Updating mizuRoute parameters (consistent)")
            if not _update_mizuroute_params_worker(params, task_data, logger, debug_info):
                return False

        # 3. Rescale forcing precipitation (pseudo-parameter, not in trialParams)
        if PRECIP_MULT_PARAM in params:
            file_manager = resolve_config_value(config, lambda: config.model.summa.filemanager, None, 'SETTINGS_SUMMA_FILEMANAGER') or 'fileManager.txt'
            if not apply_precip_multiplier(settings_dir, params[PRECIP_MULT_PARAM], logger, file_manager):
                debug_info['errors'].append(f"Failed to apply {PRECIP_MULT_PARAM}")
                return False

        # 4. Generate trial parameters file (depth, mizuRoute and forcing
        #    pseudo-parameters are not SUMMA parameters and stay out of it)
        hydrological_params = select_trial_params(params, mizuroute_params)

        if hydrological_params:
            logger.debug(f"Generating trial parameters file with: {list(hydrological_params.keys())} (consistent)")
            if not _generate_trial_params_worker(hydrological_params, settings_dir, logger, debug_info, config=config):
                return False

        logger.debug("Parameter application completed successfully (consistent)")
        return True

    except (FileNotFoundError, IOError, ValueError) as e:
        error_msg = f"Error applying parameters (consistent): {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _update_soil_depths_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Rescale the soil layers of the configured coldState file."""
    original_depths_list = task_data.get('original_depths')
    if not original_depths_list:
        logger.debug("No original depths provided, skipping soil depth update")
        return True

    names = resolve_settings_file_names(task_data.get('config'))
    return update_soil_depths(
        Path(settings_dir) / names.coldstate, original_depths_list, params, logger, debug_info)


def _update_mizuroute_params_worker(params: Dict, task_data: Dict, logger, debug_info: Dict) -> bool:
    """Write calibrated mizuRoute parameters into param.nml.default."""
    try:
        param_file = Path(task_data['mizuroute_settings_dir']) / "param.nml.default"
        return update_mizuroute_params(
            param_file, _mizuroute_params(task_data['config']), params, logger, debug_info)
    except (FileNotFoundError, IOError, ValueError) as e:
        error_msg = f"Error updating mizuRoute params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _generate_trial_params_worker(params: Dict, settings_dir: Path, logger, debug_info: Dict,
                                  config: Optional[Dict] = None) -> bool:
    """Write the configured trialParams file (in place when its layout allows)."""
    names = resolve_settings_file_names(config)
    basin_params = _configured_list(resolve_config_value(
        config, lambda: config.model.summa.basin_params_to_calibrate, None, 'BASIN_PARAMS_TO_CALIBRATE'))
    settings_dir = Path(settings_dir)
    return write_trial_params(
        settings_dir / names.trialparams, settings_dir / names.attributes, params,
        logger, debug_info, gru_level_params=basin_params)
