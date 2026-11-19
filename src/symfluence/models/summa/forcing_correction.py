# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in SUMMA forward runs.

precip_mult goes through the calibration mechanism
(``forcing_scaling.apply_precip_multiplier``:
scaled copies of the forcing, file manager redirected to them, source path
recorded), applied to the run settings directory. frozenPrecipMultip is
merged into the existing trialParams file, keeping every other value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value


def settings_dir(config: Any, project_dir: Path) -> Path:
    configured = resolve_config_value(config, lambda: config.paths.settings_summa_path,
                                      dict_key='SETTINGS_SUMMA_PATH')
    if configured and str(configured) != 'default':
        return Path(configured)
    return Path(project_dir) / 'settings' / 'SUMMA'


def _merge_trial_param(settings: Path, config: Any, name: str, value: float, logger: Any) -> None:
    import netCDF4 as nc
    import numpy as np

    from .calibration.parameter_files import write_trial_params
    from .calibration.worker_impl.parameter_application import resolve_settings_file_names

    names = resolve_settings_file_names(config, settings)
    trial = settings / names.trialparams
    params: Dict[str, Any] = {}
    if trial.exists():
        with nc.Dataset(trial) as ds:
            for var in ds.variables:
                if var not in ('hruId', 'gruId'):
                    params[var] = np.asarray(ds[var][:], dtype=float)
    params[name] = value
    if not write_trial_params(trial, settings / names.attributes, params, logger):
        raise ModelExecutionError(f'Could not write {name} to {trial}')


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    from .calibration.forcing_scaling import SOURCE_RECORD, apply_precip_multiplier

    native = info['native']
    settings = settings_dir(config, project_dir)
    file_manager = resolve_config_value(config, lambda: config.model.summa.filemanager,
                                        dict_key='SETTINGS_SUMMA_FILEMANAGER') or 'fileManager.txt'
    details: Dict[str, Any] = {'settings_dir': str(settings)}
    if 'precip_mult' in native:
        m = float(native['precip_mult'])
        if m == 1.0 and not (settings / SOURCE_RECORD).exists():
            details['precip_mult'] = 'identity, forcing untouched'
        elif not apply_precip_multiplier(settings, m, logger, file_manager):
            raise ModelExecutionError(f'Could not apply precip_mult={m} in {settings}')
        else:
            details['precip_mult'] = m
    if 'frozenPrecipMultip' in native:
        _merge_trial_param(settings, config, 'frozenPrecipMultip',
                           float(native['frozenPrecipMultip']), logger)
        details['frozenPrecipMultip'] = float(native['frozenPrecipMultip'])
    return details
