# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in PRMS forward runs.

The common values (``FC_PRECIP_MULT``, ``FC_SNOW_MULT``, ``FC_TEMP_OFFSET``)
are written into the run's ``params.dat`` through the same code path as
calibration (:func:`symfluence.models.prms.parameter_file.update_parameter_file`):
``rain_adj``/``snow_adj``/``tmax_adj``/``tmin_adj`` in station mode, the
``*_cbh_adj`` blocks in climate_hru mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.exceptions import ConfigurationError, ModelExecutionError
from symfluence.core.mixins.config import resolve_config_value

from .parameter_file import BLOCK_SEPARATOR, _block_name, forcing_correction_blocks, update_parameter_file


def write_forward_correction(param_file: Path, native: Dict[str, Any]) -> Dict[str, float]:
    """Write ``native`` (common pseudo-parameters) into ``param_file``; return the block values."""
    if not param_file.exists():
        raise ModelExecutionError(f'Parameter file not found: {param_file}')
    names = {_block_name(b) for b in param_file.read_text(encoding='utf-8').split(BLOCK_SEPARATOR)}
    blocks = forcing_correction_blocks(native, names)
    if blocks:
        try:
            update_parameter_file(param_file, dict(native))
        except ConfigurationError as e:
            raise ModelExecutionError(f'Could not apply the forcing correction to {param_file}: {e}') from e
    return blocks


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    name = resolve_config_value(config, lambda: config.model.prms.parameter_file,
                                dict_key='PRMS_PARAMETER_FILE') or 'params.dat'
    param_file = Path(project_dir) / 'settings' / 'PRMS' / str(name)
    return {'param_file': str(param_file), 'blocks': write_forward_correction(param_file, info['native'])}
