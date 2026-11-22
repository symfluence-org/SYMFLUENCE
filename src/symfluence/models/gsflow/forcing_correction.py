# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in GSFLOW forward runs.

GSFLOW's surface component is PRMS: the common values are written into the
PRMS parameter file of the run settings exactly as for PRMS
(:func:`symfluence.models.prms.forcing_correction.write_forward_correction`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.prms.forcing_correction import write_forward_correction


def settings_dir(config: Any, project_dir: Path) -> Path:
    configured = resolve_config_value(config, lambda: config.model.gsflow.settings_path,
                                      dict_key='SETTINGS_GSFLOW_PATH')
    if configured and str(configured) != 'default':
        return Path(configured)
    return Path(project_dir) / 'settings' / 'GSFLOW'


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    name = resolve_config_value(config, lambda: config.model.gsflow.parameter_file,
                                dict_key='GSFLOW_PARAMETER_FILE') or 'params.dat'
    param_file = settings_dir(config, project_dir) / str(name)
    return {'param_file': str(param_file), 'blocks': write_forward_correction(param_file, info['native'])}
