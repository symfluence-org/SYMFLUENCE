# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in Wflow forward runs: precipitation scale in the run TOML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from symfluence.core.mixins.config import resolve_config_value


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    from .calibration.parameter_manager import WflowParameterManager

    toml_name = resolve_config_value(config, lambda: config.model.wflow.config_file,
                                     dict_key='WFLOW_CONFIG_FILE')
    toml_path = Path(project_dir) / 'settings' / 'WFLOW' / str(toml_name)
    scales = {k: float(v) for k, v in info['native'].items()
              if k in WflowParameterManager.FORCING_SCALE_PARAMS}
    if scales:
        WflowParameterManager.set_forcing_scales(toml_path, scales)
    return {'toml': str(toml_path), 'scales': scales}
