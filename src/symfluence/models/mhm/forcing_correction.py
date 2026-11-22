# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in mHM forward runs.

Uses the calibration mechanism (MHMWorker._write_adjusted_precip): a
corrected pre.nc in settings/MHM/forcing_adjusted with mhm.nml
pointed at it. Other forcing-side values a calibration wrote there (the
glacier ice degree-day factor) are carried over from its
forcing_adjustment.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from symfluence.core.exceptions import ModelExecutionError


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    from .calibration.worker import MHMWorker
    from .parameters import FORCING_PARAMS

    native = {k: float(v) for k, v in info['native'].items()}
    settings = Path(project_dir) / 'settings' / 'MHM'
    summary = settings / 'forcing_adjusted' / 'forcing_adjustment.json'
    params: Dict[str, Any] = {}
    if summary.exists():
        previous = json.loads(summary.read_text())
        params.update({k: v for k, v in previous.items() if k in FORCING_PARAMS})
    elif native.get('precipCorrectionFactor', 1.0) == 1.0:
        return {'settings_dir': str(settings), 'applied': 'identity, forcing untouched'}
    params.update(native)
    flat = config
    if hasattr(config, 'model_dump'):
        from symfluence.core.config.flattening import flatten_nested_config
        flat = flatten_nested_config(config)
    worker = MHMWorker(config=flat, logger=logger)
    if not worker._write_adjusted_precip(settings, Path(project_dir), params, flat):
        raise ModelExecutionError(f'Could not write corrected mHM precipitation in {settings}')
    return {'settings_dir': str(settings), 'applied': params}
