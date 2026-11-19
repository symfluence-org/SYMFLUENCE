# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in MESH forward runs.

Writes the run forcing MESH_forcing_safe.nc in forcing/MESH_input from
its recorded unscaled source with PRECIP_MULT / TEMP_OFFSET applied
(``forcing_scaling.apply_forcing_corrections``,
the calibration mechanism; the shared source is never written through).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    from symfluence.core.mixins.project import resolve_data_subdir

    from .calibration.forcing_scaling import SOURCE_RECORD, apply_forcing_corrections

    directory = resolve_data_subdir(Path(project_dir), 'forcing') / 'MESH_input'
    native = info['native']
    identity = all(float(v) == (1.0 if k == 'PRECIP_MULT' else 0.0) for k, v in native.items())
    if identity and not (directory / SOURCE_RECORD).exists():
        return {'forcing_dir': str(directory), 'applied': 'identity, forcing untouched'}
    applied = apply_forcing_corrections(directory, native)
    return {'forcing_dir': str(directory), 'applied': applied}
