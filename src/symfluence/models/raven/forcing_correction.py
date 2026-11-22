# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in Raven forward runs.

Raven's own gauge corrections carry it: ``PRECIP_CORR`` -> ``:RainCorrection``,
``PRECIP_CORR * SNOW_CORR`` -> ``:SnowCorrection`` and ``TEMP_CORR`` ->
``:TemperatureCorrection`` in ``model.rvt`` (see :mod:`.native`). Calibration
needs nothing extra: the worker writes every parameter it is given. A forward
run edits those three lines of ``settings/RAVEN/model.rvt`` in place, keeping
any quantity the correction does not set (e.g. a calibrated ``SNOW_CORR``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator

from symfluence.core.exceptions import ModelExecutionError

_KEYS = {'rain': ':RainCorrection', 'snow': ':SnowCorrection', 'temp': ':TemperatureCorrection'}


def _values(text: str, key: str) -> list[float]:
    return [float(v) for v in re.findall(rf'^\s*{key}\s+(\S+)\s*$', text, re.MULTILINE)]


def set_gauge_corrections(rvt: Path, native: Dict[str, float]) -> Dict[str, float]:
    """Set Raven gauge corrections in ``rvt`` from native PRECIP_CORR / SNOW_CORR / TEMP_CORR.

    Returns the per-gauge values written (first gauge).
    """
    text = Path(rvt).read_text(encoding='utf-8')
    rain, snow, temp = (_values(text, _KEYS[k]) for k in ('rain', 'snow', 'temp'))
    if not rain or len(rain) != len(snow) or len(rain) != len(temp):
        raise ModelExecutionError(f'Unexpected gauge correction lines in {rvt}')
    new_rain, new_snow, new_temp = [], [], []
    for r0, s0, t0 in zip(rain, snow, temp):
        snow_corr = s0 / r0 if r0 else 1.0
        r = float(native.get('PRECIP_CORR', r0))
        sc = float(native.get('SNOW_CORR', snow_corr))
        new_rain.append(r)
        new_snow.append(r * sc)
        new_temp.append(float(native.get('TEMP_CORR', t0)))

    for key, values in (('rain', new_rain), ('snow', new_snow), ('temp', new_temp)):
        it = iter(values)

        def replace(m: 're.Match[str]', it: Iterator[float] = it) -> str:
            return f'{m.group(1)}{next(it)}{m.group(2)}'

        text = re.sub(rf'^(\s*{_KEYS[key]}\s+)\S+(\s*)$', replace, text, flags=re.MULTILINE)
    Path(rvt).write_text(text, encoding='utf-8')
    return {'RainCorrection': new_rain[0], 'SnowCorrection': new_snow[0],
            'TemperatureCorrection': new_temp[0]}


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    rvt = Path(project_dir) / 'settings' / 'RAVEN' / 'model.rvt'
    if not rvt.exists():
        raise ModelExecutionError(f'Raven gauge file not found: {rvt}')
    written = set_gauge_corrections(rvt, info.get('native', {}))
    return {'rvt': str(rvt), 'applied': written}
