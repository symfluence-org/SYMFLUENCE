# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for ParFlow-CLM.

There is no correction parameter in the lumped model, so the correction is
applied to the forcing it reads; the basin mode has
``CLMPARFLOW_BASIN_PRECIP_SCALE`` and gains a temperature offset.

Lumped mode reads precipitation twice (pre-existing design): the CLM met file
``forcing.1d`` and the Snow-17 rain-plus-melt minus PET flux on the
``z_upper`` overland boundary in the ``.pfidb``. Both are corrected:

* ``forcing.1d``: the copy in the run directory gets column 3 (APCP) multiplied
  and column 4 (temperature, K) offset (:func:`correct_forcing_1d`), done by
  the runner when it stages the run; the shared input is never written.
* boundary flux: recomputed from a corrected copy of the hourly forcing cache
  with the Snow-17 parameters in effect (:func:`correct_overland_bc`). The
  original values are recorded first, so identity restores them exactly.
  PET is kept from the cache (not recomputed for the temperature offset).

Basin mode: ``BasinParameters.precip_scale`` / ``temp_offset`` (set by the
runner, superseding ``CLMPARFLOW_BASIN_PRECIP_SCALE``). Segments of a chain
already on disk are reused as before, so change the correction only with a
fresh ``basin_runs`` directory.

The adapter's native names are the common names (``FC_PRECIP_MULT``,
``FC_TEMP_OFFSET``). There is no snowfall multiplier: CLM partitions rain and
snow itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from symfluence.core.exceptions import ModelExecutionError

PRECIP_PARAM = 'FC_PRECIP_MULT'
TEMP_PARAM = 'FC_TEMP_OFFSET'
FORCING_PARAMS = (PRECIP_PARAM, TEMP_PARAM)
FORCING_1D = 'forcing.1d'
BC_RECORD = 'forcing_correction_bc.json'
SNOW_SIDECAR = 'snow17_params.json'
CORRECTED_CACHE = 'hourly_forcing_cache_fc.npz'
_BC_PREFIX = 'Patch.z_upper.BCPressure.d'


def split_params(params: Mapping[str, Any]) -> tuple:
    """``(model params, correction values)`` with the correction keyed precip/temp."""
    values = {}
    if PRECIP_PARAM in params:
        values['precip'] = float(np.asarray(params[PRECIP_PARAM], dtype=float).ravel()[0])
    if TEMP_PARAM in params:
        values['temp'] = float(np.asarray(params[TEMP_PARAM], dtype=float).ravel()[0])
    return {k: v for k, v in params.items() if k not in FORCING_PARAMS}, values


def is_identity(values: Mapping[str, float]) -> bool:
    return float(values.get('precip', 1.0)) == 1.0 and float(values.get('temp', 0.0)) == 0.0


def correct_forcing_1d(path: Path, values: Mapping[str, float]) -> None:
    """Multiply APCP (column 3) and offset temperature (column 4) of a CLM 1-D met file in place."""
    if is_identity(values):
        return
    m = float(values.get('precip', 1.0))
    if not m > 0:
        raise ModelExecutionError(f'Precipitation multiplier must be positive, got {m}')
    data = np.loadtxt(path, ndmin=2)
    data[:, 2] *= m
    data[:, 3] += float(values.get('temp', 0.0))
    np.savetxt(path, data, fmt='%.6e')


def _bc_keys(entries: Mapping[str, str]):
    return [k for k in entries if k.startswith(_BC_PREFIX) and k.endswith('.Value')]


def correct_overland_bc(
    entries: Dict[str, str],
    settings_dir: Path,
    cache_path: Optional[Path],
    values: Mapping[str, float],
    snow_params: Optional[Mapping[str, float]] = None,
) -> bool:
    """Recompute the daily ``z_upper`` flux in ``entries`` for corrected forcing.

    Records the original flux values in ``settings_dir`` the first time;
    identity (with no Snow-17 parameters) restores them. Returns True when
    ``entries`` changed.
    """
    settings_dir = Path(settings_dir)
    keys = _bc_keys(entries)
    if not keys:
        return False
    record = settings_dir / BC_RECORD
    if not record.exists():
        if is_identity(values):
            return False
        record.write_text(json.dumps({k: entries[k] for k in keys}))
    if is_identity(values) and not snow_params:
        entries.update(json.loads(record.read_text()))
        return True
    if cache_path is None or not Path(cache_path).exists():
        raise ModelExecutionError('No hourly forcing cache to correct the ParFlow-CLM boundary flux')
    from symfluence.models.parflow.lumped_forcing import update_snow17_boundary_values

    corrected = settings_dir / CORRECTED_CACHE
    with np.load(cache_path) as cache:
        arrays = {k: cache[k] for k in cache.files}
    arrays['ppt_mm_hr'] = arrays['ppt_mm_hr'] * float(values.get('precip', 1.0))
    arrays['temp_c'] = arrays['temp_c'] + float(values.get('temp', 0.0))
    np.savez_compressed(corrected, **arrays)
    update_snow17_boundary_values(entries, dict(snow_params or {}), corrected)
    return True


def read_snow_params(settings_dir: Path) -> Dict[str, float]:
    """Snow-17 parameters last applied in ``settings_dir`` (empty: defaults)."""
    sidecar = Path(settings_dir) / SNOW_SIDECAR
    return json.loads(sidecar.read_text()) if sidecar.exists() else {}


def write_snow_params(settings_dir: Path, snow_params: Mapping[str, float]) -> None:
    (Path(settings_dir) / SNOW_SIDECAR).write_text(json.dumps({k: float(v) for k, v in snow_params.items()}))


def forward_values(config: Any, project_dir: Path, domain_name: str, experiment_id: str,
                   logger: Any) -> Optional[Dict[str, float]]:
    """Correction values for a run without calibration parameters (None: section unset)."""
    from symfluence.core.calibration.forcing_correction import forward_correction

    info = forward_correction(config, 'CLMPARFLOW', project_dir=project_dir, domain_name=domain_name,
                              experiment_id=experiment_id, log=logger)
    if info is None:
        return None
    return split_params(info['native'])[1]


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """Lumped: correct the boundary flux in ``settings/CLMPARFLOW``; the runner corrects ``forcing.1d``
    and the basin-mode forcing."""
    from symfluence.core.mixins.config import resolve_config_value

    _, values = split_params(info['native'])
    basin = resolve_config_value(config, lambda: config.model.clmparflow.basin_mode,
                                 dict_key='CLMPARFLOW_BASIN_MODE')
    if basin:
        return {'mode': 'basin', 'applied_by': 'CLMParFlowRunner._run_basin', 'values': values}
    settings = Path(project_dir) / 'settings' / 'CLMPARFLOW'
    pfidb = sorted(settings.glob('*.pfidb'))
    if not pfidb:
        return {'mode': 'lumped', 'applied': 'no .pfidb; boundary flux untouched', 'values': values}
    from symfluence.core.mixins.project import resolve_data_subdir
    from symfluence.models.parflow.calibration.parameter_manager import _read_pfidb, _write_pfidb

    entries = _read_pfidb(pfidb[0])
    cache = resolve_data_subdir(Path(project_dir), 'forcing') / 'CLMPARFLOW_input' / 'hourly_forcing_cache.npz'
    if not cache.exists():
        cache = settings / 'hourly_forcing_cache.npz'
    if correct_overland_bc(entries, settings, cache, values, read_snow_params(settings)):
        _write_pfidb(pfidb[0], entries)
    return {'mode': 'lumped', 'pfidb': str(pfidb[0]), 'values': values,
            'forcing_1d': 'corrected by CLMParFlowRunner in the run directory'}
