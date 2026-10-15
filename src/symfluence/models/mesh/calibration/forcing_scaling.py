# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibratable forcing corrections for NetCDF-forced MESH runs.

Native MESH parses ``cm=``/``ca=`` on forcing flags but applies them only to
ASCII and sequential readers, so for ``nc_subbasin`` forcing they are silently
ignored. Corrections are therefore written into a private copy of
``MESH_forcing_safe.nc`` in the run directory, always derived from a recorded,
hash-checked unscaled source. Run directories usually hold a symlink to shared
forcing; the link itself is replaced, never written through.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Mapping

from ..errors import MESHConfigurationError, MESHFileNotFoundError, MESHInputError

FORCING_FILE = 'MESH_forcing_safe.nc'
SOURCE_RECORD = 'MESH_forcing_scaling.json'
UNSCALED_FILE = 'MESH_forcing_safe.unscaled.nc'

#: name -> (native variable, operation, neutral value)
FORCING_PARAMS = {
    'PRECIP_MULT': ('PRE', 'mul', 1.0),
    'TEMP_OFFSET': ('TA', 'add', 0.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(directory: Path) -> Dict[str, str]:
    record_path = directory / SOURCE_RECORD
    target = directory / FORCING_FILE
    if record_path.exists():
        record = json.loads(record_path.read_text())
        source = Path(record['source'])
    elif target.is_symlink():
        source = target.resolve()
        record = {'source': str(source), 'source_sha256': _sha256(source)}
    elif target.exists():
        source = directory / UNSCALED_FILE
        if source.exists():
            raise MESHConfigurationError(f'{source} exists without {SOURCE_RECORD}; refusing to guess the unscaled forcing')
        target.rename(source)
        record = {'source': str(source), 'source_sha256': _sha256(source)}
    else:
        raise MESHFileNotFoundError(f'No {FORCING_FILE} in {directory}')
    if not source.exists():
        raise MESHFileNotFoundError(f'Unscaled MESH forcing source missing: {source}')
    if target.exists() and not target.is_symlink() and source.resolve() == target.resolve():
        raise MESHInputError('Unscaled forcing source and scaled run forcing are the same file')
    if _sha256(source) != record['source_sha256']:
        raise MESHInputError(f'Unscaled MESH forcing source changed since it was recorded: {source}')
    return record


def read_applied(directory: Path) -> Dict[str, float]:
    """Corrections currently written in ``directory`` (neutral when none)."""
    applied = {name: spec[2] for name, spec in FORCING_PARAMS.items()}
    record_path = Path(directory) / SOURCE_RECORD
    if record_path.exists():
        applied.update(json.loads(record_path.read_text()).get('applied', {}))
    return applied


def apply_forcing_corrections(directory: Path, values: Mapping[str, float]) -> Dict[str, float]:
    """Write ``MESH_forcing_safe.nc`` with corrections applied to the unscaled source.

    Corrections not named in ``values`` are reset to neutral, so a run never
    inherits a previous trial's forcing.
    """
    from netCDF4 import Dataset

    directory = Path(directory)
    unknown = set(values) - set(FORCING_PARAMS)
    if unknown:
        raise MESHConfigurationError(f'Unknown MESH forcing corrections: {sorted(unknown)}')
    applied = {name: float(values.get(name, spec[2])) for name, spec in FORCING_PARAMS.items()}
    if not all(math.isfinite(v) for v in applied.values()):
        raise MESHConfigurationError(f'Non-finite MESH forcing correction: {applied}')
    if applied['PRECIP_MULT'] <= 0:
        raise MESHConfigurationError('PRECIP_MULT must be positive')

    record = _resolve_source(directory)
    target = directory / FORCING_FILE
    tmp = directory / (FORCING_FILE + '.tmp')
    shutil.copyfile(record['source'], tmp)
    try:
        with Dataset(tmp, 'r+') as nc:
            for name, (variable, operation, neutral) in FORCING_PARAMS.items():
                value = applied[name]
                if value == neutral:
                    continue
                var = nc.variables[variable]
                data = var[:]
                var[:] = data * value if operation == 'mul' else data + value
                var.setncattr(f'symfluence_{name.lower()}', value)
            nc.setncattr('symfluence_forcing_source', record['source'])
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    record['applied'] = applied
    (directory / SOURCE_RECORD).write_text(json.dumps(record, indent=2))
    return applied
