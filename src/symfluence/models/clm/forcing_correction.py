# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for CLM: corrected copies of the DATM forcing.

CLM has no precipitation or temperature correction parameter, so the
correction is applied to the forcing it reads: the ``clmforc.*.nc`` files
listed in ``datm.streams.xml`` are copied into ``<settings>/forcing_corrected``
with ``PRECTmms`` multiplied and ``TBOT`` offset, and the stream file is
pointed at the copies. The original file list is recorded in
``forcing_correction_source.json`` next to the stream file, so a correction is
always derived from the unmodified forcing (repeated trials never compound)
and identity restores the original paths. Shared forcing is never written.

The adapter's native names are the common names (``FC_PRECIP_MULT``,
``FC_TEMP_OFFSET``): calibration workers receive them as parameters. CLM
partitions rain and snow itself, so there is no snowfall multiplier.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from symfluence.core.exceptions import ModelExecutionError

PRECIP_PARAM = 'FC_PRECIP_MULT'
TEMP_PARAM = 'FC_TEMP_OFFSET'
FORCING_PARAMS = (PRECIP_PARAM, TEMP_PARAM)
STREAMS_FILE = 'datm.streams.xml'
SOURCE_RECORD = 'forcing_correction_source.json'
CORRECTED_DIR = 'forcing_corrected'

_FILE_RE = re.compile(r'<file>([^<]*clmforc[^<]*\.nc)</file>')


def split_params(params: Mapping[str, Any]) -> tuple:
    """``(model params, correction values)`` with the correction keyed precip/temp."""
    values = {}
    if PRECIP_PARAM in params:
        values['precip'] = _scalar(params[PRECIP_PARAM])
    if TEMP_PARAM in params:
        values['temp'] = _scalar(params[TEMP_PARAM])
    return {k: v for k, v in params.items() if k not in FORCING_PARAMS}, values


def _scalar(value: Any) -> float:
    import numpy as np
    return float(np.asarray(value, dtype=float).ravel()[0])


def _source_files(settings_dir: Path, content: str) -> List[str]:
    record = settings_dir / SOURCE_RECORD
    if record.exists():
        return list(json.loads(record.read_text())['files'])
    return list(dict.fromkeys(_FILE_RE.findall(content)))


def apply_stream_correction(settings_dir: Path, values: Mapping[str, float], logger: Any = None) -> Dict[str, Any]:
    """Point ``datm.streams.xml`` in ``settings_dir`` at corrected forcing copies.

    ``values`` holds ``precip`` (multiplier) and/or ``temp`` (K offset);
    missing ones are identity. Identity restores the original file list.
    """
    import netCDF4 as nc

    settings_dir = Path(settings_dir)
    streams = settings_dir / STREAMS_FILE
    if not streams.exists():
        raise ModelExecutionError(f'No {STREAMS_FILE} in {settings_dir}')
    content = streams.read_text()
    current = list(dict.fromkeys(_FILE_RE.findall(content)))
    sources = _source_files(settings_dir, content)
    if len(sources) != len(current):
        raise ModelExecutionError(
            f'{STREAMS_FILE} lists {len(current)} forcing files but {SOURCE_RECORD} records {len(sources)}')
    m = float(values.get('precip', 1.0))
    dt = float(values.get('temp', 0.0))
    if not m > 0:
        raise ModelExecutionError(f'Precipitation multiplier must be positive, got {m}')
    record = settings_dir / SOURCE_RECORD
    if m == 1.0 and dt == 0.0:
        targets = sources
        if record.exists():
            record.unlink()
    else:
        out_dir = settings_dir / CORRECTED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if not record.exists():
            record.write_text(json.dumps({'files': sources}, indent=2))
        targets = []
        for src in sources:
            src_path = Path(src)
            dst = out_dir / src_path.name
            if dst.resolve() == src_path.resolve():
                raise ModelExecutionError(f'Corrected forcing would overwrite its source: {src_path}')
            tmp = dst.with_suffix('.tmp.nc')
            with nc.Dataset(src_path) as ds_in, nc.Dataset(tmp, 'w', format=ds_in.data_model) as ds_out:
                ds_out.setncatts({k: ds_in.getncattr(k) for k in ds_in.ncattrs()})
                for name, dim in ds_in.dimensions.items():
                    ds_out.createDimension(name, None if dim.isunlimited() else len(dim))
                for name, var in ds_in.variables.items():
                    fill = getattr(var, '_FillValue', None)
                    out = ds_out.createVariable(name, var.datatype, var.dimensions, fill_value=fill)
                    out.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'})
                    data = var[:]
                    if name == 'PRECTmms' and m != 1.0:
                        data = data * m
                        out.setncattr('symfluence_precip_mult', m)
                    elif name == 'TBOT' and dt != 0.0:
                        data = data + dt
                        out.setncattr('symfluence_temp_offset', dt)
                    out[:] = data
            tmp.replace(dst)
            targets.append(str(dst))
    for old, new in zip(current, targets):
        content = content.replace(f'<file>{old}</file>', f'<file>{new}</file>')
    streams.write_text(content)
    if logger is not None:
        logger.debug(f'CLM forcing correction precip x{m:g}, temp {dt:+g} K on {len(targets)} files')
    return {'precip': m, 'temp': dt, 'files': targets}


def needs_reset(settings_dir: Path) -> bool:
    """True when ``settings_dir`` holds a correction that must be undone."""
    return (Path(settings_dir) / SOURCE_RECORD).exists()


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """Correct the DATM streams in ``settings/CLM`` for a forward run (copied to the run dir)."""
    settings = Path(project_dir) / 'settings' / 'CLM'
    _, values = split_params(info['native'])
    if all(values.get(q, i) == i for q, i in (('precip', 1.0), ('temp', 0.0))) and not needs_reset(settings):
        return {'settings_dir': str(settings), 'applied': 'identity, forcing untouched'}
    return {'settings_dir': str(settings), **apply_stream_correction(settings, values, logger)}
