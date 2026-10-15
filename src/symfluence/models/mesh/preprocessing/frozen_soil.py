"""Counted native frozen-soil inputs and fail-closed runtime validation.

FRZC=0 bypasses FROZEN; it does not represent zero infiltration. Explicit
parameters are required for soil tiles; ice and mapped open water are bypassed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..errors import MESHConfigurationError, MESHInputError


def _section(lines, label):
    start = next(i for i, line in enumerate(lines) if label in line)
    count_line = next(i for i in range(start + 1, len(lines)) if re.match(r'^\s*\d+(?:\s|$)', lines[i]))
    count = int(lines[count_line].split()[0])
    rows: list[int] = []
    for i in range(count_line + 1, len(lines)):
        if len(rows) == count:
            break
        text = lines[i].split('#')[0].strip()
        if text and not text.startswith(('!', '-')):
            rows.append(i)
    if len(rows) != count:
        raise MESHInputError('Incomplete counted MESH hydrology section')
    return count_line, rows


def _context(directory):
    directory = Path(directory)
    options = (directory / 'MESH_input_run_options.ini').read_text()
    match = re.search(r'^FROZENSOILINFILFLAG\s+(\d+)', options, re.M)
    if not match or int(match[1]) == 0:
        return None
    if int(match[1]) != 1:
        raise MESHConfigurationError('Frozen-soil writer currently supports FROZENSOILINFILFLAG 1 only')
    dates = re.findall(r'^\s*(\d{4})\s+\d{1,3}\s+\d{1,2}\s+\d{1,2}\s*$', options, re.M)
    if len(dates) != 2:
        raise MESHConfigurationError('Frozen-soil opportunity time requires two native run dates')
    years = int(dates[1]) - int(dates[0]) + 1
    sand = [float(line.split()[0]) for line in (directory / 'MESH_parameters_CLASS.ini').read_text().splitlines() if '3xSAND' in line]
    if not sand:
        raise MESHInputError('Missing CLASS surface identities for frozen-soil targeting')
    soil = np.array(sand) != -4
    manifest = directory / 'spatial_units.json'
    if manifest.exists():
        units = json.loads(manifest.read_text())['units']
        if len(units) != len(soil):
            raise MESHInputError('Frozen-soil unit/CLASS count mismatch')
        soil &= np.array([u.get('modis') != 17 for u in units])
    return years, soil


def validate_frozen_soil(directory):
    context = _context(directory)
    if context is None:
        return
    years, soil = context
    lines = (Path(directory) / 'MESH_parameters_hydrology.ini').read_text().splitlines()
    fields = {}
    for label in ['GRU class independent', 'GRU class dependent']:
        _, rows = _section(lines, label)
        for i in rows:
            parts = lines[i].split('#')[0].split()
            key = parts[0].upper()
            if key in fields:
                raise MESHInputError(f'Duplicate native hydrology field {key}')
            if key in {'RCHARG', 'FRZTH'}:
                raise MESHConfigurationError(f'Unsupported native hydrology parameter {key}')
            if key == 'FRZC' and label != 'GRU class dependent':
                raise MESHConfigurationError('FRZC must be in the counted GRU dependent section')
            if key in {'SOIL_POR_MAX', 'SOIL_DEPTH', 'S0', 'T_ICE_LENS', 'T0_ACC'} and label != 'GRU class independent':
                raise MESHConfigurationError(f'{key} must be in the counted GRU independent section')
            fields[key] = parts[1:]
    ranges = {'SOIL_POR_MAX': (0, 1), 'SOIL_DEPTH': (0, float('inf')), 'S0': (0, 1), 'T_ICE_LENS': (-50, 0)}
    for key, (lo, hi) in ranges.items():
        values = np.array(fields.get(key, []), dtype=float)
        if len(values) != 1 or not np.isfinite(values).all() or not lo <= values[0] <= hi or (key != 'T_ICE_LENS' and values[0] <= 0):
            raise MESHInputError(f'Frozen-soil input {key} missing/invalid in counted section')
    t0 = np.array(fields.get('T0_ACC', []), dtype=float)
    if len(t0) != years or not np.isfinite(t0).all() or (t0 < 0).any():
        raise MESHInputError('Frozen-soil T0_ACC needs one nonnegative value per native calendar year')
    frzc = np.array(fields.get('FRZC', []), dtype=float)
    if len(frzc) != len(soil) or not np.isfinite(frzc).all() or (frzc < 0).any() or (frzc[soil] <= 0).any() or (frzc[~soil] != 0).any():
        raise MESHConfigurationError('Frozen-soil FRZC must be positive on soil and zero on ice/open-water tiles')


def prepare_frozen_soil(directory, parameters=None):
    """Write explicit assumptions, or validate existing native inputs unchanged."""
    context = _context(directory)
    if context is None:
        return
    if parameters is None:
        validate_frozen_soil(directory)
        return
    from symfluence.models.mesh.config_schema import MESHFrozenSoilConfig
    cfg = MESHFrozenSoilConfig.model_validate(parameters)
    years, soil = context
    path = Path(directory) / 'MESH_parameters_hydrology.ini'
    lines = path.read_text().splitlines()
    # Remove only inactive legacy tail lines. Active unknown fields fail closed.
    _, rows = _section(lines, 'GRU class dependent')
    last = rows[-1] if rows else _section(lines, 'GRU class dependent')[0]
    lines = [line for i, line in enumerate(lines) if not (i > last and re.match(r'^\s*(RCHARG|FRZTH)\b', line, re.I))]
    values = {'SOIL_POR_MAX': [cfg.soil_porosity], 'SOIL_DEPTH': [cfg.soil_depth], 'S0': [cfg.surface_saturation], 'T_ICE_LENS': [cfg.ice_lens_temperature], 'T0_ACC': [cfg.opportunity_time] * years}
    for label, updates in [('GRU class independent', values), ('GRU class dependent', {'FRZC': np.where(soil, cfg.infiltration_coefficient, 0.)})]:
        ci, rows = _section(lines, label)
        positions = {lines[i].split()[0].upper(): i for i in rows}
        insert = rows[-1] + 1 if rows else ci + 1
        count = len(rows)
        for key, vals in updates.items():
            line = key + ' ' + ' '.join(f'{v:.9g}' for v in vals)
            if key in positions:
                lines[positions[key]] = line
            else:
                lines.insert(insert, line)
                insert += 1
                count += 1
        lines[ci] = str(count) + ' # counted native parameters'
    original = path.read_text()
    path.write_text('\n'.join(lines) + '\n')
    try:
        validate_frozen_soil(directory)
    except (ValueError, IndexError, StopIteration):
        path.write_text(original)
        raise
