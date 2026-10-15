"""Explicit land/ice targeting for native MESH parameter arrays."""
from __future__ import annotations

import re

from ..errors import MESHConfigurationError, MESHInputError

ICE_CLASS = {
    'ICE_ALBEDO_VISIBLE': ('5xALVC/4xCMAS', 4),
    'ICE_ALBEDO_INFRARED': ('5xALIC/4xROOT', 4),
    'ICE_LNZ0': ('5xLNZ0/4xLAMN', 4),
}
ICE_HYDRO = {'FREZTH': -2., 'SWELIM': 100., 'SNDENLIM': 900.}
LAND_HYDRO = {'ZSNL', 'ZPLS', 'ZPLG', 'FRZTH', 'RCHARG', 'DRAINFRAC'}


def ice_mask(lines):
    return [float(line.split()[0]) == -4 for line in lines if '3xSAND' in line]


def update_gru_arrays(content, params, mask, land_mask=None):
    """Update exactly counted native rows; never silently edit unread tails."""
    if not params:
        return content
    if not mask:
        raise MESHConfigurationError('No native CLASS surface mapping for GRU parameters')
    if any(name in ICE_HYDRO for name in params) and not any(mask):
        raise MESHConfigurationError('Glacier parameters require at least one native ice GRU')
    if land_mask is not None and len(land_mask) != len(mask):
        raise MESHInputError('Land parameter targeting does not match native GRUs')
    lines = content.splitlines()
    header = next(i for i, line in enumerate(lines) if 'GRU class dependent hydrologic parameters' in line)
    count_idx = next(i for i in range(header+1, len(lines)) if re.match(r'^\s*\d+(?:\s|$)', lines[i]))
    count = int(lines[count_idx].split()[0])
    active: list[int] = []
    for i in range(count_idx+1, len(lines)):
        if len(active) == count:
            break
        parts = lines[i].split('#')[0].split()
        if parts and re.fullmatch(r'[A-Za-z_]\w*', parts[0]):
            active.append(i)
    if len(active) != count:
        raise MESHInputError('Incomplete native GRU parameter section')
    for name, value in params.items():
        idx = next((i for i in active if lines[i].split()[0].upper() == name), None)
        if idx is None:
            if name not in ICE_HYDRO:
                raise MESHInputError(f'{name} is not in the counted native GRU section')
            idx = active[-1]+1 if active else count_idx+1
            lines.insert(idx, name+' '+' '.join([str(ICE_HYDRO[name])]*len(mask)))
            active.append(idx)
            count += 1
        body, sep, comment = lines[idx].partition('#')
        vals = body.split()[1:]
        if len(vals) == 1:
            vals *= len(mask)
        if len(vals) != len(mask):
            raise MESHInputError(f'{name}: native GRU array length mismatch')
        for j, is_ice in enumerate(mask):
            if is_ice == (name in ICE_HYDRO) and (is_ice or land_mask is None or land_mask[j]):
                vals[j] = f'{value:.9g}'
        lines[idx] = name+' '+' '.join(vals)+((' #'+comment) if sep else '')
    lines[count_idx] = re.sub(r'^(\s*)\d+', lambda m: m[1]+str(count), lines[count_idx])
    return '\n'.join(lines)+'\n'
