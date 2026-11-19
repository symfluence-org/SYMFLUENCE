# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Read and rewrite parameter blocks of a PRMS parameter file (params.dat).

One implementation for the PRMS calibration worker, the PRMS parameter manager
and GSFLOW (whose surface component is PRMS). A block is::

    name
    ndim
    dimension name (ndim lines)
    number of values
    type code
    values (one per line)

and blocks are separated by ``####`` lines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from symfluence.core.exceptions import ConfigurationError

BLOCK_SEPARATOR = '####\n'


def _block_name(block: str) -> str:
    return block.strip().split('\n')[0].strip() if block.strip() else ''


def replace_block_values(block: str, name: str, value: float) -> str:
    """Set every value of PRMS block ``name`` (``name, ndim, dims..., n, type, values``)."""
    lines = block.strip().split('\n')
    try:
        i = next(k for k, line in enumerate(lines) if line.strip() == name)
    except StopIteration:
        return block
    ndim = int(lines[i + 1])
    count = int(lines[i + 2 + ndim])
    start = i + 4 + ndim
    out = lines[:start] + [f'{value:.6f}'] * count + lines[start + count:]
    return '\n'.join(out) + ('\n' if block.endswith('\n') else '')


def first_block_value(block: str, name: str) -> Optional[float]:
    """First value of PRMS block ``name``, or None when the block is not ``name``."""
    lines = block.strip().split('\n')
    try:
        i = next(k for k, line in enumerate(lines) if line.strip() == name)
        return float(lines[i + 4 + int(lines[i + 1])].strip())
    except (StopIteration, IndexError, ValueError):
        return None


#: Shared FORCING_CORRECTION pseudo-parameters (the common names pass through
#: the calibration unchanged and are mapped onto PRMS blocks here).
FORCING_CORRECTION_PARAMS = ('FC_PRECIP_MULT', 'FC_SNOW_MULT', 'FC_TEMP_OFFSET')

#: PRMS correction blocks per forcing mode: precip_1sta/temp_1sta (station)
#: read rain_adj/snow_adj (nmonths) and tmax_adj/tmin_adj (nhru); climate_hru
#: reads the *_cbh_adj blocks (nhru x nmonths).
STATION_CORRECTION_BLOCKS = {'rain': 'rain_adj', 'snow': 'snow_adj',
                             'tmax': 'tmax_adj', 'tmin': 'tmin_adj'}
CBH_CORRECTION_BLOCKS = {'rain': 'rain_cbh_adj', 'snow': 'snow_cbh_adj',
                         'tmax': 'tmax_cbh_adj', 'tmin': 'tmin_cbh_adj'}


def _scalar(value) -> float:
    import numpy as np
    return float(np.asarray(value, dtype=float).ravel()[0])


def forcing_correction_blocks(params: Dict[str, float], block_names) -> Dict[str, float]:
    """PRMS block values for the shared forcing-correction pseudo-parameters in ``params``.

    ``FC_PRECIP_MULT`` (m_P) multiplies rain and snow, ``FC_SNOW_MULT`` (m_S)
    snow on top of it: rain adj = m_P, snow adj = m_P * m_S.
    ``FC_TEMP_OFFSET`` sets both tmax and tmin adj. The climate_hru blocks
    are used when the file has them, otherwise the station blocks.
    """
    fc = {k: _scalar(params[k]) for k in FORCING_CORRECTION_PARAMS if k in params}
    if not fc:
        return {}
    blocks = CBH_CORRECTION_BLOCKS if 'rain_cbh_adj' in block_names else STATION_CORRECTION_BLOCKS
    out: Dict[str, float] = {}
    if 'FC_PRECIP_MULT' in fc or 'FC_SNOW_MULT' in fc:
        m_p = fc.get('FC_PRECIP_MULT', 1.0)
        out[blocks['rain']] = m_p
        out[blocks['snow']] = m_p * fc.get('FC_SNOW_MULT', 1.0)
    if 'FC_TEMP_OFFSET' in fc:
        out[blocks['tmax']] = out[blocks['tmin']] = fc['FC_TEMP_OFFSET']
    return out


def update_parameter_file(param_file: Path, params: Dict[str, float], model: str = 'PRMS') -> None:
    """Write calibration parameters into ``param_file``.

    Every value of each named block is set, whatever its dimensions (nhru,
    nmonths, nhru x nmonths). Derived parameters (``land_sca_max``) are routed
    through :mod:`symfluence.models.prms.derived_parameters`; the shared
    forcing-correction pseudo-parameters through :func:`forcing_correction_blocks`.

    Raises:
        ConfigurationError: a parameter has no block in the file. Writing
            nothing would leave the parameter inert while the optimiser keeps
            sampling it.
    """
    from .derived_parameters import apply_derived_parameters
    content = param_file.read_text(encoding='utf-8')
    if any(k in params for k in FORCING_CORRECTION_PARAMS):
        names_in_file = {_block_name(b) for b in content.split(BLOCK_SEPARATOR)}
        corrections = forcing_correction_blocks(params, names_in_file)
        params = {k: v for k, v in params.items() if k not in FORCING_CORRECTION_PARAMS}
        params.update(corrections)
    content, params = apply_derived_parameters(content, params)
    blocks = content.split(BLOCK_SEPARATOR)
    names = {_block_name(b) for b in blocks} - {''}
    missing = set(params) - names
    if missing:
        raise ConfigurationError(f'{model} parameters not in {param_file.name}: {sorted(missing)}')
    for k, b in enumerate(blocks):
        head = _block_name(b)
        if head in params:
            blocks[k] = replace_block_values(b, head, params[head])
    param_file.write_text(BLOCK_SEPARATOR.join(blocks), encoding='utf-8')
