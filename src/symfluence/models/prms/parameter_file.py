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


def update_parameter_file(param_file: Path, params: Dict[str, float], model: str = 'PRMS') -> None:
    """Write calibration parameters into ``param_file``.

    Every value of each named block is set, whatever its dimensions (nhru,
    nmonths, nhru x nmonths). Derived parameters (``land_sca_max``) are routed
    through :mod:`symfluence.models.prms.derived_parameters`.

    Raises:
        ConfigurationError: a parameter has no block in the file. Writing
            nothing would leave the parameter inert while the optimiser keeps
            sampling it.
    """
    from .derived_parameters import apply_derived_parameters
    content, params = apply_derived_parameters(param_file.read_text(encoding='utf-8'), params)
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
