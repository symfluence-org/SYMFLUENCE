"""Calibration parameters that map onto PRMS parameter blocks indirectly.

``land_sca_max`` is the maximum snow-covered fraction of land HRUs. With
``PRMS_SEPARATE_GLACIER_DEPLETION_CURVE`` the preprocessor writes two
identical 11-point ``snarea_curve`` curves (land, glacier). The land curve is
set to ``land_sca_max`` x the glacier curve, which is never scaled, so
applying a value repeatedly to the same file does not compound.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

DERIVED_PARAMETERS = ('land_sca_max',)
_POINTS = 11


def _curve_block(blocks):
    for i, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if lines and lines[0].strip() == 'snarea_curve':
            ndim = int(lines[1])
            return i, lines, 4 + ndim
    raise ValueError('land_sca_max needs a snarea_curve block')


def _two_curves(lines, start):
    values = [float(v) for v in lines[start:]]
    if len(values) != 2 * _POINTS:
        raise ValueError('land_sca_max needs separate land and glacier depletion curves '
                         '(PRMS_SEPARATE_GLACIER_DEPLETION_CURVE: true)')
    return values[:_POINTS], values[_POINTS:]


def apply_derived_parameters(content: str, params: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """Apply derived parameters to params.dat text; return (text, native params)."""
    native = {k: v for k, v in params.items() if k not in DERIVED_PARAMETERS}
    if 'land_sca_max' not in params:
        return content, native
    value = float(params['land_sca_max'])
    if not 0. < value <= 1.:
        raise ValueError(f'land_sca_max must be in (0, 1], got {value}')
    blocks = content.split('####\n')
    i, lines, start = _curve_block(blocks)
    _, glacier = _two_curves(lines, start)
    lines = lines[:start] + [f'{value * v:.6f}' for v in glacier] + [f'{v:.6f}' for v in glacier]
    blocks[i] = '\n'.join(lines) + ('\n' if blocks[i].endswith('\n') else '')
    return '####\n'.join(blocks), native


def read_derived_parameter(content: str, name: str) -> Optional[float]:
    """Current value of a derived parameter in params.dat text, if defined."""
    if name != 'land_sca_max':
        return None
    try:
        _, lines, start = _curve_block(content.split('####\n'))
        land, glacier = _two_curves(lines, start)
    except ValueError:
        return None
    return land[-1] / glacier[-1] if glacier[-1] else None
