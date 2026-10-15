# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Write GSFLOW calibration parameters into PRMS and MODFLOW input files.

Single code path for the calibration worker and the parameter manager:

* ``K``, ``SY``      MODFLOW UPW (HK = VKA = K; SY)
* ``strhc1``         SFR streambed hydraulic conductivity of every reach
* ``land_sca_max``   derived PRMS land snow-cover maximum (needs separate
                     land/glacier depletion curves, see PRMS derived parameters)
* anything else      PRMS parameter block; every value of the block is set,
                     whatever its dimensions (nhru, nmonths, nhru x nmonths)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from symfluence.core.exceptions import ConfigurationError, ValidationError

MODFLOW_PARAMS = ('K', 'SY')
SFR_PARAMS = ('strhc1',)


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


def update_prms_parameters(param_file: Path, params: Dict[str, float]) -> None:
    from symfluence.models.prms.derived_parameters import apply_derived_parameters
    content, params = apply_derived_parameters(param_file.read_text(encoding='utf-8'), params)
    blocks = content.split('####\n')
    names = {b.strip().split('\n')[0].strip() for b in blocks if b.strip()}
    missing = set(params) - names
    if missing:
        raise ConfigurationError(f'GSFLOW parameters not in {param_file.name}: {sorted(missing)}')
    for k, b in enumerate(blocks):
        head = b.strip().split('\n')[0].strip() if b.strip() else ''
        if head in params:
            blocks[k] = replace_block_values(b, head, params[head])
    param_file.write_text('####\n'.join(blocks), encoding='utf-8')


def _constant_lines(lines):
    return [i for i, line in enumerate(lines) if line.strip().upper().startswith('CONSTANT')]


def update_upw(modflow_dir: Path, params: Dict[str, float]) -> None:
    """K -> UPW HK and VKA; SY -> UPW SY (must stay below UZF THTS)."""
    upw = next(iter(sorted(modflow_dir.glob('*.upw'))), None)
    uzf = next(iter(sorted(modflow_dir.glob('*.uzf'))), None)
    if upw is None:
        raise ConfigurationError(f'No UPW package in {modflow_dir}')
    if 'SY' in params and uzf is not None:
        consts = [float(uzf.read_text().split('\n')[i].split()[1]) for i in _constant_lines(uzf.read_text().split('\n'))]
        thts = consts[4] if len(consts) >= 5 and uzf.read_text().startswith('# GSFLOW UZF Package') else None
        if thts is not None and not 0 < params['SY'] < thts:
            raise ValidationError('Specific yield must be positive and below UZF saturated water content')
    lines = upw.read_text(encoding='utf-8').split('\n')
    ci = _constant_lines(lines)
    if len(ci) < 4:
        raise ConfigurationError(f'{upw.name} must give HK, VKA, SS and SY as CONSTANT arrays')
    if 'K' in params:
        lines[ci[0]] = lines[ci[1]] = f"  CONSTANT  {params['K']:.6e}"
    if 'SY' in params:
        lines[ci[3]] = f"  CONSTANT  {params['SY']:.6e}"
    upw.write_text('\n'.join(lines), encoding='utf-8')


def update_sfr_streambed(modflow_dir: Path, strhc1: float) -> None:
    """Set STRHC1 (10th field) on every REACHINPUT reach line of the SFR package."""
    sfr = next(iter(sorted(modflow_dir.glob('*.sfr'))), None)
    if sfr is None:
        raise ConfigurationError(f'No SFR package in {modflow_dir}')
    lines = sfr.read_text().split('\n')
    body = [k for k, line in enumerate(lines) if line.strip() and not line.startswith('#')]
    upper = [lines[k].strip().upper() for k in body]
    i = body[upper.index('END') + 1] if upper and upper[0] == 'OPTIONS' else body[0]   # item 1
    nstrm = abs(int(lines[i].split()[0]))
    for k in range(i + 1, i + 1 + nstrm):
        f = lines[k].split()
        if len(f) < 10:
            raise ValidationError(f'{sfr.name}: reach line {k + 1} has no STRHC1 field')
        f[9] = f'{strhc1:.6g}'
        lines[k] = '    ' + '  '.join(f)
    sfr.write_text('\n'.join(lines))


def apply_gsflow_parameters(settings_dir: Path, params: Dict[str, float], param_file: str = 'params.dat') -> None:
    """Apply calibration parameters to a GSFLOW settings tree (raises on failure)."""
    modflow_dir = settings_dir / 'modflow'
    if not modflow_dir.exists():
        modflow_dir = settings_dir
    prms = {k: v for k, v in params.items() if k not in MODFLOW_PARAMS + SFR_PARAMS}
    if prms:
        update_prms_parameters(settings_dir / param_file, prms)
    mf = {k: v for k, v in params.items() if k in MODFLOW_PARAMS}
    if mf:
        update_upw(modflow_dir, mf)
    if 'strhc1' in params:
        update_sfr_streambed(modflow_dir, params['strhc1'])
