# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Calibratable precipitation multiplier for SUMMA.

SUMMA only exposes ``frozenPrecipMultip``, which scales snowfall. ``precip_mult``
scales total ``pptrate`` (rain and snow) in the forcing files, e.g. to absorb
gauge undercatch or reanalysis precipitation bias. Scaled copies of the forcing
are written next to the settings directory and the file manager's
``forcingPath`` is redirected to them. The original forcing path is recorded in
``precip_mult_source.json`` inside the settings directory so repeated
applications always scale the unmodified source.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PRECIP_MULT_PARAM = 'precip_mult'
FORCING_PARAMS = (PRECIP_MULT_PARAM,)
FORCING_PARAM_BOUNDS = {PRECIP_MULT_PARAM: {'min': 0.7, 'max': 1.6}}
FORCING_PARAM_DEFAULTS = {PRECIP_MULT_PARAM: 1.0}

SOURCE_RECORD = 'precip_mult_source.json'
SCALED_DIR_NAME = 'forcing_precip_scaled'
APPLIED_RECORD = 'applied_precip_mult.json'

# Source pptrate cache per (source dir, file name); avoids re-reading every trial.
_SOURCE_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def split_forcing_params(names: List[str]) -> Tuple[List[str], List[str]]:
    """Split a parameter list into (other, forcing) parameter names."""
    forcing = [n for n in names if n in FORCING_PARAMS]
    return [n for n in names if n not in FORCING_PARAMS], forcing


def _read_file_manager(fm_path: Path) -> Tuple[List[str], Dict[str, str]]:
    lines = fm_path.read_text(encoding='utf-8').splitlines(keepends=True)
    entries = {}
    for line in lines:
        parts = line.split(None, 1)
        if len(parts) == 2 and "'" in parts[1]:
            entries[parts[0]] = parts[1].split("'")[1]
    return lines, entries


def apply_precip_multiplier(
    settings_dir: Path,
    multiplier: float,
    logger: Optional[logging.Logger] = None,
    file_manager: str = 'fileManager.txt',
) -> bool:
    """Write forcing with ``pptrate`` scaled by ``multiplier`` and point SUMMA at it."""
    import netCDF4 as nc

    logger = logger or logging.getLogger(__name__)
    settings_dir = Path(settings_dir)
    multiplier = float(np.atleast_1d(multiplier)[0])
    if not np.isfinite(multiplier) or multiplier <= 0:
        logger.error(f"Invalid {PRECIP_MULT_PARAM}: {multiplier}")
        return False

    fm_path = settings_dir / file_manager
    if not fm_path.exists():
        logger.error(f"File manager not found for {PRECIP_MULT_PARAM}: {fm_path}")
        return False
    lines, entries = _read_file_manager(fm_path)

    record = settings_dir / SOURCE_RECORD
    if record.exists():
        source_dir = Path(json.loads(record.read_text())['forcing_path'])
    else:
        source_dir = Path(entries['forcingPath'])
        record.write_text(json.dumps({'forcing_path': str(source_dir)}, indent=2))

    target_dir = settings_dir.parent.parent / SCALED_DIR_NAME
    if target_dir.resolve() == source_dir.resolve():
        logger.error(f"{PRECIP_MULT_PARAM} source and target forcing directories coincide: {source_dir}")
        return False
    target_dir.mkdir(parents=True, exist_ok=True)

    list_file = settings_dir / entries.get('forcingListFile', 'forcingFileList.txt')
    names = [n.strip().strip("'") for n in list_file.read_text().splitlines() if n.strip()]

    applied = target_dir / APPLIED_RECORD
    previous = json.loads(applied.read_text()) if applied.exists() else {}
    if (previous.get('multiplier') != multiplier or previous.get('source') != str(source_dir)
            or previous.get('files') != names):
        if applied.exists():
            applied.unlink()  # invalidate while files are being rewritten
        for name in names:
            src, dst = source_dir / name, target_dir / name
            key = (str(source_dir), name)
            if key not in _SOURCE_CACHE:
                with nc.Dataset(src) as ds:
                    _SOURCE_CACHE[key] = np.asarray(ds['pptrate'][:], dtype='f8')
            if not dst.exists():
                shutil.copyfile(src, dst)
            with nc.Dataset(dst, 'r+') as ds:
                ds['pptrate'][:] = _SOURCE_CACHE[key] * multiplier
                ds['pptrate'].setncattr('precip_mult', multiplier)
        applied.write_text(json.dumps(
            {'multiplier': multiplier, 'source': str(source_dir), 'files': names}, indent=2))

    target = f"{target_dir}/"
    if entries.get('forcingPath') != target:
        updated = [f"forcingPath          '{target}'\n" if line.split(None, 1)[:1] == ['forcingPath'] else line
                   for line in lines]
        fm_path.write_text(''.join(updated), encoding='utf-8')
    logger.debug(f"Applied {PRECIP_MULT_PARAM}={multiplier:.4f} to {len(names)} forcing files in {target_dir}")
    return True
