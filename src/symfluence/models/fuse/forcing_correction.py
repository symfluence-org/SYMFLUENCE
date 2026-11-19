# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for FUSE: corrected copies of the FUSE forcing.

FUSE's own rainfall-error parameters (``RFERR_MLT``/``RFERR_ADD``) act only
under the matching ``RFERR`` model decision and are fixed by the constraints
file, so the correction is applied to the forcing instead: ``precip`` is
multiplied by ``FC_PRECIP_MULT`` and ``temp`` (degC) offset by
``FC_TEMP_OFFSET`` in a copy. The shared ``forcing/FUSE_input`` files are never
written. PET in the forcing was computed from the uncorrected temperature and
is left as it is.

* Calibration: the worker receives ``FC_PRECIP_MULT`` / ``FC_TEMP_OFFSET`` as
  pseudo-parameters, records them in its settings directory
  (:data:`TRIAL_RECORD`) and replaces the run directory's ``sim_input.nc``
  symlink by a corrected copy before FUSE runs.
* Forward runs: :func:`apply_forward` writes corrected copies to
  ``forcing/FUSE_input_corrected``, which the runner reads instead of
  ``FUSE_input`` while FORCING_CORRECTION is configured.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from symfluence.core.exceptions import ModelExecutionError

PRECIP = 'FC_PRECIP_MULT'
TEMP = 'FC_TEMP_OFFSET'
PSEUDO_PARAMS = (PRECIP, TEMP)

#: Correction for the next calibration run, in the worker's FUSE settings dir.
TRIAL_RECORD = 'fuse_forcing_correction.json'
#: Directory of corrected forward-run forcing, next to ``FUSE_input``.
CORRECTED_DIR = 'FUSE_input_corrected'
#: Record in :data:`CORRECTED_DIR` describing the correction and its source.
CORRECTED_RECORD = 'forcing_correction.json'


def split_params(params: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:
    """``(model parameters, forcing corrections)`` from a parameter set."""
    model = {k: v for k, v in params.items() if k not in PSEUDO_PARAMS}
    corrections = {k: float(params[k]) for k in PSEUDO_PARAMS if k in params}
    return model, corrections


def is_identity(corrections: Mapping[str, float]) -> bool:
    return (float(corrections.get(PRECIP, 1.0)) == 1.0
            and float(corrections.get(TEMP, 0.0)) == 0.0)


def write_corrected(source: Path, target: Path, corrections: Mapping[str, float],
                    in_place: bool = False) -> None:
    """Write ``source`` to ``target`` with ``precip`` scaled and ``temp`` offset.

    Written through a temporary file and renamed, so ``target`` is never
    half-written. ``target`` must not be ``source`` (or a link to it) unless
    ``in_place`` (a private copy).
    """
    import netCDF4 as nc
    import numpy as np

    source, target = Path(source).resolve(), Path(target)
    # A symlink is replaced, never written through; a real file must not be the source.
    if not in_place and not target.is_symlink() and target.exists() and target.resolve() == source:
        raise ModelExecutionError(f'Refusing to write the forcing correction into its source {source}')
    tmp = target.with_name(target.name + f'.tmp{os.getpid()}')
    shutil.copyfile(source, tmp)
    try:
        with nc.Dataset(tmp, 'r+') as ds:
            m = float(corrections.get(PRECIP, 1.0))
            dt = float(corrections.get(TEMP, 0.0))
            if m != 1.0:
                if 'precip' not in ds.variables:
                    raise ModelExecutionError(f"No 'precip' variable in {source}")
                ds['precip'][:] = np.asarray(ds['precip'][:]) * m
                ds['precip'].setncattr('symfluence_precip_mult', m)
            if dt != 0.0:
                if 'temp' not in ds.variables:
                    raise ModelExecutionError(f"No 'temp' variable in {source}")
                ds['temp'][:] = np.asarray(ds['temp'][:]) + dt
                ds['temp'].setncattr('symfluence_temp_offset', dt)
        if target.exists() or target.is_symlink():
            target.unlink()
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


# -- calibration -------------------------------------------------------------

def record_trial(fuse_settings_dir: Path, corrections: Mapping[str, float]) -> None:
    """Record the corrections for the next run from ``fuse_settings_dir`` (clear when none)."""
    record = Path(fuse_settings_dir) / TRIAL_RECORD
    if corrections and not is_identity(corrections):
        record.write_text(json.dumps(dict(corrections)), encoding='utf-8')
    else:
        record.unlink(missing_ok=True)


def apply_trial(execution_cwd: Path, run_id: str) -> Optional[Dict[str, float]]:
    """Replace ``<run_id>_input.nc`` in ``execution_cwd`` by a corrected copy, if recorded."""
    record = Path(execution_cwd) / TRIAL_RECORD
    if not record.exists():
        return None
    corrections = json.loads(record.read_text(encoding='utf-8'))
    link = Path(execution_cwd) / f'{run_id}_input.nc'
    if link.is_symlink():
        write_corrected(link.resolve(), link, corrections)
    else:  # a private copy (no symlink support): correct it in place
        write_corrected(link, link, corrections, in_place=True)
    return corrections


# -- forward runs ------------------------------------------------------------

def corrected_forcing_dir(config: Any, project_forcing_dir: Path) -> Optional[Path]:
    """Corrected forcing directory the runner should read, or None.

    Only while FORCING_CORRECTION is configured, so removing the section
    returns the runner to the shared ``FUSE_input``.
    """
    from symfluence.core.calibration.forcing_correction import configured_mode

    directory = Path(project_forcing_dir) / CORRECTED_DIR
    if configured_mode(config) is None or not (directory / CORRECTED_RECORD).exists():
        return None
    return directory


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """Write corrected copies of the FUSE forcing for a forward run (or remove them for identity)."""
    from symfluence.core.mixins.project import resolve_data_subdir

    forcing_dir = resolve_data_subdir(Path(project_dir), 'forcing')
    source_dir = forcing_dir / 'FUSE_input'
    target_dir = forcing_dir / CORRECTED_DIR
    _, corrections = split_params(info.get('native', {}))
    if is_identity(corrections):
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return {'forcing_dir': str(source_dir), 'applied': 'identity, shared forcing used'}
    if not source_dir.is_dir():
        raise ModelExecutionError(f'FUSE forcing directory not found: {source_dir}')
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / CORRECTED_RECORD).unlink(missing_ok=True)  # invalid while rewriting
    written = []
    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        dst = target_dir / src.name
        if src.name.endswith('_input.nc'):
            write_corrected(src, dst, corrections)
            written.append(src.name)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
    if not written:
        raise ModelExecutionError(f'No *_input.nc forcing in {source_dir}')
    (target_dir / CORRECTED_RECORD).write_text(json.dumps(
        {'source': str(source_dir), 'corrections': corrections, 'files': written}, indent=2),
        encoding='utf-8')
    return {'forcing_dir': str(target_dir), 'applied': corrections, 'files': written}
