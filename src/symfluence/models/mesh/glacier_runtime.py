"""Fail-closed execution checks for the experimental serial finite-ice build."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from .errors import MESHConfigurationError, MESHInputError, MESHOutputError

PROTOCOL = 'SYMFLUENCE_GLACIER_V1'


def _run_dates(inputs):
    lines = (inputs / 'MESH_input_run_options.ini').read_text().splitlines()
    dates = [list(map(int, line.split())) for line in lines
             if re.fullmatch(r'\s*\d{4}\s+\d{1,3}\s+\d{1,2}\s+\d{1,2}\s*', line)]
    if len(dates) != 2:
        raise MESHInputError('Cannot identify finite glacier run dates')
    return dates


def check_finite_runtime(executable: Path, inputs: Path, mode: str) -> bool:
    store = inputs / 'MESH_glacier_storage.ini'
    finite = mode == 'finite_ice_guard'
    if store.exists() != finite:
        raise MESHConfigurationError('MESH finite_ice_guard mode and MESH_glacier_storage.ini must agree')
    if not finite:
        return False
    metadata = executable.with_name(executable.name + '.glacier.json')
    if not metadata.exists():
        raise MESHConfigurationError('Finite glacier runs require a versioned glacier-capable MESH binary manifest')
    data = json.loads(metadata.read_text())
    binary = executable.read_bytes()
    if (data.get('protocol') != PROTOCOL or
            data.get('sha256') != hashlib.sha256(binary).hexdigest() or
            PROTOCOL.encode() not in binary):
        raise MESHConfigurationError('MESH glacier binary capability/hash mismatch')
    flags = {}
    for line in (inputs / 'MESH_input_run_options.ini').read_text().splitlines():
        parts = line.split('#')[0].split()
        if parts and parts[0] in ('RESUMEFLAG', 'SAVERESUMEFLAG'):
            flags[parts[0]] = parts[1:]
    # Only full sequential restart is covered by the native integration test.
    for name in ('RESUMEFLAG', 'SAVERESUMEFLAG'):
        if flags.get(name) not in (['0'], ['4']):
            raise MESHConfigurationError(f'Finite glacier mode supports only {name} 0 or 4')
    if flags['RESUMEFLAG'] == ['4']:
        sidecar = inputs / 'int_statVariables.seq.runclass36.glacier-v1'
        if not sidecar.exists():
            raise MESHInputError('Finite glacier restart is missing its persistent ice sidecar')
        clock = list(map(int, sidecar.read_text().splitlines()[-1].split()))
        if clock != _run_dates(inputs)[0]:
            raise MESHConfigurationError('Glacier restart clock must equal the requested forcing start date')
    return True


def check_finite_output(inputs: Path) -> None:
    import numpy as np
    import pandas as pd

    if (inputs / 'MESH_glacier_depletion.txt').exists():
        raise MESHOutputError('Glacier exhausted: output cannot be scored')
    path = inputs / 'MESH_glacier_ledger.csv'
    if not path.exists():
        raise MESHOutputError('Finite glacier run produced no native ice ledger')
    data = pd.read_csv(path)
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise MESHOutputError('Empty/nonfinite native glacier ledger')
    if (data.end_kgm2 < 0).any() or (data.residual_kgm2.abs() > 1e-7).any():
        raise MESHOutputError('Native glacier store failed its mass-balance check')
    dates = [datetime(year, 1, 1) + timedelta(days=day-1, hours=hour, minutes=minute)
             for year, day, hour, minute in _run_dates(inputs)]
    options = (inputs / 'MESH_input_run_options.ini').read_text()
    timestep = re.search(r'^TIMESTEPFLAG\s+(\d+)\b', options, re.MULTILINE)
    if not timestep or int(timestep[1]) <= 0:
        raise MESHConfigurationError('Finite glacier coverage requires an explicit positive TIMESTEPFLAG')
    expected = (dates[1] - dates[0]).total_seconds() / (int(timestep[1]) * 60)
    initial = (inputs / 'MESH_glacier_storage.ini').read_text().splitlines()
    expected_ids = {int(row.split()[0]) for row in initial[1:]}
    if set(data.mid) != expected_ids:
        raise MESHOutputError('Native glacier ledger tile identities are incomplete')
    for _, tile in data.groupby('tile'):
        if len(tile) != expected:
            raise MESHOutputError('Incomplete native glacier output coverage')
        if not (np.diff(tile.step) == 1).all():
            raise MESHConfigurationError('Missing/repeated native glacier timesteps')
        if not np.allclose(tile.start_kgm2.to_numpy()[1:], tile.end_kgm2.to_numpy()[:-1], rtol=0, atol=1e-7):
            raise MESHOutputError('Discontinuous persistent glacier storage')
