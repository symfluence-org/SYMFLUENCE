"""Finite glacier workflow cannot silently fall back to infinite-ice execution."""
from __future__ import annotations

import hashlib
import json

import pytest

from symfluence.models.mesh.glacier_runtime import PROTOCOL, check_finite_output, check_finite_runtime


def test_binary_and_input_agreement(tmp_path):
    exe = tmp_path/'mesh'
    exe.write_bytes(PROTOCOL.encode())
    (tmp_path/'MESH_glacier_storage.ini').write_text('1\n11 10\n')
    (tmp_path/'MESH_input_run_options.ini').write_text('RESUMEFLAG 0\nSAVERESUMEFLAG 0\n')
    with pytest.raises(ValueError, match='must agree'):
        check_finite_runtime(exe, tmp_path, 'none')
    with pytest.raises(ValueError, match='manifest'):
        check_finite_runtime(exe, tmp_path, 'finite_ice_guard')
    (tmp_path/'mesh.glacier.json').write_text(json.dumps({'protocol': PROTOCOL,
                                  'sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}))
    assert check_finite_runtime(exe, tmp_path, 'finite_ice_guard')
    exe.write_bytes(b'old binary')
    with pytest.raises(ValueError, match='mismatch'):
        check_finite_runtime(exe, tmp_path, 'finite_ice_guard')


def test_partial_ledger_cannot_be_scored(tmp_path):
    (tmp_path/'MESH_glacier_storage.ini').write_text('1\n11 10\n')
    (tmp_path/'MESH_input_run_options.ini').write_text('TIMESTEPFLAG 60\n2013 182 0 0\n2013 182 2 0\n')
    (tmp_path/'MESH_glacier_ledger.csv').write_text(
        'step,tile,mid,start_kgm2,end_kgm2,residual_kgm2\n1,2,11,10,9,0\n')
    with pytest.raises(ValueError, match='coverage'):
        check_finite_output(tmp_path)
