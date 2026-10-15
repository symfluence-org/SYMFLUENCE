"""Compile and execute the actual native store, including fresh-process restart."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[4]/'src/symfluence/models/mesh/patches/glacier_ledger.f90'
DRIVER = '''program test
use glacier_ledger
implicit none
character(30)::arg
integer::k,n
call get_command_argument(1,arg)
if(trim(arg)=='resume') call ledger_read('checkpoint')
call ledger_init([10,11],[40.,-4.],1,2)
n=2
if(trim(arg)=='split'.or.trim(arg)=='resume') n=1
do k=1,n
 call ledger_begin()
 call ledger_flux(2,1,2.)
 call ledger_flux(2,2,.1)
 call ledger_flux(2,3,.5)
 call ledger_flux(2,4,.2)
 call ledger_commit()
enddo
call ledger_save('checkpoint')
end program
'''


@pytest.fixture
def executable(tmp_path):
    compiler = shutil.which('gfortran')
    if not compiler:
        pytest.skip('Native ledger tests require gfortran')
    (tmp_path/'test.f90').write_text(DRIVER)
    exe = tmp_path/'test'
    subprocess.run([compiler, '-fcheck=all', '-ffree-line-length-none', str(SOURCE),
                    'test.f90', '-o', str(exe)], cwd=tmp_path, check=True, capture_output=True)
    return exe


def test_restart_conservation(executable, tmp_path):
    checkpoints = []
    for case in ('full', 'split'):
        root = tmp_path/case
        root.mkdir()
        (root/'MESH_glacier_storage.ini').write_text('1\n11 10\n')
        subprocess.run([executable, case], cwd=root, check=True, capture_output=True)
        if case == 'split':
            subprocess.run([executable, 'resume'], cwd=root, check=True, capture_output=True)
        checkpoints.append((root/'checkpoint').read_bytes())
        data = np.genfromtxt(root/'MESH_glacier_ledger.csv', delimiter=',', names=True)
        assert np.max(abs(data['residual_kgm2'])) < 1e-12
        # Single-precision process fluxes accumulated in double precision.
        assert float(np.atleast_1d(data['end_kgm2'])[-1]) == pytest.approx(6.8, abs=1e-7)
    assert checkpoints[0] == checkpoints[1]


def test_depletion_rejects_unfunded_melt(executable, tmp_path):
    (tmp_path/'MESH_glacier_storage.ini').write_text('1\n11 .01\n')
    result = subprocess.run([executable, 'full'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'GLACIER_DEPLETION' in result.stderr
    assert not (tmp_path/'checkpoint').exists()
    assert len((tmp_path/'MESH_glacier_ledger.csv').read_text().splitlines()) == 1


def test_missing_restart_is_not_a_cold_start(executable, tmp_path):
    (tmp_path/'MESH_glacier_storage.ini').write_text('1\n11 10\n')
    result = subprocess.run([executable, 'resume'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'missing restart sidecar' in result.stderr
