"""`symfluence binary install gsflow --patched` applies the glacier conservation patch."""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from symfluence.models.gsflow import build_instructions as bi

PATCHED_FILES = {
    'GSFLOW/src/gsflow/gsflow_module.f90',
    'GSFLOW/src/prms/glacr_melt.f90',
    'GSFLOW/src/prms/sm_read_control_file.f90',
    'GSFLOW/src/prms/snowcomp.f90',
    'GSFLOW/src/prms/srunoff.f90',
}


def test_patch_ships_with_package_and_targets_expected_files():
    text = bi.GLACIER_PATCH.read_text()
    touched = {line[len('+++ b/'):] for line in text.splitlines() if line.startswith('+++ b/')}
    assert touched == PATCHED_FILES
    assert "'glacier_dynamics'" in text
    assert len(bi.GLACIER_PATCH_BASE) == 40


def test_patch_step_runs_before_the_build_and_pins_the_base_commit():
    commands = bi.get_gsflow_build_instructions()['build_commands']
    step = bi._glacier_patch_command()
    assert step in commands and commands.index(step) == 1   # after the common environment
    assert bi.GLACIER_PATCH_BASE in step
    assert 'git apply --check' in step and str(bi.GLACIER_PATCH) in step
    assert 'SYMFLUENCE_PATCHED' in step


@pytest.mark.skipif(shutil.which('bash') is None, reason='bash required')
def test_patch_step_is_a_no_op_for_ordinary_installs(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != 'SYMFLUENCE_PATCHED'}
    r = subprocess.run(['bash', '-c', bi._glacier_patch_command()], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(shutil.which('bash') is None or shutil.which('git') is None, reason='bash and git required')
def test_patched_step_fails_loudly_outside_the_pinned_source(tmp_path):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    r = subprocess.run(['bash', '-c', bi._glacier_patch_command()], cwd=tmp_path,
                       env=dict(os.environ, SYMFLUENCE_PATCHED='1', GIT_TERMINAL_PROMPT='0'),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert not (tmp_path / 'bin').exists()
