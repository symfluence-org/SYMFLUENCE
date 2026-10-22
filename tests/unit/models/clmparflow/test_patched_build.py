# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""`symfluence binary install clmparflow --patched` builds ParFlow v3.15.0 with the SYMFLUENCE CLM patches."""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from symfluence.models.clmparflow import build_instructions as bi

# On Windows runners `bash` may be the WSL launcher (no distribution installed).
POSIX_BASH = os.name != 'nt' and shutil.which('bash') is not None

PATCHED_FILES = {
    'parflow3150_clm_glacier_water.patch': {'pfsimulator/clm/CMakeLists.txt', 'pfsimulator/clm/clm.F90',
                                            'pfsimulator/clm/clm_main.F90', 'pfsimulator/clm/clm_hydro_ice.F90'},
    'parflow3150_clm_restart_read.patch': {'pfsimulator/clm/drv_restart.F90'},
    'parflow3150_clm_ice_albedo.patch': {'pfsimulator/clm/clm_soilalb.F90'},
}


def test_patches_ship_with_the_package_and_touch_only_clm():
    assert {p.name for p in bi.PATCHES} == set(PATCHED_FILES)
    for p in bi.PATCHES:
        touched = {line[len('+++ b/'):] for line in p.read_text().splitlines() if line.startswith('+++ b/')}
        assert touched == PATCHED_FILES[p.name]
    assert 'clm_hydro_ice' in bi.PATCHES[0].read_text() and 'glacier_ice.txt.' in bi.PATCHES[0].read_text()


def test_patched_step_runs_first_and_pins_the_release():
    commands = bi.get_clmparflow_build_instructions()['build_commands']
    step = bi._patched_build_command()
    assert commands[0] == step
    assert f'git checkout -q "{bi.PATCH_BASE_TAG}"' in step and bi.HYPRE_TAG in step
    assert all(f'git apply --check "{p.as_posix()}"' in step for p in bi.PATCHES)
    assert '-DPARFLOW_HAVE_CLM=ON' in step and 'SYMFLUENCE_PATCHED' in step
    assert 'bin/.symfluence_patched' in commands[1]  # the unpatched download/build is skipped afterwards


@pytest.mark.skipif(not POSIX_BASH, reason='POSIX bash required')
def test_patched_step_is_a_no_op_for_ordinary_installs(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != 'SYMFLUENCE_PATCHED'}
    r = subprocess.run(['bash', '-c', bi._patched_build_command()], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not POSIX_BASH or shutil.which('git') is None, reason='POSIX bash and git required')
def test_patched_step_fails_loudly_outside_the_parflow_source(tmp_path):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    r = subprocess.run(['bash', '-c', bi._patched_build_command()], cwd=tmp_path,
                       env=dict(os.environ, SYMFLUENCE_PATCHED='1', GIT_TERMINAL_PROMPT='0'),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert not (tmp_path / 'bin' / '.symfluence_patched').exists()
