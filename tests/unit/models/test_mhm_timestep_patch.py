"""`symfluence binary install mhm --patched` applies the non-hourly time-step patch,
and the preprocessor refuses MHM_TIMESTEP != 1 without it."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.mhm import build_instructions as bi
from symfluence.models.mhm.preprocessor import MHMPreProcessor

# On Windows runners `bash` may be the WSL launcher (no distribution installed);
# real Windows builds run these steps under the installer's MSYS bash.
POSIX_BASH = os.name != 'nt' and shutil.which('bash') is not None


def test_patch_ships_with_package_and_targets_the_meteo_handler():
    text = bi.TIMESTEP_PATCH.read_text()
    touched = {line[len('+++ b/'):] for line in text.splitlines() if line.startswith('+++ b/')}
    assert touched == {'src/meteo/mo_meteo_handler.f90'}
    assert text.count('+        self%iMeteoTS = ceiling(real(tt, dp) / real(max(1_i4, '
                      'self%nTStepDay / self%nTstepForcingDay), dp))') == 1
    assert len(bi.TIMESTEP_PATCH_BASE) == 40


def test_patch_step_runs_before_the_build_and_pins_the_base_commit():
    commands = bi.get_mhm_build_instructions()['build_commands']
    step = bi._timestep_patch_command()
    assert step in commands and commands.index(step) == 1   # after the common environment
    assert bi.TIMESTEP_PATCH_BASE in step
    assert 'git apply --check' in step and bi.TIMESTEP_PATCH.as_posix() in step
    assert f'bin/{bi.TIMESTEP_PATCH_MARKER}' in step


@pytest.mark.skipif(not POSIX_BASH, reason='POSIX bash required')
def test_patch_step_is_a_no_op_for_ordinary_installs(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != 'SYMFLUENCE_PATCHED'}
    r = subprocess.run(['bash', '-c', bi._timestep_patch_command()], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not POSIX_BASH or shutil.which('git') is None, reason='POSIX bash and git required')
def test_patched_step_fails_loudly_outside_the_pinned_source(tmp_path):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    r = subprocess.run(['bash', '-c', bi._timestep_patch_command()], cwd=tmp_path,
                       env=dict(os.environ, SYMFLUENCE_PATCHED='1', GIT_TERMINAL_PROMPT='0'),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert not (tmp_path / 'bin').exists()


def _preprocessor(timestep: int, install: Path):
    cfg = {'MHM_TIMESTEP': timestep, 'MHM_INSTALL_PATH': str(install)}
    pre = SimpleNamespace(config=None)
    pre._get_config_value = lambda accessor, default=None, dict_key=None: cfg.get(dict_key, default)
    return pre


def test_hourly_step_needs_no_patch(tmp_path):
    assert MHMPreProcessor._model_timestep(_preprocessor(1, tmp_path)) == 1


def test_daily_step_refused_on_an_unpatched_install(tmp_path):
    with pytest.raises(ConfigurationError, match='--patched'):
        MHMPreProcessor._model_timestep(_preprocessor(24, tmp_path))


def test_daily_step_allowed_on_a_patched_install(tmp_path):
    (tmp_path / bi.TIMESTEP_PATCH_MARKER).write_text('')
    assert MHMPreProcessor._model_timestep(_preprocessor(24, tmp_path)) == 24
