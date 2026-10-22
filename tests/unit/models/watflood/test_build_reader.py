"""Verify the generated build selects the current native parameter reader."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from symfluence.models.watflood.build_instructions import get_watflood_build_instructions


@pytest.mark.parametrize('modern', [False, True])
def test_generated_common_library_selects_one_reader(tmp_path, modern):
    if not shutil.which('cmake'):
        pytest.skip('CMake unavailable')
    commands = get_watflood_build_instructions()['build_commands']
    script = next(s for s in commands if '# Common CMakeLists' in s)
    section = script.split('# Common CMakeLists', 1)[1].split('CMAKE_EOF\n', 1)[1].split('\nCMAKE_EOF', 1)[0]
    # Render the same shell heredoc substitutions used by the installer.
    sources = 'read_par_parser.f;other.f;'
    (tmp_path / 'read_par_parser.f').write_text('')
    if modern:
        sources += 'read_par_parser.f90;'
        (tmp_path / 'read_par_parser.f90').write_text('')
    rendered = section.replace('${COMMON_SOURCES}', sources, 1).replace('\\$', '$')
    # Exercise the actual generated CMake list selection without compiling the
    # unrelated native model or requiring its external libraries.
    selection = rendered.split('set(COMMON_SOURCES', 1)[1].split('add_library', 1)[0]
    cmake = tmp_path / 'select.cmake'
    cmake.write_text('set(COMMON_SOURCES' + selection + '\nfile(WRITE "selected.txt" "${COMMON_SOURCES}")\n')
    subprocess.run(['cmake', '-P', str(cmake)], cwd=tmp_path, check=True)
    selected = (tmp_path / 'selected.txt').read_text().split(';')
    assert ('read_par_parser.f90' in selected) == modern
    assert ('read_par_parser.f' in selected) == (not modern)
    assert 'other.f' in selected
