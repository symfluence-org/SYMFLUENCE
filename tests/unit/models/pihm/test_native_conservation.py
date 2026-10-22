"""Compile actual native PIHM edge fluxes, rather than a Python reimplementation."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def test_native_surface_flux_cancels_across_shared_edge(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = Path(os.environ.get('PIHM_SOURCE_DIR', root / 'output/tungnaa_pihm/source'))
    compiler = shutil.which('cc')
    if not compiler or not (source / 'cvode/instdir/include').is_dir():
        pytest.skip('Native PIHM source, CVODE headers and C compiler required')
    executable = tmp_path / 'shared_edge'
    subprocess.run([
        compiler, '-I' + str(source / 'src/include'),
        '-I' + str(source / 'cvode/instdir/include'),
        str(Path(__file__).parent / 'fixtures/check_shared_edge.c'),
        str(source / 'src/lat_flow.c'), '-lm', '-o', str(executable),
    ], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
