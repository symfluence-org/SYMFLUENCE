"""OutputFileLocator: case-insensitive model table, whole-token experiment ids."""
from __future__ import annotations

import os
import time

from symfluence.core.modeling.output_file_locator import OutputFileLocator


def test_mixed_case_model_keys_are_reachable(tmp_path):
    (tmp_path / 'rhessys_basin.daily').write_text('x')
    (tmp_path / 'other_output.nc').write_text('x')
    locator = OutputFileLocator()
    for spelling in ('RHESSys', 'RHESSYS', 'rhessys'):
        files = locator._find_model_specific(tmp_path, spelling, 'streamflow')
        assert [f.name for f in files] == ['rhessys_basin.daily'], spelling

    mizu = tmp_path / 'mizuRoute'
    mizu.mkdir()
    (mizu / 'case.h.2010-01-01.nc').write_text('x')
    for spelling in ('mizuRoute', 'MIZUROUTE'):
        files = locator._find_model_specific(tmp_path, spelling, 'streamflow')
        assert [f.name for f in files] == ['case.h.2010-01-01.nc'], spelling


def test_experiment_id_matches_whole_tokens_only(tmp_path):
    """'run_1' must not pick up 'run_10_timestep.nc', even when that is newer."""
    own = tmp_path / 'run_1_timestep.nc'
    other = tmp_path / 'run_10_timestep.nc'
    own.write_text('x')
    other.write_text('x')
    now = time.time()
    os.utime(own, (now - 100, now - 100))
    os.utime(other, (now, now))

    files = OutputFileLocator(experiment_id='run_1')._sort_by_mtime([own, other])
    assert files == [own]

    files = OutputFileLocator(experiment_id='run_10')._sort_by_mtime([own, other])
    assert files == [other]

    # no file carries the id -> plain newest-first, as before
    files = OutputFileLocator(experiment_id='run_7')._sort_by_mtime([own, other])
    assert files == [other, own]
