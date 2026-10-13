"""Ensure calibration cannot corrupt HRU-by-month PRMS parameter blocks."""
from __future__ import annotations

import pytest

from symfluence.models.prms.calibration.parameter_manager import PRMSParameterManager
from symfluence.models.prms.calibration.worker import PRMSWorker


@pytest.mark.parametrize('dimensions,count', [('nmonths',12), ('nhru\nnmonths',60)])
def test_updates_preserve_dimensions(dimensions,count):
    ndim=len(dimensions.splitlines())
    block=f'snow_adj\n{ndim}\n{dimensions}\n{count}\n2\n'+('1.0\n'*count)
    manager=object.__new__(PRMSParameterManager)
    worker=object.__new__(PRMSWorker)
    for changed in [manager._update_block_values(block,'snow_adj',1.7),worker._replace_block_values(block,'snow_adj',1.7)]:
        lines=changed.splitlines()
        assert lines[:ndim+4]==block.splitlines()[:ndim+4]
        assert len(lines[ndim+4:])==count
        assert all(float(v)==1.7 for v in lines[ndim+4:])
        assert manager._extract_first_value(changed,'snow_adj')==1.7
        assert changed.endswith('\n')
