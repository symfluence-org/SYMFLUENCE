"""Ensure calibration cannot corrupt HRU-by-month PRMS parameter blocks."""
from __future__ import annotations

import pytest

from symfluence.models.gsflow import parameter_files as gsflow_parameter_files
from symfluence.models.prms.parameter_file import first_block_value, replace_block_values


@pytest.mark.parametrize('dimensions,count', [('nmonths',12), ('nhru\nnmonths',60)])
def test_updates_preserve_dimensions(dimensions,count):
    ndim=len(dimensions.splitlines())
    block=f'snow_adj\n{ndim}\n{dimensions}\n{count}\n2\n'+('1.0\n'*count)
    changed=replace_block_values(block,'snow_adj',1.7)
    lines=changed.splitlines()
    assert lines[:ndim+4]==block.splitlines()[:ndim+4]
    assert len(lines[ndim+4:])==count
    assert all(float(v)==1.7 for v in lines[ndim+4:])
    assert first_block_value(changed,'snow_adj')==1.7
    assert first_block_value(changed,'rain_adj') is None
    assert changed.endswith('\n')


def test_prms_and_gsflow_share_one_block_writer():
    assert gsflow_parameter_files.replace_block_values is replace_block_values
