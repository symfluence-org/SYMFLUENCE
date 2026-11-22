# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for GSFLOW (PRMS surface blocks)."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.models.gsflow import forcing_correction as gsflow_fc
from symfluence.models.gsflow.parameter_files import apply_gsflow_parameters

LOG = logging.getLogger('test_gsflow_fc')

_spec = importlib.util.spec_from_file_location(
    '_prms_fc_helpers', Path(__file__).parents[1] / 'prms' / 'test_forcing_correction_prms.py')
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
write_params, read_block = _helpers.write_params, _helpers.read_block


def test_calibration_path_writes_prms_blocks(tmp_path):
    settings = tmp_path / 'GSFLOW'
    p = write_params(settings / 'params.dat')
    apply_gsflow_parameters(settings, {'soil_moist_max': 3.0, 'FC_PRECIP_MULT': 1.15,
                                       'FC_TEMP_OFFSET': -1.0})
    assert read_block(p, 'rain_adj') == pytest.approx([1.15] * 12)
    assert read_block(p, 'snow_adj') == pytest.approx([1.15] * 12)
    assert read_block(p, 'tmin_adj') == pytest.approx([-1.0, -1.0])
    assert read_block(p, 'soil_moist_max') == pytest.approx([3.0, 3.0])


def test_worker_applies_pseudo_parameters(tmp_path):
    from symfluence.models.gsflow.calibration.worker import GSFLOWWorker

    settings = tmp_path / 'proc' / 'GSFLOW'
    p = write_params(settings / 'params.dat', cbh=True)
    cfg = {'DOMAIN_NAME': 'd', 'SYMFLUENCE_DATA_DIR': str(tmp_path / 'none'),
           'GSFLOW_PARAMETER_FILE': 'params.dat'}
    assert GSFLOWWorker(config=cfg, logger=LOG).apply_parameters(
        {'FC_PRECIP_MULT': 1.2, 'FC_SNOW_MULT': 1.5}, settings, config=cfg)
    assert read_block(p, 'rain_cbh_adj') == pytest.approx([1.2] * 24)
    assert read_block(p, 'snow_cbh_adj') == pytest.approx([1.8] * 24)


def test_apply_forward_and_discovery(tmp_path):
    p = write_params(tmp_path / 'settings' / 'GSFLOW' / 'params.dat')
    details = gsflow_fc.apply_forward({'GSFLOW_PARAMETER_FILE': 'params.dat'}, LOG, tmp_path,
                                      {'native': {'FC_PRECIP_MULT': 1.0, 'FC_TEMP_OFFSET': 0.0}})
    assert details['blocks'] == {'rain_adj': 1.0, 'snow_adj': 1.0, 'tmax_adj': 0.0, 'tmin_adj': 0.0}
    gsflow_fc.apply_forward({'GSFLOW_PARAMETER_FILE': 'params.dat'}, LOG, tmp_path,
                            {'native': {'FC_PRECIP_MULT': 0.8}})
    assert read_block(p, 'rain_adj') == pytest.approx([0.8] * 12)
    runner = type('GSFLOWRunner', (), {'__module__': 'symfluence.models.gsflow.runner'})
    assert fc.find_forward_applier(runner) is gsflow_fc.apply_forward


def test_settings_path_override(tmp_path):
    custom = tmp_path / 'custom'
    p = write_params(custom / 'params.dat')
    gsflow_fc.apply_forward({'SETTINGS_GSFLOW_PATH': str(custom), 'GSFLOW_PARAMETER_FILE': 'params.dat'},
                            LOG, tmp_path, {'native': {'FC_PRECIP_MULT': 1.4}})
    assert read_block(p, 'rain_adj') == pytest.approx([1.4] * 12)
