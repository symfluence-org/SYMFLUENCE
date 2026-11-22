# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for PRMS: pseudo-parameters -> PRMS correction blocks."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.models.prms import forcing_correction as prms_fc
from symfluence.models.prms.parameter_file import forcing_correction_blocks, update_parameter_file

LOG = logging.getLogger('test_prms_fc')
NHRU = 2


def _block(name, dims, values):
    lines = [name, str(len(dims)), *dims, str(len(values)), '2', *[f'{v:.6f}' for v in values]]
    return '\n'.join(lines) + '\n'


def write_params(path: Path, cbh: bool = False) -> Path:
    blocks = [
        _block('soil_moist_max', ['nhru'], [6.0] * NHRU),
        _block('tmax_adj', ['nhru'], [0.0] * NHRU),
        _block('tmin_adj', ['nhru'], [0.0] * NHRU),
        _block('rain_adj', ['nmonths'], [1.0] * 12),
        _block('snow_adj', ['nmonths'], [1.0] * 12),
    ]
    if cbh:
        for name, v in (('rain_cbh_adj', 1.0), ('snow_cbh_adj', 1.0), ('tmax_cbh_adj', 0.0), ('tmin_cbh_adj', 0.0)):
            blocks.append(_block(name, ['nhru', 'nmonths'], [v] * (NHRU * 12)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('Header\nVersion: 1.7\n** Parameters **\n####\n' + '####\n'.join(blocks), encoding='utf-8')
    return path


def read_block(path: Path, name: str):
    for b in path.read_text().split('####\n'):
        lines = b.strip().split('\n')
        if name in [line.strip() for line in lines]:
            i = [line.strip() for line in lines].index(name)
            ndim = int(lines[i + 1])
            n = int(lines[i + 2 + ndim])
            start = i + 4 + ndim
            return [float(v) for v in lines[start:start + n]]
    raise KeyError(name)


def test_blocks_station_and_cbh():
    params = {'FC_PRECIP_MULT': 1.2, 'FC_SNOW_MULT': 1.5, 'FC_TEMP_OFFSET': -0.5}
    assert forcing_correction_blocks(params, {'rain_adj'}) == pytest.approx(
        {'rain_adj': 1.2, 'snow_adj': 1.8, 'tmax_adj': -0.5, 'tmin_adj': -0.5})
    assert forcing_correction_blocks(params, {'rain_adj', 'rain_cbh_adj'}) == pytest.approx(
        {'rain_cbh_adj': 1.2, 'snow_cbh_adj': 1.8, 'tmax_cbh_adj': -0.5, 'tmin_cbh_adj': -0.5})
    assert forcing_correction_blocks({'FC_SNOW_MULT': 1.3}, set()) == pytest.approx(
        {'rain_adj': 1.0, 'snow_adj': 1.3})
    assert forcing_correction_blocks({'soil_moist_max': 3.0}, set()) == {}


def test_update_parameter_file_station_mode(tmp_path):
    p = write_params(tmp_path / 'params.dat')
    update_parameter_file(p, {'soil_moist_max': 5.0, 'FC_PRECIP_MULT': 1.1, 'FC_TEMP_OFFSET': 0.7})
    assert read_block(p, 'soil_moist_max') == pytest.approx([5.0] * NHRU)
    assert read_block(p, 'rain_adj') == pytest.approx([1.1] * 12)
    assert read_block(p, 'snow_adj') == pytest.approx([1.1] * 12)
    assert read_block(p, 'tmax_adj') == pytest.approx([0.7] * NHRU)
    assert read_block(p, 'tmin_adj') == pytest.approx([0.7] * NHRU)
    assert 'FC_PRECIP_MULT' not in p.read_text()


def test_update_parameter_file_climate_hru_mode(tmp_path):
    p = write_params(tmp_path / 'params.dat', cbh=True)
    update_parameter_file(p, {'FC_PRECIP_MULT': 1.25})
    assert read_block(p, 'rain_cbh_adj') == pytest.approx([1.25] * NHRU * 12)
    assert read_block(p, 'snow_cbh_adj') == pytest.approx([1.25] * NHRU * 12)
    assert read_block(p, 'rain_adj') == pytest.approx([1.0] * 12)  # station blocks unused


def test_worker_applies_pseudo_parameters(tmp_path):
    from symfluence.models.prms.calibration.worker import PRMSWorker

    settings = tmp_path / 'proc' / 'PRMS'
    p = write_params(settings / 'params.dat')
    cfg = {'DOMAIN_NAME': 'd', 'SYMFLUENCE_DATA_DIR': str(tmp_path / 'none'),
           'PRMS_PARAMETER_FILE': 'params.dat'}
    worker = PRMSWorker(config=cfg, logger=LOG)
    assert worker.apply_parameters({'soil_moist_max': 4.0, 'FC_PRECIP_MULT': 0.9}, settings, config=cfg)
    assert read_block(p, 'rain_adj') == pytest.approx([0.9] * 12)
    assert read_block(p, 'soil_moist_max') == pytest.approx([4.0] * NHRU)


def test_calibrated_plan_pseudo_parameters_round_trip(monkeypatch):
    monkeypatch.setitem(fc.ADAPTERS, 'PRMS', fc.ForcingCorrectionAdapter(
        'PRMS', precip='FC_PRECIP_MULT', snow='FC_SNOW_MULT', temp='FC_TEMP_OFFSET'))
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated',
                          'FORCING_CORRECTION_QUANTITIES': 'precip,snow,temp'}, 'PRMS')
    native = plan.to_native({'FC_PRECIP_MULT': 1.2, 'FC_SNOW_MULT': 1.1, 'FC_TEMP_OFFSET': 1.0})
    assert native == {'FC_PRECIP_MULT': 1.2, 'FC_SNOW_MULT': 1.1, 'FC_TEMP_OFFSET': 1.0}
    assert plan.values_from_native(native) == {'precip': 1.2, 'snow': 1.1, 'temp': 1.0}


def test_apply_forward_and_discovery(tmp_path):
    p = write_params(tmp_path / 'settings' / 'PRMS' / 'params.dat')
    details = prms_fc.apply_forward({'PRMS_PARAMETER_FILE': 'params.dat'}, LOG, tmp_path,
                                    {'native': {'FC_PRECIP_MULT': 1.3}})
    assert details['blocks'] == pytest.approx({'rain_adj': 1.3, 'snow_adj': 1.3})
    assert read_block(p, 'snow_adj') == pytest.approx([1.3] * 12)
    # idempotent
    prms_fc.apply_forward({'PRMS_PARAMETER_FILE': 'params.dat'}, LOG, tmp_path,
                          {'native': {'FC_PRECIP_MULT': 1.3}})
    assert read_block(p, 'rain_adj') == pytest.approx([1.3] * 12)

    runner = type('PRMSRunner', (), {'__module__': 'symfluence.models.prms.runner'})
    assert fc.find_forward_applier(runner) is prms_fc.apply_forward


def test_apply_forward_missing_file_raises(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError
    with pytest.raises(ModelExecutionError):
        prms_fc.apply_forward({}, LOG, tmp_path, {'native': {'FC_PRECIP_MULT': 1.3}})
