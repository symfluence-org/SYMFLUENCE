# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in forward (non-calibration) runs."""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.core.exceptions import ConfigurationError

LOG = logging.getLogger('test_forcing_correction_forward')


def _cfg(tmp_path, **kw):
    return {'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'exp1', 'SYMFLUENCE_DATA_DIR': str(tmp_path), **kw}


def _write_report(project: Path, model: str, folder: str, applied, mode='calibrated'):
    d = project / 'optimization' / model / folder
    d.mkdir(parents=True)
    (d / fc.REPORT_NAME).write_text(json.dumps({'mode': mode, 'applied': applied}))
    return d / fc.REPORT_NAME


def test_forward_absent_section(tmp_path):
    assert fc.forward_correction(_cfg(tmp_path), 'GR', tmp_path) is None
    assert fc.apply_forward_forcing_correction('GR', _cfg(tmp_path), LOG, tmp_path) is None
    assert not (tmp_path / 'simulations').exists()


def test_forward_fixed_values(tmp_path):
    cfg = _cfg(tmp_path, FORCING_CORRECTION_MODE='fixed', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.2,
               FORCING_CORRECTION_TEMP_OFFSET=-0.4)
    info = fc.forward_correction(cfg, 'GR', tmp_path)
    assert info['native'] == {'PCORR': 1.2, 'TCORR': -0.4}
    assert info['values'] == {'FC_PRECIP_MULT': 1.2, 'FC_TEMP_OFFSET': -0.4}
    assert info['source'] == 'config'


def test_forward_off_resets_to_identity(tmp_path):
    info = fc.forward_correction(_cfg(tmp_path, FORCING_CORRECTION_MODE='off'), 'MESH', tmp_path)
    assert info['native'] == {'PRECIP_MULT': 1.0, 'TEMP_OFFSET': 0.0}
    info = fc.forward_correction(_cfg(tmp_path, FORCING_CORRECTION_MODE='off'), 'HYPE', tmp_path)
    assert info['native'] == {'pcurain': 0.0, 'pcusnow': 0.0}


def test_forward_calibrated_uses_newest_report_of_experiment(tmp_path):
    import os
    old = _write_report(tmp_path, 'GR', 'dds_exp1', {'FC_PRECIP_MULT': 1.1})
    os.utime(old, (1, 1))
    _write_report(tmp_path, 'GR', 'de_exp1', {'FC_PRECIP_MULT': 1.3})
    _write_report(tmp_path, 'GR', 'dds_other', {'FC_PRECIP_MULT': 1.9})
    cfg = _cfg(tmp_path, FORCING_CORRECTION_MODE='calibrated')
    info = fc.forward_correction(cfg, 'GR', tmp_path, experiment_id='exp1')
    assert info['native'] == {'PCORR': 1.3}
    assert info['source'].endswith('de_exp1/forcing_correction.json')


def test_forward_calibrated_without_report_warns(tmp_path, caplog):
    cfg = _cfg(tmp_path, FORCING_CORRECTION_MODE='calibrated', FORCING_CORRECTION_TEMP_OFFSET=0.5)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        info = fc.forward_correction(cfg, 'GR', tmp_path, experiment_id='exp1', log=LOG)
    assert 'no calibration report' in caplog.text
    assert info['native'] == {'TCORR': 0.5}


def test_calibrated_report_round_trip_matches_final_evaluation(tmp_path):
    """The values a calibration reports at its best set are what forward runs apply."""
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated'}, 'SUMMA')
    d = tmp_path / 'optimization' / 'SUMMA' / 'dds_exp1'
    fc.write_report(d / fc.REPORT_NAME, plan, {'precip_mult': [1.27, 1.27], 'k_snow': 0.1})
    info = fc.forward_correction(_cfg(tmp_path, FORCING_CORRECTION_MODE='calibrated'), 'SUMMA',
                                 tmp_path, experiment_id='exp1')
    assert info['native'] == pytest.approx({'precip_mult': 1.27})


@pytest.fixture
def fake_model(monkeypatch):
    """A model package with a runner and a forcing_correction module."""
    calls = []
    pkg = types.ModuleType('fakemodelpkg')
    pkg.__path__ = []
    runner = types.ModuleType('fakemodelpkg.runner')

    class FakeRunner:
        pass
    FakeRunner.__module__ = 'fakemodelpkg.runner'
    runner.FakeRunner = FakeRunner
    applier = types.ModuleType('fakemodelpkg.forcing_correction')

    def apply_forward(config, logger, project_dir, info):
        calls.append(dict(info['native']))
        return {'wrote': True}
    applier.apply_forward = apply_forward
    for name, mod in (('fakemodelpkg', pkg), ('fakemodelpkg.runner', runner),
                      ('fakemodelpkg.forcing_correction', applier)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(fc.ADAPTERS, 'FAKE', fc.ForcingCorrectionAdapter('FAKE', precip='PM'))
    return FakeRunner, calls


def test_apply_forward_dispatches_and_records(tmp_path, fake_model):
    runner_class, calls = fake_model
    cfg = _cfg(tmp_path, FORCING_CORRECTION_MODE='fixed', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.15)
    info = fc.apply_forward_forcing_correction('FAKE', cfg, LOG, tmp_path, runner_class=runner_class)
    assert calls == [{'PM': 1.15}]
    assert info['details'] == {'wrote': True}
    record = json.loads((tmp_path / 'simulations' / 'exp1' / 'FAKE' / fc.FORWARD_RECORD_NAME).read_text())
    assert record['values'] == {'FC_PRECIP_MULT': 1.15} and record['mode'] == 'fixed'


def test_apply_forward_without_applier_for_supported_model_raises(tmp_path):
    class Runner:
        pass
    Runner.__module__ = 'nonexistent_pkg_xyz.runner'
    cfg = _cfg(tmp_path, FORCING_CORRECTION_MODE='fixed', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.1)
    with pytest.raises(ConfigurationError, match='forward-run applier'):
        fc.apply_forward_forcing_correction('GR', cfg, LOG, tmp_path, runner_class=Runner)


def test_apply_forward_unsupported_model_identity_ok(tmp_path):
    cfg = _cfg(tmp_path, FORCING_CORRECTION_MODE='off')
    info = fc.apply_forward_forcing_correction('NOAHMP', cfg, LOG, tmp_path, runner_class=None)
    assert info['native'] == {} and info['details']['applied'] is False


def test_every_adapter_has_a_forward_applier():
    from symfluence.core.registries import R
    for model in fc.ADAPTERS:
        runner_class = R.runners.get(model)
        assert runner_class is not None, model
        assert fc.find_forward_applier(runner_class) is not None, model


def test_model_manager_applies_before_running(tmp_path, fake_model, monkeypatch):
    from symfluence.core.registries import R
    from symfluence.project.model_manager import ModelManager

    runner_class, calls = fake_model
    order = []

    class Runner(runner_class):
        def __init__(self, config, logger, reporting_manager=None):
            order.append(('init', list(calls)))

        def run(self):
            order.append('run')
    Runner.__module__ = 'fakemodelpkg.runner'
    monkeypatch.setattr(R.runners, 'get', lambda key, default=None: Runner)
    monkeypatch.setattr(R.runners, 'meta', lambda key: {})
    mm = object.__new__(ModelManager)
    mm.config = _cfg(tmp_path, FORCING_CORRECTION_MODE='fixed', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.05)
    mm.logger = LOG
    mm.reporting_manager = None
    monkeypatch.setattr(ModelManager, 'project_dir', tmp_path, raising=False)
    mm._run_sequential(['FAKE'])
    assert order == [('init', [{'PM': 1.05}]), 'run']
