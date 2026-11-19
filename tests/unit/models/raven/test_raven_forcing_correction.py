# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for Raven: gauge corrections in calibration and forward runs."""
from __future__ import annotations

import json
import logging

import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.models.raven import native
from symfluence.models.raven.forcing_correction import apply_forward, set_gauge_corrections

RAVEN = fc.ForcingCorrectionAdapter('RAVEN', precip='PRECIP_CORR', snow='SNOW_CORR', temp='TEMP_CORR')


@pytest.fixture
def settings(tmp_path):
    d = tmp_path / 'settings' / 'RAVEN'
    d.mkdir(parents=True)
    bands = [dict(lat=64.0, lon=-18.0, elevation=600.0), dict(lat=64.1, lon=-18.1, elevation=900.0)]
    hrus = [dict(id=1, band=0, ice=False, area_km2=1.0, **bands[0]),
            dict(id=2, band=1, ice=True, area_km2=1.0, **bands[1])]
    (d / 'geometry.json').write_text(json.dumps(dict(hrus=hrus, bands=bands, forcing_dir='f', ice_mm=1.0)))
    return d


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(fc.ADAPTERS, 'RAVEN', RAVEN)


def _corrections(rvt):
    return {k: [float(line.split()[1]) for line in rvt.splitlines() if line.strip().startswith(k)]
            for k in (':RainCorrection', ':SnowCorrection', ':TemperatureCorrection')}


def test_calibration_natives_reach_every_gauge(settings, adapter):
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated',
                          'FORCING_CORRECTION_QUANTITIES': 'precip,snow,temp'}, 'RAVEN')
    assert plan.superseded_native_names() == ['PRECIP_CORR', 'TEMP_CORR', 'SNOW_CORR']
    native_params = plan.to_native({'FC_PRECIP_MULT': 1.2, 'FC_SNOW_MULT': 0.9, 'FC_TEMP_OFFSET': -0.5})
    assert native_params == {'PRECIP_CORR': 1.2, 'SNOW_CORR': 0.9, 'TEMP_CORR': -0.5}
    native.parameters(settings, native_params)  # what RavenWorker.apply_parameters writes
    got = _corrections((settings / 'model.rvt').read_text())
    assert got[':RainCorrection'] == [1.2, 1.2]
    assert got[':SnowCorrection'] == pytest.approx([1.08, 1.08])
    assert got[':TemperatureCorrection'] == [-0.5, -0.5]
    assert plan.values_from_native(native_params) == pytest.approx({'precip': 1.2, 'snow': 0.9, 'temp': -0.5})


def test_forward_keeps_calibrated_snow_and_is_idempotent(settings, adapter):
    native.parameters(settings, dict(PRECIP_CORR=1.3, SNOW_CORR=0.8, TEMP_CORR=1.0, MELT_FACTOR=4.0))
    rvp_before = (settings / 'model.rvp').read_text()
    info = fc.forward_correction({'FORCING_CORRECTION_MODE': 'fixed',
                                  'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.1}, 'RAVEN')
    assert info['native'] == {'PRECIP_CORR': 1.1}
    for _ in range(2):
        details = apply_forward({}, logging.getLogger('t'), settings.parent.parent, info)
    got = _corrections((settings / 'model.rvt').read_text())
    assert got[':RainCorrection'] == [1.1, 1.1]
    assert got[':SnowCorrection'] == pytest.approx([1.1 * 0.8] * 2)
    assert got[':TemperatureCorrection'] == [1.0, 1.0]
    assert details['applied']['RainCorrection'] == 1.1
    assert (settings / 'model.rvp').read_text() == rvp_before  # other parameters untouched


def test_forward_off_resets_precip_and_temp(settings, adapter):
    native.parameters(settings, dict(PRECIP_CORR=1.3, SNOW_CORR=0.8, TEMP_CORR=1.0))
    info = fc.forward_correction({'FORCING_CORRECTION_MODE': 'off'}, 'RAVEN')
    apply_forward({}, logging.getLogger('t'), settings.parent.parent, info)
    got = _corrections((settings / 'model.rvt').read_text())
    assert got == {':RainCorrection': [1.0, 1.0], ':SnowCorrection': pytest.approx([0.8, 0.8]),
                   ':TemperatureCorrection': [0.0, 0.0]}


def test_set_gauge_corrections_rejects_unexpected_file(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError
    (tmp_path / 'model.rvt').write_text(':Gauge g\n:EndGauge\n')
    with pytest.raises(ModelExecutionError):
        set_gauge_corrections(tmp_path / 'model.rvt', {'PRECIP_CORR': 1.1})


def test_forward_applier_found_from_runner_module():
    from symfluence.models.raven.adapter import RavenRunner
    assert fc.find_forward_applier(RavenRunner) is apply_forward
