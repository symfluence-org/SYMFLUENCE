# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for FUSE: corrected forcing copies in calibration and forward runs."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from symfluence.core.calibration import forcing_correction as fc
from symfluence.models.fuse import forcing_correction as ffc

FUSE = fc.ForcingCorrectionAdapter('FUSE', precip='FC_PRECIP_MULT', temp='FC_TEMP_OFFSET')
LOG = logging.getLogger('t')


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(fc.ADAPTERS, 'FUSE', FUSE)


def _forcing(path: Path):
    ds = xr.Dataset({'precip': (('time',), np.array([1.0, 2.0, 0.0])),
                     'temp': (('time',), np.array([-2.0, 0.0, 3.0])),
                     'pet': (('time',), np.array([0.5, 0.5, 0.5]))},
                    coords={'time': np.arange(3)})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _read(path):
    with xr.open_dataset(path) as ds:
        return ds['precip'].values.copy(), ds['temp'].values.copy(), ds['pet'].values.copy()


def test_calibration_passes_common_names_through(adapter):
    plan = fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated',
                          'FORCING_CORRECTION_QUANTITIES': 'precip,temp'}, 'FUSE')
    assert plan.superseded_native_names() == ['FC_PRECIP_MULT', 'FC_TEMP_OFFSET']
    assert plan.to_native({'FC_PRECIP_MULT': 1.2, 'FC_TEMP_OFFSET': -1.0}) == \
        {'FC_PRECIP_MULT': 1.2, 'FC_TEMP_OFFSET': -1.0}
    with pytest.raises(fc.ConfigurationError):
        fc.build_plan({'FORCING_CORRECTION_MODE': 'calibrated', 'FORCING_CORRECTION_QUANTITIES': 'snow'},
                      'FUSE')


def test_trial_corrects_private_copy_never_the_shared_file(tmp_path):
    shared = tmp_path / 'forcing' / 'FUSE_input' / 'd_input.nc'
    _forcing(shared)
    run = tmp_path / 'proc' / 'FUSE'
    run.mkdir(parents=True)
    model, corr = ffc.split_params({'MAXWATR_1': 100.0, 'FC_PRECIP_MULT': 1.5, 'FC_TEMP_OFFSET': 1.0})
    assert model == {'MAXWATR_1': 100.0} and corr == {'FC_PRECIP_MULT': 1.5, 'FC_TEMP_OFFSET': 1.0}
    ffc.record_trial(run, corr)
    for _ in range(2):  # prepare_input_files re-links before every run
        link = run / 'sim_input.nc'
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(shared)
        assert ffc.apply_trial(run, 'sim') == corr
    assert not (run / 'sim_input.nc').is_symlink()
    p, t, pet = _read(run / 'sim_input.nc')
    np.testing.assert_allclose(p, [1.5, 3.0, 0.0])
    np.testing.assert_allclose(t, [-1.0, 1.0, 4.0])
    np.testing.assert_allclose(pet, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(_read(shared)[0], [1.0, 2.0, 0.0])
    # identity / no correction clears the record
    ffc.record_trial(run, {'FC_PRECIP_MULT': 1.0})
    assert ffc.apply_trial(run, 'sim') is None


def test_trial_corrects_plain_copy_in_place(tmp_path):
    run = tmp_path / 'run'
    _forcing(run / 'sim_input.nc')
    ffc.record_trial(run, {'FC_PRECIP_MULT': 2.0})
    ffc.apply_trial(run, 'sim')
    np.testing.assert_allclose(_read(run / 'sim_input.nc')[0], [2.0, 4.0, 0.0])


def test_worker_keeps_pseudo_params_out_of_constraints(tmp_path):
    from symfluence.models.fuse.calibration.worker import FUSEWorker

    settings = tmp_path / 'FUSE'
    settings.mkdir()
    src = Path(ffc.__file__).parent / 'base_settings' / 'fuse_zConstraints_snow.txt'
    shutil.copy(src, settings / src.name)
    before = (settings / src.name).read_text()
    worker = FUSEWorker(config={}, logger=LOG)
    assert worker.apply_parameters({'FC_PRECIP_MULT': 1.3}, settings)
    assert (settings / src.name).read_text() == before
    assert (settings / ffc.TRIAL_RECORD).exists()
    assert worker.apply_parameters({}, settings)
    assert not (settings / ffc.TRIAL_RECORD).exists()


def test_forward_writes_corrected_copy_and_runner_dir_follows_config(tmp_path, adapter):
    project = tmp_path / 'domain_d'
    src_dir = project / 'data' / 'forcing' / 'FUSE_input'
    _forcing(src_dir / 'd_input.nc')
    (src_dir / 'd_elev_bands.nc').write_bytes(b'bands')
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.25,
           'FORCING_CORRECTION_TEMP_OFFSET': -0.5}
    info = fc.forward_correction(cfg, 'FUSE')
    assert info['native'] == {'FC_PRECIP_MULT': 1.25, 'FC_TEMP_OFFSET': -0.5}
    details = ffc.apply_forward(cfg, LOG, project, info)
    out = project / 'data' / 'forcing' / ffc.CORRECTED_DIR
    assert details['forcing_dir'] == str(out) and details['files'] == ['d_input.nc']
    p, t, _ = _read(out / 'd_input.nc')
    np.testing.assert_allclose(p, [1.25, 2.5, 0.0])
    np.testing.assert_allclose(t, [-2.5, -0.5, 2.5])
    assert (out / 'd_elev_bands.nc').read_bytes() == b'bands'
    np.testing.assert_allclose(_read(src_dir / 'd_input.nc')[0], [1.0, 2.0, 0.0])
    forcing_root = project / 'data' / 'forcing'
    assert ffc.corrected_forcing_dir(cfg, forcing_root) == out
    assert ffc.corrected_forcing_dir({}, forcing_root) is None  # section removed -> shared forcing
    # off -> identity removes the corrected copy
    off = fc.forward_correction({'FORCING_CORRECTION_MODE': 'off'}, 'FUSE')
    ffc.apply_forward({'FORCING_CORRECTION_MODE': 'off'}, LOG, project, off)
    assert not out.exists()
    assert ffc.corrected_forcing_dir(cfg, forcing_root) is None


def test_forward_applier_found_from_runner_module():
    from symfluence.models.fuse.runner import FUSERunner
    assert fc.find_forward_applier(FUSERunner) is ffc.apply_forward
