# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Forward-run appliers of the shared FORCING_CORRECTION (GR, SUMMA, MESH, Wflow, mHM, HYPE)."""
from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import pytest

from symfluence.core.calibration import forcing_correction as fc

LOG = logging.getLogger('test_forcing_correction_appliers')


def _info(model, **cfg):
    config = {'FORCING_CORRECTION_MODE': 'fixed', **cfg}
    return config, fc.forward_correction(config, model)


# --------------------------------------------------------------------------- GR

def test_gr_runner_uses_shared_values_without_parameter_set(tmp_path):
    from symfluence.models.gr.runner import GRRunner
    config, _ = _info('GR', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.2, FORCING_CORRECTION_TEMP_OFFSET=0.3)
    stub = SimpleNamespace(config=config, project_dir=tmp_path, domain_name='d', experiment_id='e',
                           logger=LOG, _external_params=None)
    assert GRRunner._forcing_corrections(stub) == {'PCORR': 1.2, 'TCORR': 0.3}
    # a calibration parameter set wins
    stub = SimpleNamespace(config=config, project_dir=tmp_path, domain_name='d', experiment_id='e',
                           logger=LOG, _external_params={'PCORR': 0.9})
    assert GRRunner._forcing_corrections(stub) == {'PCORR': 0.9, 'TCORR': 0.0}
    # legacy: no section, identity
    stub = SimpleNamespace(config={}, project_dir=tmp_path, domain_name='d', experiment_id='e',
                           logger=LOG, _external_params=None)
    assert GRRunner._forcing_corrections(stub) == {'PCORR': 1.0, 'TCORR': 0.0}


# ------------------------------------------------------------------------ SUMMA

def _summa_settings(tmp_path):
    settings = tmp_path / 'settings' / 'SUMMA'
    forcing = tmp_path / 'forcing' / 'SUMMA_input'
    settings.mkdir(parents=True)
    forcing.mkdir(parents=True)
    with nc.Dataset(forcing / 'f1.nc', 'w') as ds:
        ds.createDimension('time', 3)
        ds.createDimension('hru', 1)
        v = ds.createVariable('pptrate', 'f8', ('time', 'hru'))
        v[:] = np.full((3, 1), 2.0)
    (settings / 'fileManager.txt').write_text(
        f"forcingPath          '{forcing}/'\nforcingListFile      'forcingFileList.txt'\n")
    (settings / 'forcingFileList.txt').write_text("'f1.nc'\n")
    return settings, forcing


def test_summa_forward_scales_copy_and_keeps_source(tmp_path):
    from symfluence.models.summa.forcing_correction import apply_forward
    settings, forcing = _summa_settings(tmp_path)
    config, info = _info('SUMMA', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.5)
    details = apply_forward(config, LOG, tmp_path, info)
    assert details['precip_mult'] == 1.5
    fm = (settings / 'fileManager.txt').read_text()
    assert 'forcing_precip_scaled' in fm
    with nc.Dataset(tmp_path / 'forcing_precip_scaled' / 'f1.nc') as ds:
        assert np.allclose(ds['pptrate'][:], 3.0)
    with nc.Dataset(forcing / 'f1.nc') as ds:
        assert np.allclose(ds['pptrate'][:], 2.0)


def test_summa_forward_identity_leaves_untouched_setup_alone(tmp_path):
    from symfluence.models.summa.forcing_correction import apply_forward
    settings, _ = _summa_settings(tmp_path)
    before = (settings / 'fileManager.txt').read_text()
    config, info = _info('SUMMA', FORCING_CORRECTION_MODE='off')
    apply_forward(config, LOG, tmp_path, info)
    assert (settings / 'fileManager.txt').read_text() == before
    assert not (tmp_path / 'forcing_precip_scaled').exists()


def test_summa_forward_snow_merges_into_trial_params(tmp_path, monkeypatch):
    from symfluence.models.summa import forcing_correction as summa_fc
    from symfluence.models.summa.calibration import parameter_files
    settings, _ = _summa_settings(tmp_path)
    with nc.Dataset(settings / 'trialParams.nc', 'w') as ds:
        ds.createDimension('hru', 2)
        ds.createVariable('hruId', 'i4', ('hru',))[:] = [1, 2]
        ds.createVariable('k_snow', 'f8', ('hru',))[:] = [0.1, 0.2]
    captured = {}

    def fake_write(path, attrs, params, logger, *a, **k):
        captured.update(params)
        return True
    monkeypatch.setattr(parameter_files, 'write_trial_params', fake_write)
    config, info = _info('SUMMA', FORCING_CORRECTION_SNOW_MULTIPLIER=1.3)
    summa_fc.apply_forward(config, LOG, tmp_path, info)
    assert set(captured) == {'k_snow', 'frozenPrecipMultip'}
    assert np.allclose(captured['k_snow'], [0.1, 0.2]) and captured['frozenPrecipMultip'] == 1.3


# ------------------------------------------------------------------------- MESH

def test_mesh_forward_writes_private_copy(tmp_path):
    from symfluence.models.mesh.forcing_correction import apply_forward
    run_dir = tmp_path / 'data' / 'forcing' / 'MESH_input'
    run_dir.mkdir(parents=True)
    shared = tmp_path / 'shared.nc'
    with nc.Dataset(shared, 'w') as ds:
        ds.createDimension('time', 2)
        ds.createVariable('PRE', 'f8', ('time',))[:] = [1.0, 2.0]
        ds.createVariable('TA', 'f8', ('time',))[:] = [270.0, 280.0]
    (run_dir / 'MESH_forcing_safe.nc').symlink_to(shared)
    config, info = _info('MESH', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.1, FORCING_CORRECTION_TEMP_OFFSET=-1.0)
    apply_forward(config, LOG, tmp_path, info)
    with nc.Dataset(run_dir / 'MESH_forcing_safe.nc') as ds:
        assert np.allclose(ds['PRE'][:], [1.1, 2.2]) and np.allclose(ds['TA'][:], [269.0, 279.0])
    with nc.Dataset(shared) as ds:
        assert np.allclose(ds['PRE'][:], [1.0, 2.0])
    # 'off' afterwards restores the unscaled forcing
    config, info = _info('MESH', FORCING_CORRECTION_MODE='off')
    apply_forward(config, LOG, tmp_path, info)
    with nc.Dataset(run_dir / 'MESH_forcing_safe.nc') as ds:
        assert np.allclose(ds['PRE'][:], [1.0, 2.0]) and np.allclose(ds['TA'][:], [270.0, 280.0])


# ------------------------------------------------------------------------ Wflow

def test_wflow_forward_sets_scale(tmp_path):
    from symfluence.models.wflow.forcing_correction import apply_forward
    settings = tmp_path / 'settings' / 'WFLOW'
    settings.mkdir(parents=True)
    toml = settings / 'wflow_sbm.toml'
    toml.write_text('[input.forcing]\natmosphere_water__precipitation_volume_flux = "precip"\n')
    config, info = _info('WFLOW', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.25,
                         WFLOW_CONFIG_FILE='wflow_sbm.toml')
    apply_forward(config, LOG, tmp_path, info)
    forcing = tomllib.loads(toml.read_text())['input']['forcing']
    assert forcing['atmosphere_water__precipitation_volume_flux']['scale'] == 1.25


# ------------------------------------------------------------------------- HYPE

def test_hype_forward_writes_par_txt(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError
    from symfluence.models.hype.forcing_correction import apply_forward
    settings = tmp_path / 'settings' / 'HYPE'
    settings.mkdir(parents=True)
    par = settings / 'par.txt'
    par.write_text('!! header\npcurain\t0.0\npcusnow\t0.0\t0.0\nrrcs1\t0.1\n')
    config, info = _info('HYPE', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.2,
                         FORCING_CORRECTION_SNOW_MULTIPLIER=1.5)
    apply_forward(config, LOG, tmp_path, info)
    lines = {ln.split('\t')[0]: ln.split('\t')[1:] for ln in par.read_text().splitlines()[1:]}
    assert float(lines['pcurain'][0]) == pytest.approx(0.2)
    assert [float(v) for v in lines['pcusnow']] == pytest.approx([0.8, 0.8])
    assert lines['rrcs1'] == ['0.1']
    par.write_text('rrcs1\t0.1\n')
    with pytest.raises(ModelExecutionError):
        apply_forward(config, LOG, tmp_path, info)
    config, info = _info('HYPE', FORCING_CORRECTION_MODE='off')
    assert 'identity' in apply_forward(config, LOG, tmp_path, info)['applied']


# -------------------------------------------------------------------------- mHM

def test_mhm_forward_reuses_worker_and_keeps_glacier_factor(tmp_path, monkeypatch):
    from symfluence.models.mhm.calibration.worker import MHMWorker
    from symfluence.models.mhm.forcing_correction import apply_forward
    adjusted = tmp_path / 'settings' / 'MHM' / 'forcing_adjusted'
    adjusted.mkdir(parents=True)
    (adjusted / 'forcing_adjustment.json').write_text(json.dumps(
        {'precipCorrectionFactor': 1.4, 'glacierIceDDF': 7.5, 'glacier_cells': 3}))
    seen = {}

    def fake_write(self, settings_dir, project_dir, params, config=None):
        seen.update(params)
        return True
    monkeypatch.setattr(MHMWorker, '_write_adjusted_precip', fake_write)
    config, info = _info('MHM', FORCING_CORRECTION_PRECIP_MULTIPLIER=1.1)
    apply_forward(config, LOG, tmp_path, info)
    assert seen == {'precipCorrectionFactor': 1.1, 'glacierIceDDF': 7.5}


def test_mhm_forward_identity_without_previous_adjustment_is_noop(tmp_path, monkeypatch):
    from symfluence.models.mhm.calibration.worker import MHMWorker
    from symfluence.models.mhm.forcing_correction import apply_forward
    monkeypatch.setattr(MHMWorker, '_write_adjusted_precip',
                        lambda *a, **k: pytest.fail('should not write'))
    config, info = _info('MHM', FORCING_CORRECTION_MODE='off')
    assert 'identity' in apply_forward(config, LOG, Path(tmp_path), info)['applied']
