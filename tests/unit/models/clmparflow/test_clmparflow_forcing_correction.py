# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for ParFlow-CLM (lumped forcing copies, boundary flux, basin mode)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symfluence.models.clmparflow import forcing_correction as pfc
from symfluence.models.parflow.calibration.parameter_manager import _read_pfidb, _write_pfidb
from symfluence.models.parflow.lumped_forcing import SnowBands, save_forcing_cache

LOG = logging.getLogger('test_clmparflow_fc')


def _project(tmp_path: Path):
    forcing = tmp_path / 'domain_d' / 'data' / 'forcing' / 'CLMPARFLOW_input'
    forcing.mkdir(parents=True)
    times = pd.date_range('2001-01-01', periods=24 * 20, freq='h')
    temp = 2.0 + 5.0 * np.sin(np.arange(len(times)) / 24.0)
    save_forcing_cache(forcing, np.full(len(times), 0.5), np.full(len(times), 0.05), temp, times.values,
                       SnowBands((500.0,), (1.0,), 500.0, 'test'), 64.0)
    settings = tmp_path / 'domain_d' / 'settings' / 'CLMPARFLOW'
    settings.mkdir(parents=True)
    entries = {'Solver.LSM': 'CLM', 'Geom.domain.Perm.Value': '5'}
    for day in pd.date_range('2001-01-01', periods=20, freq='D'):
        entries[f"Patch.z_upper.BCPressure.d{day:%Y%m%d}.Value"] = '-0.0001'
    _write_pfidb(settings / 'run.pfidb', entries)
    return settings, forcing / 'hourly_forcing_cache.npz'


def test_forcing_1d_columns(tmp_path):
    met = tmp_path / 'forcing.1d'
    np.savetxt(met, np.tile([200., 300., 2.0, 273.15, 2., .5, 8e4, .004], (3, 1)), fmt='%.6e')
    pfc.correct_forcing_1d(met, {'precip': 1.25, 'temp': -2.0})
    data = np.loadtxt(met)
    assert np.allclose(data[:, 2], 2.5) and np.allclose(data[:, 3], 271.15) and np.allclose(data[:, 0], 200.)
    before = met.read_text()
    pfc.correct_forcing_1d(met, {})
    assert met.read_text() == before


def test_overland_bc_scaled_and_restored(tmp_path):
    settings, cache = _project(tmp_path)
    entries = _read_pfidb(settings / 'run.pfidb')
    original = dict(entries)
    assert not pfc.correct_overland_bc(entries, settings, cache, {})  # identity, nothing recorded
    assert pfc.correct_overland_bc(entries, settings, cache, {'precip': 1.3})
    key = 'Patch.z_upper.BCPressure.d20010110.Value'
    wetter = float(entries[key])
    pfc.correct_overland_bc(entries, settings, cache, {'precip': 1.0}, {'SNOW17_SCF': 1.0})
    assert float(entries[key]) > wetter  # less negative flux = less infiltration input
    pfc.correct_overland_bc(entries, settings, cache, {})
    assert entries == original


def test_worker_strips_fc_params_and_corrects_bc(tmp_path):
    from symfluence.models.clmparflow.calibration.worker import CLMParFlowWorker

    settings, _ = _project(tmp_path)
    cfg = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'DOMAIN_NAME': 'd', 'CLMPARFLOW_PARAMS_TO_CALIBRATE': 'K_SAT'}
    worker = CLMParFlowWorker(config=cfg, logger=LOG)
    assert worker.apply_parameters({'FC_PRECIP_MULT': 1.2, 'FC_TEMP_OFFSET': 0.5}, settings, config=cfg)
    entries = _read_pfidb(settings / 'run.pfidb')
    assert not any(k.startswith('FC_') or 'FC_' in v for k, v in entries.items())
    assert entries['Patch.z_upper.BCPressure.d20010110.Value'] != '-0.0001'
    assert (settings / pfc.BC_RECORD).exists()
    # identity restores the preprocessor's values
    assert worker.apply_parameters({'FC_PRECIP_MULT': 1.0}, settings, config=cfg)
    assert _read_pfidb(settings / 'run.pfidb')['Patch.z_upper.BCPressure.d20010110.Value'] == '-0.0001'


def test_runner_corrects_staged_met_file_only(tmp_path):
    from symfluence.models.clmparflow.runner import CLMParFlowRunner

    settings, cache = _project(tmp_path)
    met = cache.parent / 'forcing.1d'
    np.savetxt(met, np.tile([200., 300., 2.0, 273.15, 2., .5, 8e4, .004], (2, 1)), fmt='%.6e')
    runner = CLMParFlowRunner.__new__(CLMParFlowRunner)
    runner.settings_dir = settings
    runner.project_dir = tmp_path / 'domain_d'
    runner.forcing_correction_values = {'precip': 2.0}
    sim = tmp_path / 'sim'
    sim.mkdir()
    for _ in range(2):  # restaging never compounds
        runner._setup_sim_directory(sim)
    assert np.allclose(np.loadtxt(sim / 'forcing.1d')[:, 2], 4.0)
    assert np.allclose(np.loadtxt(met)[:, 2], 2.0)


def test_basin_parameters_temp_offset(tmp_path):
    import xarray as xr

    from symfluence.models.clmparflow.basin import pfio
    from symfluence.models.clmparflow.basin.case import BasinParameters, write_forcing
    from symfluence.models.clmparflow.basin.grid import BasinGrid

    assert 'temp_offset' not in BasinParameters().record()
    ny, nx = 2, 2
    mask = np.ones((ny, nx), bool)
    unit = np.zeros((ny, nx), int)
    zeros = np.zeros((ny, nx))
    g = BasinGrid(mask=mask, z=zeros, zc=zeros, sx=zeros, sy=zeros, unit=unit, ice=np.zeros((ny, nx), bool),
                  icefrac=zeros, x0=0., y0=0., dx=1000., crs='EPSG:3057', units=['land'], info={})
    times = pd.date_range('2013-10-01', periods=24, freq='h')
    names = {'DSWR': 'sw', 'DLWR': 'lw', 'APCP': 'pr', 'Temp': 'ta', 'UGRD': 'ws', 'Press': 'ps', 'SPFH': 'q'}
    xr.Dataset({v: (('hru', 'time'), np.full((1, 24), 270.0)) for v in names.values()},
               coords={'time': times, 'hru': [0]}).assign(hruId=('hru', ['land'])).to_netcdf(tmp_path / 'f.nc')
    write_forcing(tmp_path / 'out', g, BasinParameters(precip_scale=1.1, temp_offset=-1.5), tmp_path / 'f.nc',
                  '2013-10-01', '2013-10-02', unit_variable='hruId', names=names)
    assert pfio.read_pfb(tmp_path / 'out' / 'CARRA.Temp.000001_to_000024.pfb')[0, 0, 0] == pytest.approx(268.5)
    assert pfio.read_pfb(tmp_path / 'out' / 'CARRA.APCP.000001_to_000024.pfb')[0, 0, 0] == pytest.approx(297.0)


def test_forward_applier_lumped(tmp_path, monkeypatch):
    from symfluence.core.calibration import forcing_correction as fc
    from symfluence.models.clmparflow.runner import CLMParFlowRunner

    monkeypatch.setitem(fc.ADAPTERS, 'CLMPARFLOW', fc.ForcingCorrectionAdapter(
        'CLMPARFLOW', precip='FC_PRECIP_MULT', temp='FC_TEMP_OFFSET'))
    settings, _ = _project(tmp_path)
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.2,
           'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'e', 'SYMFLUENCE_DATA_DIR': str(tmp_path)}
    info = fc.apply_forward_forcing_correction('CLMPARFLOW', cfg, LOG, tmp_path / 'domain_d',
                                               runner_class=CLMParFlowRunner)
    assert info['details']['mode'] == 'lumped'
    assert _read_pfidb(settings / 'run.pfidb')['Patch.z_upper.BCPressure.d20010110.Value'] != '-0.0001'
    cfg['FORCING_CORRECTION_MODE'] = 'off'
    fc.apply_forward_forcing_correction('CLMPARFLOW', cfg, LOG, tmp_path / 'domain_d',
                                        runner_class=CLMParFlowRunner)
    assert _read_pfidb(settings / 'run.pfidb')['Patch.z_upper.BCPressure.d20010110.Value'] == '-0.0001'
