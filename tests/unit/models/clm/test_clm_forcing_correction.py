# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION for CLM: corrected copies of the DATM forcing."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from symfluence.core.calibration import forcing_correction as fc
from symfluence.models.clm import forcing_correction as clm_fc

LOG = logging.getLogger('test_clm_fc')


def _setup(tmp_path: Path):
    forcing = tmp_path / 'data' / 'forcing' / 'CLM_input'
    forcing.mkdir(parents=True)
    files = []
    for year in (2001, 2002):
        path = forcing / f'clmforc.{year}.nc'
        with nc.Dataset(path, 'w') as ds:
            ds.createDimension('time', None)
            ds.createDimension('lat', 1)
            for name, value in (('PRECTmms', 1e-4), ('TBOT', 270.0), ('QBOT', 0.003)):
                var = ds.createVariable(name, 'f4', ('time', 'lat'))
                var.units = 'x'
                var[:] = np.full((4, 1), value)
        files.append(path)
    settings = tmp_path / 'settings' / 'CLM'
    settings.mkdir(parents=True)
    entries = '\n'.join(f'      <file>{f}</file>' for f in files)
    (settings / 'datm.streams.xml').write_text(
        f'<file>\n<stream_info name="P">\n<datafiles>\n{entries}\n</datafiles>\n</stream_info>\n'
        f'<stream_info name="T">\n<datafiles>\n{entries}\n</datafiles>\n</stream_info>\n'
        f'<stream_info name="topo">\n<datafiles>\n      <file>{forcing}/topo_forcing.nc</file>\n'
        '</datafiles>\n</stream_info>\n</file>\n')
    return settings, files


def _streamed(settings: Path):
    return clm_fc._FILE_RE.findall((settings / 'datm.streams.xml').read_text())


def test_split_params():
    rest, values = clm_fc.split_params({'FC_PRECIP_MULT': np.array([1.2]), 'FC_TEMP_OFFSET': -1.0, 'FMAX': .4})
    assert rest == {'FMAX': .4} and values == {'precip': 1.2, 'temp': -1.0}


def test_correction_writes_copies_and_never_compounds(tmp_path):
    settings, files = _setup(tmp_path)
    clm_fc.apply_stream_correction(settings, {'precip': 1.2, 'temp': -1.0})
    streamed = _streamed(settings)
    assert len(streamed) == 4 and all('forcing_corrected' in f for f in streamed)
    assert 'topo_forcing.nc' in (settings / 'datm.streams.xml').read_text()
    with nc.Dataset(streamed[0]) as ds:
        assert np.allclose(ds['PRECTmms'][:], 1.2e-4) and np.allclose(ds['TBOT'][:], 269.0)
        assert np.allclose(ds['QBOT'][:], 0.003) and ds['PRECTmms'].units == 'x'
    # a second trial derives from the recorded source, not the corrected copy
    clm_fc.apply_stream_correction(settings, {'precip': 0.9})
    with nc.Dataset(_streamed(settings)[0]) as ds:
        assert np.allclose(ds['PRECTmms'][:], 0.9e-4) and np.allclose(ds['TBOT'][:], 270.0)
    # shared forcing untouched
    with nc.Dataset(files[0]) as ds:
        assert np.allclose(ds['PRECTmms'][:], 1e-4)
    # identity restores the original file list and drops the record
    clm_fc.apply_stream_correction(settings, {})
    assert _streamed(settings) == [str(f) for f in files] * 2
    assert not clm_fc.needs_reset(settings)


def test_forward_run_found_next_to_runner(tmp_path, monkeypatch):
    from symfluence.models.clm.runner import CLMRunner

    monkeypatch.setitem(fc.ADAPTERS, 'CLM', fc.ForcingCorrectionAdapter(
        'CLM', precip='FC_PRECIP_MULT', temp='FC_TEMP_OFFSET'))
    settings, _ = _setup(tmp_path)
    cfg = {'FORCING_CORRECTION_MODE': 'fixed', 'FORCING_CORRECTION_PRECIP_MULTIPLIER': 1.1,
           'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'run1'}
    info = fc.apply_forward_forcing_correction('CLM', cfg, LOG, tmp_path, runner_class=CLMRunner)
    assert info['native'] == {'FC_PRECIP_MULT': 1.1}
    with nc.Dataset(_streamed(settings)[0]) as ds:
        assert np.allclose(ds['PRECTmms'][:], 1.1e-4)
    record = json.loads((tmp_path / 'simulations' / 'run1' / 'CLM' / 'forcing_correction.json').read_text())
    assert record['details']['precip'] == 1.1
    # 'off' resets
    cfg['FORCING_CORRECTION_MODE'] = 'off'
    fc.apply_forward_forcing_correction('CLM', cfg, LOG, tmp_path, runner_class=CLMRunner)
    assert all('forcing_corrected' not in f for f in _streamed(settings))


def test_bad_multiplier_raises(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError
    settings, _ = _setup(tmp_path)
    with pytest.raises(ModelExecutionError):
        clm_fc.apply_stream_correction(settings, {'precip': 0.0})
