# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""KiWIS share-channel streamflow handler (no network: the client is patched)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.core.registries import R
from symfluence.data.observation.handlers import kiwis as handler_mod
from symfluence.data.observation.handlers.kiwis import KiWISStreamflowHandler

pytestmark = [pytest.mark.unit, pytest.mark.data]

CHANNEL = 'https://example.invalid/KiWebPortal/rest/shares/channels/abc'


def _config(tmp_path, **extra):
    cfg = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path), 'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain', 'EXPERIMENT_ID': 'e',
        'EXPERIMENT_TIME_START': '2026-01-01 00:00', 'EXPERIMENT_TIME_END': '2026-01-02 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped', 'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'ERA5', 'HYDROLOGICAL_MODEL': 'SUMMA', 'FORCING_TIME_STEP_SIZE': 86400,
        'STREAMFLOW_DATA_PROVIDER': 'KIWIS',
    }
    cfg.update(extra)
    return cfg


def _fake_fetch(calls):
    def fetch(channel, ts_id, start, end, **kwargs):
        calls.append((channel, ts_id, start, end))
        t = pd.date_range(start, end, freq='5min', inclusive='left')     # [start, end)
        return pd.DataFrame({'value': 5.0, 'quality': 200.0}, index=pd.DatetimeIndex(t, name='time'))
    return fetch


def test_registered_for_provider_and_additional_observations():
    assert R.observation_handlers.get('kiwis') is KiWISStreamflowHandler
    assert R.observation_handlers.get('kiwis_streamflow') is KiWISStreamflowHandler


def test_channel_from_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(handler_mod.kiwis, 'fetch', _fake_fetch(calls))
    monkeypatch.setenv('KIWIS_CHANNEL_URL', CHANNEL)
    monkeypatch.setenv('KIWIS_TS_ID', 'TS1')
    h = KiWISStreamflowHandler(_config(tmp_path), logging.getLogger('kiwis'))

    raw = h.acquire()

    assert calls[0][:2] == (CHANNEL, 'TS1')
    assert 'TS1' not in raw.name
    out = pd.read_csv(h.process(raw), index_col='datetime', parse_dates=True)
    assert list(out.columns) == ['discharge_cms']
    assert list(out.index) == [pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02')]
    np.testing.assert_allclose(out['discharge_cms'], 5.0)


def test_missing_channel_raises(tmp_path, monkeypatch):
    monkeypatch.delenv('KIWIS_CHANNEL_URL', raising=False)
    monkeypatch.delenv('KIWIS_TS_ID', raising=False)
    h = KiWISStreamflowHandler(_config(tmp_path), logging.getLogger('kiwis'))
    with pytest.raises(DataAcquisitionError):
        h.acquire()


def test_update_refetches_only_the_revision_window(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(handler_mod.kiwis, 'fetch', _fake_fetch(calls))
    cfg = _config(tmp_path, KIWIS_CHANNEL_URL=CHANNEL, KIWIS_TS_ID='TS1', KIWIS_REVISE_DAYS=0.5)
    h = KiWISStreamflowHandler(cfg, logging.getLogger('kiwis'))
    raw = h.acquire()
    first = pd.read_csv(raw, index_col=0, parse_dates=True)

    calls.clear()
    h.acquire()

    assert calls[0][2] == first.index.max() - pd.Timedelta(hours=12)
    n_first = len(first)
    assert len(pd.read_csv(raw)) == n_first          # replaced, not duplicated
