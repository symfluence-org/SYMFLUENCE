"""Missing discharge must not truncate the GR simulation forcing clock."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.gr import preprocessor as gr

pytestmark = pytest.mark.unit


def test_forcing_continues_after_observations_end(tmp_path, monkeypatch):
    obs_dir = tmp_path / 'streamflow/preprocessed'
    obs_dir.mkdir(parents=True)
    dates = pd.date_range('2023-09-28', periods=5)
    pd.DataFrame({'datetime': dates[:3], 'discharge_cms': [10., 11., 12.]}).to_csv(
        obs_dir / 'test_streamflow_processed.csv', index=False)
    basin = tmp_path / 'test_riverBasins_lumped.shp'
    basin.touch()
    monkeypatch.setattr(gr.gpd, 'read_file', lambda _: pd.DataFrame({'GRU_area': [1e8]}))
    monkeypatch.setattr(gr, 'ForcingDataProcessor', lambda *args: SimpleNamespace(
        resample_to_frequency=lambda ds, **kwargs: ds))
    obj = SimpleNamespace(config={}, logger=logging.getLogger('test.gr.coverage'),
        domain_name='test', project_observations_dir=tmp_path, forcing_gr_path=tmp_path,
        _get_config_value=lambda *args, default=None, **kwargs: default,
        _get_default_path=lambda *args: tmp_path, _get_method_suffix=lambda: 'lumped',
        get_catchment_path=lambda: tmp_path / 'absent.shp',
        calculate_pet_oudin=lambda temp, lat: xr.zeros_like(temp))
    ds = xr.Dataset({'P': ('time', np.ones(5)), 'T': ('time', np.zeros(5))}, coords={'time': dates})
    result = pd.read_csv(gr.GRPreProcessor._prepare_lumped_forcing(obj, ds))
    assert len(result) == 5
    assert result.time.iloc[-1] == '2023-10-02'
    assert result.q_obs.iloc[:3].notna().all()
    assert result.q_obs.iloc[3:].isna().all()
    assert np.isfinite(result[['pr', 'temp', 'pet']]).all().all()


def test_fuse_forcing_continues_after_observations_end(monkeypatch):
    from symfluence.models.fuse import preprocessor as fuse
    monkeypatch.setattr(fuse.gpd, 'read_file', lambda _: None)
    obj = SimpleNamespace(catchment_path=None, logger=logging.getLogger('test.fuse.coverage'),
        calculate_catchment_centroid=lambda _: (-18., 64.))
    dates = pd.date_range('2023-09-28', periods=5)
    ds = xr.Dataset({'precip': ('time', np.ones(5)), 'temp': ('time', np.zeros(5))}, coords={'time': dates})
    obs = xr.Dataset({'q_obs': ('time', [1., 2., 3.])}, coords={'time': dates[:3]})
    out = fuse.FUSEPreProcessor._create_lumped_dataset(obj, ds, xr.zeros_like(ds.temp), obs, {'time_label': 'day'})
    assert len(out.time) == 5
    assert np.all(out.q_obs.values.squeeze()[3:] == -9999.)
    assert np.isfinite(out.precip).all()
