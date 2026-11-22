"""MizuRouteRemapGenerator id columns and the lumped SUMMA GRU id."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.mizuroute.remap_generator import MizuRouteRemapGenerator


def _pp(tmp_path, **overrides):
    kw = dict(
        POUR_POINT_COORDS='51.17/-115.57',
        EXPERIMENT_TIME_START='2020-01-01 00:00',
        EXPERIMENT_TIME_END='2020-12-31 23:00',
        DOMAIN_DEFINITION_METHOD='delineate',
        SETTINGS_MIZU_NEEDS_REMAP=True,
    )
    kw.update(overrides)
    cfg = SymfluenceConfig.from_minimal('t', 'SUMMA', **kw)

    def get(acc, default=None, dict_key=None):
        if default is None:
            return resolve_config_value(cfg, acc, dict_key=dict_key)
        return resolve_config_value(cfg, acc, default, dict_key)

    return SimpleNamespace(
        config=cfg, logger=logging.getLogger('test_remap'), _get_config_value=get,
        project_dir=tmp_path, setup_dir=tmp_path, mizu_remap_file='remap.nc',
        mizu_topology_file='topology.nc',
        domain_definition_method='lumped', experiment_id='exp', domain_name='t',
    )


def test_routing_basin_column_defaults_to_gru_id(tmp_path):
    gen = MizuRouteRemapGenerator(_pp(tmp_path))
    assert gen._routing_basin_id_field() == 'GRU_ID'

    df = pd.DataFrame({'S_1_GRU_ID': [1, 1, 2], 'S_2_GRU_ID': [10, 11, 11],
                       'AP1N': [0.4, 0.6, 1.0]})
    gen._process_remap_variables(df)
    assert gen.nc_rnhruid.ravel().tolist() == [1, 2]
    assert gen.nc_noverlaps.ravel().tolist() == [2, 1]
    assert gen.nc_hmgruid == [10, 11, 11]
    assert gen.nc_weight == [0.4, 0.6, 1.0]


def _make_area_weighted(tmp_path, pp):
    xr.Dataset({'hruId': ('hru', np.array([101, 102], dtype='i4'))}).to_netcdf(
        tmp_path / 'topology.nc')
    pp.subcatchment_weights = np.array([0.25, 0.75])
    pp.subcatchment_gru_ids = np.array([101, 102])
    MizuRouteRemapGenerator(pp).create_area_weighted_remap_file()
    with nc4.Dataset(tmp_path / 'remap.nc') as ds:
        return ds['HM_hruId'][:].tolist(), ds['RN_hruId'][:].tolist()


def test_lumped_hm_id_read_from_summa_attributes(tmp_path):
    settings = tmp_path / 'settings' / 'SUMMA'
    settings.mkdir(parents=True)
    xr.Dataset({'gruId': ('gru', np.array([7], dtype='i4')),
                'hruId': ('hru', np.array([7], dtype='i4'))}).to_netcdf(settings / 'attributes.nc')

    hm, rn = _make_area_weighted(tmp_path, _pp(tmp_path))
    assert hm == [7, 7]
    assert rn == [101, 102]


def test_lumped_hm_id_falls_back_to_1_without_attributes(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger='test_remap'):
        hm, _ = _make_area_weighted(tmp_path, _pp(tmp_path))
    assert hm == [1, 1]
    assert 'assuming lumped GRU id 1' in caplog.text
