"""Explicit network paths must be honoured for multiple-HRU SUMMA GRUs."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString, box

from symfluence.core.config.models.paths import PathsConfig
from symfluence.models.mizuroute.topology_generator import MizuRouteTopologyGenerator


def test_explicit_network_preserves_gru_routing_and_areas(tmp_path):
    network = tmp_path / 'custom_network'
    network.mkdir()
    gpd.GeoDataFrame(
        {'LINKNO': [7, 65], 'DSLINKNO': [65, -9999], 'Slope': [.01, .02], 'Length': [100., 200.]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])], crs=32627,
    ).to_file(network / 'channels.gpkg')
    gpd.GeoDataFrame({'GRU_ID': [7, 65]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs=32627).to_file(network / 'basins.gpkg')
    settings = tmp_path / 'settings/SUMMA'
    settings.mkdir(parents=True)
    xr.Dataset({'gruId': ('gru', [7, 65]), 'hru2gruId': ('hru', [7, 7, 65]),
                'HRUarea': ('hru', [10., 20., 40.])}).to_netcdf(settings / 'attributes.nc')
    paths = PathsConfig(river_network_path=str(network), river_network_name='channels.gpkg',
                        river_basins_path=str(network), river_basins_name='basins.gpkg')

    def get_value(accessor, default=None, **kwargs):
        try:
            return accessor()
        except AttributeError:
            return default

    pp = SimpleNamespace(
        config=SimpleNamespace(paths=paths, domain=SimpleNamespace(delineation=SimpleNamespace(routing='river_network'))),
        domain_definition_method='semidistributed', domain_name='test', project_dir=tmp_path,
        logger=logging.getLogger(__name__), _get_config_value=get_value,
        _get_method_suffix=lambda: 'semidistributed', mizu_topology_file='topology.nc',
        setup_dir=tmp_path, mizu_make_outlet=None, river_segid_col='LINKNO',
        river_downsegid_col='DSLINKNO', river_slope_col='Slope', river_length_col='Length',
    )
    MizuRouteTopologyGenerator(pp).create_network_topology_file()
    with xr.open_dataset(tmp_path / 'topology.nc') as ds:
        np.testing.assert_array_equal(ds.hruId, [7, 65])
        np.testing.assert_array_equal(ds.hruToSegId, [7, 65])
        np.testing.assert_allclose(ds.area, [30., 40.])
        np.testing.assert_array_equal(ds.downSegId, [65, -9999])
    assert pp.summa_uses_gru_runoff
