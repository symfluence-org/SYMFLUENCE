"""Routing topology: grid branch selection, cycles, HRU-to-segment references."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, box

from symfluence.core.exceptions import ValidationError
from symfluence.core.modeling.utilities.base_topology_generator import BaseTopologyGenerator
from symfluence.models.mizuroute.topology_generator import MizuRouteTopologyGenerator
from symfluence.models.troute.topology_generator import TRouteTopologyGenerator


def _get_value(accessor, default=None, **kwargs):
    try:
        value = accessor()
    except AttributeError:
        return default
    return default if value is None else value


def _pp(tmp_path, method, suffix, **extra):
    paths = SimpleNamespace(
        river_network_path='default', river_network_name='default',
        river_basins_path='default', river_basins_name='default',
    )
    values = dict(
        config=SimpleNamespace(
            paths=paths,
            domain=SimpleNamespace(delineation=SimpleNamespace(routing='lumped'), grid_cell_size=1000.0),
            model=SimpleNamespace(troute=SimpleNamespace(), mizuroute=SimpleNamespace()),
        ),
        domain_definition_method=method, domain_name='test', project_dir=tmp_path,
        experiment_id='run_1', logger=logging.getLogger(__name__), _get_config_value=_get_value,
        _get_method_suffix=lambda: suffix, mizu_topology_file='topology.nc',
        setup_dir=tmp_path, mizu_make_outlet=None,
        river_segid_col='LINKNO', river_downsegid_col='DSLINKNO',
        river_slope_col='Slope', river_length_col='Length',
        basin_gruid_col='GRU_ID', basin_hru_to_seg_col='gru_to_seg', basin_area_col='GRU_area',
        calculate_feature_centroids=lambda gdf: gdf.geometry.centroid,
    )
    values.update(extra)
    return SimpleNamespace(**values)


def _write_grid(tmp_path, suffix, downstream):
    """Three grid cells; *downstream* is the D8 column the delineator writes."""
    basins = tmp_path / 'shapefiles' / 'river_basins'
    network = tmp_path / 'shapefiles' / 'river_network'
    basins.mkdir(parents=True)
    network.mkdir(parents=True)
    cells = [box(i, 0, i + 1, 1) for i in range(3)]
    gpd.GeoDataFrame(
        {'GRU_ID': [1, 2, 3], 'downstream': downstream, 'gru_to_seg': [1, 2, 3],
         'GRU_area': [1e6, 1e6, 1e6], 'elev_mean': [30.0, 10.0, 20.0], 'slope': [.01, .01, .01]},
        geometry=cells, crs=32627,
    ).to_file(basins / f'test_riverBasins_{suffix}.shp')
    gpd.GeoDataFrame(
        {'LINKNO': [1, 2, 3], 'DSLINKNO': downstream, 'Length': [1000.] * 3,
         'Slope': [.01] * 3, 'GRU_ID': [1, 2, 3]},
        geometry=[c.centroid for c in cells], crs=32627,
    ).to_file(network / f'test_riverNetwork_{suffix}.shp')


def test_normalised_distributed_method_takes_the_grid_branch_and_breaks_cycles(tmp_path):
    """'distributed' (what the typed config yields) with the delineator's file name.

    The D8 graph has a cycle 1 -> 2 -> 3 -> 1; the grid branch breaks it at the
    lowest cell (2). Before, 'distributed' never matched the 'distribute' test,
    the vector path ran, and the cycle went to mizuRoute untouched.
    """
    _write_grid(tmp_path, 'distributed_1000m', downstream=[2, 3, 1])
    pp = _pp(tmp_path, 'distributed', 'distributed_1000m')

    MizuRouteTopologyGenerator(pp).create_network_topology_file()

    with xr.open_dataset(tmp_path / 'topology.nc') as ds:
        down = dict(zip(ds.segId.values.tolist(), ds.downSegId.values.tolist()))
    assert down[1] == 2 and down[3] == 1
    assert down[2] not in (1, 2, 3)  # cell 2 became the outlet
    assert pp.summa_uses_gru_runoff is True


def test_base_generator_detects_grid_for_both_spellings(tmp_path):
    _write_grid(tmp_path, 'distributed_1000m', downstream=[2, 3, 1])
    for method in ('distribute', 'distributed'):
        generator = TRouteTopologyGenerator(_pp(tmp_path, method, 'distributed_1000m'))
        assert generator.detect_domain_type() == 'grid'
        topology = generator.build_topology()
        assert topology.domain_type == 'grid'
        assert not BaseTopologyGenerator.find_routing_cycles(topology.seg_ids, topology.down_seg_ids)


def test_vector_network_with_a_cycle_is_refused(tmp_path):
    basins = tmp_path / 'shapefiles' / 'river_basins'
    network = tmp_path / 'shapefiles' / 'river_network'
    basins.mkdir(parents=True)
    network.mkdir(parents=True)
    gpd.GeoDataFrame(
        {'LINKNO': [1, 2, 3], 'DSLINKNO': [2, 3, 1], 'Length': [100.] * 3, 'Slope': [.01] * 3},
        geometry=[LineString([(i, 0), (i + 1, 0)]) for i in range(3)], crs=32627,
    ).to_file(network / 'test_riverNetwork_semidistributed.shp')
    gpd.GeoDataFrame(
        {'GRU_ID': [1, 2, 3], 'gru_to_seg': [1, 2, 3], 'GRU_area': [1e6] * 3},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)], crs=32627,
    ).to_file(basins / 'test_riverBasins_semidistributed.shp')

    pp = _pp(tmp_path, 'semidistributed', 'semidistributed')
    with pytest.raises(ValidationError, match='routing cycle'):
        MizuRouteTopologyGenerator(pp).create_network_topology_file()
    with pytest.raises(ValidationError, match='routing cycle'):
        TRouteTopologyGenerator(pp).build_topology()


def _write_vector(tmp_path, hru_to_seg):
    basins = tmp_path / 'shapefiles' / 'river_basins'
    network = tmp_path / 'shapefiles' / 'river_network'
    basins.mkdir(parents=True)
    network.mkdir(parents=True)
    gpd.GeoDataFrame(
        {'LINKNO': [10, 20], 'DSLINKNO': [20, 0], 'Length': [100.] * 2, 'Slope': [.01] * 2},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])], crs=32627,
    ).to_file(network / 'test_riverNetwork_semidistributed.shp')
    gpd.GeoDataFrame(
        {'GRU_ID': [1, 2], 'gru_to_seg': hru_to_seg, 'GRU_area': [1e6] * 2},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=32627,
    ).to_file(basins / 'test_riverBasins_semidistributed.shp')
    (tmp_path / 'shapefiles' / 'pour_point').mkdir()
    gpd.GeoDataFrame({'id': [1]}, geometry=[Point(2, 0)], crs=32627).to_file(
        tmp_path / 'shapefiles' / 'pour_point' / 'pp.shp')


def test_hru_draining_to_an_unknown_segment_raises_instead_of_being_rewired(tmp_path):
    """Segment 19 does not exist; it used to be rewired to 20 ('closest id')."""
    _write_vector(tmp_path, hru_to_seg=[10, 19])
    pp = _pp(tmp_path, 'semidistributed', 'semidistributed')
    with pytest.raises(ValidationError, match='HRU 2 -> segment 19'):
        MizuRouteTopologyGenerator(pp).create_network_topology_file()
    with pytest.raises(ValidationError, match='HRU 2 -> segment 19'):
        TRouteTopologyGenerator(pp).build_topology()


def test_valid_vector_network_still_builds(tmp_path):
    _write_vector(tmp_path, hru_to_seg=[10, 20])
    pp = _pp(tmp_path, 'semidistributed', 'semidistributed')
    MizuRouteTopologyGenerator(pp).create_network_topology_file()
    with xr.open_dataset(tmp_path / 'topology.nc') as ds:
        np.testing.assert_array_equal(ds.hruToSegId.values, [10, 20])
    topology = TRouteTopologyGenerator(pp).build_topology()
    np.testing.assert_array_equal(topology.hru_to_seg_ids, [10, 20])


def test_find_routing_cycles():
    find = BaseTopologyGenerator.find_routing_cycles
    assert find(np.array([1, 2, 3]), np.array([2, 3, 0])) == []
    assert find(np.array([1, 2, 3]), np.array([2, 3, 1])) == [[1, 2, 3]]
    assert find(np.array([4, 1, 2]), np.array([1, 2, 1])) == [[1, 2]]
    assert find(np.array([5]), np.array([5])) == [[5]]
    # an outlet in a zero-based numbering is not a self-loop
    assert find(np.array([0, 1]), np.array([0, 0])) == []


def test_remap_fallback_defines_gru_ids(tmp_path):
    """The shapefile fallback used to hit NameError: gru_ids."""
    from symfluence.models.mizuroute.remap_generator import MizuRouteRemapGenerator

    catchment = tmp_path / 'shapefiles' / 'catchment'
    catchment.mkdir(parents=True)
    gpd.GeoDataFrame(
        {'GRU_ID': [11, 12], 'avg_subbas': [0.25, 0.75]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=32627,
    ).to_file(catchment / 'test_catchment_delineated.shp')
    xr.Dataset({'hruId': ('hru', [11, 12])}).to_netcdf(tmp_path / 'topology.nc')
    pp = _pp(tmp_path, 'lumped', 'lumped', mizu_remap_file='remap.nc')

    MizuRouteRemapGenerator(pp).create_area_weighted_remap_file()

    with xr.open_dataset(tmp_path / 'remap.nc') as ds:
        np.testing.assert_array_equal(ds.RN_hruId.values, [11, 12])
        np.testing.assert_allclose(ds.weight.values, [0.25, 0.75])
