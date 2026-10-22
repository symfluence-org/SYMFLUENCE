"""Routing polygons must retain the TauDEM IDs used by the channel network."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from symfluence.geospatial.discretization.core import DomainDiscretizationRunner


def test_delineated_catchments_preserve_noncontiguous_network_ids(tmp_path):
    basin_dir = tmp_path / 'shapefiles/river_basins'
    basin_dir.mkdir(parents=True)
    source = gpd.GeoDataFrame(
        {'GRU_ID': [65, 7], 'gru_to_seg': [65, 7]},
        geometry=[box(499000, 399000, 500000, 400000), box(500000, 399000, 502000, 400000)],
        crs=3057,
    )
    source.to_file(basin_dir / 'test_riverBasins_delineate.shp')
    runner = object.__new__(DomainDiscretizationRunner)
    runner.logger = logging.getLogger(__name__)
    runner.discretizer = SimpleNamespace(
        domain_name='test', project_dir=tmp_path,
        _get_catchment_subpath=lambda filename: 'shapefiles/catchment/lumped/test',
    )
    path = runner._discretize_delineated_domain('elevation')
    output = gpd.read_file(path)
    assert output.GRU_ID.tolist() == [65, 7]
    assert output.gru_to_seg.tolist() == [65, 7]
    np.testing.assert_allclose(output.avg_subbas.sum(), 1)
