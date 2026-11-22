"""CARRA/CERRA forcing-grid subsetting must find the catchment shapefile.

The handlers used to look in ``shapefile_path.parent / 'catchment'`` for a
hard-coded ``{domain}_HRUs_GRUs.shp``. ``shapefile_path`` is
``shapefiles/forcing`` (or ``shapefiles/forcing/<signature>``) and the catchment
lives under ``shapefiles/catchment/<method>/<experiment_id>/``, so the bbox
filter was always skipped and the whole reanalysis domain was processed.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.data.preprocessing.dataset_handlers.carra_utils import CARRAHandler
from symfluence.data.preprocessing.dataset_handlers.cerra_utils import CERRAHandler

CONFIG = {
    'DOMAIN_NAME': 'Iceland_test',
    'DOMAIN_DEFINITION_METHOD': 'semidistributed',
    'EXPERIMENT_ID': 'run_1',
    'SUB_GRID_DISCRETIZATION': 'elevation',
}


@pytest.mark.parametrize('handler_cls', [CARRAHandler, CERRAHandler])
def test_catchment_shapefile_found_in_nested_layout(tmp_path, handler_cls):
    nested = tmp_path / 'shapefiles' / 'catchment' / 'semidistributed' / 'run_1'
    nested.mkdir(parents=True)
    shp = nested / 'Iceland_test_HRUs_elevation.shp'
    shp.write_bytes(b'')
    handler = handler_cls(dict(CONFIG), logging.getLogger('test'), tmp_path)
    assert handler._find_catchment_shapefile() == shp


def test_catchment_shapefile_honours_explicit_catchment_path(tmp_path):
    custom = tmp_path / 'elsewhere'
    custom.mkdir()
    shp = custom / 'my_hrus.shp'
    shp.write_bytes(b'')
    config = dict(CONFIG, CATCHMENT_PATH=str(custom), CATCHMENT_SHP_NAME='my_hrus.shp')
    handler = CERRAHandler(config, logging.getLogger('test'), tmp_path)
    assert handler._find_catchment_shapefile() == shp


def test_catchment_shapefile_missing_returns_none(tmp_path):
    handler = CARRAHandler(dict(CONFIG), logging.getLogger('test'), tmp_path)
    assert handler._find_catchment_shapefile() is None
