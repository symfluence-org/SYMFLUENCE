"""Glacier mode builds inputs from explicit geometry or fails; it never invents them.

The removed raster/shapefile route assumed a 50 m ice thickness, set the bed to
the DEM surface, and on failure left ``attributes_glac.nc`` with glacier grids
next to a plain copy of ``coldState.nc``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.summa.preprocessor import SummaPreProcessor

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _preprocessor(tmp_path: Path, **summa) -> SummaPreProcessor:
    settings = dict(glacier_mode=False, filemanager='fileManager.txt',
                    glacier_attributes='attributes_glac.nc', glacier_coldstate='coldState_glac.nc',
                    glacier_outline_path=None, glacier_thickness_dir=None, glacier_grid_resolution=100.0)
    settings.update(summa)
    pre = SummaPreProcessor.__new__(SummaPreProcessor)
    pre._config = SimpleNamespace(model=SimpleNamespace(summa=SimpleNamespace(**settings)))
    pre.logger = logging.getLogger('test_summa_glacier')
    pre.project_dir = tmp_path
    pre.domain_name = 'toy'
    pre.setup_dir = tmp_path / 'settings' / 'SUMMA'
    pre.setup_dir.mkdir(parents=True)
    pre.attribute_name, pre.coldstate_name = 'attributes.nc', 'coldState.nc'
    pre.catchment_path, pre.catchment_name = tmp_path / 'shapefiles' / 'catchment', 'toy_HRUs.shp'
    pre.dem_path = tmp_path / 'dem.tif'
    pre.hruId = 'HRU_ID'
    return pre


def _plain_files(setup_dir: Path) -> None:
    xr.Dataset({'hruId': ('hru', [1]), 'gruId': ('gru', [1])}).to_netcdf(setup_dir / 'attributes.nc')
    xr.Dataset({'mLayerDepth': (('midToto', 'hru'), np.ones((3, 1)))}).to_netcdf(setup_dir / 'coldState.nc')


def _legacy_rasters(pre: SummaPreProcessor) -> None:
    glacier_dir = pre.project_attributes_dir / 'glaciers'
    glacier_dir.mkdir(parents=True)
    for name in ('domain_type.tif', 'hru_id.tif', 'rgi_id.tif'):
        (glacier_dir / f'domain_toy_{name}').write_bytes(b'')


def test_explicit_geometry_is_passed_through_unchanged(tmp_path):
    pre = _preprocessor(tmp_path, glacier_mode=True, glacier_outline_path=str(tmp_path / 'ice.gpkg'),
                        glacier_thickness_dir=str(tmp_path / 'thickness'), glacier_grid_resolution=50.0)

    with patch('symfluence.models.summa.glacier_geometry.prepare_glacier_geometry') as prepare:
        pre.run_glacier_preprocessing()

    prepare.assert_called_once_with(
        pre.setup_dir / 'attributes.nc', pre.setup_dir / 'coldState.nc',
        tmp_path / 'shapefiles' / 'catchment' / 'toy_HRUs.shp', tmp_path / 'dem.tif',
        tmp_path / 'ice.gpkg', tmp_path / 'thickness', 50.0, 'HRU_ID')


def test_only_one_explicit_input_is_rejected(tmp_path):
    pre = _preprocessor(tmp_path, glacier_mode=True, glacier_outline_path=str(tmp_path / 'ice.gpkg'))
    with pytest.raises(ConfigurationError, match='both outlines and thickness'):
        pre.run_glacier_preprocessing()


@pytest.mark.parametrize('summa', [dict(glacier_mode=True), dict(filemanager='fileManager_glac.txt')])
def test_glacier_mode_without_explicit_inputs_raises(tmp_path, summa):
    pre = _preprocessor(tmp_path, **summa)
    _plain_files(pre.setup_dir)
    _legacy_rasters(pre)   # the removed route would have run on these

    assert pre._is_glacier_mode_enabled()
    with pytest.raises(ConfigurationError, match='SUMMA_GLACIER_OUTLINE_PATH.*SUMMA_GLACIER_THICKNESS_DIR'):
        pre.run_glacier_preprocessing()

    # nothing is copied under the glacier names
    assert not (pre.setup_dir / 'attributes_glac.nc').exists()
    assert not (pre.setup_dir / 'coldState_glac.nc').exists()


def test_one_provided_glacier_file_is_not_completed_from_the_base_file(tmp_path):
    pre = _preprocessor(tmp_path, glacier_mode=True)
    _plain_files(pre.setup_dir)
    xr.Dataset({'nGlac': ('gru', [1])}).to_netcdf(pre.setup_dir / 'attributes_glac.nc')

    with pytest.raises(ConfigurationError, match='missing: coldState_glac.nc'):
        pre.run_glacier_preprocessing()
    assert not (pre.setup_dir / 'coldState_glac.nc').exists()


def test_glacier_attributes_with_ordinary_cold_state_is_rejected(tmp_path):
    """The pair the removed route left behind when it failed half-way."""
    pre = _preprocessor(tmp_path, glacier_mode=True)
    xr.Dataset({'nGlac': ('gru', [1]), 'glacierMask': (('ygrid', 'xgrid', 'grid', 'gru'), np.ones((2, 2, 1, 1)))}
               ).to_netcdf(pre.setup_dir / 'attributes_glac.nc')
    xr.Dataset({'mLayerDepth': (('midToto', 'hru'), np.ones((3, 1)))}).to_netcdf(pre.setup_dir / 'coldState_glac.nc')

    with pytest.raises(ConfigurationError, match="no 'dom' dimension"):
        pre.run_glacier_preprocessing()


def test_complete_provided_glacier_files_are_used_as_they_are(tmp_path):
    pre = _preprocessor(tmp_path, glacier_mode=True)
    xr.Dataset({'nGlac': ('gru', [1])}).to_netcdf(pre.setup_dir / 'attributes_glac.nc')
    xr.Dataset({'DOMarea': (('scalarv', 'hru', 'dom'), np.ones((1, 1, 2)))}).to_netcdf(pre.setup_dir / 'coldState_glac.nc')
    before = {p.name: p.read_bytes() for p in pre.setup_dir.iterdir()}

    pre.run_glacier_preprocessing()

    assert {p.name: p.read_bytes() for p in pre.setup_dir.iterdir()} == before


def test_detected_glacier_data_alone_does_not_fabricate_glacier_files(tmp_path, caplog):
    """Glacier rasters in the project (e.g. acquired for another model) are not a request."""
    pre = _preprocessor(tmp_path)
    _plain_files(pre.setup_dir)
    _legacy_rasters(pre)
    before = sorted(p.name for p in pre.setup_dir.iterdir())

    assert pre._is_glacier_mode_enabled()
    with caplog.at_level(logging.WARNING, logger='test_summa_glacier'):
        pre.run_glacier_preprocessing()

    assert sorted(p.name for p in pre.setup_dir.iterdir()) == before
    assert any('running SUMMA without glaciers' in record.getMessage() for record in caplog.records)


def test_no_glacier_mode_without_request_or_data(tmp_path):
    assert not _preprocessor(tmp_path)._is_glacier_mode_enabled()
