# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GR distributed mode: forcing corrections, per-HRU hypsometry, glacier guard.

These tests work on the generated airGR script and its inputs; they do not
start R.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.mask
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.exceptions import ConfigurationError, ModelExecutionError
from symfluence.core.mixins.config import SCHEMA_DEFAULT, fallback_default
from symfluence.models.gr.calibration.parameter_manager import GRParameterManager
from symfluence.models.gr.glacier import require_lumped_for_glacier
from symfluence.models.gr.hypsometry import catchment_hypsometry, flat_hypsometry
from symfluence.models.gr.runner import GRRunner

pytestmark = [pytest.mark.unit, pytest.mark.quick]

X0, Y0, PIXEL, N = 500000.0, 7100000.0, 100.0, 40
CRS = 'EPSG:32627'


def _write_dem(path: Path) -> np.ndarray:
    """40x40 DEM rising 10 m per column: west half 0-190 m, east half 200-390 m."""
    dem = (np.ones((N, 1)) * np.arange(N)[None, :] * 10.0).astype('float32')
    dem[0, 0] = -9999.0
    with rasterio.open(path, 'w', driver='GTiff', height=N, width=N, count=1, dtype='float32',
                       crs=CRS, transform=from_origin(X0, Y0, PIXEL, PIXEL), nodata=-9999.0) as dst:
        dst.write(dem, 1)
    return dem


def _half(i: int):
    half = N * PIXEL / 2
    return box(X0 + i * half, Y0 - N * PIXEL, X0 + (i + 1) * half, Y0)


def _runner(tmp_path: Path, catchment: gpd.GeoDataFrame) -> GRRunner:
    runner = object.__new__(GRRunner)
    runner.project_dir = tmp_path
    dem_dir = runner.project_attributes_dir / 'elevation' / 'dem'
    dem_dir.mkdir(parents=True)
    _write_dem(dem_dir / 'domain_toy_elv.tif')
    shp_dir = tmp_path / 'shp'
    shp_dir.mkdir()
    catchment.to_file(shp_dir / 'hrus.shp')
    runner.logger = logging.getLogger('test_gr_distributed')
    runner.domain_name = 'toy'
    runner.catchment_path = shp_dir
    runner.catchment_name = 'hrus.shp'
    runner._get_config_value = lambda getter, default=SCHEMA_DEFAULT, dict_key=None, **kw: fallback_default(default, dict_key)
    return runner


def test_catchment_hypsometry_matches_direct_dem_statistics(tmp_path):
    """The shared helper reproduces the masking the lumped runner used inline."""
    dem_path = tmp_path / 'dem.tif'
    _write_dem(dem_path)
    catchment = gpd.GeoDataFrame({'GRU_ID': [1]}, geometry=[_half(0).union(_half(1))], crs=CRS)

    with rasterio.open(dem_path) as src:
        image, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, nodata=src.nodata)
        reference = image[0][image[0] != src.nodata]
    hypso, z_mean, elevations = catchment_hypsometry(dem_path, catchment)

    assert elevations.size == N * N - 1
    np.testing.assert_array_equal(hypso, np.percentile(reference, np.arange(0, 101, 1)))
    assert z_mean == np.mean(reference)
    assert str(z_mean) == str(np.mean(reference))   # same text in the R script


def test_catchment_without_dem_pixels_raises(tmp_path):
    dem_path = tmp_path / 'dem.tif'
    _write_dem(dem_path)
    elsewhere = gpd.GeoDataFrame({'GRU_ID': [1]}, geometry=[box(0, 0, 10, 10)], crs=CRS)
    with pytest.raises(ModelExecutionError, match='No valid DEM pixels'):
        catchment_hypsometry(dem_path, elsewhere)


def test_distributed_script_applies_forcing_corrections_and_layers(tmp_path):
    hypso, z_mean = flat_hypsometry(640.0)
    script = GRRunner._distributed_r_script(
        tmp_path / 'hru_3_temp.csv', 3, hypso, z_mean, 7,
        {'PCORR': 1.3, 'TCORR': -1.5}, 'c(X1=1, X2=0, X3=1, X4=1)',
        ('2000-01-01', '2000-12-31'), ('2000-01-01', '2002-12-31'),
    )
    assert re.search(r'Precip = BasinObs\$pr \* 1\.3,', script)
    assert re.search(r'TempMean = BasinObs\$temp \+ -1\.5,', script)
    assert re.search(r'ZInputs = 640\.0,\s+NLayers = 7\s', script)
    assert 'HypsoData = c(' + ', '.join(['640.0'] * 101) + ')' in script
    assert 'GR date mismatch for HRU 3' in script


def test_identity_corrections_leave_forcing_unchanged(tmp_path):
    hypso, z_mean = flat_hypsometry(100.0)
    script = GRRunner._distributed_r_script(
        tmp_path / 'x.csv', 1, hypso, z_mean, 5, {'PCORR': 1.0, 'TCORR': 0.0}, 'c(X1=1)',
        ('2000-01-01', '2000-12-31'), ('2000-01-01', '2002-12-31'),
    )
    assert 'Precip = BasinObs$pr * 1.0,' in script
    assert 'TempMean = BasinObs$temp + 0.0,' in script


def test_each_hru_gets_its_own_hypsometry(tmp_path):
    catchment = gpd.GeoDataFrame({'GRU_ID': [11, 22]}, geometry=[_half(0), _half(1)], crs=CRS)
    runner = _runner(tmp_path, catchment)
    (west_hypso, west_z), (east_hypso, east_z) = runner._distributed_hypsometries([11, 22], True)
    assert west_hypso[0] == 0.0 and west_hypso[-1] == 190.0
    assert east_hypso[0] == 200.0 and east_hypso[-1] == 390.0
    assert float(east_z) == pytest.approx(295.0)
    assert west_hypso[-1] < east_hypso[0]     # no HRU spans the whole catchment's range


def test_hypsometry_follows_hru_id_not_row_order(tmp_path):
    catchment = gpd.GeoDataFrame({'GRU_ID': [22, 11]}, geometry=[_half(1), _half(0)], crs=CRS)
    runner = _runner(tmp_path, catchment)
    (first_hypso, _), (second_hypso, _) = runner._distributed_hypsometries([11, 22], True)
    assert first_hypso[-1] == 190.0 and second_hypso[0] == 200.0


def test_hru_without_dem_pixels_uses_its_mean_elevation_flat(tmp_path):
    outside = box(X0 + 10 * N * PIXEL, Y0 - N * PIXEL, X0 + 11 * N * PIXEL, Y0)
    catchment = gpd.GeoDataFrame(
        {'GRU_ID': [1, 2], 'elev_mean': [95.0, 1234.0]}, geometry=[_half(0), outside], crs=CRS
    )
    runner = _runner(tmp_path, catchment)
    _, (hypso, z_mean) = runner._distributed_hypsometries([1, 2], True)
    assert z_mean == 1234.0
    np.testing.assert_array_equal(hypso, np.full(101, 1234.0))


def test_hru_without_any_elevation_raises(tmp_path):
    outside = box(X0 + 10 * N * PIXEL, Y0 - N * PIXEL, X0 + 11 * N * PIXEL, Y0)
    catchment = gpd.GeoDataFrame({'GRU_ID': [1, 2]}, geometry=[_half(0), outside], crs=CRS)
    runner = _runner(tmp_path, catchment)
    with pytest.raises(ModelExecutionError, match='GR HRU 2'):
        runner._distributed_hypsometries([1, 2], True)


def test_hru_count_mismatch_raises(tmp_path):
    catchment = gpd.GeoDataFrame({'GRU_ID': [1, 2]}, geometry=[_half(0), _half(1)], crs=CRS)
    runner = _runner(tmp_path, catchment)
    with pytest.raises(ModelExecutionError, match='3 HRUs'):
        runner._distributed_hypsometries([1, 2, 3], True)


def test_glacier_guard():
    require_lumped_for_glacier(True, True, ['X1', 'ICE_DDF'])        # lumped: fine
    require_lumped_for_glacier(False, False, ['X1', 'PCORR'])        # distributed, no glacier
    with pytest.raises(ConfigurationError, match='ICE_DDF'):
        require_lumped_for_glacier(False, False, ['X1', 'ICE_DDF'])
    with pytest.raises(ConfigurationError, match='GR_GLACIER_ENABLED'):
        require_lumped_for_glacier(False, True, ['X1'])


def _manager_config(**overrides):
    config = {
        'DOMAIN_NAME': 'toy', 'EXPERIMENT_ID': 'run_1',
        'GR_PARAMS_TO_CALIBRATE': 'X1,X2,PCORR,TCORR,ICE_DDF',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
    }
    config.update(overrides)
    return config


def test_parameter_manager_rejects_glacier_parameter_in_distributed_mode(tmp_path):
    log = logging.getLogger('test_gr_distributed')
    with pytest.raises(ConfigurationError, match='lumped-only'):
        GRParameterManager(_manager_config(DOMAIN_DEFINITION_METHOD='delineate'), log, tmp_path)
    with pytest.raises(ConfigurationError, match='lumped-only'):
        GRParameterManager(_manager_config(GR_SPATIAL_MODE='distributed'), log, tmp_path)
    # Lumped (auto or explicit, even on a delineated domain) keeps ICE_DDF.
    assert 'ICE_DDF' in GRParameterManager(_manager_config(), log, tmp_path).gr_params
    lumped = _manager_config(DOMAIN_DEFINITION_METHOD='delineate', GR_SPATIAL_MODE='lumped')
    assert 'ICE_DDF' in GRParameterManager(lumped, log, tmp_path).gr_params
    # Distributed without glacier parameters is unaffected.
    plain = _manager_config(DOMAIN_DEFINITION_METHOD='delineate', GR_PARAMS_TO_CALIBRATE='X1,PCORR,TCORR')
    assert GRParameterManager(plain, log, tmp_path).gr_params == ['X1', 'PCORR', 'TCORR']


def test_runner_rejects_glacier_parameters_in_distributed_run():
    runner = object.__new__(GRRunner)
    runner.logger = logging.getLogger('test_gr_distributed')
    runner.spatial_mode = 'distributed'
    runner._get_config_value = lambda getter, default=SCHEMA_DEFAULT, dict_key=None, **kw: fallback_default(default, dict_key)
    runner.config = SimpleNamespace()
    with pytest.raises(ConfigurationError, match='ICE_DDF'):
        runner.run_gr(params={'X1': 300.0, 'ICE_DDF': 5.0})
