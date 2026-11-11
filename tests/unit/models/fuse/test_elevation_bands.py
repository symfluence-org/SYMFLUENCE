# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FUSE elevation bands come from real hypsometry, per spatial unit."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.fuse.elevation_band_manager import (
    FuseElevationBandManager,
    equal_area_bands,
    read_band_parameters,
    write_unit_bands,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]

X0, Y0, PIXEL, N = 500000.0, 7100000.0, 100.0, 40
CRS = 'EPSG:32627'


def _half(i: int):
    half = N * PIXEL / 2
    return box(X0 + i * half, Y0 - N * PIXEL, X0 + (i + 1) * half, Y0)


def _manager(tmp_path: Path, catchment: gpd.GeoDataFrame, *, mode='lumped', n_bands=4,
             dem=True, subcatchment_ids=None) -> FuseElevationBandManager:
    """Manager over a toy project: DEM rising 10 m per column (0..390 m)."""
    project = tmp_path / 'domain_toy'
    forcing = project / 'forcing' / 'FUSE_input'
    forcing.mkdir(parents=True)
    shp = project / 'shapefiles' / 'catchment' / 'toy_HRUs.shp'
    shp.parent.mkdir(parents=True)
    catchment.to_file(shp)

    manager = FuseElevationBandManager(
        config={'DOMAIN_NAME': 'toy', 'EXPERIMENT_ID': 'run_1', 'FUSE_SPATIAL_MODE': mode,
                'FUSE_N_ELEVATION_BANDS': n_bands},
        logger=logging.getLogger('test_fuse_bands'),
        project_dir=project,
        forcing_fuse_path=forcing,
        catchment_path=shp,
        domain_name='toy',
        calculate_catchment_centroid_callback=lambda gdf: (-19.0, 64.0),
    )
    if dem:
        dem_path = manager._dem_path()
        dem_path.parent.mkdir(parents=True)
        values = (np.ones((N, 1)) * np.arange(N)[None, :] * 10.0).astype('float32')
        with rasterio.open(dem_path, 'w', driver='GTiff', height=N, width=N, count=1,
                           dtype='float32', crs=CRS, nodata=-9999.0,
                           transform=from_origin(X0, Y0, PIXEL, PIXEL)) as dst:
            dst.write(values, 1)
    if subcatchment_ids is not None:
        times = pd.date_range('2000-01-01', periods=2)
        xr.Dataset(
            {'pr': (('time', 'latitude', 'longitude'), np.zeros((2, 1, len(subcatchment_ids))))},
            coords={'time': times, 'latitude': [64.0],
                    'longitude': np.asarray(subcatchment_ids, dtype=float)},
        ).to_netcdf(forcing / 'toy_input.nc')
    return manager


def _whole(**columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {k: [v] for k, v in columns.items()} or {'GRU_ID': [1]},
        geometry=[_half(0).union(_half(1))], crs=CRS,
    )


# --------------------------------------------------------------------------
# equal_area_bands
# --------------------------------------------------------------------------

def test_equal_area_bands_are_band_means_not_range_endpoints():
    z = np.arange(0.0, 400.0, 10.0)                      # uniform 0..390 m
    elev, frac = equal_area_bands(z, None, 4)
    np.testing.assert_allclose(elev, [45.0, 145.0, 245.0, 345.0])
    np.testing.assert_allclose(frac, 0.25)
    assert elev[0] > z.min() and elev[-1] < z.max()
    assert float(np.sum(elev * frac)) == pytest.approx(z.mean())


def test_equal_area_bands_follow_a_skewed_hypsometry():
    """Mostly-low terrain: three of four bands sit low, not spread evenly over the range."""
    z = np.concatenate([np.full(90, 100.0), np.full(10, 1100.0)])
    elev, frac = equal_area_bands(z, None, 4)
    np.testing.assert_allclose(elev, [100.0, 100.0, 100.0, 500.0])
    assert float(np.sum(elev * frac)) == pytest.approx(z.mean())


def test_equal_area_bands_weight_by_area():
    elev, frac = equal_area_bands([200.0, 1000.0], [3.0, 1.0], 2)
    np.testing.assert_allclose(elev, [200.0, 600.0])     # upper half: 1/4 at 200 m, 1/4 at 1000 m
    assert float(np.sum(elev * frac)) == pytest.approx(400.0)


# --------------------------------------------------------------------------
# lumped
# --------------------------------------------------------------------------

def test_lumped_bands_from_dem(tmp_path):
    manager = _manager(tmp_path, _whole(elev_mean=9999.0, elev_range=5.0), n_bands=4)
    with xr.open_dataset(manager.create_elevation_bands()) as ds:
        assert ds['mean_elev'].dims == ('elevation_band', 'latitude', 'longitude')
        np.testing.assert_allclose(ds['mean_elev'].values.ravel(), [45.0, 145.0, 245.0, 345.0])
        np.testing.assert_allclose(ds['area_frac'].values.ravel(), 0.25)
        np.testing.assert_allclose(ds['prec_frac'].values.ravel(), 0.25)
    bands = read_band_parameters(manager.forcing_fuse_path / 'toy_elev_bands.nc')
    assert bands['z_forcing'] == pytest.approx(195.0)    # true mean elevation of the DEM


def test_lumped_single_band_sits_at_mean_elevation(tmp_path):
    manager = _manager(tmp_path, _whole(), n_bands=1)
    bands = read_band_parameters(manager.create_elevation_bands())
    assert bands['n_bands'] == 1
    np.testing.assert_allclose(bands['mean_elev'], [195.0])
    np.testing.assert_allclose(bands['area_frac'], [1.0])


def test_mean_elevation_only_gives_one_band_and_no_invented_range(tmp_path, caplog):
    """A low-relief basin at 100 m must not get bands from -150 to 350 m."""
    manager = _manager(tmp_path, _whole(elev_mean=100.0), n_bands=5, dem=False)
    with caplog.at_level(logging.WARNING, logger='test_fuse_bands'):
        bands = read_band_parameters(manager.create_elevation_bands())
    assert bands['n_bands'] == 1
    np.testing.assert_allclose(bands['mean_elev'], [100.0])
    np.testing.assert_allclose(bands['area_frac'], [1.0])
    assert any('single elevation band' in r.getMessage() for r in caplog.records)


def test_no_elevation_at_all_raises(tmp_path):
    manager = _manager(tmp_path, _whole(), dem=False)
    with pytest.raises(ModelExecutionError, match='No elevation information'):
        manager.create_elevation_bands()
    assert not (manager.forcing_fuse_path / 'toy_elev_bands.nc').exists()


def test_lumped_bands_from_hru_elevation_table_without_dem(tmp_path):
    hrus = gpd.GeoDataFrame(
        {'GRU_ID': [1, 1], 'elev_mean': [200.0, 1000.0], 'HRU_area': [3.0e6, 1.0e6]},
        geometry=[_half(0), _half(1)], crs=CRS,
    )
    manager = _manager(tmp_path, hrus, n_bands=2, dem=False)
    bands = read_band_parameters(manager.create_elevation_bands())
    np.testing.assert_allclose(bands['mean_elev'], [200.0, 600.0])
    assert bands['z_forcing'] == pytest.approx(400.0)


# --------------------------------------------------------------------------
# distributed
# --------------------------------------------------------------------------

def test_distributed_bands_are_per_subcatchment(tmp_path):
    """Each unit gets its own hypsometry; they are not one linspace broadcast to all."""
    # Shapefile order differs from the forcing order: matching is by GRU_ID.
    hrus = gpd.GeoDataFrame({'GRU_ID': [20, 10]}, geometry=[_half(1), _half(0)], crs=CRS)
    manager = _manager(tmp_path, hrus, mode='distributed', n_bands=2, subcatchment_ids=[10, 20])
    path = manager.create_elevation_bands()
    with xr.open_dataset(path) as ds:
        assert ds['mean_elev'].shape == (2, 1, 2)
        west = ds['mean_elev'].sel(longitude=10.0).values.ravel()
        east = ds['mean_elev'].sel(longitude=20.0).values.ravel()
        np.testing.assert_allclose(west, [45.0, 145.0])      # columns 0..190 m
        np.testing.assert_allclose(east, [245.0, 345.0])     # columns 200..390 m
        np.testing.assert_allclose(ds['area_frac'].sum('elevation_band').values, 1.0)

    # Scalar FUSE parameters come from one unit, not from all units flattened.
    bands = read_band_parameters(path)
    np.testing.assert_allclose(bands['mean_elev'], west)
    assert bands['z_forcing'] == pytest.approx(95.0)

    # A per-subcatchment run gets that subcatchment's own bands.
    target = write_unit_bands(path, tmp_path / 'subcat_20.nc', 'longitude', 20)
    with xr.open_dataset(target) as unit:
        assert unit['mean_elev'].shape == (2, 1, 1)
        np.testing.assert_allclose(unit['mean_elev'].values.ravel(), east)


def test_distributed_mean_only_units_use_their_own_mean(tmp_path):
    hrus = gpd.GeoDataFrame(
        {'GRU_ID': [10, 20], 'elev_mean': [300.0, 900.0]}, geometry=[_half(0), _half(1)], crs=CRS
    )
    manager = _manager(tmp_path, hrus, mode='distributed', n_bands=3, dem=False,
                       subcatchment_ids=[10, 20])
    with xr.open_dataset(manager.create_elevation_bands()) as ds:
        assert ds.sizes['elevation_band'] == 1
        np.testing.assert_allclose(ds['mean_elev'].values.ravel(), [300.0, 900.0])
        np.testing.assert_allclose(ds['area_frac'].values.ravel(), [1.0, 1.0])


def test_distributed_unmatched_subcatchment_raises(tmp_path):
    hrus = gpd.GeoDataFrame({'GRU_ID': [10, 20]}, geometry=[_half(0), _half(1)], crs=CRS)
    manager = _manager(tmp_path, hrus, mode='distributed', subcatchment_ids=[10, 30])
    with pytest.raises(ModelExecutionError, match='30'):
        manager.create_elevation_bands()
