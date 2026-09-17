"""The point forcing must follow geography, not storage order."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.preprocessing.resampling.point_scale_extractor import PointScaleForcingExtractor


@pytest.fixture
def extractor(tmp_path):
    config = dict(DOMAIN_NAME='point', EXPERIMENT_ID='test', SYMFLUENCE_DATA_DIR=str(tmp_path),
                  SYMFLUENCE_CODE_DIR=str(tmp_path), HYDROLOGICAL_MODEL='SUMMA', FORCING_DATASET='ERA5',
                  EXPERIMENT_TIME_START='2023-01-01 00:00', EXPERIMENT_TIME_END='2023-01-02 00:00',
                  POUR_POINT_COORDS='32.37/77.25')
    return PointScaleForcingExtractor(config, tmp_path,
        SimpleNamespace(get_coordinate_names=lambda: ('latitude', 'longitude')),
        logging.getLogger(__name__))


def test_nearest_cell_independent_of_axis_order(extractor):
    ds = xr.Dataset({'air_temperature': (('latitude', 'longitude'), [[260, 261, 262], [270, 271, 272]])},
                    coords={'latitude': [32.5, 32.25], 'longitude': [77., 77.25, 77.5]})
    for candidate in [ds, ds.isel(latitude=slice(None, None, -1), longitude=slice(None, None, -1))]:
        result = candidate.isel(extractor._point_indexers(candidate))
        assert result.air_temperature.item() == 271
        assert result.latitude.item() == 32.25
        assert result.longitude.item() == 77.25


def test_curvilinear_grid(extractor):
    ds = xr.Dataset(coords={'latitude': (('y', 'x'), [[31, 31], [32.4, 33]]),
                            'longitude': (('y', 'x'), [[76, 77], [77.2, 78]])})
    assert extractor._point_indexers(ds) == {'y': 1, 'x': 0}


def test_selected_cell_provenance_is_written(extractor, tmp_path, monkeypatch):
    ds = xr.Dataset({'air_temperature': (('time', 'latitude', 'longitude'), np.arange(6).reshape(1, 2, 3))},
                    coords={'time': [pd.Timestamp('2023-01-01')], 'latitude': [32.5, 32.25], 'longitude': [77., 77.25, 77.5]})
    src, dst = tmp_path / 'source.nc', tmp_path / 'point.nc'
    ds.to_netcdf(src, engine='h5netcdf')
    monkeypatch.setattr(extractor, '_standardize_if_raw', lambda d, f: d)
    extractor._process_single_file(src, dst, tmp_path / 'absent.csv')
    with xr.open_dataset(dst) as result:
        assert result.air_temperature.item() == 4
        assert result.attrs['point_forcing_latitude'] == 32.25
        assert result.attrs['point_forcing_longitude'] == 77.25
        assert result.attrs['point_target_latitude'] == 32.37


def test_elevation_uses_selected_cell_not_grid_midpoint(extractor, tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    src = tmp_path / 'forcing.nc'
    xr.Dataset(coords={'latitude': [32.5, 32.25], 'longitude': [77., 77.25, 77.5]}).to_netcdf(src, engine='h5netcdf')
    dem = tmp_path / 'local_dem.tif'
    with rasterio.open(dem, 'w', driver='GTiff', width=2, height=2, count=1,
                       dtype='float32', crs='EPSG:4326',
                       transform=from_origin(77.24, 32.38, 0.01, 0.01)) as dst:
        dst.write(np.full((1, 2, 2), 3950, dtype='float32'))
    # The grid midpoint is inside this small DEM, but the selected cell is not.
    # Sampling the midpoint would fabricate an elevation for a different cell.
    assert extractor._get_forcing_elevation_from_dem([src], dem) is None
