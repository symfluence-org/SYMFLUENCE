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


# --- small multi-cell grids are area-weighted, not collapsed to one cell -----

def _small_grid_case(tmp_path, definition_method='lumped'):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    config = dict(DOMAIN_NAME='small', EXPERIMENT_ID='test', SYMFLUENCE_DATA_DIR=str(tmp_path),
                  SYMFLUENCE_CODE_DIR=str(tmp_path), HYDROLOGICAL_MODEL='SUMMA', FORCING_DATASET='ERA5',
                  EXPERIMENT_TIME_START='2023-01-01 00:00', EXPERIMENT_TIME_END='2023-01-02 00:00',
                  POUR_POINT_COORDS='64.05/-19.45', DOMAIN_DEFINITION_METHOD=definition_method,
                  CATCHMENT_SHP_HRUID='HRU_ID')
    ex = PointScaleForcingExtractor(
        config, tmp_path, SimpleNamespace(get_coordinate_names=lambda: ('latitude', 'longitude')),
        logging.getLogger(__name__))

    # 2 x 4 grid, 0.25 degree; cell value = 10 * row + column
    lat = [64.25, 64.0]
    lon = [-19.5, -19.25, -19.0, -18.75]
    values = np.array([[0., 1., 2., 3.], [10., 11., 12., 13.]])
    forcing = tmp_path / 'forcing.nc'
    xr.Dataset(
        {'air_temperature': (('time', 'latitude', 'longitude'), np.stack([values, values + 100.0]))},
        coords={'time': pd.date_range('2023-01-01', periods=2, freq='h'), 'latitude': lat, 'longitude': lon},
    ).to_netcdf(forcing, engine='h5netcdf')

    # HRU 1: exactly the cell (row 1, col 3); HRU 2: half in (1,0) and half in (1,1)
    catchment = tmp_path / 'catchment.shp'
    gpd.GeoDataFrame(
        {'HRU_ID': [1, 2], 'GRU_ID': [1, 1], 'elev_mean': [500.0, 600.0]},
        geometry=[box(-18.875, 63.875, -18.625, 64.125), box(-19.475, 63.95, -19.275, 64.05)],
        crs='EPSG:4326',
    ).to_file(catchment)

    # DEM: 100 m west of -19.0, 900 m east of it
    dem = tmp_path / 'dem.tif'
    elev = np.full((60, 120), 100.0, dtype='float32')
    elev[:, 65:] = 900.0
    with rasterio.open(dem, 'w', driver='GTiff', width=120, height=60, count=1, dtype='float32',
                       crs='EPSG:4326', transform=from_origin(-19.65, 64.4, 0.01, 0.01),
                       nodata=-9999.0) as dst:
        dst.write(elev, 1)
    return ex, forcing, catchment, dem


def test_small_grid_is_area_weighted_per_hru(tmp_path):
    ex, forcing, catchment, dem = _small_grid_case(tmp_path)
    assert ex.should_use_point_scale(tmp_path)        # still bypasses EASYMORE
    out_dir = tmp_path / 'out'
    ex.process([forcing], out_dir, catchment, lambda f: out_dir / f.name, dem_path=dem)

    with xr.open_dataset(out_dir / 'forcing.nc') as result:
        assert result.attrs['point_extraction_method'] == 'area_weighted_v1'
        assert result['hruId'].values.tolist() == [1, 2]
        temp = result['air_temperature'].values
    # Old behaviour: both HRUs got the cell nearest the pour point (value 10)
    np.testing.assert_allclose(temp[0], [13.0, 10.5], atol=0.02)
    np.testing.assert_allclose(temp[1], [113.0, 110.5], atol=0.02)

    csv = pd.read_csv(tmp_path / 'shapefiles/catchment_intersection/with_forcing/small_ERA5_intersected_shapefile.csv')
    np.testing.assert_allclose(csv.groupby('HRU_ID')['weight'].sum().values, 1.0)
    assert len(csv[csv['HRU_ID'] == 2]) == 2
    # forcing elevation is the cell-mean DEM: 900 m for the eastern cell
    assert csv[csv['HRU_ID'] == 1]['S_2_elev_m'].iloc[0] == pytest.approx(900.0, abs=1.0)
    assert csv[csv['HRU_ID'] == 2]['S_2_elev_m'].iloc[0] == pytest.approx(100.0, abs=1.0)


def test_point_domain_on_small_grid_keeps_nearest_cell(tmp_path):
    ex, forcing, catchment, dem = _small_grid_case(tmp_path, definition_method='point')
    out_dir = tmp_path / 'out'
    ex.process([forcing], out_dir, catchment, lambda f: out_dir / f.name, dem_path=dem)
    with xr.open_dataset(out_dir / 'forcing.nc') as result:
        assert result.attrs['point_extraction_method'] == 'nearest_coordinate_v1'
        np.testing.assert_allclose(result['air_temperature'].values[0], [10.0, 10.0])


def test_point_forcing_elevation_is_cell_mean_not_one_pixel(tmp_path):
    ex, forcing, catchment, dem = _small_grid_case(tmp_path, definition_method='point')
    import rasterio
    with rasterio.open(dem, 'r+') as dst:
        elev = dst.read(1)
        elev[:] = 100.0
        elev[:, :15] = 500.0          # western half of the selected cell (-19.625..-19.375)
        dst.write(elev, 1)
    # selected cell (row 1, col 0): the pixel at its centre (64.0, -19.5) is 500 m,
    # the cell mean is about 0.5 * 500 + 0.5 * 100
    assert ex._get_forcing_elevation_from_dem([forcing], dem) == pytest.approx(300.0, abs=15.0)


def test_hru_ids_beyond_int32_are_rejected(extractor):
    with pytest.raises(ValueError, match='32-bit'):
        extractor._hru_ids_int32([7120034520])
