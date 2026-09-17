"""Snow-pixel fractions must exclude clouds and off-footprint pixels."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from symfluence.data.observation.handlers.modis_snow import MODISSnowHandler

pytestmark = [pytest.mark.unit, pytest.mark.data]


def handler(tmp_path, **extra):
    config = dict(SYMFLUENCE_DATA_DIR=str(tmp_path), SYMFLUENCE_CODE_DIR=str(tmp_path),
                  DOMAIN_NAME='test', EXPERIMENT_ID='snow',
                  EXPERIMENT_TIME_START='2023-01-01 00:00', EXPERIMENT_TIME_END='2023-01-03 00:00',
                  HYDROLOGICAL_MODEL='SUMMA', FORCING_DATASET='ERA5',
                  MODIS_SCA_OBSERVATION_MODE='snow_fraction', MODIS_MIN_PIXELS=1,
                  MODIS_SCA_MIN_VALID_RATIO=0.5, **extra)
    return MODISSnowHandler(config, logging.getLogger(__name__))


def dataset():
    return xr.Dataset({
        'NDSI_Snow_Cover': (('time', 'lat', 'lon'), [[[10, 0, 100, 100]], [[50, 250, 100, 100]], [[80, 50, 100, 100]]]),
        'NDSI_Snow_Cover_Basic_QA': (('time', 'lat', 'lon'), [[[0, 1, 0, 0]], [[0, 0, 0, 0]], [[2, 3, 0, 0]]]),
    }, coords={'time': pd.date_range('2023-01-01', periods=3), 'lat': [30.0], 'lon': [70., 71., 72., 73.]})


def test_mask_qa_and_clouds(tmp_path, monkeypatch):
    h = handler(tmp_path, MODIS_SCA_USE_CATCHMENT_MASK=True)
    shp = tmp_path / 'basin.shp'
    gpd.GeoDataFrame(geometry=[box(69.5, 29.5, 71.5, 30.5)], crs=4326).to_file(shp)
    monkeypatch.setattr(h, '_resolve_catchment_shapefile', lambda: shp)
    ds = dataset()
    result = h._extract_snow_fraction(ds.NDSI_Snow_Cover, ds)
    # Two footprint pixels, not four bounding-box pixels. 10 means snow,
    # not 10% cover. Exactly 50% valid coverage is retained.
    np.testing.assert_allclose(result.sca, [0.5, 1.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.valid_ratio, [1., 0.5, 0.])
    assert result.valid_pixels.tolist() == [2, 1, 0]
    path = tmp_path / 'modis.nc'
    ds.to_netcdf(path)
    saved = pd.read_csv(h._process_netcdf(path))
    np.testing.assert_allclose(saved.sca, result.sca, equal_nan=True)


def test_requires_qa(tmp_path):
    h = handler(tmp_path)
    ds = dataset().drop_vars('NDSI_Snow_Cover_Basic_QA')
    with pytest.raises(ValueError, match='Basic_QA'):
        h._extract_snow_fraction(ds.NDSI_Snow_Cover, ds)


def test_missing_requested_mask_is_error(tmp_path, monkeypatch):
    h = handler(tmp_path, MODIS_SCA_USE_CATCHMENT_MASK=True)
    monkeypatch.setattr(h, '_resolve_catchment_shapefile', lambda: None)
    ds = dataset()
    with pytest.raises(ValueError, match='catchment shapefile'):
        h._extract_snow_fraction(ds.NDSI_Snow_Cover, ds)


def test_cosine_latitude_weights_and_threshold(tmp_path):
    h = handler(tmp_path, MODIS_SCA_NDSI_THRESHOLD=10)
    ds = xr.Dataset({
        'NDSI_Snow_Cover': (('time', 'lat', 'lon'), [[[11], [10]]]),
        'NDSI_Snow_Cover_Basic_QA': (('time', 'lat', 'lon'), [[[0], [1]]]),
    }, coords={'time': [pd.Timestamp('2023-01-01')], 'lat': [0., 60.], 'lon': [70.]})
    result = h._extract_snow_fraction(ds.NDSI_Snow_Cover, ds)
    assert result.sca.iloc[0] == pytest.approx(2 / 3)
