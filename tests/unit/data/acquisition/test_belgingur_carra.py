from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.acquisition.belgingur_carra import VARIABLES, normalize_belgingur


def sample():
    d=xr.Dataset({'Times':('Time',np.array(['2017-07-01_00:00:00','2017-07-01_03:00:00'],dtype='S19')),
                  'XLAT':(('south_north','west_east'),np.array([[64.,64.],[64.02,64.02]])),
                  'XLONG':(('south_north','west_east'),np.array([[-19.,-18.98],[-19.,-18.98]]))})
    values=[10.,1000.,3.6,75.,4.,100.,300.]
    for (name,_,unit,*_),v in zip(VARIABLES,values):
        d[name]=(('Time','south_north','west_east'),np.full((2,2,2),v));d[name].attrs['units']=unit
    d.attrs.update(MAP_PROJ4_STR='+proj=lcc +lat_0=72 +lon_0=-36 +lat_1=72 +lat_2=72 +R=6367470',DX=2500,DY=2500)
    return d


def convert(d):
    return normalize_belgingur(d,dict(lat_min=63.9,lat_max=64.1,lon_min=-19.1,lon_max=-18.9),'2017-07-01','2017-07-01 03:00','test')


def test_conversions_preserve_rates_grid_and_clock():
    d=convert(sample())
    assert d.air_temperature.values[0,0,0]==pytest.approx(283.15)
    assert d.surface_air_pressure.values[0,0,0]==pytest.approx(100000)
    assert d.precipitation_flux.values[0,0,0]==pytest.approx(.001)
    assert d.surface_downwelling_shortwave_flux.values[0,0,0]==100
    assert .005 < float(d.specific_humidity[0,0,0]) < .006
    assert d.latitude.ndim==2
    assert pd.DatetimeIndex(d.time.values).equals(pd.date_range('2017-07-01',periods=2,freq='3h'))


def test_rejects_unknown_precipitation_units():
    d=sample();d.lwe_precipitation_rate.attrs['units']='mm'
    with pytest.raises(ValueError,match='expected units'):convert(d)


def test_rejects_missing_time():
    d=sample();d['Times']=('Time',np.array(['2017-07-01_00:00:00','2017-07-01_06:00:00'],dtype='S19'))
    with pytest.raises(ValueError,match='three-hour'):convert(d)


def test_curvilinear_longitudes_normalize_without_sorting_native_grid(tmp_path):
    import logging

    from symfluence.data.preprocessing.dataset_handlers.carra_utils import CARRAHandler
    d=convert(sample())
    expected=d.longitude.values.copy()
    d=d.assign_coords(longitude=(d.longitude.dims,expected+360))
    path=tmp_path/'month.nc';d.to_netcdf(path)
    handler=CARRAHandler.__new__(CARRAHandler)
    handler.logger=logging.getLogger('test')
    handler.open_dataset=xr.open_dataset
    handler._normalize_coordinates(tmp_path)
    with xr.open_dataset(path) as result:
        np.testing.assert_allclose(result.longitude,expected)
        np.testing.assert_array_equal(result.air_temperature,d.air_temperature)
        assert result.longitude.dims==('y','x')
