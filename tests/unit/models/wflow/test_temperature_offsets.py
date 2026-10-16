"""Seasonal temperature corrections shift temperatures and keep Oudin PET consistent."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
from symfluence.models.wflow.distributed import offset_temperature

OFFSETS = {'DJF': 1.0, 'MAM': 0.5, 'JJA': 0.25, 'SON': 0.75}


def _forcing():
    time = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    temp = np.linspace(-10, 10, len(time))[:, None] * np.ones((1, 2))
    ds = xr.Dataset({k: (('time', 'hru'), temp.copy()) for k in ('temp', 'tmin', 'tmax')},
                    coords={'time': time, 'hru': [1, 2]})
    ds['pet'] = ds.temp * 0
    ds['precip'] = ds.temp * 0 + 1
    return ds, gpd.GeoDataFrame({'center_lat': [64.0, 64.2]})


def test_seasonal_offsets_and_pet():
    ds, bands = _forcing()
    out = offset_temperature(ds, bands, OFFSETS)
    diff = (out.temp - ds.temp).isel(hru=0).to_series()
    assert diff['2020-01-15'] == 1.0 and diff['2020-04-15'] == 0.5
    assert diff['2020-07-15'] == 0.25 and diff['2020-10-15'] == 0.75
    assert np.allclose(out.tmax - ds.tmax, out.temp - ds.temp)
    expected = PETCalculatorMixin.oudin_pet_numpy(out.temp.values[:, 1], out.time.dt.dayofyear.values, 64.2)
    assert np.allclose(out.pet.values[:, 1], expected)
    assert np.array_equal(out.precip, ds.precip)


def test_incomplete_offsets_rejected():
    ds, bands = _forcing()
    with pytest.raises(ValidationError):
        offset_temperature(ds, bands, {'DJF': 1.0})
