"""Check actual NetCDF writer output against VIC image's kPa contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.models.vic.preprocessor import VICPreProcessor


def setup_writer(tmp_path):
    pre = object.__new__(VICPreProcessor)
    pre.domain_name = 'test'
    pre.forcing_dir = tmp_path
    pre._get_catchment_properties = lambda: dict(elev=750., lat=64., lon=-19.)
    times = pd.date_range('2014-01-01', periods=2, freq='h')
    ds = xr.Dataset({v: ('time', [x, x]) for v, x in
                     dict(AIR_TEMP=2., PREC=0., SWDOWN=50., LWDOWN=300., WIND=5.).items()},
                    coords={'time': times})
    domain = xr.Dataset({'mask': (('lat', 'lon'), [[1]])}, coords={'lat': [64.], 'lon': [-19.]})
    return pre, ds, domain


@pytest.mark.parametrize('units,scale', [('Pa', 1.), ('kPa', .001), ('hPa', .01)])
@pytest.mark.parametrize('humidity', ['direct', 'specific', 'estimated'])
def test_written_pressure_and_humidity(tmp_path, units, scale, humidity):
    pre, ds, domain = setup_writer(tmp_path)
    ds['PRESSURE'] = ('time', [90000.*scale]*2, {'units': units})
    if humidity == 'direct':
        ds['VP'] = ('time', [500.*scale]*2, {'units': units})
        expected_vp = .5
    elif humidity == 'specific':
        ds['specific_humidity'] = ('time', [.004]*2)
        expected_vp = .004*90./(.622+.378*.004)
    else:
        expected_vp = .6*.6108*np.exp(17.27*2./(2.+237.3))
    pre._write_forcing_files(ds, domain, ds.time.values[0], ds.time.values[-1])
    with xr.open_dataset(tmp_path/'test_forcing.2014.nc') as written:
        np.testing.assert_allclose(written.PRESSURE, 90.)
        np.testing.assert_allclose(written.VP, expected_vp)
        assert written.PRESSURE.attrs['units'] == written.VP.attrs['units'] == 'kPa'
        # Native driver multiplies by 1000, yielding physical Pa and density.
        density = written.PRESSURE.values*1000/(287.*(2.+273.15))
        assert np.all((density > 1.) & (density < 1.3))


def test_fallback_and_synthetic_pressure(tmp_path):
    pre, ds, domain = setup_writer(tmp_path)
    pre._write_forcing_files(ds, domain, ds.time.values[0], ds.time.values[-1])
    expected = 101.325*np.exp(-750./8500.)
    with xr.open_dataset(tmp_path/'test_forcing.2014.nc') as written:
        np.testing.assert_allclose(written.PRESSURE, expected)
    pre._generate_synthetic_forcing(pd.Timestamp('2014-01-01'), pd.Timestamp('2014-01-01T01:00'))
    with xr.open_dataset(tmp_path/'test_forcing.2014.nc') as written:
        np.testing.assert_allclose(written.PRESSURE, expected)
        assert written.VP.max() < 2.
        assert written.VP.attrs['units'] == 'kPa'


def test_unknown_pressure_units_rejected(tmp_path):
    pre, ds, domain = setup_writer(tmp_path)
    ds['PRESSURE'] = ('time', [90., 90.], {'units': 'barfoo'})
    with pytest.raises(ValidationError, match='Unsupported pressure units'):
        pre._write_forcing_files(ds, domain, ds.time.values[0], ds.time.values[-1])
