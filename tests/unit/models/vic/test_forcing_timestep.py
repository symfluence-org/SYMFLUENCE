"""VIC forcing: non-hourly sources, gaps, missing inputs and partial years."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import FileOperationError, ValidationError
from symfluence.models.vic.preprocessor import VICPreProcessor


def _pre(tmp_path):
    pre = object.__new__(VICPreProcessor)
    pre.domain_name = 'toy'
    pre.forcing_dir = tmp_path
    return pre


def _domain():
    return xr.Dataset({'mask': (('lat', 'lon'), [[1]])}, coords={'lat': [64.], 'lon': [-19.]})


def _forcing(times, precip_rate=1.0 / 3600.0):
    n = len(times)

    def var(value, units):
        return (('time', 'hru'), np.full((n, 1), value), {'units': units})

    return xr.Dataset(
        {
            'precipitation_flux': var(precip_rate, 'kg m-2 s-1'),
            'air_temperature': var(280.0, 'K'),
            'surface_downwelling_shortwave_flux': var(100.0, 'W m-2'),
            'surface_downwelling_longwave_flux': var(300.0, 'W m-2'),
            'surface_air_pressure': var(90000.0, 'Pa'),
            'specific_humidity': var(0.004, 'kg kg-1'),
            'wind_speed': var(3.0, 'm s-1'),
        },
        coords={'time': times},
    )


@pytest.mark.parametrize('step_h', [1, 3, 6])
def test_annual_precipitation_independent_of_source_step(tmp_path, step_h):
    """1 mm/h is 8760 mm/yr whether the source is 1-, 3- or 6-hourly."""
    times = pd.date_range('2001-01-01', '2001-12-31 23:00', freq=f'{step_h}h')
    pre = _pre(tmp_path)
    pre._write_forcing_files(_forcing(times), _domain(), times[0], times[-1])
    with xr.open_dataset(tmp_path / 'toy_forcing.2001.nc') as out:
        assert out.sizes['time'] == 8760
        np.testing.assert_allclose(float(out.PREC.sum()), 8760.0, rtol=1e-9)
        np.testing.assert_allclose(out.PREC.values, 1.0)
        np.testing.assert_allclose(out.AIR_TEMP.values, 280.0 - 273.15)
        assert not np.isnan(out.SWDOWN.values).any()


def test_gap_in_forcing_raises(tmp_path):
    times = pd.date_range('2001-01-01', '2001-01-10 23:00', freq='h')
    times = times.delete(slice(50, 60))
    with pytest.raises(ValueError, match='irregular'):
        _pre(tmp_path)._write_forcing_files(_forcing(times), _domain(), times[0], times[-1])
    assert not list(tmp_path.glob('*.nc'))


def test_partial_years_are_not_padded(tmp_path):
    """Files hold only real timesteps; the run is limited to whole covered days."""
    times = pd.date_range('2001-10-01 06:00', '2002-03-15 11:00', freq='h')
    pre = _pre(tmp_path)
    pre._write_forcing_files(_forcing(times), _domain(), times[0], times[-1])
    with xr.open_dataset(tmp_path / 'toy_forcing.2001.nc') as first, \
            xr.open_dataset(tmp_path / 'toy_forcing.2002.nc') as last:
        assert pd.Timestamp(first.time.values[0]) == times[0]
        assert pd.Timestamp(last.time.values[-1]) == times[-1]
        assert first.sizes['time'] + last.sizes['time'] == len(times)
        # Subsequent years start on 1 January, as VIC requires.
        assert pd.Timestamp(last.time.values[0]) == pd.Timestamp('2002-01-01')

    start, end = pre._clip_to_forcing_coverage(pd.Timestamp('2001-01-01'), pd.Timestamp('2002-12-31'))
    assert start == pd.Timestamp('2001-10-02')
    assert end == pd.Timestamp('2002-03-14')
    # A period inside the coverage is left alone.
    start, end = pre._clip_to_forcing_coverage(pd.Timestamp('2001-11-01'), pd.Timestamp('2002-02-01 23:00'))
    assert (start, end) == (pd.Timestamp('2001-11-01'), pd.Timestamp('2002-02-01'))
    with pytest.raises(ValidationError, match='does not cover'):
        pre._clip_to_forcing_coverage(pd.Timestamp('2005-01-01'), pd.Timestamp('2005-12-31'))


@pytest.mark.parametrize('dropped,named', [
    ('precipitation_flux', 'PREC'),
    ('air_temperature', 'AIR_TEMP'),
    ('wind_speed', 'WIND'),
    ('surface_downwelling_shortwave_flux', 'SWDOWN'),
    ('surface_downwelling_longwave_flux', 'LWDOWN'),
])
def test_missing_variable_raises_naming_it(tmp_path, dropped, named):
    times = pd.date_range('2001-01-01', periods=48, freq='h')
    forcing = _forcing(times).drop_vars(dropped)
    with pytest.raises(ValidationError, match=named):
        _pre(tmp_path)._write_forcing_files(forcing, _domain(), times[0], times[-1])
    assert not list(tmp_path.glob('*.nc'))


def test_missing_domain_file_raises_instead_of_synthetic_forcing(tmp_path):
    pre = _pre(tmp_path)
    pre.params_dir = tmp_path / 'params'
    pre._get_simulation_dates = lambda: (pd.Timestamp('2001-01-01'), pd.Timestamp('2001-01-02'))
    with pytest.raises(FileOperationError, match='domain file not found'):
        pre._generate_forcing_files()
    assert not list(tmp_path.glob('*.nc'))


def test_unloadable_forcing_propagates(tmp_path):
    pre = _pre(tmp_path)
    pre.params_dir = tmp_path
    _domain().to_netcdf(tmp_path / 'vic_domain.nc')
    pre._get_simulation_dates = lambda: (pd.Timestamp('2001-01-01'), pd.Timestamp('2001-01-02'))

    def _no_forcing():
        raise FileNotFoundError('No forcing data found in nowhere')

    pre._load_forcing_data = _no_forcing
    with pytest.raises(FileNotFoundError, match='No forcing data found'):
        pre._generate_forcing_files()
    assert not list(tmp_path.glob('*forcing*.nc'))
