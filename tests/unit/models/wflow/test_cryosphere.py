from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.wflow.cryosphere import calculate_constraints


def fixture(tmp_path):
    settings = tmp_path / 'settings'
    output = tmp_path / 'output'
    obs = tmp_path / 'data/observations/cryosphere'
    for p in [settings, output, obs]:
        p.mkdir(parents=True)
    coords = dict(y=[0.], x=[0., 1.])
    maps = xr.Dataset({
        'wflow_subcatch': (('y', 'x'), [[1., 1.]]),
        'wflow_cellarea': (('y', 'x'), [[1e6, 1e6]]),
        'wflow_glacierfrac': (('y', 'x'), [[0., 1.]]),
    }, coords=coords)
    maps.to_netcdf(settings / 'wflow_staticmaps.nc')
    dates = pd.date_range('2012-09-30', '2014-10-01')
    shape = (len(dates), 1, 2)
    ice = np.zeros(shape)
    ice[:, 0, 1] = 10000 - np.arange(len(dates))
    data = xr.Dataset({
        'ice': (('time', 'y', 'x'), ice),
        'snow': (('time', 'y', 'x'), np.zeros(shape)),
        'snowwater': (('time', 'y', 'x'), np.zeros(shape)),
    }, coords=dict(time=dates + pd.Timedelta(days=1), **coords))
    data.to_netcdf(output / 'cryosphere.nc')
    pd.DataFrame(dict(YYYY=dates.year, MM=dates.month, DD=dates.day,
                      fsca_outside_glaciers=np.zeros(len(dates)))).to_csv(
        obs / 'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';', index=False)
    pd.DataFrame(dict(annual_net_MB=[-.365, -999.], g_area_dyn=[1., 1.]), index=[2013, 2014]).to_csv(
        obs / 'glacier_timeseries.csv', sep=';')
    return settings, output, obs, data


def test_complete_first_year_and_no_evaluation_leakage(tmp_path):
    settings, output, obs, data = fixture(tmp_path)
    # An extreme evaluation year and its states must not affect the objective.
    data['ice'].loc[dict(time=slice('2013-10-02', None))] = 1e8
    data.to_netcdf(output / 'cryosphere.nc')
    result = calculate_constraints(output, settings, tmp_path, '2012-10-01, 2013-09-30')
    assert result['snow_score'] == 1
    assert result['glacier_score'] == pytest.approx(1)
    assert result['glacier_balance_year_count'] == 1
    saved = json.loads((output / 'cryosphere_metrics.json').read_text())
    assert [r['year'] for r in saved['annual_glacier_balance']] == [2013]


@pytest.mark.parametrize('failure', ['missing_day', 'nonfinite', 'missing_boundary'])
def test_rejects_incomplete_or_invalid_states(tmp_path, failure):
    settings, output, obs, data = fixture(tmp_path)
    if failure == 'missing_day':
        data = data.isel(time=np.arange(data.sizes['time']) != 100)
    elif failure == 'missing_boundary':
        data = data.isel(time=slice(1, None))
    else:
        data['ice'].values[100, 0, 1] = np.nan
    data.to_netcdf(output / 'cryosphere.nc')
    with pytest.raises(ValueError):
        calculate_constraints(output, settings, tmp_path, '2012-10-01, 2013-09-30')
