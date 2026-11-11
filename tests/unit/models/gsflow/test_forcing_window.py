"""GSFLOW daily inputs cover the whole end day."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.gsflow.basin_inputs import load_hru_forcing


def test_cbh_forcing_covers_whole_end_day_for_midnight_end(tmp_path):
    """An end date without a time of day used to leave one timestep in the
    last day unless the caller padded it by 23 h."""
    times = pd.date_range('2010-01-01', '2010-01-03 21:00', freq='3h')
    n = len(times)
    xr.Dataset({
        'precipitation_flux': (('time', 'hru'), np.full((n, 2), 1e-4)),
        'air_temperature': (('time', 'hru'), 273.15 + np.tile(times.hour.values[:, None], (1, 2)).astype(float)),
        'surface_downwelling_shortwave_flux': (('time', 'hru'), np.full((n, 2), 100.)),
        'hruId': (('hru',), [1, 2]),
    }, coords={'time': times}, attrs={'timestep_seconds': 10800.}).to_netcdf(tmp_path / 'forcing.nc')
    for end in (datetime(2010, 1, 2), datetime(2010, 1, 2, 21)):
        daily = load_hru_forcing(tmp_path, [1, 2], datetime(2010, 1, 1), end)
        assert len(daily['precip']) == 2  # and not the following day
        assert daily['precip'].iloc[-1].tolist() == pytest.approx([8 * 1e-4 * 10800] * 2)
        assert daily['tmax'].iloc[-1].tolist() == pytest.approx([21.0, 21.0])
        assert daily['tmin'].iloc[-1].tolist() == pytest.approx([0.0, 0.0])
