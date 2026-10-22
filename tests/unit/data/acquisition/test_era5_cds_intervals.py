"""CDS hourly totals must not be differenced or shift the first interval."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.data.acquisition.handlers.era5_processing import era5_to_summa_schema


def test_cds_hourly_totals_preserve_first_value_and_volume():
    dates = pd.date_range('2013-07-01',periods=8,freq='h')
    rain = np.arange(1,9,dtype=float) / 1000
    ds = xr.Dataset({
        'tp': ('time',rain), 'ssrd': ('time',np.arange(8)*3600.),
        'strd': ('time',np.full(8,300.*3600)),
        't2m': ('time',np.full(8,280.)), 'sp': ('time',np.full(8,95000.)),
        'd2m': ('time',np.full(8,275.)),
        'u10': ('time',np.ones(8)), 'v10': ('time',np.ones(8)),
    },coords={'time':dates})
    out = era5_to_summa_schema(ds,source='cds')
    np.testing.assert_allclose(out.precipitation_flux*3600,rain*1000)
    np.testing.assert_allclose(out.surface_downwelling_shortwave_flux,np.arange(8))
    np.testing.assert_allclose(out.surface_downwelling_longwave_flux,300)
    assert pd.DatetimeIndex(out.time.values).equals(dates)
