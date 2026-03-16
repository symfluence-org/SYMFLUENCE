from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.constants import PhysicalConstants
from symfluence.data.preprocessing.dataset_handlers.rdrs_utils import RDRSHandler


@pytest.fixture
def rdrs_handler():
    config = {'DOMAIN_NAME': 'test_domain'}
    import logging
    logger = logging.getLogger('test')
    project_dir = Path('/tmp/test_project')
    return RDRSHandler(config, logger, project_dir)

def test_rdrs_v21_processing(rdrs_handler):
    # Mock RDRS v2.1 dataset
    times = pd.date_range('2015-01-01', periods=2, freq='h')
    ds = xr.Dataset(
        data_vars={
            'RDRS_v2.1_P_TT_1.5m': (['time'], [10.0, 15.0]), # Celsius
            'RDRS_v2.1_P_P0_SFC': (['time'], [1013.0, 1012.0]), # mb
            'RDRS_v2.1_A_PR0_SFC': (['time'], [1.0, 2.0]), # mm/hr
        },
        coords={'time': times}
    )

    processed = rdrs_handler.process_dataset(ds)

    assert processed.air_temperature.values[0] == 10.0 + PhysicalConstants.KELVIN_OFFSET
    assert processed.surface_air_pressure.values[0] == 1013.0 * 100
    assert processed.precipitation_flux.values[0] == 1.0 / 3600.0

def test_rdrs_v31_processing(rdrs_handler):
    # Mock RDRS v3.1 dataset (short names, already in standard units)
    times = pd.date_range('2015-01-01', periods=2, freq='h')
    ds = xr.Dataset(
        data_vars={
            'TT': (['time'], [283.15, 288.15]), # Kelvin
            'P0': (['time'], [101325.0, 101200.0]), # Pa
            'PR0': (['time'], [0.0001, 0.0002]), # mm/s
        },
        coords={'time': times}
    )

    processed = rdrs_handler.process_dataset(ds)

    assert processed.air_temperature.values[0] == 283.15
    assert processed.surface_air_pressure.values[0] == 101325.0
    assert processed.precipitation_flux.values[0] == 0.0001
