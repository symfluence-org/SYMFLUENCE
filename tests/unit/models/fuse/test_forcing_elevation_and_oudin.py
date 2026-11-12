"""FUSE_FORCING_ELEVATION reaches Z_FORCING; distributed Oudin uses the shared routine."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.modeling.mixins.pet_calculator import PETCalculatorMixin
from symfluence.models.fuse.config_schema import FUSEConfig
from symfluence.models.fuse.forcing_processor import FuseForcingProcessor
from symfluence.models.fuse.runner import FUSERunner

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _runner(forcing_elevation):
    runner = FUSERunner.__new__(FUSERunner)
    fuse = FUSEConfig(**({} if forcing_elevation is None else {'FUSE_FORCING_ELEVATION': forcing_elevation}))
    runner._config = SimpleNamespace(model=SimpleNamespace(fuse=fuse))
    runner.logger = logging.getLogger('test_fuse_forcing_elevation')
    return runner


def test_configured_forcing_elevation_overrides_band_mean():
    assert _runner(812.5)._forcing_elevation({'z_forcing': 400.0}) == 812.5


def test_band_mean_when_not_configured():
    assert _runner(None)._forcing_elevation({'z_forcing': 400.0}) == 400.0


def test_vectorized_oudin_matches_shared_routine_and_is_finite_at_the_poles():
    times = pd.date_range('2021-01-01', '2021-12-31', freq='D')
    lats = np.array([45.0, 64.0, 80.0])
    temp = xr.DataArray(
        np.full((len(times), 3), 283.15), dims=('time', 'hru'),
        coords={'time': times, 'hru': [1, 2, 3]})
    pet = FuseForcingProcessor._vectorized_oudin(None, temp, lats)
    assert pet.dims == ('time', 'hru') and np.all(np.isfinite(pet.values))
    doy = np.asarray(times.dayofyear)
    for j, lat in enumerate(lats):
        np.testing.assert_allclose(
            pet.values[:, j], PETCalculatorMixin.oudin_pet_numpy(np.full(len(times), 10.0), doy, lat))
