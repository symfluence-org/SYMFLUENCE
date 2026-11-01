# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RHESSys climate generation resamples non-hourly source to hourly cadence.

RHESSys converts precipitation to mm/hour and sums to daily totals (and counts
rain-duration hours), treating one forcing step as one hour. A 3-hourly source
must therefore be resampled to hourly at load so daily totals and hour counts
are correct.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.modeling.model_ready.forcing_reader import forcing_timestep_seconds
from symfluence.models.rhessys.climate_generator import RHESSysClimateGenerator


def _gen(tmp_path):
    return RHESSysClimateGenerator({}, tmp_path / "domain_test", "test")


def _write_forcing(gen, freq, n):
    fdir = gen.forcing_basin_path
    fdir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-01", periods=n, freq=freq)
    xr.Dataset(
        {
            "precipitation_flux": (("time", "hru"), np.full((n, 1), 1e-4)),
            "air_temperature": (("time", "hru"), np.full((n, 1), 283.15)),
        },
        coords={"time": times, "hru": [0]},
    ).to_netcdf(fdir / "forcing.nc")


def _spacing_seconds(ds):
    diffs = np.diff(pd.DatetimeIndex(ds["time"].values).values).astype("timedelta64[s]").astype(int)
    assert len(set(diffs)) == 1, f"non-uniform: {set(diffs)}"
    return int(diffs[0])


def test_three_hourly_source_is_resampled_to_hourly(tmp_path):
    gen = _gen(tmp_path)
    _write_forcing(gen, "3h", n=40)
    ds = gen._load_forcing_data(datetime(2005, 1, 1), datetime(2005, 1, 6))
    assert forcing_timestep_seconds(ds) == 3600.0
    assert _spacing_seconds(ds) == 3600


def test_hourly_source_passes_through_unchanged(tmp_path):
    gen = _gen(tmp_path)
    _write_forcing(gen, "h", n=120)
    ds = gen._load_forcing_data(datetime(2005, 1, 1), datetime(2005, 1, 6))
    assert _spacing_seconds(ds) == 3600
