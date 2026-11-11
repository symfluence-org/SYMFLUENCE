# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RHESSys climate files: no synthetic climate, rain precision, rain duration."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import ValidationError
from symfluence.models.rhessys.climate_generator import RHESSysClimateGenerator


def _gen(tmp_path):
    return RHESSysClimateGenerator({}, tmp_path / "domain_test", "test")


def _values(path):
    return [float(v) for v in path.read_text().split("\n")[1:] if v]


def _write_forcing(gen, precip_mm_h, sw=None, drop=()):
    fdir = gen.forcing_basin_path
    fdir.mkdir(parents=True, exist_ok=True)
    n = len(precip_mm_h)
    times = pd.date_range("2005-01-01", periods=n, freq="h")
    data = {
        "precipitation_flux": (("time", "hru"), np.asarray(precip_mm_h)[:, None] / 3600.0,
                               {"units": "kg m-2 s-1"}),
        "air_temperature": (("time", "hru"), np.full((n, 1), 283.15)),
    }
    if sw is not None:
        data["surface_downwelling_shortwave_flux"] = (("time", "hru"), np.asarray(sw)[:, None])
    ds = xr.Dataset(data, coords={"time": times, "hru": [0]}).drop_vars(list(drop))
    ds.to_netcdf(fdir / "forcing.nc")


def test_missing_forcing_raises_instead_of_synthetic_climate(tmp_path):
    gen = _gen(tmp_path)
    with pytest.raises(FileNotFoundError, match="No forcing data found"):
        gen.generate_climate_files(datetime(2005, 1, 1), datetime(2005, 1, 3))
    assert not list(gen.climate_dir.glob("*.rain"))
    assert not hasattr(gen, "_create_synthetic_climate")


@pytest.mark.parametrize("dropped,named", [
    ("precipitation_flux", "precipitation"),
    ("air_temperature", "temperature"),
])
def test_missing_variable_raises(tmp_path, dropped, named):
    gen = _gen(tmp_path)
    _write_forcing(gen, np.zeros(48), drop=(dropped,))
    with pytest.raises(ValidationError, match=named):
        gen.generate_climate_files(datetime(2005, 1, 1), datetime(2005, 1, 3))


def test_rain_file_keeps_sub_tenth_millimetre_depths(tmp_path):
    """Rain is written in m/day; 4 decimals would quantise it to 0.1 mm/day."""
    gen = _gen(tmp_path)
    daily_mm = np.array([0.04, 0.049, 0.16, 12.34])
    _write_forcing(gen, np.repeat(daily_mm / 24.0, 24))
    gen.generate_climate_files(datetime(2005, 1, 1), datetime(2005, 1, 4, 23))
    written_mm = np.array(_values(gen.climate_dir / "test_base.rain")) * 1000.0
    np.testing.assert_allclose(written_mm, daily_mm, atol=1e-4)


def test_daytime_rain_duration_counts_wet_daylight_hours_only(tmp_path):
    """Trace drizzle and night-time rain do not count towards the duration."""
    gen = _gen(tmp_path)
    hours = np.arange(48) % 24
    sw = np.where((hours >= 8) & (hours < 16), 200.0, 0.0)
    precip = np.full(48, 0.01)            # drizzle every hour, both days
    precip[[2, 3]] = 1.0                  # day 1: two night hours of rain
    precip[[10, 11, 12]] = 1.0            # day 1: three daylight hours of rain
    _write_forcing(gen, precip, sw=sw)
    gen.generate_climate_files(datetime(2005, 1, 1), datetime(2005, 1, 2, 23))
    assert _values(gen.climate_dir / "test_base.daytime_rain_duration") == [3.0, 0.0]


def test_rain_duration_without_shortwave_uses_trace_threshold_only(tmp_path):
    gen = _gen(tmp_path)
    precip = np.full(24, 0.01)
    precip[[2, 3, 10, 11, 12]] = 1.0
    _write_forcing(gen, precip)
    gen.generate_climate_files(datetime(2005, 1, 1), datetime(2005, 1, 1, 23))
    assert _values(gen.climate_dir / "test_base.daytime_rain_duration") == [5.0]
