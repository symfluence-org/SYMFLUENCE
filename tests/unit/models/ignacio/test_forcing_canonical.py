# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical-forcing-reader behaviour for the IGNACIO (fire) preprocessor.

IGNACIO converted CF ``precipitation_flux`` to its weather CSV with a hardcoded
``* 3600`` (hourly assumption). With the canonical reader it scales by the
declared timestep instead, so daily forcing is no longer undercounted 24x.
"""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.ignacio.preprocessor import IGNACIOPreProcessor


@pytest.fixture
def ignacio_config(tmp_path):
    return SymfluenceConfig(**{
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "test_run",
        "EXPERIMENT_TIME_START": "2005-01-01 00:00",
        "EXPERIMENT_TIME_END": "2005-03-01 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "IGNACIO",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 86400,
    })


def _write_cf_forcing(forcing_dir, pptrate_value, *, n=10, drop=(), name="ERA5_forcing.nc"):
    forcing_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-02", periods=n, freq="D")
    data = {
        "precipitation_flux": ("time", np.full(n, pptrate_value, dtype="f8")),  # kg m-2 s-1
        "air_temperature": ("time", np.full(n, 283.15, dtype="f8")),            # K -> 10 degC
        "wind_speed": ("time", np.full(n, 2.0, dtype="f8")),
        "specific_humidity": ("time", np.full(n, 0.005, dtype="f8")),
        "surface_air_pressure": ("time", np.full(n, 101325.0, dtype="f8")),
    }
    xr.Dataset({k: v for k, v in data.items() if k not in drop},
               coords={"time": times}).to_netcdf(forcing_dir / name)


def test_precip_scaled_by_declared_timestep(ignacio_config):
    pp = IGNACIOPreProcessor(ignacio_config, Mock())
    pp.ignacio_input_dir.mkdir(parents=True, exist_ok=True)
    _write_cf_forcing(pp.forcing_basin_path, 1e-4)  # daily, 1e-4 kg/m2/s

    weather_csv = pp._convert_era5_to_weather_csv({})

    assert weather_csv is not None and weather_csv.exists()
    df = pd.read_csv(weather_csv)
    # 1e-4 kg/m2/s * 86400 s = 8.64 mm per step (old *3600 gave 0.36)
    assert df["PRECIP"].mean() == pytest.approx(8.64, rel=1e-3)
    assert df["TEMP"].mean() == pytest.approx(10.0, abs=1e-3)


def _preprocessor(ignacio_config):
    pp = IGNACIOPreProcessor(ignacio_config, Mock())
    pp.ignacio_input_dir.mkdir(parents=True, exist_ok=True)
    return pp


def test_humidity_and_wind_from_forcing_not_constants(ignacio_config):
    pp = _preprocessor(ignacio_config)
    _write_cf_forcing(pp.forcing_basin_path, 0.0)

    df = pd.read_csv(pp._convert_era5_to_weather_csv({}))

    # q = 5 g/kg at 1013 hPa and 10 degC is ~66% RH (the old fallback was a flat 50%).
    assert df["RH"].iloc[0] == pytest.approx(66.2, abs=0.5)
    assert df["WS"].iloc[0] == pytest.approx(7.2)  # 2 m/s in km/h


@pytest.mark.parametrize("dropped", [
    "air_temperature", "wind_speed", "precipitation_flux", "specific_humidity",
])
def test_missing_weather_variable_raises(ignacio_config, dropped):
    pp = _preprocessor(ignacio_config)
    _write_cf_forcing(pp.forcing_basin_path, 1e-4, drop=(dropped,))

    with pytest.raises(ModelExecutionError, match="lacks"):
        pp._prepare_weather_data({})
    assert not (pp.ignacio_input_dir / "static_weather.csv").exists()


def test_relative_humidity_used_directly(ignacio_config):
    pp = _preprocessor(ignacio_config)
    _write_cf_forcing(pp.forcing_basin_path, 0.0,
                      drop=("specific_humidity", "surface_air_pressure"))
    path = pp.forcing_basin_path / "ERA5_forcing.nc"
    ds = xr.load_dataset(path)
    ds["relative_humidity"] = ("time", np.full(ds.sizes["time"], 0.3))
    ds["relative_humidity"].attrs["units"] = "1"
    ds.to_netcdf(path)

    df = pd.read_csv(pp._convert_era5_to_weather_csv({}))
    assert df["RH"].iloc[0] == pytest.approx(30.0)


def test_unreadable_forcing_raises_instead_of_synthetic(ignacio_config):
    pp = _preprocessor(ignacio_config)
    pp.forcing_basin_path.mkdir(parents=True, exist_ok=True)
    (pp.forcing_basin_path / "ERA5_forcing.nc").write_bytes(b"not netcdf")

    with pytest.raises(ModelExecutionError, match="Cannot read"):
        pp._prepare_weather_data({})
    assert not (pp.ignacio_input_dir / "static_weather.csv").exists()


def test_missing_station_path_raises(ignacio_config, tmp_path):
    pp = _preprocessor(ignacio_config)
    with pytest.raises(ModelExecutionError, match="IGNACIO_STATION_PATH"):
        pp._prepare_weather_data({"station_path": str(tmp_path / "nope.csv")})


def test_no_forcing_falls_back_to_reproducible_synthetic_weather(ignacio_config):
    pp = _preprocessor(ignacio_config)
    cfg = {"ignition_date": "2014-07-15", "random_seed": 7}

    first = pd.read_csv(pp._prepare_weather_data(cfg))
    second = pd.read_csv(pp._prepare_weather_data(cfg))

    pd.testing.assert_frame_equal(first, second)
    assert any("SYNTHETIC" in str(c.args[0]) for c in pp.logger.warning.call_args_list)


def test_wind_direction_reproducible_from_seed(ignacio_config):
    pp = _preprocessor(ignacio_config)
    _write_cf_forcing(pp.forcing_basin_path, 0.0)

    a = pd.read_csv(pp._convert_era5_to_weather_csv({"random_seed": 3}))["WD"]
    b = pd.read_csv(pp._convert_era5_to_weather_csv({"random_seed": 3}))["WD"]
    pd.testing.assert_series_equal(a, b)
