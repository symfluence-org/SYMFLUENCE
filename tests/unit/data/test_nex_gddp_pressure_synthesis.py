# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for NEX-GDDP-CMIP6 surface-pressure synthesis.

Co-author NB reported that SUMMA preprocessing on a NEX-GDDP-CMIP6
configuration (config_paradise_gddp_access_cm2.yaml) failed with:

    ERROR Execution failed: Failed during SUMMA preprocessing:
    Missing required forcing variables: [surface_air_pressure]

Root cause: NEX-GDDP-CMIP6 publishes {pr, tas, tasmax, tasmin, huss,
hurs, rlds, rsds, sfcWind} but NOT ``ps``. SUMMA requires airpres,
so the handler previously just dropped through without producing
a pressure variable and later model-ready preprocessing errored.

Pin the fix: when pressure is absent from the input dataset, the
handler synthesizes a climatological estimate via the International
Standard Atmosphere (altitude from mean temperature via lapse rate,
then ISA pressure-altitude relation), emits a clear warning, and
writes a full-like DataArray so downstream code sees a normal
forcing field rather than an empty/missing one.
"""

import logging

import numpy as np
import pytest
import xarray as xr

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_handler():
    """Build an NEXGDDPCMIP6Handler without running its
    config-dependent __init__."""
    from symfluence.data.preprocessing.dataset_handlers.nex_gddp_utils import (
        NEXGDDPCMIP6Handler,
    )
    h = NEXGDDPCMIP6Handler.__new__(NEXGDDPCMIP6Handler)
    h.logger = logging.getLogger("test_nex_pressure")
    return h


def _make_ds_without_pressure(t_mean_K=285.0):
    """Build a minimal daily NEX-GDDP-style dataset with raw CMIP
    names (pr, tas, huss, rlds, rsds, sfcWind) but no ps."""
    times = xr.cftime_range(start="2015-01-01", periods=3, freq="D")
    lats = np.array([64.0, 64.5])
    lons = np.array([-22.0, -21.5])
    shape = (len(times), len(lats), len(lons))
    rng = np.random.default_rng(0)

    def _arr(mean, spread=0.5):
        return mean + spread * rng.standard_normal(shape).astype("float32")

    ds = xr.Dataset(
        data_vars=dict(
            pr=(("time", "lat", "lon"), np.clip(_arr(1.5e-5, 5e-6), 0, None)),
            tas=(("time", "lat", "lon"), _arr(t_mean_K, 2.0)),
            huss=(("time", "lat", "lon"), np.clip(_arr(3e-3, 1e-3), 0, None)),
            rlds=(("time", "lat", "lon"), _arr(250.0, 10.0)),
            rsds=(("time", "lat", "lon"), _arr(150.0, 20.0)),
            sfcWind=(("time", "lat", "lon"), np.clip(_arr(5.0, 1.0), 0, None)),
        ),
        coords=dict(time=times, lat=lats, lon=lons),
    )
    return ds


def test_handler_synthesizes_pressure_when_ps_missing(caplog):
    """NEX-GDDP-CMIP6 doesn't ship ps; handler must synthesize
    surface_air_pressure so SUMMA preprocessing doesn't fail with a
    missing-variable error."""
    h = _make_handler()
    ds_in = _make_ds_without_pressure(t_mean_K=285.0)
    caplog.set_level(logging.WARNING)
    ds_out = h.process_dataset(ds_in)

    assert "surface_air_pressure" in ds_out.data_vars, \
        "process_dataset must emit surface_air_pressure even when ps is absent"
    # ISA from T_mean=285K => z ≈ 477m => P ≈ 95600 Pa
    p = float(ds_out["surface_air_pressure"].mean().values)
    assert 92000 < p < 100000, f"synthesized pressure {p:.0f} Pa outside expected 92–100 kPa range for T≈285 K"

    warning_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "NEX-GDDP-CMIP6 does not publish surface pressure" in warning_text
    assert "synthesizing from mean air temperature" in warning_text


def test_synthesized_pressure_tracks_altitude(caplog):
    """Colder mean T → higher inferred altitude → lower pressure.
    The synthesis must be monotone in T so users can sanity-check
    the output against expected orographic pressure differences."""
    h = _make_handler()
    caplog.set_level(logging.WARNING)
    p_warm = float(h.process_dataset(_make_ds_without_pressure(t_mean_K=285.0))["surface_air_pressure"].mean().values)
    p_cold = float(h.process_dataset(_make_ds_without_pressure(t_mean_K=275.0))["surface_air_pressure"].mean().values)
    assert p_cold < p_warm, \
        f"colder T should yield lower P (higher altitude); got warm={p_warm:.0f}, cold={p_cold:.0f}"


def test_degenerate_temperature_falls_back_to_sea_level(caplog):
    """If mean T is unphysical (e.g. all-NaN input) we can't infer
    altitude — fall back to sea-level P0 with a clear warning so the
    user isn't left guessing why pressure looks constant."""
    h = _make_handler()
    ds = _make_ds_without_pressure(t_mean_K=285.0)
    ds["tas"] = xr.full_like(ds["tas"], float("nan"))
    caplog.set_level(logging.WARNING)
    ds_out = h.process_dataset(ds)
    p = float(ds_out["surface_air_pressure"].mean().values)
    assert abs(p - 101325.0) < 1.0, f"expected sea-level fallback 101325 Pa, got {p}"
    warn = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "could not infer altitude" in warn
