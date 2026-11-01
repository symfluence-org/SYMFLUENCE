# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the canonical forcing reader's CFIF (CF standard name) contract.

``open_canonical_forcing`` must return forcing under canonical CF names
(``precipitation_flux``, ``air_temperature``, ...) regardless of whether the
source used SUMMA-native shorthand (``pptrate``/``airtemp``) or other dataset
aliases, and must expose the timestep via ``ds.attrs['timestep_seconds']``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")
import xarray as xr

from symfluence.core.modeling.model_ready.cf_conventions import (
    CANONICAL_FORCING,
    resolve_forcing_var,
)
from symfluence.core.modeling.model_ready.forcing_reader import (
    forcing_timestep_seconds,
    open_canonical_forcing,
)

CF_NAMES = {
    "precipitation_flux", "air_temperature", "surface_downwelling_shortwave_flux",
    "surface_downwelling_longwave_flux", "wind_speed", "specific_humidity",
    "surface_air_pressure",
}


def _write(tmp_path, data_vars):
    times = pd.date_range("2020-01-01", periods=8, freq="D")
    ds = xr.Dataset(
        {k: ("time", np.linspace(1, 8, 8)) for k in data_vars},
        coords={"time": times},
    )
    path = tmp_path / "forcing.nc"
    ds.to_netcdf(path)
    return path


def test_canonical_forcing_is_keyed_by_cf_names():
    assert set(CANONICAL_FORCING) == CF_NAMES
    # Each entry still records its SUMMA-native shorthand for model-native layers.
    assert CANONICAL_FORCING["precipitation_flux"]["summa"] == "pptrate"
    assert CANONICAL_FORCING["air_temperature"]["summa"] == "airtemp"


def test_summa_source_names_are_renamed_to_cf(tmp_path):
    path = _write(tmp_path, ["pptrate", "airtemp", "SWRadAtm"])
    ds = open_canonical_forcing(path)
    assert "precipitation_flux" in ds
    assert "air_temperature" in ds
    assert "surface_downwelling_shortwave_flux" in ds
    assert "pptrate" not in ds and "airtemp" not in ds


def test_era5_style_aliases_are_renamed_to_cf(tmp_path):
    path = _write(tmp_path, ["tp", "t2m", "ssrd"])
    ds = open_canonical_forcing(path)
    assert "precipitation_flux" in ds
    assert "air_temperature" in ds
    assert "surface_downwelling_shortwave_flux" in ds


def test_already_cf_named_source_is_unchanged(tmp_path):
    path = _write(tmp_path, ["precipitation_flux", "air_temperature"])
    ds = open_canonical_forcing(path)
    assert "precipitation_flux" in ds and "air_temperature" in ds


def test_mixed_vocabulary_store_is_coalesced(tmp_path):
    """A partially-regenerated store can spread spellings of one variable across
    files (some carry 'pptrate', others 'precipitation_flux'); the by-coords merge
    then leaves each spelling NaN where the other supplied data. open_canonical_forcing
    must coalesce them into a single, gap-free canonical variable.
    """
    t_old = pd.date_range("2002-01-01", periods=4, freq="D")
    t_new = pd.date_range("2002-01-05", periods=4, freq="D")
    # File A: SUMMA-shorthand spelling, first half of the period.
    xr.Dataset({"pptrate": ("time", np.full(4, 1.0))},
               coords={"time": t_old}).to_netcdf(tmp_path / "a.nc")
    # File B: canonical CF spelling, second half.
    xr.Dataset({"precipitation_flux": ("time", np.full(4, 2.0))},
               coords={"time": t_new}).to_netcdf(tmp_path / "b.nc")

    ds = open_canonical_forcing([tmp_path / "a.nc", tmp_path / "b.nc"])

    assert "precipitation_flux" in ds
    assert "pptrate" not in ds  # redundant alias dropped after coalescing
    vals = ds["precipitation_flux"].values
    assert not np.isnan(vals).any(), "coalesced variable must have no gaps"
    assert vals[0] == 1.0 and vals[-1] == 2.0  # values taken from whichever source had them


def test_timestep_inferred_from_daily_axis(tmp_path):
    path = _write(tmp_path, ["pptrate"])
    ds = open_canonical_forcing(path)
    assert forcing_timestep_seconds(ds) == pytest.approx(86400.0)


def test_resolve_forcing_var_accepts_cf_key_and_aliases(tmp_path):
    path = _write(tmp_path, ["pptrate"])
    ds = xr.open_dataset(path)
    try:
        # Resolves the SUMMA source under the CF canonical key.
        assert resolve_forcing_var(ds, "precipitation_flux") == "pptrate"
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# resample_canonical_forcing
# ---------------------------------------------------------------------------
from symfluence.core.modeling.model_ready.forcing_reader import resample_canonical_forcing  # noqa: E402


def _forcing(times, pptrate, airtemp):
    return xr.Dataset(
        {
            "precipitation_flux": ("time", np.asarray(pptrate, dtype="f8")),
            "air_temperature": ("time", np.asarray(airtemp, dtype="f8")),
        },
        coords={"time": pd.DatetimeIndex(times)},
    )


def test_resample_is_noop_when_already_at_target():
    times = pd.date_range("2020-01-01", periods=6, freq="h")
    ds = _forcing(times, np.full(6, 1e-4), np.full(6, 283.15))
    out = resample_canonical_forcing(ds, 3600)
    assert out["time"].size == 6
    np.testing.assert_array_equal(out["precipitation_flux"].values, ds["precipitation_flux"].values)


def test_resample_3hourly_to_hourly_conserves_precip_total():
    # 4 source steps at 3h spacing, constant rate.
    times = pd.date_range("2020-01-01", periods=4, freq="3h")
    rate = 2e-4
    ds = _forcing(times, np.full(4, rate), np.full(4, 283.15))

    out = resample_canonical_forcing(ds, 3600)

    # 4 intervals * 3 h = 12 hourly steps covering the full source span.
    assert out["time"].size == 12
    assert float(out.attrs["timestep_seconds"]) == 3600.0
    # Rate is a step function -> every hour carries the source rate.
    np.testing.assert_allclose(out["precipitation_flux"].values, rate, rtol=1e-9)
    # Accumulated total conserved: source sum(rate*3h) == hourly sum(rate*1h).
    src_total = float((ds["precipitation_flux"] * 10800).sum())
    out_total = float((out["precipitation_flux"] * 3600).sum())
    assert out_total == pytest.approx(src_total, rel=1e-9)


def test_resample_interpolates_state_variables():
    # Temperature ramps 280 -> 286 over two 3h steps; hourly interp is linear.
    times = pd.date_range("2020-01-01", periods=3, freq="3h")
    ds = _forcing(times, np.full(3, 1e-4), [280.0, 283.0, 286.0])

    out = resample_canonical_forcing(ds, 3600)
    temps = out["air_temperature"].values

    # First hour equals the first source value; interior is linearly interpolated.
    assert temps[0] == pytest.approx(280.0)
    assert temps[1] == pytest.approx(281.0, abs=1e-6)   # 1/3 of the way 280->283
    assert temps[3] == pytest.approx(283.0, abs=1e-6)   # the second source point
    assert not np.isnan(temps).any()


def test_irregular_axis_is_rejected(tmp_path):
    """[00:00, 01:00, 04:00] must not be declared a 3600 s cadence."""
    times = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 04:00"])
    ds = xr.Dataset({"pptrate": ("time", np.ones(3))}, coords={"time": times})
    path = tmp_path / "gappy.nc"
    ds.to_netcdf(path)

    with pytest.raises(ValueError, match="irregular"):
        open_canonical_forcing(path)


def test_declared_timestep_does_not_mask_irregular_axis(tmp_path):
    """A declared timestep_seconds attribute must not bypass axis validation."""
    times = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 04:00"])
    ds = xr.Dataset({"pptrate": ("time", np.ones(3))}, coords={"time": times})
    ds.attrs["timestep_seconds"] = 3600.0
    with pytest.raises(ValueError, match="irregular"):
        forcing_timestep_seconds(ds)


def test_declared_timestep_cross_checked_against_axis():
    """A wrong declared attribute loses to the measured cadence."""
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    ds = xr.Dataset({"pptrate": ("time", np.ones(5))}, coords={"time": times})
    ds.attrs["timestep_seconds"] = 3600.0  # lies: the axis is daily
    assert forcing_timestep_seconds(ds) == pytest.approx(86400.0)


def test_duplicate_timestamps_are_rejected(tmp_path):
    times = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00"])
    ds = xr.Dataset({"pptrate": ("time", np.ones(3))}, coords={"time": times})
    path = tmp_path / "dupes.nc"
    ds.to_netcdf(path)

    with pytest.raises(ValueError, match="strictly increasing"):
        open_canonical_forcing(path)


def test_resample_refuses_irregular_source():
    times = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 04:00"])
    ds = _forcing(times, np.ones(3) * 1e-4, np.full(3, 283.15))
    with pytest.raises(ValueError, match="irregular"):
        resample_canonical_forcing(ds, 3600)


def test_resample_daily_to_hourly_conserves_precip():
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    rate = 1e-4
    ds = _forcing(times, np.full(3, rate), np.full(3, 283.15))
    out = resample_canonical_forcing(ds, 3600)
    assert out["time"].size == 72  # 3 days * 24 h
    out_total = float((out["precipitation_flux"] * 3600).sum())
    src_total = float((ds["precipitation_flux"] * 86400).sum())
    assert out_total == pytest.approx(src_total, rel=1e-9)


# --- conflicting-discretization guard (issue #339) --------------------------

def _write_hru_forcing(path, n_hru, periods=8):
    """Write a minimal forcing file with an ``hru`` spatial dimension."""
    times = pd.date_range("2020-01-01", periods=periods, freq="D")
    ds = xr.Dataset(
        {"precipitation_flux": (("time", "hru"), np.ones((periods, n_hru)) * 1e-4)},
        coords={"time": times, "hru": np.arange(n_hru)},
    )
    ds.to_netcdf(path)
    return path


def test_conflicting_hru_dims_raise_actionable_error(tmp_path):
    """A store mixing hru=1 and hru=12 must fail loudly, naming the stray file.

    Reproduces the domain_Bow_at_Banff_lumped_era5 corruption: a lumped hru=1
    forcing and a 12-band elevation hru=12 remap collided in one store. Instead
    of xarray's cryptic "conflicting dimension sizes: {1, 12}", the reader must
    raise a clear message that identifies the offending file and points at #339.
    """
    lumped = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_CDS_2002_2009.nc", 1)
    stray = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_4ae454551262b9b7.nc", 12)
    with pytest.raises(ValueError) as exc:
        open_canonical_forcing([lumped, stray])
    msg = str(exc.value)
    assert "hru" in msg
    assert "1" in msg and "12" in msg
    assert "4ae454551262b9b7" in msg  # names the stray file
    assert "#339" in msg


def test_consistent_hru_dims_pass_the_guard(tmp_path):
    """Two time-chunks that share the same hru size must NOT trip the guard."""
    from symfluence.core.modeling.model_ready.forcing_reader import assert_consistent_spatial_dims

    jan = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_2020-01.nc", 12)
    feb = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_2020-02.nc", 12)
    # Must not raise: same discretization across a legitimately-chunked store.
    assert_consistent_spatial_dims([jan, feb])


def test_store_builder_rejects_mixed_discretization(tmp_path):
    """The store writer must refuse to publish a mixed-hru source set."""
    from symfluence.data.model_ready.forcings_builder import ForcingsStoreBuilder

    source = tmp_path / "domain_x" / "forcing" / "basin_averaged_data"
    source.mkdir(parents=True)
    _write_hru_forcing(source / "x_ERA5_remapped_CDS_2002_2009.nc", 1)
    _write_hru_forcing(source / "x_ERA5_remapped_4ae454551262b9b7.nc", 12)

    builder = ForcingsStoreBuilder(
        project_dir=tmp_path / "domain_x",
        domain_name="x",
        forcing_dataset="ERA5",
    )
    with pytest.raises(ValueError, match="#339"):
        builder.build()


# --- discretization-namespaced forcing store (issue #339 true-source fix) ----

def test_discretization_token_sanitizes():
    from symfluence.core.modeling.model_ready.forcing_reader import discretization_token

    assert discretization_token("lumped") == "lumped"
    assert discretization_token("Elevation") == "elevation"
    # Comma-separated composite discretizations flatten to a single safe token.
    assert discretization_token("elevation,landclass") == "elevation-landclass"
    # Empty / unknown falls back to a stable, letter-initial default.
    assert discretization_token(None) == "default"
    assert discretization_token("") == "default"


def test_select_forcing_files_picks_matching_discretization():
    """Each discretization selects only its own namespaced files."""
    from symfluence.core.modeling.model_ready.forcing_reader import select_forcing_files

    files = [
        "Bow_ERA5_remapped_lumped_2002-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_lumped_2003-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_elevation_2002-01-01-00-00-00.nc",
    ]
    lumped = [p.name for p in select_forcing_files(files, "lumped")]
    elevation = [p.name for p in select_forcing_files(files, "elevation")]
    assert lumped == [
        "Bow_ERA5_remapped_lumped_2002-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_lumped_2003-01-01-00-00-00.nc",
    ]
    assert elevation == ["Bow_ERA5_remapped_elevation_2002-01-01-00-00-00.nc"]


def test_select_forcing_files_falls_back_for_legacy_untokened_store():
    """A store predating namespacing (no token) is returned whole, not empty."""
    from symfluence.core.modeling.model_ready.forcing_reader import select_forcing_files

    legacy = [
        "Bow_ERA5_remapped_2002-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_2003-01-01-00-00-00.nc",
    ]
    # No file carries the 'lumped' token -> fall back to the full list so
    # single-discretization / pre-fix stores never regress or read as empty.
    assert [p.name for p in select_forcing_files(legacy, "lumped")] == legacy
    # A falsy discretization is an explicit "do not scope".
    assert [p.name for p in select_forcing_files(legacy, None)] == legacy


def test_open_canonical_forcing_selects_by_discretization(tmp_path):
    """Each model reads the forcing matching ITS discretization from one store.

    The store holds a lumped hru=1 forcing beside a 12-band elevation hru=12
    forcing (the domain_Bow_at_Banff_lumped_era5 case). With the run's
    discretization supplied, the reader must return only that discretization's
    data instead of colliding on ``conflicting dimension sizes: {1, 12}``.
    """
    lumped = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_lumped_2020-01.nc", 1)
    elevation = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_elevation_2020-01.nc", 12)
    store = [lumped, elevation]

    ds_lumped = open_canonical_forcing(store, discretization="lumped")
    assert ds_lumped.sizes["hru"] == 1

    ds_elev = open_canonical_forcing(store, discretization="elevation")
    assert ds_elev.sizes["hru"] == 12


def test_namespaced_discretizations_coexist_but_intra_namespace_collision_fails(tmp_path):
    """The write-boundary check allows distinct namespaces, rejects a real collision."""
    from symfluence.core.modeling.model_ready.forcing_reader import (
        assert_consistent_within_discretization,
    )

    lumped = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_lumped_2020-01.nc", 1)
    elevation = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_elevation_2020-01.nc", 12)
    # Distinct namespaces (hru=1 vs hru=12) coexist without error.
    assert_consistent_within_discretization([lumped, elevation])

    # A genuine collision WITHIN one namespace (same token, disagreeing sizes)
    # must still fail loudly.
    bad = _write_hru_forcing(tmp_path / "Bow_ERA5_remapped_lumped_2021-01.nc", 12)
    with pytest.raises(ValueError, match="#339"):
        assert_consistent_within_discretization([lumped, bad])


def test_store_builder_allows_namespaced_multi_discretization(tmp_path):
    """The store writer must PUBLISH a namespaced multi-discretization source set."""
    from symfluence.data.model_ready.forcings_builder import ForcingsStoreBuilder

    source = tmp_path / "domain_x" / "forcing" / "basin_averaged_data"
    source.mkdir(parents=True)
    _write_hru_forcing(source / "x_ERA5_remapped_lumped_2002.nc", 1)
    _write_hru_forcing(source / "x_ERA5_remapped_elevation_2002.nc", 12)

    builder = ForcingsStoreBuilder(
        project_dir=tmp_path / "domain_x",
        domain_name="x",
        forcing_dataset="ERA5",
        strategy="copy",  # symlinks may be unavailable in CI
    )
    target = builder.build()
    assert target is not None
    linked = {p.name for p in target.glob("*.nc")}
    assert linked == {"x_ERA5_remapped_lumped_2002.nc", "x_ERA5_remapped_elevation_2002.nc"}
    # And each model reads only its own discretization back out of the store.
    files = sorted(target.glob("*.nc"))
    assert open_canonical_forcing(files, discretization="lumped").sizes["hru"] == 1
    assert open_canonical_forcing(files, discretization="elevation").sizes["hru"] == 12
