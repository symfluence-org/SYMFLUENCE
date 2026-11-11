# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for HYPE elevation-banded sub-basin expansion.

Covers ``HYPEGeoDataManager._build_banded_geodata`` — the pure DataFrame
transform that splits each sub-basin into a vertical cascade of elevation-band
sub-basins (semi-distributed HYPE). No file I/O or model execution.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from symfluence.models.hype.geodata_manager import HYPEGeoDataManager


@pytest.fixture
def manager(tmp_path):
    return HYPEGeoDataManager({}, logging.getLogger("test"), tmp_path, {})


@pytest.fixture
def lumped_basin():
    """Single outlet sub-basin (id 100), 1000 area units, 2% glacier, 2 SLCs."""
    return pd.DataFrame([{
        "subid": 100, "maindown": 0, "grwdown": 0, "rivlen": 5000.0,
        "slope_mean": 0.01, "area": 1000.0, "latitude": 51.0,
        "longitude": -115.0, "elev_mean": 2000.0, "glacier_fraction": 0.02,
        "SLC_1": 0.6, "SLC_2": 0.4,
    }])


def _three_bands():
    # Intentionally unsorted to exercise low->high ordering.
    return {100: [
        {"elev_mean": 2600, "area_frac": 0.2},
        {"elev_mean": 1800, "area_frac": 0.5},
        {"elev_mean": 2200, "area_frac": 0.3},
    ]}


def test_expands_to_one_row_per_band(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    assert len(out) == 3


def test_band_ids_unique_and_positive(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    assert out["subid"].is_unique
    assert (out["subid"] > 0).all()
    # parent*100 + 1..3
    assert sorted(out["subid"]) == [10001, 10002, 10003]


def test_single_outlet_and_vertical_cascade(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    # Exactly one outlet (the lowest band drains out of the domain).
    assert (out["maindown"] == 0).sum() == 1
    by_elev = out.sort_values("elev_mean").reset_index(drop=True)
    # Lowest band -> outlet; each higher band -> the band directly below it.
    assert by_elev.loc[0, "maindown"] == 0
    assert by_elev.loc[1, "maindown"] == by_elev.loc[0, "subid"]
    assert by_elev.loc[2, "maindown"] == by_elev.loc[1, "subid"]
    # grwdown mirrors maindown.
    assert (out["grwdown"] == out["maindown"]).all()


def test_area_conserved_and_split_by_fraction(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    assert out["area"].sum() == pytest.approx(1000.0)
    by_elev = out.sort_values("elev_mean")
    assert list(by_elev["area"]) == pytest.approx([500.0, 300.0, 200.0])


def test_elevation_increases_with_band(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    by_id = out.sort_values("subid")
    assert list(by_id["elev_mean"]) == [1800, 2200, 2600]


def test_glacier_conserved_and_concentrated_in_top_band(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    # Total glacier area preserved: 2% of 1000 = 20.
    assert (out["area"] * out["glacier_fraction"]).sum() == pytest.approx(20.0)
    by_elev = out.sort_values("elev_mean").reset_index(drop=True)
    # All glacier sits in the highest band (200 area, 20 glacier -> 0.1).
    assert by_elev.loc[0, "glacier_fraction"] == 0.0
    assert by_elev.loc[1, "glacier_fraction"] == 0.0
    assert by_elev.loc[2, "glacier_fraction"] == pytest.approx(0.1)


def test_slc_fractions_inherited(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    assert out["SLC_1"].unique().tolist() == [0.6]
    assert out["SLC_2"].unique().tolist() == [0.4]


def test_channel_length_only_on_valley_band(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands())
    by_elev = out.sort_values("elev_mean").reset_index(drop=True)
    assert by_elev.loc[0, "rivlen"] == 5000.0          # valley keeps the reach
    assert (by_elev.loc[1:, "rivlen"] == 100.0).all()  # internal vertical links


def test_no_band_info_passes_through_unchanged(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, {})
    pd.testing.assert_frame_equal(out, lumped_basin.reset_index(drop=True))


def test_too_many_bands_keeps_subbasin_lumped(manager, lumped_basin):
    # >= multiplier bands would collide with id scheme -> keep lumped.
    bands = {100: [{"elev_mean": 1000 + i, "area_frac": 1.0}
                   for i in range(manager._BAND_ID_MULTIPLIER)]}
    out = manager._build_banded_geodata(lumped_basin, bands)
    assert len(out) == 1
    assert out.loc[0, "subid"] == 100


def test_glacier_spills_into_second_band_when_top_too_small(manager, lumped_basin):
    # Top band area (10) < glacier area (20) -> remainder spills to next band.
    bands = {100: [
        {"elev_mean": 1800, "area_frac": 0.90},
        {"elev_mean": 2600, "area_frac": 0.01},  # 10 area units
        {"elev_mean": 2200, "area_frac": 0.09},  # 90 area units
    ]}
    out = manager._build_banded_geodata(lumped_basin, bands)
    assert (out["area"] * out["glacier_fraction"]).sum() == pytest.approx(20.0)
    by_elev = out.sort_values("elev_mean").reset_index(drop=True)
    # Highest band fully glaciated; spill into the middle band; valley clean.
    assert by_elev.loc[2, "glacier_fraction"] == pytest.approx(1.0)   # 10/10
    assert by_elev.loc[1, "glacier_fraction"] == pytest.approx(10.0 / 90.0)
    assert by_elev.loc[0, "glacier_fraction"] == 0.0


def _three_bands_with_hru_id():
    # hru_id present -> bands keyed by hru_id (matches forcing columns).
    return {100: [
        {"hru_id": 3, "elev_mean": 2600, "area_frac": 0.2},
        {"hru_id": 1, "elev_mean": 1800, "area_frac": 0.5},
        {"hru_id": 2, "elev_mean": 2200, "area_frac": 0.3},
    ]}


def test_hru_id_used_as_subid(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands_with_hru_id())
    # sub-basin ids are the HRU ids, not parent*100+j
    assert sorted(out["subid"]) == [1, 2, 3]


def test_hru_id_cascade_by_elevation(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands_with_hru_id())
    by_elev = out.sort_values("elev_mean").reset_index(drop=True)
    # lowest elev (hru 1) is the outlet; each higher band drains to the next lower
    assert by_elev.loc[0, "subid"] == 1 and by_elev.loc[0, "maindown"] == 0
    assert by_elev.loc[1, "subid"] == 2 and by_elev.loc[1, "maindown"] == 1
    assert by_elev.loc[2, "subid"] == 3 and by_elev.loc[2, "maindown"] == 2
    # exactly one outlet
    assert (out["maindown"] == 0).sum() == 1


def test_hru_id_conserves_area_and_glacier(manager, lumped_basin):
    out = manager._build_banded_geodata(lumped_basin, _three_bands_with_hru_id())
    assert out["area"].sum() == pytest.approx(1000.0)
    assert (out["area"] * out["glacier_fraction"]).sum() == pytest.approx(20.0)


def test_multi_subbasin_drains_to_downstream_valley_band(manager):
    """An upstream sub-basin's valley band drains to the downstream band-1."""
    base = pd.DataFrame([
        {"subid": 1, "maindown": 2, "grwdown": 2, "rivlen": 100.0,
         "slope_mean": 0.01, "area": 100.0, "latitude": 51.0,
         "longitude": -115.0, "elev_mean": 2000.0, "glacier_fraction": 0.0,
         "SLC_1": 1.0},
        {"subid": 2, "maindown": 0, "grwdown": 0, "rivlen": 200.0,
         "slope_mean": 0.01, "area": 200.0, "latitude": 51.0,
         "longitude": -115.0, "elev_mean": 1500.0, "glacier_fraction": 0.0,
         "SLC_1": 1.0},
    ])
    bands = {
        1: [{"elev_mean": 1900, "area_frac": 0.5}, {"elev_mean": 2100, "area_frac": 0.5}],
        2: [{"elev_mean": 1400, "area_frac": 0.5}, {"elev_mean": 1600, "area_frac": 0.5}],
    }
    out = manager._build_banded_geodata(base, bands)
    # Sub-basin 1's lowest band (101) should drain to sub-basin 2's band-1 (201).
    valley_1 = out[out["subid"] == 101].iloc[0]
    assert valley_1["maindown"] == 201
    # Sub-basin 2's lowest band (201) is the true outlet.
    valley_2 = out[out["subid"] == 201].iloc[0]
    assert valley_2["maindown"] == 0

def _two_parents(n_bands_downstream=0):
    base = pd.DataFrame([
        {"subid": 10, "maindown": 20, "grwdown": 20, "rivlen": 100.0,
         "slope_mean": 0.01, "area": 100.0, "latitude": 51.0,
         "longitude": -115.0, "elev_mean": 2000.0, "glacier_fraction": 0.0,
         "SLC_1": 1.0},
        {"subid": 20, "maindown": 0, "grwdown": 0, "rivlen": 200.0,
         "slope_mean": 0.01, "area": 200.0, "latitude": 51.0,
         "longitude": -115.0, "elev_mean": 1500.0, "glacier_fraction": 0.0,
         "SLC_1": 1.0},
    ])
    bands = {10: [{"elev_mean": 1900, "area_frac": 0.5}, {"elev_mean": 2100, "area_frac": 0.5}]}
    if n_bands_downstream:
        bands[20] = [{"elev_mean": 1000 + k, "area_frac": 1.0} for k in range(n_bands_downstream)]
    return base, bands


def test_valley_band_drains_to_unbanded_downstream_parent(manager):
    """Regression: with no bands for the downstream parent the upstream valley
    band got maindown 0 and was disconnected from the network."""
    base, bands = _two_parents()
    out = manager._build_banded_geodata(base, bands).set_index("subid")
    assert out.loc[1001, "maindown"] == 20 and out.loc[1001, "grwdown"] == 20
    assert out.loc[1002, "maindown"] == 1001
    assert out.loc[20, "maindown"] == 0
    assert set(out["maindown"]) - {0} <= set(out.index)


def test_valley_band_drains_to_downstream_parent_kept_lumped(manager):
    """Regression: a parent with too many bands is kept lumped, yet upstream
    bands were routed to its (non-existent) first band id."""
    base, bands = _two_parents(n_bands_downstream=manager._BAND_ID_MULTIPLIER)
    out = manager._build_banded_geodata(base, bands).set_index("subid")
    assert 20 in out.index and len(out) == 3
    assert out.loc[1001, "maindown"] == 20
    assert set(out["maindown"]) - {0} <= set(out.index)


def test_unbanded_upstream_parent_drains_to_downstream_valley_band(manager):
    base, bands = _two_parents()
    bands = {20: [{"elev_mean": 1400, "area_frac": 0.5}, {"elev_mean": 1600, "area_frac": 0.5}]}
    out = manager._build_banded_geodata(base, bands).set_index("subid")
    assert out.loc[10, "maindown"] == 2001 and out.loc[10, "grwdown"] == 2001
    assert set(out["maindown"]) - {0} <= set(out.index)


def test_valley_band_of_missing_downstream_parent_is_an_outlet(manager):
    base, bands = _two_parents()
    base = base[base["subid"] == 10]
    out = manager._build_banded_geodata(base, bands).set_index("subid")
    assert out.loc[1001, "maindown"] == 0


def test_id_shift_moves_groundwater_target_with_the_subids(manager):
    """Regression: the 0-based -> 1-based shift updated subid and maindown but
    left grwdown, so each sub-basin's groundwater target was itself."""
    df = pd.DataFrame({"subid": [0, 1, 2], "maindown": [1, 2, -1]})
    df["grwdown"] = df["maindown"]
    out = manager._shift_ids_if_needed(df)
    assert out["subid"].tolist() == [1, 2, 3]
    assert out["maindown"].tolist() == [2, 3, 0]
    assert out["grwdown"].tolist() == [2, 3, 0]


def test_one_based_ids_are_not_shifted(manager):
    df = pd.DataFrame({"subid": [1, 2], "maindown": [2, 0], "grwdown": [2, 0]})
    assert manager._shift_ids_if_needed(df.copy()).equals(df)


# ── Forcing-side band expansion (lapse) ──────────────────────────────────────

@pytest.fixture
def forcing_processor(tmp_path):
    from symfluence.models.hype.forcing_processor import HYPEForcingProcessor
    fp = HYPEForcingProcessor(
        config={}, logger=logging.getLogger("test"),
        forcing_input_dir=tmp_path, output_path=tmp_path, cache_path=tmp_path,
    )
    # three bands: 1000 m (low), 2000 m (ref-ish), 3000 m (high), equal area
    fp.set_elevation_bands([
        {"hru_id": 1, "elev_mean": 1000.0, "area": 1.0},
        {"hru_id": 2, "elev_mean": 2000.0, "area": 1.0},
        {"hru_id": 3, "elev_mean": 3000.0, "area": 1.0},
    ], lapse_rate=0.0065)
    return fp


def _one_col_df():
    idx = pd.date_range("2002-01-01", periods=4, freq="D")
    return pd.DataFrame({1: [10.0, 10.0, 10.0, 10.0]}, index=idx)


def test_temperature_expands_with_lapse(forcing_processor):
    out = forcing_processor._expand_columns_to_bands(_one_col_df(), "Tobs")
    # 3 columns keyed by hru_id
    assert list(out.columns) == [1, 2, 3]
    # ref_elev = 2000 (equal-area mean). Low band warmer, high band colder.
    assert out[1].iloc[0] == pytest.approx(10.0 + 0.0065 * (2000 - 1000))  # +6.5
    assert out[2].iloc[0] == pytest.approx(10.0)                            # at ref
    assert out[3].iloc[0] == pytest.approx(10.0 + 0.0065 * (2000 - 3000))  # -6.5


def test_tmax_tmin_also_lapsed(forcing_processor):
    for var in ("TMAXobs", "TMINobs"):
        out = forcing_processor._expand_columns_to_bands(_one_col_df(), var)
        assert out[1].iloc[0] > out[2].iloc[0] > out[3].iloc[0]


def test_precip_replicated_unchanged(forcing_processor):
    out = forcing_processor._expand_columns_to_bands(_one_col_df(), "Pobs")
    assert list(out.columns) == [1, 2, 3]
    for c in (1, 2, 3):
        assert (out[c].to_numpy() == 10.0).all()


def test_no_bands_returns_input_unchanged(tmp_path):
    from symfluence.models.hype.forcing_processor import HYPEForcingProcessor
    fp = HYPEForcingProcessor(
        config={}, logger=logging.getLogger("test"),
        forcing_input_dir=tmp_path, output_path=tmp_path, cache_path=tmp_path,
    )
    # No bands set -> expansion is a no-op guarded at the call site; the helper
    # with <2 bands returns the frame as-is.
    fp.set_elevation_bands([{"hru_id": 1, "elev_mean": 1000.0, "area": 1.0}])
    df = _one_col_df()
    out = fp._expand_columns_to_bands(df, "Tobs")
    pd.testing.assert_frame_equal(out, df)
