# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Round-7 audit regressions: SLC soil-0 duplicates, outlet coding, subid map."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from symfluence.models.hype.geodata_manager import (
    GROUNDWATER_OUT_OF_DOMAIN_SUBID,
    HYPEGeoDataManager,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


@pytest.fixture
def manager(tmp_path):
    return HYPEGeoDataManager({}, logging.getLogger("hype-r7"), tmp_path, {})


def test_soil_zero_does_not_duplicate_slc_classes(manager):
    """(lc, 0) and (lc, 1) used to become two SLC rows that both matched the basin."""
    base = pd.DataFrame({"subid": [1, 2], "maindown": [2, 0]})
    landcover = pd.DataFrame({"IGBP_10": [0.5, 1.0], "IGBP_12": [0.5, 0.0]}, index=[1, 2])
    soil = pd.DataFrame({"majority": [0, 1]}, index=[1, 2])

    slc, out = manager._process_slc(base.copy(), landcover, soil, 0.05)

    assert not slc.duplicated(["landcover", "soil"]).any()
    assert set(slc["soil"]) == {1}
    cols = [c for c in out if c.startswith("SLC_")]
    fractions = out[cols].div(out[cols].sum(axis=1), axis=0)
    row = fractions.iloc[0]
    assert sorted(row[row > 0].round(6)) == [0.5, 0.5]


def test_shift_keeps_outlets_as_outlets(manager):
    """0-based ids: a self-reference or the -1 marker is an outlet, not a link."""
    df = pd.DataFrame({"subid": [0, 1, 2], "maindown": [0, 0, 1], "grwdown": [0, 0, 1]})
    out = manager._shift_ids_if_needed(df)
    assert list(out["subid"]) == [1, 2, 3]
    assert list(out["maindown"]) == [0, 1, 2]  # basin 0 is the outlet; 1 -> 0; 2 -> 1
    assert not (out["maindown"] == out["subid"]).any()

    df = pd.DataFrame({"subid": [0, 1], "maindown": [manager._OUTLET, 0], "grwdown": [-1, 0]})
    out = manager._shift_ids_if_needed(df)
    assert list(out["maindown"]) == [0, 1]


def test_out_of_domain_groundwater_code_survives(manager):
    df = pd.DataFrame({
        "subid": [0, 1],
        "maindown": [1, -1],
        "grwdown": [GROUNDWATER_OUT_OF_DOMAIN_SUBID, 0],
    })
    out = manager._shift_ids_if_needed(df)
    assert out.loc[0, "grwdown"] == GROUNDWATER_OUT_OF_DOMAIN_SUBID
    assert out.loc[1, "grwdown"] == 1


def test_subid_map_points_at_each_original_ids_outlet(manager, tmp_path):
    # Parent 3 split into bands 301 (valley) <- 302 (upper); parent 5 unbanded.
    final = pd.DataFrame({
        "subid": [301, 302, 5],
        "maindown": [5, 301, 0],
    })
    manager._write_subid_map(pd.Series([3, 3, 5]), final)
    mapping = pd.read_csv(tmp_path / HYPEGeoDataManager.SUBID_MAP_FILE)
    assert dict(zip(mapping["original_id"], mapping["hype_subid"])) == {3: 301, 5: 5}


def test_postprocessor_translates_sim_reach_id(tmp_path):
    from symfluence.models.hype.postprocessor import HYPEPostProcessor

    (tmp_path / "settings" / "HYPE").mkdir(parents=True)
    pd.DataFrame({"original_id": [0, 1], "hype_subid": [1, 2]}).to_csv(
        tmp_path / "settings" / "HYPE" / "subid_map.csv", index=False)

    pp = HYPEPostProcessor.__new__(HYPEPostProcessor)
    pp.project_dir = tmp_path
    pp.logger = logging.getLogger("hype-r7")
    pp._get_config_value = lambda accessor=None, default=None, dict_key=None: 1
    df = pd.DataFrame({"DATE": ["2000-01-01"], "1": [5.0], "2": [9.0]})
    # Original id 1 is HYPE subid 2; the untranslated "1" is a different basin.
    assert pp._get_flow_column(df) == "2"
