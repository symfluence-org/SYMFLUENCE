# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Lapse-rate guard tests for the SUMMA forcing processor.

A forcing cell whose elevation failed to populate lands as 0.0 or -9999 in the
forcing<->catchment intersection. Left unguarded it drives a large fabricated
lapse-rate cooling (roughly -9 K for a ~1400 m site). These tests pin the guard
that falls such cells back to the catchment elevation (zero lapse).
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from symfluence.models.summa.forcing_processor import SummaForcingProcessor

LAPSE_RATE = 0.0065  # K/m


def _make_processor(tmp_path: Path) -> SummaForcingProcessor:
    config = SimpleNamespace(forcing=SimpleNamespace(lapse_rate=LAPSE_RATE))
    return SummaForcingProcessor(
        config=config,
        logger=logging.getLogger("test.summa.lapse"),
        forcing_basin_path=tmp_path,
        forcing_summa_path=tmp_path,
        intersect_path=tmp_path,
        catchment_path=tmp_path,
        project_dir=tmp_path,
        setup_dir=tmp_path,
        domain_name="test_domain",
        forcing_dataset="ERA5",
        data_step=3600,
        gruId="HRU_ID",
        hruId="HRU_ID",
        catchment_name="catchment.shp",
    )


def _topo(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.mark.parametrize("bad_forcing_elev", [0.0, -9999.0, float("nan"), 9000.0])
def test_invalid_forcing_elevation_yields_zero_lapse(tmp_path, bad_forcing_elev):
    """A cell with an unpopulated/implausible forcing elevation must not cool."""
    proc = _make_processor(tmp_path)
    topo = _topo([
        {"S_1_HRU_ID": 1, "S_1_elev_m": 1450.0, "S_2_elev_m": bad_forcing_elev, "weight": 1.0},
    ])
    lapse_values, lapse_rate = proc._precalculate_lapse_corrections(topo)
    assert lapse_rate == pytest.approx(LAPSE_RATE)
    # Guard -> forcing elevation collapses to the catchment elevation -> no bias.
    assert lapse_values.loc[1, "lapse_values"] == pytest.approx(0.0)


def test_valid_forcing_elevation_applies_real_lapse(tmp_path):
    """A cell with a good forcing elevation still gets the true lapse correction."""
    proc = _make_processor(tmp_path)
    topo = _topo([
        {"S_1_HRU_ID": 1, "S_1_elev_m": 1450.0, "S_2_elev_m": 1350.0, "weight": 1.0},
    ])
    lapse_values, _ = proc._precalculate_lapse_corrections(topo)
    # 1.0 * 0.0065 * (1350 - 1450) = -0.65 K
    assert lapse_values.loc[1, "lapse_values"] == pytest.approx(LAPSE_RATE * (1350.0 - 1450.0))


def test_mixed_cells_guard_only_the_invalid_one(tmp_path):
    """Weighted mix: the good cell contributes lapse, the zeroed cell contributes nothing."""
    proc = _make_processor(tmp_path)
    topo = _topo([
        {"S_1_HRU_ID": 1, "S_1_elev_m": 1450.0, "S_2_elev_m": 1350.0, "weight": 0.5},
        {"S_1_HRU_ID": 1, "S_1_elev_m": 1450.0, "S_2_elev_m": 0.0, "weight": 0.5},
    ])
    lapse_values, _ = proc._precalculate_lapse_corrections(topo)
    # Only the valid half-weight cell contributes: 0.5 * 0.0065 * (1350 - 1450).
    assert lapse_values.loc[1, "lapse_values"] == pytest.approx(0.5 * LAPSE_RATE * (1350.0 - 1450.0))


def test_hru_filter_uses_configured_id_column(tmp_path, monkeypatch):
    processor = _make_processor(tmp_path)
    processor.catchment_name = 'catchment.shp'
    monkeypatch.setattr('symfluence.models.summa.forcing_processor.gpd.read_file',
                        lambda path: pd.DataFrame({'HRU_ID': [1, 2]}))
    assert processor._filter_forcing_hru_ids([1, 3]) == [1]
