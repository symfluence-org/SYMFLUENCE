# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The shared forcing reader must not double a series from overlapping files.

A store that holds more than one file for the same period — a duplicate or
stray remap left under the shared ``{domain}_{forcing}_remapped_*`` namespace
— merges into a series with duplicate timestamps. Left unhandled that either
raises on a non-unique time index or silently feeds a 2x-length, garbled
forcing to the model (observed: 140254 = 2x70127 steps collapsed a TOPMODEL
calibration to KGE -20). Generalizes the per-model mHM fix (#338) to the
reader every model shares.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.modeling.model_ready.forcing_reader import _dedupe_forcing_time


def _ds(times, values):
    return xr.Dataset({"pr": ("time", list(values))}, coords={"time": list(times)})


def test_overlapping_files_are_deduped_keeping_first():
    t = pd.date_range("2002-01-01", periods=5, freq="D")
    # two identical periods concatenated (the duplicate-remap case)
    ds = _ds(list(t) + list(t), list(range(5)) + [99, 98, 97, 96, 95])
    out = _dedupe_forcing_time(ds, [Path("a_CDS.nc"), Path("a_4ae.nc")])
    assert out.sizes["time"] == 5
    assert len(np.unique(out["time"].values)) == 5
    assert list(out["pr"].values) == [0, 1, 2, 3, 4]  # first occurrence wins


def test_nonoverlapping_chunks_pass_through_unchanged():
    ds = _ds(pd.date_range("2002-01-01", periods=6, freq="D"), range(6))
    out = _dedupe_forcing_time(ds, [Path("y2002.nc"), Path("y2003.nc")])
    assert out.sizes["time"] == 6
    assert list(out["pr"].values) == [0, 1, 2, 3, 4, 5]


def test_dedupe_warns_only_on_duplicates(caplog):
    t = pd.date_range("2002-01-01", periods=3, freq="D")
    with caplog.at_level(logging.WARNING):
        _dedupe_forcing_time(_ds(list(t) + list(t[:1]), [0, 1, 2, 9]),
                             [Path("a.nc"), Path("b.nc")])
    assert any("duplicate timestep" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _dedupe_forcing_time(_ds(t, [0, 1, 2]), [Path("a.nc")])
    assert not caplog.records


def test_no_time_dim_is_a_noop():
    ds = xr.Dataset({"x": ("hru", [1, 2, 3])})
    assert _dedupe_forcing_time(ds, [Path("a.nc")]) is ds
