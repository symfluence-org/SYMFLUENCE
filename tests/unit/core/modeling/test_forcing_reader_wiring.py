# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model forcing readers select the run's HRU set (issue #339)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.modeling.model_ready.forcing_reader import open_canonical_forcing
from symfluence.core.modeling.utilities.time_window_manager import TimeWindowManager
from symfluence.models.summa.forcing_processor import SummaForcingProcessor


def _write(path: Path, hru: int, t0: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(t0, periods=4, freq="h")
    xr.Dataset(
        {"air_temperature": (("time", "hru"), np.full((4, hru), 280.0))},
        coords={"time": times, "hru": np.arange(hru)},
    ).to_netcdf(path)
    return path


def test_open_canonical_forcing_passes_geometry(tmp_path, monkeypatch):
    from symfluence.core.modeling import forcing_naming
    from symfluence.core.modeling.model_ready import forcing_reader

    a = _write(tmp_path / "d_ERA5_remapped_grus-h1-aaaaaaaa_2010.nc", 1, "2010-01-01")
    seen = {}

    def spy(files, discretization=None, geometry=None):
        seen["geometry"] = geometry
        return forcing_naming.select_forcing_files(files, discretization, geometry)

    monkeypatch.setattr(forcing_reader, "select_forcing_files", spy)
    geometry = forcing_naming.HruGeometry(count=1, digest="a" * 64)
    with open_canonical_forcing([a], "grus", geometry=geometry) as ds:
        assert ds.sizes["hru"] == 1
    assert seen["geometry"] == geometry


def test_time_window_reads_only_the_run_discretization(tmp_path):
    _write(tmp_path / "d_ERA5_remapped_lumped_2010.nc", 1, "2010-01-01")
    _write(tmp_path / "d_ERA5_remapped_elevation_2000.nc", 12, "2000-01-01")
    twm = TimeWindowManager({"SUB_GRID_DISCRETIZATION": "lumped"}, logging.getLogger("t"))
    start, _ = twm.get_forcing_time_range(tmp_path)
    assert start == pd.Timestamp("2010-01-01")


def test_summa_stale_removal_keeps_other_hru_sets(tmp_path):
    proc = SummaForcingProcessor.__new__(SummaForcingProcessor)
    proc.domain_name, proc.forcing_dataset = "d", "ERA5"
    proc.forcing_summa_path = tmp_path
    proc.logger = logging.getLogger("t")
    ours = tmp_path / "d_ERA5_remapped_grus-h1-aaaaaaaa_2010.nc"
    old_ours = tmp_path / "d_ERA5_remapped_grus-h1-aaaaaaaa_2009.nc"
    other = tmp_path / "d_ERA5_remapped_elevation-h12-bbbbbbbb_2010.nc"
    for f in (ours, old_ours, other):
        f.write_text("x")
    proc._remove_stale_forcing_files({ours.name})
    assert ours.exists() and other.exists() and not old_ours.exists()


def test_forcing_processor_selector_is_optional():
    proc = SummaForcingProcessor.__new__(SummaForcingProcessor)
    files = [Path("a.nc"), Path("b.nc")]
    assert proc._select_run_files(files) == files
    proc.forcing_selector = lambda fs: fs[:1]
    assert proc._select_run_files(files) == files[:1]
