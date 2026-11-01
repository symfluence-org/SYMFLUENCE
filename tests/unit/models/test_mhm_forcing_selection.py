# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Deterministic forcing-file selection for the mHM preprocessor.

The basin-averaged forcing store is shared across every model in a domain and can
accumulate more than one NetCDF: a different forcing dataset, a remap produced
under a different discretization (a 12-band elevation split alongside the 1-HRU
lumped basin), or a stale artefact from an earlier run. ``_load_forcing_data``
used to glob every ``*.nc`` and hand them all to ``open_canonical_forcing``,
which merges/aligns them -- a stray, incompatibly-shaped file then crashed on hru
alignment (the ``{1, 12}`` mismatch) and once triggered a silent
synthetic-forcing fallback.

These tests pin the replacement: select by (active forcing dataset, spatial mode)
using the framework's ``{domain}_{forcing_dataset}_remapped_*`` naming, merge
same-shape files (genuine time chunks), and raise a clear error -- never a blind
merge -- when incompatibly-shaped candidates genuinely remain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import FileOperationError
from symfluence.models.mhm.preprocessor import MHMPreProcessor


class _Sel(MHMPreProcessor):
    """Bypass __init__; exercise only the forcing-selection logic."""

    def __init__(self, store, *, domain="test_domain", dataset="ERA5", mode="lumped"):
        self.forcing_basin_path = store
        self.domain_name = domain          # mixin setter -> self._domain_name
        self.forcing_dataset = dataset     # mixin setter -> override
        self.spatial_mode = mode


def _write_forcing(path, hru, *, n=6, t0="2005-01-01"):
    """Write a minimal canonical forcing NetCDF with ``hru`` spatial points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(t0, periods=n, freq="D")
    xr.Dataset(
        {
            "precipitation_flux": (("time", "hru"), np.zeros((n, hru), dtype="f8")),
            "air_temperature": (("time", "hru"), np.full((n, hru), 283.15, dtype="f8")),
        },
        coords={"time": times, "hru": np.arange(hru)},
    ).to_netcdf(path)
    return path


def _names(paths):
    return sorted(p.name for p in paths)


def test_single_file_is_selected_without_inspection(tmp_path):
    """One file in the store -> returned as-is (no ambiguity to resolve)."""
    store = tmp_path / "store"
    _write_forcing(store / "test_domain_ERA5_remapped_CDS_2002_2009.nc", hru=1)
    h = _Sel(store)
    got = h._select_forcing_files(list(store.glob("*.nc")))
    assert _names(got) == ["test_domain_ERA5_remapped_CDS_2002_2009.nc"]


def test_stray_multi_hru_dropped_for_lumped(tmp_path):
    """The exact bug: lumped store holds hru=1 correct + hru=12 stray remap.

    Selection must return only the single-HRU file, not merge {1, 12}.
    """
    store = tmp_path / "store"
    _write_forcing(store / "test_domain_ERA5_remapped_CDS_2002_2009.nc", hru=1)
    _write_forcing(store / "test_domain_ERA5_remapped_4ae454551262b9b7.nc", hru=12)
    h = _Sel(store, mode="lumped")
    got = h._select_forcing_files(list(store.glob("*.nc")))
    assert _names(got) == ["test_domain_ERA5_remapped_CDS_2002_2009.nc"]


def test_wrong_forcing_dataset_excluded(tmp_path):
    """A stray RDRS remap must not be selected for an ERA5 run."""
    store = tmp_path / "store"
    _write_forcing(store / "test_domain_ERA5_remapped_CDS.nc", hru=1)
    _write_forcing(store / "test_domain_RDRS_remapped_abc.nc", hru=1)
    h = _Sel(store, dataset="ERA5")
    got = h._select_forcing_files(list(store.glob("*.nc")))
    assert _names(got) == ["test_domain_ERA5_remapped_CDS.nc"]


def test_incompatible_shapes_raise_instead_of_merging(tmp_path):
    """Distributed store with mixed hru shapes -> clear error, never a merge."""
    store = tmp_path / "store"
    _write_forcing(store / "test_domain_ERA5_remapped_a.nc", hru=1)
    _write_forcing(store / "test_domain_ERA5_remapped_b.nc", hru=12)
    h = _Sel(store, mode="distributed")
    with pytest.raises(FileOperationError, match="incompatible spatial shapes"):
        h._select_forcing_files(list(store.glob("*.nc")))


def test_same_shape_time_chunks_all_kept(tmp_path):
    """Genuine per-period chunks of one grid stay mergeable."""
    store = tmp_path / "store"
    _write_forcing(store / "test_domain_ERA5_remapped_2005.nc", hru=1, t0="2005-01-01")
    _write_forcing(store / "test_domain_ERA5_remapped_2006.nc", hru=1, t0="2006-01-01")
    h = _Sel(store, mode="lumped")
    got = h._select_forcing_files(list(store.glob("*.nc")))
    assert _names(got) == [
        "test_domain_ERA5_remapped_2005.nc",
        "test_domain_ERA5_remapped_2006.nc",
    ]


def test_distributed_same_multi_hru_chunks_all_kept(tmp_path):
    """Distributed domain with consistent hru=12 chunks -> all kept."""
    store = tmp_path / "store"
    _write_forcing(store / "test_domain_ERA5_remapped_2005.nc", hru=12, t0="2005-01-01")
    _write_forcing(store / "test_domain_ERA5_remapped_2006.nc", hru=12, t0="2006-01-01")
    h = _Sel(store, mode="distributed")
    got = h._select_forcing_files(list(store.glob("*.nc")))
    assert len(got) == 2


def test_selection_prevents_silent_shape_contamination(tmp_path):
    """End-to-end guard: a blind merge fails loudly; selection isolates the file.

    Historically, merging the whole store silently outer-joined to hru=12 with
    NaN-filled cells -- the quiet wrong-shape that fed fabricated weather
    downstream. The #339 read-boundary guard now rejects a mixed-discretization
    set outright, so the blind path fails loudly and actionably; per-model
    selection remains the way to obtain the right file from a shared store.
    """
    from symfluence.core.modeling.model_ready.forcing_reader import open_canonical_forcing

    store = tmp_path / "store"
    good = _write_forcing(store / "test_domain_ERA5_remapped_CDS.nc", hru=1)
    _write_forcing(store / "test_domain_ERA5_remapped_stray.nc", hru=12)

    # A blind merge of the whole mixed store is refused at the read boundary.
    with pytest.raises(ValueError, match="cross-discretization forcing collision"):
        open_canonical_forcing(sorted(store.glob("*.nc")))

    # Selection isolates the correct file, which then opens as a true lumped grid.
    h = _Sel(store, mode="lumped")
    selected = h._select_forcing_files(list(store.glob("*.nc")))
    assert selected == [good]
    ds = open_canonical_forcing(selected)
    assert ds.sizes.get("hru", 1) == 1
    ds.close()
