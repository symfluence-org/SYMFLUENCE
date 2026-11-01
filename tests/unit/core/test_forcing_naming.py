# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the core discretization-namespacing contract (issue #339).

The naming convention is shared across layers — ``data`` writes the namespaced
filenames, ``core.modeling`` and the model adapters select by them — so it lives
in ``core`` and must stay import-free of the data layer.
"""
from __future__ import annotations

from symfluence.core.modeling.forcing_naming import (
    discretization_key_from_name,
    discretization_token,
    forcing_name_matches_discretization,
    select_forcing_files,
)


def test_discretization_token_sanitizes():
    assert discretization_token("lumped") == "lumped"
    assert discretization_token("Elevation") == "elevation"
    assert discretization_token("elevation,landclass") == "elevation-landclass"
    assert discretization_token(None) == "default"
    assert discretization_token("") == "default"


def test_forcing_name_matches_discretization():
    assert forcing_name_matches_discretization("Bow_ERA5_remapped_lumped_2002-01-01.nc", "lumped")
    assert forcing_name_matches_discretization("Bow_ERA5_remapped_lumped.nc", "lumped")
    assert not forcing_name_matches_discretization("Bow_ERA5_remapped_elevation_2002.nc", "lumped")


def test_discretization_key_from_name_returns_none_for_legacy_names():
    assert discretization_key_from_name("Bow_ERA5_remapped_elevation_2002-01-01.nc") == "elevation"
    # A date-tag-only legacy remap carries no token; it must not be mistaken for one.
    assert discretization_key_from_name("Bow_ERA5_remapped_2002-01-01-00-00-00.nc") is None


def test_select_forcing_files_picks_matching_discretization():
    files = [
        "Bow_ERA5_remapped_lumped_2002-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_lumped_2003-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_elevation_2002-01-01-00-00-00.nc",
    ]
    assert [p.name for p in select_forcing_files(files, "lumped")] == files[:2]
    assert [p.name for p in select_forcing_files(files, "elevation")] == files[2:]


def test_select_forcing_files_falls_back_for_legacy_untokened_store():
    legacy = [
        "Bow_ERA5_remapped_2002-01-01-00-00-00.nc",
        "Bow_ERA5_remapped_2003-01-01-00-00-00.nc",
    ]
    assert [p.name for p in select_forcing_files(legacy, "lumped")] == legacy
    assert [p.name for p in select_forcing_files(legacy, None)] == legacy
