# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Contract tests for core-owned forcing artifact selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.core.modeling.forcing_selection import (
    discretization_token,
    select_forcing_files,
)

pytestmark = [pytest.mark.unit]


def test_discretization_token_is_stable():
    assert discretization_token('Elevation, Landclass') == 'elevation-landclass'
    assert discretization_token(None) == 'default'


def test_selection_prefers_matching_namespace_and_preserves_legacy_fallback():
    lumped = Path('domain_era5_remapped_lumped_2000.nc')
    elevation = Path('domain_era5_remapped_elevation_2000.nc')
    files = [elevation, lumped]

    assert select_forcing_files(files, 'lumped') == [lumped]


def test_selection_raises_when_namespaced_store_lacks_the_discretization():
    from symfluence.core.exceptions import FileOperationError

    files = [Path('domain_era5_remapped_elevation_2000.nc'),
             Path('domain_era5_remapped_lumped_2000.nc')]
    with pytest.raises(FileOperationError, match="soilclass"):
        select_forcing_files(files, 'soilclass')


def test_selection_falls_back_only_for_untokened_legacy_store():
    legacy = [Path('domain_era5_remapped_2000-01-01-00-00-00.nc'),
              Path('domain_era5_remapped_2000-02-01-00-00-00.nc')]
    assert select_forcing_files(legacy, 'soilclass') == legacy
