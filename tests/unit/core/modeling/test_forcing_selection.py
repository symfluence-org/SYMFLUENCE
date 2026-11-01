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
    assert select_forcing_files(files, 'soilclass') == files
