# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""EXPERIMENT_BACKUP_SETTINGS: unquoted YAML yes/no is a bool and used to fail validation."""

from __future__ import annotations

import pytest

from symfluence.models.summa.config_schema import SUMMAConfig

pytestmark = [pytest.mark.unit, pytest.mark.quick]


@pytest.mark.parametrize("value, expected", [
    (True, "yes"), (False, "no"), ("yes", "yes"), ("No", "no"), (" YES ", "yes"),
])
def test_backup_settings_accepts_yaml_bools(value, expected):
    assert SUMMAConfig(EXPERIMENT_BACKUP_SETTINGS=value).backup_settings == expected
