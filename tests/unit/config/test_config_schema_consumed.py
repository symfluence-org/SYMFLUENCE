# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Phase C guard 2 — every registered config schema is consumed (RTI item 26).

Phase B made ``R.config_schemas`` the authority for model-specific typed configs:
``ModelConfig.auto_populate_model_configs`` instantiates the registered schema for a
selected model, and ``config_resolution.get_config_schema`` resolves it. A schema can
be registered (via each model package's manifest/``register()`` or an external plugin's ``model_manifest(config_schema=...)``)
yet never actually applied — e.g. under a key the resolution path never queries. That
"accepted but unread" state is silent: the config validates, but the typed schema is
ignored.

This guard fails the build if any registered schema is not consumed by ``core/config``:
for every entry in ``R.config_schemas`` it asserts (a) the resolution helper can reach
it and (b) selecting that model yields a *typed* instance of exactly that schema.
"""

from __future__ import annotations

import pytest

import symfluence  # noqa: F401 - triggers bootstrap + config-schema registration
import symfluence.core.config.models.model_configs  # noqa: F401 - ensures bridge ran
from symfluence.core.config.config_resolution import get_config_schema
from symfluence.core.config.models.model_configs import ModelConfig
from symfluence.core.registries import R

REGISTERED_SCHEMAS = sorted(R.config_schemas.keys())


def test_some_schemas_registered():
    """Sanity: Phase B bridge populated the registry (guards against an empty sweep)."""
    assert len(REGISTERED_SCHEMAS) >= 20, (
        f"only {len(REGISTERED_SCHEMAS)} config schemas registered: {REGISTERED_SCHEMAS}"
    )


@pytest.mark.parametrize("name", REGISTERED_SCHEMAS)
def test_registered_schema_is_resolvable(name):
    """Every registered schema is reachable through the resolution helper."""
    assert get_config_schema(name) is not None, (
        f"{name} is in R.config_schemas but get_config_schema('{name}') returns None — "
        "registered but unreadable by core/config (RTI item 26)."
    )


@pytest.mark.parametrize("name", REGISTERED_SCHEMAS)
def test_registered_schema_is_applied(name):
    """Selecting a model yields a typed instance of its registered schema.

    Exercises the same path Phase B uses in production: ``auto_populate_model_configs``
    auto-creates ``schema_cls()`` for each selected model. If a registered schema is
    never applied here, it is accepted-but-unread.
    """
    expected = R.config_schemas.get(name)
    mc = ModelConfig.model_validate({"HYDROLOGICAL_MODEL": name})
    instance = mc.model_specific.get(name.lower())
    assert isinstance(instance, expected), (
        f"{name}: schema {expected!r} registered but auto_populate produced "
        f"{type(instance)!r} for config.model.{name.lower()} — accepted but unread "
        "(RTI item 26)."
    )
