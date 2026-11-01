# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RECOGNIZED_FLAT_KEYS is a closed set.

The flat-key audit (docs/adr/config_flat_key_audit.md) promoted every
formerly-recognized key into a typed Pydantic field, an alias, or deleted
it as dead. The set now holds only deprecated keys consumed by
compatibility validators. These tests keep it that way: new config keys
must get Pydantic fields (or plugin-declared schemas, ADR-0002) so they
are typed, documented in the comprehensive template, and typo-checked —
adding to RECOGNIZED_FLAT_KEYS instead would bypass all three.
"""
from __future__ import annotations

# The only sanctioned entries: deprecated keys that a compatibility
# validator still consumes. Each must name its consumer and removal target.
SANCTIONED_DEPRECATED_KEYS = {
    # Migrated to NGEN_MODULES_SELECTED by NGENConfig._migrate_enable_flags;
    # remove at 2.0
    "ENABLE_NOAH",
    "ENABLE_PET",
    "ENABLE_SLOTH",
}


def test_recognized_flat_keys_is_closed():
    from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS

    unexpected = set(RECOGNIZED_FLAT_KEYS) - SANCTIONED_DEPRECATED_KEYS
    assert not unexpected, (
        f"New entries in RECOGNIZED_FLAT_KEYS: {sorted(unexpected)}. "
        "This set is closed — give the key a Pydantic field on the owning "
        "config model (it will be recognized via the transform map and must "
        "be documented in config_template_comprehensive.yaml), or declare it "
        "in a plugin schema. Only deprecated keys consumed by compatibility "
        "validators belong here (add them to SANCTIONED_DEPRECATED_KEYS with "
        "consumer + removal target)."
    )


def test_sanctioned_keys_still_have_consumers():
    """The deprecated toggles must keep their compatibility consumer until
    they are removed outright (don't let the validator disappear while the
    keys stay recognized)."""
    import inspect

    from symfluence.models.ngen.config_schema import NGENConfig

    source = inspect.getsource(NGENConfig)
    for key in ("ENABLE_NOAH", "ENABLE_PET", "ENABLE_SLOTH"):
        assert key in source, (
            f"{key} is in RECOGNIZED_FLAT_KEYS but NGENConfig no longer "
            "consumes it — remove it from the recognized set too."
        )
