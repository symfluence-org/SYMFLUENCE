# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for config coercion helpers (review item 17 — core/config/coercion.py).

These convert dict/SymfluenceConfig inputs at the ~60 call sites that accept either,
and were previously untested.
"""

from __future__ import annotations

import pytest

from symfluence.core.config.coercion import coerce_config, ensure_config
from symfluence.core.config.models import SymfluenceConfig

from .conftest import build_minimal_config

# ---- ensure_config -------------------------------------------------------


def test_ensure_config_passes_through_typed(minimal_config):
    assert ensure_config(minimal_config) is minimal_config


def test_ensure_config_coerces_valid_dict(minimal_flat_config):
    cfg = ensure_config(minimal_flat_config)
    assert isinstance(cfg, SymfluenceConfig)
    assert cfg.domain.name == "test_basin"


def test_ensure_config_rejects_non_mapping():
    with pytest.raises(TypeError):
        ensure_config(42)  # type: ignore[arg-type]


def test_ensure_config_raises_on_invalid_dict():
    # A dict missing required fields is still a dict, so ensure_config attempts
    # construction and surfaces the validation error (not a silent fallback).
    with pytest.raises(Exception):  # pydantic ValidationError (ValueError subclass)
        ensure_config({"DOMAIN_NAME": "x"})


# ---- coerce_config -------------------------------------------------------


def test_coerce_config_passes_through_typed(minimal_config):
    assert coerce_config(minimal_config) is minimal_config


def test_coerce_config_coerces_valid_dict(minimal_flat_config):
    cfg = coerce_config(minimal_flat_config)
    assert isinstance(cfg, SymfluenceConfig)


def test_coerce_config_partial_falls_back_with_warning():
    partial = {"DOMAIN_NAME": "x"}
    with pytest.warns(DeprecationWarning):
        result = coerce_config(partial, warn=True)
    assert result is partial  # original dict returned unchanged


def test_coerce_config_partial_no_warn_is_silent(recwarn):
    partial = {"DOMAIN_NAME": "x"}
    result = coerce_config(partial, warn=False)
    assert result is partial
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_coerce_config_strict_raises():
    with pytest.raises((TypeError, ValueError)):
        coerce_config({"DOMAIN_NAME": "x"}, strict=True)


def test_coerce_config_strict_via_env(monkeypatch):
    monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", "true")
    with pytest.raises((TypeError, ValueError)):
        coerce_config({"DOMAIN_NAME": "x"})  # strict resolved from env


def test_coerce_config_env_false_allows_fallback(monkeypatch):
    monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", "false")
    result = coerce_config({"DOMAIN_NAME": "x"}, warn=False)
    assert result == {"DOMAIN_NAME": "x"}


def test_round_trip_flatten_then_coerce(minimal_config):
    """A config flattened and re-coerced preserves key fields."""
    from symfluence.core.config.flattening import flatten_nested_config

    flat = flatten_nested_config(build_minimal_config())
    cfg2 = ensure_config(flat)
    assert cfg2.domain.name == minimal_config.domain.name
    assert cfg2.model.hydrological_model == minimal_config.model.hydrological_model
