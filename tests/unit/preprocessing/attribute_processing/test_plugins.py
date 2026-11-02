# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for external attribute-processor plugin discovery and dispatch."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.data.preprocessing.attribute_processor import attributeProcessor
from symfluence.data.preprocessing.attribute_processors import plugins
from symfluence.data.preprocessing.attribute_processors.base import BaseAttributeProcessor

pytestmark = [pytest.mark.unit, pytest.mark.quick]


class _GoodPlugin(BaseAttributeProcessor):
    """A valid plugin that bypasses the heavy base __init__ for testing."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def process(self):
        return {"climate.koppen_code": "Csb", "climate.holdridge_zone": "Boreal wet forest"}


class _BrokenPlugin(BaseAttributeProcessor):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def process(self):
        raise RuntimeError("boom")


class _NotAProcessor:
    """Does not subclass BaseAttributeProcessor; must be rejected by discovery."""


def _fake_ep(name, obj, raises=False):
    ep = MagicMock()
    ep.name = name
    ep.load.side_effect = RuntimeError("load failed") if raises else None
    if not raises:
        ep.load.return_value = obj
    return ep


# --- discover_attribute_plugins ---------------------------------------------

def test_discovery_keeps_only_valid_subclasses():
    eps = [
        _fake_ep("good", _GoodPlugin),
        _fake_ep("not_a_processor", _NotAProcessor),
        _fake_ep("explodes", None, raises=True),
    ]
    with patch.object(plugins.metadata, "entry_points", return_value=eps):
        found = plugins.discover_attribute_plugins(logging.getLogger("t"))
    assert [n for n, _ in found] == ["good"]
    assert found[0][1] is _GoodPlugin


# --- attributeProcessor._process_plugin_attributes --------------------------

def _stub_processor(enabled=True, exclude=None):
    """Minimal duck-typed host exposing just what the method touches."""
    stub = MagicMock(spec=attributeProcessor)
    stub.logger = logging.getLogger("t")
    stub._config = {}

    def _cfg(accessor, default=None, dict_key=None):
        if dict_key == "ATTRIBUTE_PLUGINS_ENABLED":
            return enabled
        if dict_key == "ATTRIBUTE_PLUGINS_EXCLUDE":
            return exclude if exclude is not None else []
        return default

    stub._get_config_value.side_effect = _cfg
    # Bind the real method under test to the stub.
    stub._process_plugin_attributes = attributeProcessor._process_plugin_attributes.__get__(stub)
    return stub


def test_plugins_aggregate_results():
    stub = _stub_processor()
    with patch.object(plugins, "discover_attribute_plugins", return_value=[("good", _GoodPlugin)]):
        out = stub._process_plugin_attributes()
    assert out["climate.koppen_code"] == "Csb"


def test_plugins_disabled_returns_empty():
    stub = _stub_processor(enabled=False)
    with patch.object(plugins, "discover_attribute_plugins", return_value=[("good", _GoodPlugin)]) as disc:
        out = stub._process_plugin_attributes()
    assert out == {}
    disc.assert_not_called()  # short-circuits before discovery


def test_plugins_exclude_by_name():
    stub = _stub_processor(exclude=["good"])
    with patch.object(plugins, "discover_attribute_plugins", return_value=[("good", _GoodPlugin)]):
        out = stub._process_plugin_attributes()
    assert out == {}


def test_plugin_failure_is_swallowed():
    stub = _stub_processor()
    order = [("broken", _BrokenPlugin), ("good", _GoodPlugin)]
    with patch.object(plugins, "discover_attribute_plugins", return_value=order):
        out = stub._process_plugin_attributes()
    # Broken plugin is skipped; the good one's attributes still land.
    assert out == {"climate.koppen_code": "Csb", "climate.holdridge_zone": "Boreal wet forest"}
