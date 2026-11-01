"""The single typed-first config lookup used by every _get_config_value variant."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from symfluence.core.mixins.config import resolve_config_value

pytestmark = pytest.mark.unit


def test_typed_value_wins():
    cfg = SimpleNamespace(domain=SimpleNamespace(name="typed"), get=lambda k: "flat")
    assert resolve_config_value(cfg, lambda: cfg.domain.name, "d", "DOMAIN_NAME") == "typed"


def test_dict_config_falls_back_to_dict_key():
    cfg = {"DOMAIN_NAME": "flat"}
    assert resolve_config_value(cfg, lambda: cfg.domain.name, "d", "DOMAIN_NAME") == "flat"


def test_none_typed_value_falls_back():
    cfg = SimpleNamespace(domain=SimpleNamespace(name=None), get=lambda k: None)
    assert resolve_config_value(cfg, lambda: cfg.domain.name, "d", "DOMAIN_NAME") == "d"


def test_no_accessor_and_no_key_returns_default():
    assert resolve_config_value({}, None, 7) == 7
    assert resolve_config_value(None, lambda: None, 7, "X") == 7


def test_non_callable_accessor_is_ignored():
    assert resolve_config_value({"X": 1}, "not callable", 0, "X") == 1


@pytest.mark.parametrize("exc", [AttributeError, KeyError, TypeError])
def test_lookup_errors_fall_back(exc):
    def boom():
        raise exc("x")
    assert resolve_config_value({"X": 2}, boom, 0, "X") == 2


def test_other_errors_propagate():
    def boom():
        raise ValueError("real bug")
    with pytest.raises(ValueError):
        resolve_config_value({}, boom, 0, "X")
