"""Every typed config path passed to ``_get_config_value`` must exist.

``_get_config_value(lambda: self.config.a.b, default)`` swallows the
``AttributeError`` a mistyped path raises and quietly returns the default (or
the ``dict_key`` fallback). A wrong field name therefore never fails -- the
user's setting is just ignored. E.g. ``paths.pour_point_shp_path`` (the field
is ``pour_point_path``) made ``POUR_POINT_SHP_PATH`` a no-op, and
``optimization.number_of_iterations`` pinned FUSE's MAXN to 100.

This test statically resolves each such lambda against ``SymfluenceConfig``.
Options that only exist as flat keys should use
``_get_config_value(lambda: None, default=..., dict_key='FLAT_KEY')``.
``model.*`` paths are skipped: model sections are resolved per model at
runtime.
"""
from __future__ import annotations

import ast
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from symfluence.core.config.models import SymfluenceConfig

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "src" / "symfluence"
_ACCESSORS = {"_get_config_value", "get_config_value"}


def _resolves(cls: type, parts: list[str]) -> bool:
    for part in parts:
        fields = getattr(cls, "model_fields", {})
        if part not in fields:
            return hasattr(cls, part)  # property / method
        annotation = fields[part].annotation
        nxt = next(
            (a for a in (annotation, *typing.get_args(annotation))
             if isinstance(a, type) and issubclass(a, BaseModel)),
            None,
        )
        if nxt is None:
            return True  # scalar / dict leaf
        cls = nxt
    return True


def _attr_chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return parts[::-1]


def _unresolvable_paths() -> list[str]:
    bad = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr in _ACCESSORS
                    and call.args and isinstance(call.args[0], ast.Lambda)):
                continue
            chain = _attr_chain(call.args[0].body)
            if "config" not in chain:
                continue
            cfg_path = chain[chain.index("config") + 1:]
            if not cfg_path or cfg_path[0] == "model":
                continue
            if not _resolves(SymfluenceConfig, cfg_path):
                rel = path.relative_to(SRC.parent)
                bad.append(f"{rel}:{call.lineno}: config.{'.'.join(cfg_path)}")
    return bad


def test_typed_config_paths_resolve():
    bad = _unresolvable_paths()
    assert not bad, (
        "Typed config paths that don't exist on SymfluenceConfig (they silently "
        "return the default). Fix the path, or use `lambda: None` with a "
        "dict_key for flat-only options:\n  " + "\n  ".join(bad)
    )


def test_checker_catches_a_bad_path():
    assert not _resolves(SymfluenceConfig, ["paths", "pour_point_shp_path"])
    assert _resolves(SymfluenceConfig, ["paths", "pour_point_path"])
