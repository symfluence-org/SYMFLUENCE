"""Every typed config path passed to ``_get_config_value`` must exist.

``_get_config_value(lambda: self.config.a.b, default)`` (and
``resolve_config_value(config, lambda: config.a.b, ...)``) swallows the
``AttributeError`` a mistyped path raises and quietly returns the default (or
the ``dict_key`` fallback). A wrong field name therefore never fails -- the
user's setting is just ignored. E.g. ``paths.pour_point_shp_path`` (the field
is ``pour_point_path``) made ``POUR_POINT_SHP_PATH`` a no-op, and
``optimization.number_of_iterations`` pinned FUSE's MAXN to 100.

This test statically resolves each such lambda against ``SymfluenceConfig``.
Options that only exist as flat keys should use
``_get_config_value(lambda: None, default=..., dict_key='FLAT_KEY')``.
``model.<name>.*`` paths are checked against the per-model schema registered
in ``R.config_schemas`` -- the same registry ``ModelConfig.__getattr__`` uses
to resolve ``config.model.<name>`` at runtime.
"""
from __future__ import annotations

import ast
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.config.models.model_configs import ModelConfig
from symfluence.core.registries import R

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "src" / "symfluence"
_ACCESSORS = {"_get_config_value", "get_config_value", "resolve_config_value"}

def _resolves(cls: type, parts: list[str]) -> bool:
    for part in parts:
        fields = getattr(cls, "model_fields", {})
        if part not in fields and cls is ModelConfig:
            # ``config.model.<name>`` resolves through R.config_schemas
            schema = R.config_schemas.get(part.upper())
            if schema is None:
                return False
            cls = schema
            continue
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
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _config_paths(body: ast.AST) -> list[list[str]]:
    """Every ``config.a.b`` path read anywhere in a lambda body.

    Looking only at the body itself missed guarded reads such as
    ``lambda: self.config.model.troute.topology if self.config.model else None``
    (an ``IfExp``, not an ``Attribute``), which is how a non-existent
    ``troute.topology`` went unnoticed.
    """
    inner = {
        id(node.value) for node in ast.walk(body) if isinstance(node, ast.Attribute)
    }
    paths = []
    for node in ast.walk(body):
        # outermost attribute of each chain only (a.b.c, not its prefix a.b)
        if not isinstance(node, ast.Attribute) or id(node) in inner:
            continue
        chain = _attr_chain(node)
        if "config" not in chain:
            continue
        cfg_path = chain[chain.index("config") + 1:]
        if cfg_path and cfg_path not in paths:
            paths.append(cfg_path)
    return paths


def _unresolvable_paths() -> list[str]:
    bad = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _ACCESSORS:
                continue
            # _get_config_value(lambda, ...) / resolve_config_value(config, lambda, ...)
            lam = next((a for a in call.args[:2] if isinstance(a, ast.Lambda)), None)
            if lam is None:
                continue
            rel = path.relative_to(SRC.parent)
            for cfg_path in _config_paths(lam.body):
                if not _resolves(SymfluenceConfig, cfg_path):
                    bad.append(f"{rel}:{call.lineno}: config.{'.'.join(cfg_path)}")
    return bad


# Guarded reads (``... if self.config.model else None``) that the checker did
# not look inside until it learnt to. Each names an option with no schema field,
# so the setting is ignored on typed configs. Fix the read (add the field or use
# a flat ``dict_key``) and delete the entry; do not add to this list.
_KNOWN_UNRESOLVED = {
    "symfluence/models/fuse/calibration/optimizer.py: config.model.fuse.forcing_elevation",
    "symfluence/models/gnn/runner.py: config.model.gnn.checkpoint_interval",
    "symfluence/models/hype/preprocessor.py: config.model.hype.timeshift",
    "symfluence/models/hype/preprocessor.py: config.model.hype.frac_threshold",
}


def _without_line(entry: str) -> str:
    location, cfg_path = entry.rsplit(": ", 1)
    return f"{Path(location.rsplit(':', 1)[0]).as_posix()}: {cfg_path}"


def test_typed_config_paths_resolve():
    found = _unresolvable_paths()
    bad = [entry for entry in found if _without_line(entry) not in _KNOWN_UNRESOLVED]
    stale = _KNOWN_UNRESOLVED - {_without_line(entry) for entry in found}
    assert not stale, f"Fixed -- remove from _KNOWN_UNRESOLVED: {sorted(stale)}"
    assert not bad, (
        "Typed config paths that don't exist on SymfluenceConfig (they silently "
        "return the default). Fix the path, or use `lambda: None` with a "
        "dict_key for flat-only options:\n  " + "\n  ".join(bad)
    )


def test_checker_catches_a_bad_path():
    assert not _resolves(SymfluenceConfig, ["paths", "pour_point_shp_path"])
    assert _resolves(SymfluenceConfig, ["paths", "pour_point_path"])


def test_checker_sees_paths_inside_guarded_lambdas():
    lam = ast.parse(
        "lambda: self.config.model.troute.topology "
        "if self.config.model and self.config.model.troute else None"
    ).body[0].value
    assert _config_paths(lam.body) == [
        ["model", "troute", "topology"], ["model"], ["model", "troute"],
    ]
    plain = ast.parse("lambda: self.config.domain.name").body[0].value
    assert _config_paths(plain.body) == [["domain", "name"]]
    assert _config_paths(ast.parse("lambda: None").body[0].value.body) == []


def test_checker_resolves_model_sections_through_the_schema_registry():
    assert _resolves(SymfluenceConfig, ["model", "summa", "exe"])
    assert _resolves(SymfluenceConfig, ["model", "hydrological_model"])
    assert not _resolves(SymfluenceConfig, ["model", "summa", "no_such_field"])
    assert not _resolves(SymfluenceConfig, ["model", "no_such_model", "exe"])
