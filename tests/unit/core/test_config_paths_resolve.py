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

# Pre-existing model paths with no schema field (and, mostly, no dict_key), so
# they always return their default. Each needs a schema field or a dict_key
# decision, which is a behaviour change, so they are listed rather than fixed
# here. Keyed by (file, typed path); remove an entry when you fix it.
KNOWN_UNRESOLVED = {
    # option has no field and no dict_key: always the default
    ("symfluence/core/modeling/utilities/base_topology_generator.py", "model.mizuroute.grid_cell_size"),
    ("symfluence/models/mizuroute/topology_generator.py", "model.mizuroute.grid_cell_size"),
    ("symfluence/models/lstm/preprocessor.py", "model.flash.lookback"),  # legacy FLASH name
    ("symfluence/models/ngen/config_generator.py", "model.ngen.forcing_provider"),
    ("symfluence/models/rhessys/runner.py", "model.rhessys.std_scale"),
    ("symfluence/models/rhessys/runner.py", "model.rhessys.use_subsurface_gw"),
    ("symfluence/models/wflow/preprocessor.py", "model.wflow.soil_layer_thickness"),
    # probably meant ``observed_perimeter_path`` (IGNACIO_OBSERVED_PERIMETER_PATH)
    ("symfluence/models/ignacio/postprocessor.py", "model.ignacio.observed_perimeter"),
    # models/mhm/ is excluded from the config-access codemod (in-flight work)
    ("symfluence/models/mhm/calibration/optimizer.py", "model.mhm.routing_integration"),
}


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
            chain = _attr_chain(lam.body)
            if "config" not in chain:
                continue
            cfg_path = chain[chain.index("config") + 1:]
            if not cfg_path:
                continue
            rel = path.relative_to(SRC.parent)
            if (rel.as_posix(), ".".join(cfg_path)) in KNOWN_UNRESOLVED:
                continue
            if not _resolves(SymfluenceConfig, cfg_path):
                bad.append(f"{rel}:{call.lineno}: config.{'.'.join(cfg_path)}")
    return bad


def test_typed_config_paths_resolve():
    bad = _unresolvable_paths()
    assert not bad, (
        "Typed config paths that don't exist on SymfluenceConfig (they silently "
        "return the default). Fix the path, or use `lambda: None` with a "
        "dict_key for flat-only options:\n  " + "\n  ".join(bad)
    )


def test_known_unresolved_is_not_stale():
    still_bad = set()
    for rel, dotted in KNOWN_UNRESOLVED:
        used = f"config.{dotted}" in (SRC.parent / rel).read_text(encoding="utf-8")
        if used and not _resolves(SymfluenceConfig, dotted.split(".")):
            still_bad.add((rel, dotted))
    assert still_bad == KNOWN_UNRESOLVED, (
        "These are fixed or gone -- drop them from KNOWN_UNRESOLVED: "
        f"{sorted(KNOWN_UNRESOLVED - still_bad)}"
    )


def test_checker_catches_a_bad_path():
    assert not _resolves(SymfluenceConfig, ["paths", "pour_point_shp_path"])
    assert _resolves(SymfluenceConfig, ["paths", "pour_point_path"])


def test_checker_resolves_model_sections_through_the_schema_registry():
    assert _resolves(SymfluenceConfig, ["model", "summa", "exe"])
    assert _resolves(SymfluenceConfig, ["model", "hydrological_model"])
    assert not _resolves(SymfluenceConfig, ["model", "summa", "no_such_field"])
    assert not _resolves(SymfluenceConfig, ["model", "no_such_model", "exe"])
