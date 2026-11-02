"""A dict_key must be the flat key its typed path flattens to.

``_get_config_value(lambda: self.config.a.b, default, dict_key='KEY')`` falls
back to ``config.get('KEY')`` when config is a flat dict -- which is what
calibration workers receive (built by ``to_dict(flatten=True)``). If ``KEY``
isn't the canonical flat name of ``a.b``, workers silently get the default:
e.g. ``dict_key='MPI_PROCESSES'`` for ``system.num_processes`` (flattened as
``NUM_PROCESSES``) pinned WRF-Hydro to 1 process in workers, and
``OPTIMIZATION_POPULATION_SIZE`` ignored ``POPULATION_SIZE``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import symfluence  # noqa: F401  (registers model config schemas)
from symfluence.core.config.transformers import get_flat_to_nested_map

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "src" / "symfluence"
_ACCESSORS = {"_get_config_value", "get_config_value", "resolve_config_value"}

# Deliberate secondary keys (not the field's flat name), with the reason.
ALLOWED = {
    # CLI `data` commands set DATA_DIR as a fallback when SYMFLUENCE_DATA_DIR is absent.
    ("geospatial/geofabric/base/base_delineator.py", "system.data_dir", "DATA_DIR"),
}


def _chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _mismatches() -> list[str]:
    path_to_keys: dict[str, set[str]] = {}
    for key, nested in get_flat_to_nested_map().items():
        path_to_keys.setdefault(".".join(nested), set()).add(key)

    bad = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for call in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _ACCESSORS:
                continue
            lam = next((a for a in call.args[:2] if isinstance(a, ast.Lambda)), None)
            if lam is None:
                continue
            chain = _chain(lam.body)
            if "config" not in chain:
                continue
            typed = ".".join(chain[chain.index("config") + 1:])
            dk = next((k.value for k in call.keywords if k.arg == "dict_key"), None)
            if dk is None and name == "resolve_config_value" and len(call.args) >= 4:
                dk = call.args[3]
            if not (isinstance(dk, ast.Constant) and isinstance(dk.value, str)):
                continue
            keys = path_to_keys.get(typed)
            if keys and dk.value not in keys and (rel, typed, dk.value) not in ALLOWED:
                bad.append(f"{rel}:{call.lineno}: {typed} dict_key={dk.value!r}, expected one of {sorted(keys)}")
    return bad


def test_dict_keys_match_typed_paths():
    bad = _mismatches()
    assert not bad, (
        "dict_key doesn't match the typed path's flat key, so flat-dict configs "
        "(calibration workers) silently get the default:\n  " + "\n  ".join(bad)
    )
