"""Ratchet: no new raw flat-key config reads outside models/, cli/ and config internals.

Config reaches code both as a typed ``SymfluenceConfig`` and as a flat dict,
which is why reads were spread over several styles. The canonical forms are:

* ``self.config.section.field`` where config is known to be typed;
* ``self._get_config_value(lambda: self.config.section.field, default, dict_key='KEY')``
  or ``resolve_config_value(config, lambda: config.section.field, default, 'KEY')``
  where it may be a dict (both funnel through ``core.mixins.config.resolve_config_value``).

Raw ``config.get('KEY')`` / ``config['KEY']`` reads bypass the typed path, so
a renamed field isn't caught by ``test_config_paths_resolve``. The sites below
are what remains, mostly typed-first code whose dict fallback is explicit, or
keys with no typed field. Counts may only go down: lower the number (or drop the
entry) when you convert a site.

``models/`` (~600 sites) is the next codemod; ``cli/``, ``gui/`` and
``core/config/`` legitimately operate on raw YAML dicts.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[3] / "src" / "symfluence"
PACKAGES = ("core", "data", "geospatial", "evaluation", "optimization", "reporting", "project")
EXCLUDED_PREFIXES = ("core/config/",)
_CONFIG_NAMES = {"config", "config_dict", "_config", "flat_config", "cfg"}
_KEY = re.compile(r"^[A-Z][A-Z0-9_]+$")

BASELINE = {
    "core/calibration/coupled/optimizer.py": 1,
    "core/calibration/coupled/worker.py": 1,
    "core/calibration/regionalization/strategies.py": 1,  # TRANSFER_FUNCTION_PARAM_CONFIG (no field)
    "core/calibration/workers/base_worker.py": 5,  # workers receive flat dicts
    "core/metrics/streamflow_metrics.py": 1,  # FIXED_CATCHMENT_AREA (no field)
    "core/modeling/utilities/forcing_data_processor.py": 4,
    "core/path_resolver.py": 7,
    "data/acquisition/handlers/daymet.py": 3,
    "data/acquisition/utils.py": 3,  # credentials (no fields)
    "geospatial/geofabric/processors/taudem_executor.py": 1,  # MPI_LAUNCHER (no field)
    "optimization/calibration_targets/base.py": 1,
    "project/pour_point_workflow.py": 1,
}


def _is_config(node: ast.AST) -> bool:
    name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
    return name in _CONFIG_NAMES


def _is_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(_KEY.match(node.value))


def _raw_reads(path: Path) -> int:
    count = 0
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and _is_config(node.func.value)
                and node.args and _is_key(node.args[0])):
            count += 1
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
              and _is_config(node.value) and _is_key(node.slice)):
            count += 1
    return count


def _current_counts() -> dict[str, int]:
    counts = {}
    for pkg in PACKAGES:
        for path in (SRC / pkg).rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            if rel.startswith(EXCLUDED_PREFIXES):
                continue
            n = _raw_reads(path)
            if n:
                counts[rel] = n
    return counts


def test_raw_config_reads_do_not_grow():
    grown = {
        rel: (n, BASELINE.get(rel, 0))
        for rel, n in _current_counts().items()
        if n > BASELINE.get(rel, 0)
    }
    assert not grown, (
        "New raw config.get('KEY') / config['KEY'] reads (current, allowed). Use "
        "resolve_config_value(config, lambda: config.section.field, default, 'KEY') "
        "or self._get_config_value(...) instead:\n  "
        + "\n  ".join(f"{rel}: {n} > {allowed}" for rel, (n, allowed) in sorted(grown.items()))
    )


def test_baseline_is_not_stale():
    counts = _current_counts()
    stale = {rel: (counts.get(rel, 0), n) for rel, n in BASELINE.items() if counts.get(rel, 0) < n}
    assert not stale, (
        "Raw config reads were removed -- lower BASELINE to lock in the progress:\n  "
        + "\n  ".join(f"{rel}: now {now}, baseline {n}" for rel, (now, n) in sorted(stale.items()))
    )
