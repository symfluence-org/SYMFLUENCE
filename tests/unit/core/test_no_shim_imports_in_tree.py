# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""In-tree code must import canonical core paths, never back-compat shims.

The service-decomposition promotions left shim modules at historical import
paths so external packages keep working. The ones no plugin imports were
deleted; the rest warn on import and go before 1.0 (ADR-0011). In-tree code was migrated to the canonical
``symfluence.core.*`` paths — this guard keeps it that way, so the shims'
lifecycle is a purely external-facing decision.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "symfluence"

# Deprecated shim module prefixes (canonical homes are under symfluence.core).
SHIM_PREFIXES = (
    # Kept only for the external model plugins (see
    # docs/adr/0011-retire-unused-back-compat-shims.md); each emits a
    # DeprecationWarning on import.
    "symfluence.models.base",
    "symfluence.models.mixins",
    "symfluence.models.execution",
    "symfluence.models.state",
    "symfluence.models.utilities",
    "symfluence.models.adapters",
    "symfluence.models.spatial_modes",
    "symfluence.optimization.optimizers",
    "symfluence.optimization.workers.base_worker",
    "symfluence.optimization.workers.inmemory_worker",
    "symfluence.optimization.core.base_parameter_manager",
    "symfluence.optimization.core.parameter_bounds_registry",
    "symfluence.optimization.regionalization.strategies",
    "symfluence.evaluation.metrics",
    "symfluence.data.utils.netcdf_utils",
    "symfluence.geospatial.geometry_utils",
)
# ``symfluence.optimization.parameter_managers`` is a deprecated aggregate too,
# but optimization_manager still imports it to seed parameter-manager discovery
# (``from symfluence.optimization import parameter_managers``), so it is not
# listed here until that seeding moves.

# The shim files themselves (they self-reference their canonical target and
# legitimately live at the deprecated paths).
_SHIM_FILE_MARKER = "Back-compat shim"
_SHIM_DIR_HINTS = (
    "models/base", "models/mixins", "models/execution", "models/state",
    "models/templates", "models/utilities", "models/adapters",
    "optimization/optimizers", "optimization/mixins", "optimization/workers",
    "optimization/core", "optimization/parameter_managers",
    "cli/services", "evaluation", "geospatial",
)


def _is_shim_file(path: Path, text: str) -> bool:
    rel = str(path.relative_to(SRC)).replace("\\", "/")
    return _SHIM_FILE_MARKER in text and any(h in rel for h in _SHIM_DIR_HINTS)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom):
    """Absolute module named by an ``ImportFrom``, resolving relative imports.

    Relative imports matter: ``models/__init__.py`` once pulled in the
    ``models.execution`` shim as ``from .execution import ...``, which an
    absolute-only scan cannot see.
    """
    if node.level == 0:
        return node.module
    package = module.split(".") if is_package else module.split(".")[:-1]
    if node.level > 1:
        package = package[: len(package) - (node.level - 1)]
    return ".".join(package + ([node.module] if node.module else []))


@pytest.mark.unit
def test_no_in_tree_imports_of_deprecated_shim_paths():
    offenders = []
    for py in SRC.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if _is_shim_file(py, text):
            continue
        tree = ast.parse(text)
        module = _module_name(py)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                base = _resolve_from(module, py.name == "__init__.py", node)
                if base:
                    # ``from pkg import sub`` imports the submodule ``pkg.sub``.
                    names = [base] + [f"{base}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if any(name == p or name.startswith(p + ".") for p in SHIM_PREFIXES):
                    offenders.append(
                        f"{py.relative_to(SRC)}:{node.lineno}: {name}"
                    )
    assert offenders == [], (
        "in-tree imports of deprecated shim paths (use the canonical "
        "symfluence.core.* homes):\n" + "\n".join(offenders)
    )
