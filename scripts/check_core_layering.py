#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Layering guard for the ``core`` package (RTI review item 19).

The architecture intends ``symfluence.core`` to depend on nothing above it:
every other top-level package (``models``, ``evaluation``, ``fews``,
``geospatial``, ``project``, ``data``, ``optimization``, ``cli``, ``agent``,
``gui``, ``tui`` ...) may depend on ``core``, but ``core`` must not depend on
them. This script walks the AST of every module under ``src/symfluence/core``
and flags imports of those upper layers.

Two severities:

* **Module-level imports** (executed at import time) are forbidden outright.
  These create hard, import-time coupling and are what inverts the layering.
* **Deferred imports** (inside a function/method body) are the accepted
  inversion-of-control pattern -- a small number of registry-seeding and
  data-driven-validation seams genuinely need to reach up at call time. Each
  one must be listed in ``ALLOWED_DEFERRED`` with a reason. A new deferred
  edge that is not on the list fails the check, forcing a conscious decision.

Two *kinds* of edge are scanned:

* **Static imports** -- ``import x`` / ``from x import y`` AST nodes.
* **String-built imports** -- module paths assembled as string literals or
  f-strings and resolved at runtime through ``importlib.import_module`` or a
  registry's ``add_lazy``. An AST-only scan is blind to these, so a genuine
  runtime upward edge such as
  ``importlib.import_module(f"symfluence.models.{model.lower()}.calibration.worker")``
  used to pass the guard silently -- which is exactly how a "removable" layer
  quietly becomes non-removable. Both kinds share the same severity split and
  the same allow-list.

Run directly (``python scripts/check_core_layering.py``) or via the mirrored
unit test ``tests/unit/core/test_core_layering.py``. Exit code is non-zero on
any violation.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Tuple

# Top-level symfluence subpackages that ``core`` must not depend on.
UPPER_LAYERS: Set[str] = {
    "models",
    "evaluation",
    "fews",
    "geospatial",
    "project",
    "data",
    "optimization",
    "cli",
    "agent",
    "gui",
    "tui",
    # ``coupling`` was omitted here for a long time, which meant core's edges
    # into the dCoupler stack were invisible to this guard *by configuration*
    # rather than because the scan could not see them — the ALLOWED_DEFERRED
    # entry for ``modeling/coupling.py`` below was inert as a result. Every
    # existing edge is a lazy IoC seam and stays legal, but a NEW one now has
    # to be a conscious decision, which is the whole point of the guard.
    "coupling",
    # ``testing`` is the public test-support surface. It depends on core by
    # design; core importing it would drag pytest-shaped helpers into the runtime
    # import graph and invert the dependency. Listed here so that edge cannot be
    # added by accident.
    "testing",
}

# Accepted deferred (call-time) edges: (core-relative path, imported module prefix, reason).
# These are inversion-of-control seams -- core reaches up only at call time to
# seed a registry or validate a value against an upper-layer catalogue. Keep this
# list short and justified; adding to it should be a deliberate decision.
ALLOWED_DEFERRED: List[Tuple[str, str, str]] = [
    # (_bootstrap.py seeding of R.metrics is no longer an upward edge: the
    # metric registry moved into core.metrics with the calibration promotion.)
    (
        "calibration/targets.py",
        "symfluence.optimization",
        "Model adapters request generic calibration targets through a core "
        "capability facade; the host implementation is resolved only when used.",
    ),
    (
        "modeling/coupling.py",
        "symfluence.coupling.graph_builder",
        "The model-facing facade resolves the optional coupling capability only "
        "when graph execution is explicitly requested.",
    ),
    (
        "_bootstrap.py",
        "symfluence.coupling.adapters",
        "BMI/dCoupler component adapters are registered as lazy dotted paths "
        "(R.bmi_adapters.add_lazy), so nothing in the coupling layer is imported "
        "until a graph actually resolves an adapter. Same inversion-of-control "
        "seam as the other registry seeding here; the strings are declarations, "
        "not imports.",
    ),
    (
        "config/factories.py",
        "symfluence.cli.init_presets",
        "from_preset factory resolves a named CLI init preset at call time; the "
        "preset catalogue transitively needs the models layer, and from_preset "
        "is a public SymfluenceConfig classmethod, so this stays a call-time seam.",
    ),
    (
        "contracts.py",
        "symfluence.data.backends.contract",
        "contract_version('acquisition') surfaces the acquisition family's "
        "PROTOCOL_VERSION (owned by data/, where external services import it) "
        "for a uniform read-only view; resolved at call time only.",
    ),
    (
        "calibration/optimizers/component_factory.py",
        "symfluence.optimization.calibration_targets",
        "The component factory resolves calibration targets through the "
        "optimization facade at call time; targets wrap evaluation evaluators, "
        "which core must not import at module level.",
    ),
    # ---- String-built edges (first surfaced when the scan was extended) ----
    (
        "_bootstrap.py",
        "symfluence.data",
        "Registry seeders: R.acquisition_handlers / R.observation_handlers call "
        "importlib.import_module('symfluence.data.*') on first lookup so the "
        "handler decorators fire. The canonical IoC seam -- core never names a "
        "handler, only the package whose import populates the registry.",
    ),
    (
        "_bootstrap.py",
        "symfluence.geospatial",
        "Registry seeder for delineation strategies; same first-lookup IoC "
        "pattern as the data handlers above.",
    ),
    (
        "_bootstrap.py",
        "symfluence.evaluation",
        "Registry seeder for evaluators; same first-lookup IoC pattern.",
    ),
    (
        "_bootstrap.py",
        "symfluence.optimization",
        "Registry seeder for in-tree model optimizers; same first-lookup IoC "
        "pattern.",
    ),
    (
        "_bootstrap.py",
        "symfluence.models",
        "NOT an import: ``ep.value.startswith('symfluence.models.')`` is a "
        "prefix test on an entry-point value, used only to count how many "
        "in-tree model plugins loaded. Module-path-shaped by nature, so the "
        "string scan sees it; nothing is imported.",
    ),
    (
        "calibration/parameters/parameter_bounds_registry.py",
        "symfluence.models",
        "NOT imports: the ``_owned_by_package(model, 'symfluence.models.<x>')`` "
        "package names are interpolated into the KeyError text that tells the "
        "operator which package failed to register bounds. Core resolves the "
        "bounds through the registry and never imports these modules.",
    ),
    (
        "calibration/mixins/parallel/execution_strategies/mpi.py",
        "symfluence.optimization",
        "GENUINE upward edge, reported rather than fixed (file is owned "
        "elsewhere): _get_worker_info() falls back to a hardcoded "
        "'symfluence.optimization.workers.summa_parallel_workers' module name "
        "that the generated MPI worker script imports in a subprocess. Only "
        "reachable when the worker callable has no __module__, and the fallback "
        "is SUMMA-specific -- it should come from the caller, not a constant.",
    ),
    (
        "calibration/mixins/parallel/execution_strategies/mpi_persistent.py",
        "symfluence.optimization",
        "Same hardcoded SUMMA MPI-worker module fallback as mpi.py above; same "
        "disposition.",
    ),
]


class Violation(NamedTuple):
    path: str
    lineno: int
    module: str
    deferred: bool
    #: True when the edge was built from a string literal / f-string resolved
    #: at runtime (importlib, registry ``add_lazy``) rather than a static import.
    string_built: bool = False

    def describe(self) -> str:
        kind = "deferred (not allow-listed)" if self.deferred else "MODULE-LEVEL"
        how = "string-built import" if self.string_built else "import"
        return f"{self.path}:{self.lineno}: {kind} {how} of upper layer '{self.module}'"


# A string literal is treated as a module path only if it is *shaped* like one:
# dotted identifiers, nothing else. This keeps prose (docstrings, log messages,
# error text that merely mentions a module) out of the scan while still
# catching every real ``import_module`` / ``add_lazy`` target.
_MODULE_PATH_RE = re.compile(r"^symfluence\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\.?$")


def _joinedstr_literal_prefix(node: ast.JoinedStr) -> Optional[str]:
    """Constant leading segment of an f-string, or None if it starts dynamic.

    ``f"symfluence.models.{model}.worker"`` yields ``"symfluence.models."`` --
    enough to classify the target layer even though the rest is only known at
    runtime. ``f"{base}.preprocessor"`` yields None: nothing can be concluded.
    """
    if not node.values:
        return None
    first = node.values[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    return first.value


def _string_module_target(node: ast.AST) -> Optional[str]:
    """Module path a string literal / f-string targets, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
        return value if _MODULE_PATH_RE.match(value) else None
    if isinstance(node, ast.JoinedStr):
        prefix = _joinedstr_literal_prefix(node)
        if prefix is None:
            return None
        # The prefix ends mid-path (at the '{'), so allow a trailing dot.
        return prefix if _MODULE_PATH_RE.match(prefix) else None
    return None


def _upper_layer_of(module: str | None) -> str | None:
    """Return the upper-layer package name if *module* targets one, else None."""
    if not module or not module.startswith("symfluence."):
        return None
    parts = module.split(".")
    if len(parts) >= 2 and parts[1] in UPPER_LAYERS:
        return module
    return None


def _is_allowed_deferred(rel_path: str, module: str) -> bool:
    for path_suffix, mod_prefix, _reason in ALLOWED_DEFERRED:
        if rel_path.replace("\\", "/").endswith(path_suffix) and module.startswith(mod_prefix):
            return True
    return False


def _scan_file(py_file: Path, core_root: Path) -> List[Violation]:
    rel_path = str(py_file.relative_to(core_root))
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

    # Map each import node to whether it is nested inside a function body or an
    # ``if TYPE_CHECKING:`` block (the latter is erased at runtime, so it is a
    # type-only edge — treated like a deferred import, requiring an allowance).
    # Nodes carrying an edge: static imports plus the string literals /
    # f-strings a runtime import can be built from.
    edge_types = (ast.Import, ast.ImportFrom, ast.Constant, ast.JoinedStr)

    deferred_nodes: Set[int] = set()
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(func):
                if isinstance(child, edge_types):
                    deferred_nodes.add(id(child))
        elif isinstance(func, ast.If):
            test = func.test
            is_type_checking = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            ) or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_type_checking:
                for child in ast.walk(func):
                    if isinstance(child, edge_types):
                        deferred_nodes.add(id(child))

    # ``ast.walk`` yields a JoinedStr *and* the Constant pieces it is made of,
    # which would report one f-string edge twice. The JoinedStr is the node
    # that carries the meaning, so its literal parts are skipped.
    fstring_parts: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.Constant):
                    fstring_parts.add(id(value))

    violations: List[Violation] = []
    for node in ast.walk(tree):
        if id(node) in fstring_parts:
            continue
        modules: List[str] = []
        string_built = False
        if isinstance(node, ast.ImportFrom):
            # level > 0 -> relative import inside symfluence (not an absolute upper-layer dep)
            if node.level == 0 and node.module:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, (ast.Constant, ast.JoinedStr)):
            target = _string_module_target(node)
            if target is None:
                continue
            modules.append(target)
            string_built = True
        else:
            continue

        for module in modules:
            if _upper_layer_of(module) is None:
                continue
            deferred = id(node) in deferred_nodes
            # The allow-list governs deferred static imports, and string-built
            # edges at EITHER position. A module-level string literal executes
            # nothing by itself -- it is data in a lazy-import map that some
            # ``__getattr__`` resolves at call time -- so it cannot be
            # import-time coupling and must remain allow-listable. Its position
            # is still reported (MODULE-LEVEL vs deferred) so the distinction
            # stays visible in the output.
            if (deferred or string_built) and _is_allowed_deferred(rel_path, module):
                continue
            violations.append(
                Violation(rel_path, node.lineno, module, deferred, string_built)
            )
    return violations


def find_violations(core_root: Path | None = None) -> List[Violation]:
    if core_root is None:
        core_root = Path(__file__).resolve().parent.parent / "src" / "symfluence" / "core"
    violations: List[Violation] = []
    for py_file in sorted(core_root.rglob("*.py")):
        violations.extend(_scan_file(py_file, core_root))
    return violations


# ---------------------------------------------------------------------------
# Package-boundary rules (monorepo prep for service decomposition)
# ---------------------------------------------------------------------------
# Each rule: (scanned package, forbidden import prefixes, allowed deferred
# edges). Module-level imports of a forbidden prefix are always violations;
# deferred (function-body) imports must be allow-listed with a reason.
#
# Note: the deprecated lazy shims under optimization/ (parameter_managers,
# the calibration_targets re-export map, workers/__init__) resolve via importlib.import_module with string paths. The scan now SEES
# those string-built edges (it used to be AST-imports only), so each is
# explicitly allow-listed below with its reason rather than passing unnoticed.

BoundaryRule = Tuple[str, Tuple[str, ...], List[Tuple[str, str, str]]]

BOUNDARY_RULES: List[BoundaryRule] = [
    (
        # The models layer must be removable: nothing may import it at module
        # level, so the framework imports cleanly with the model suite absent.
        "optimization",
        ("symfluence.models",),
        [
            (
                "_autodiscover.py",
                "symfluence.models",
                "Auto-discovery iterates installed model packages at call time; "
                "tolerates absence via try/except ImportError.",
            ),
            (
                "workers/utilities/__init__.py",
                "symfluence.models.utilities.routing_decider",
                "Deprecated lazy re-export (PEP 562) of RoutingDecider from its "
                "canonical home in the models package.",
            ),
            # ---- String-built edges (first surfaced when the scan was
            # extended). These are the deprecated PEP-562 shims the header
            # comment above already described as invisible to the AST scan;
            # they are now visible and explicitly allowed. Each resolves its
            # target inside __getattr__, so the models layer stays removable at
            # import time -- the property this rule exists to protect. They go
            # away when the shims are dropped, not before.
            (
                "calibration_targets/__init__.py",
                "symfluence.models",
                "Deprecated lazy re-export map (_MODEL_TARGET_EXPORTS); the "
                "module-level strings are data, resolved by __getattr__ via "
                "importlib at call time.",
            ),
            (
                "parameter_managers/__init__.py",
                "symfluence.models",
                "Deprecated lazy re-export map (_MANAGERS); module-level strings "
                "are data, resolved by __getattr__ via importlib at call time.",
            ),
            (
                "workers/__init__.py",
                "symfluence.models",
                "Deprecated lazy re-export map for the per-model calibration "
                "workers, resolved at call time.",
            ),
        ],
    ),
    (
        # Models must not depend on interface layers, nor on the geospatial
        # capability package: the geometry utilities model preprocessors need
        # are part of the core contract surface (core.geometry_utils).
        # Build-environment helpers live in core.build; everything else the
        # adapters need is in core.
        "models",
        (
            "symfluence.cli",
            "symfluence.gui",
            "symfluence.tui",
            "symfluence.agent",
            "symfluence.fews",
            "symfluence.geospatial",
            # Bundled assets. Seven model modules used to import
            # ``symfluence.resources.get_base_settings_dir`` for their own
            # settings data; the resolution now lives in
            # ``core.modeling.base_settings`` (which also broke the
            # resources <-> core cycle). ``resources`` deliberately stays OUT
            # of UPPER_LAYERS -- core reading bundled data is a downward edge --
            # but models must reach settings through the core contract so the
            # models distribution depends on ``symfluence.core`` alone.
            "symfluence.resources",
            # ModelManager was promoted to project/; the (since removed) shim at
            # models/model_manager.py imported it at module level, so importing
            # the models package pulled in the orchestration layer. Forbidding
            # the prefix keeps that from creeping back once models ships as its
            # own distribution.
            "symfluence.project",
        ),
        [
            (
                "model_manager.py",
                "symfluence.project",
                "Back-compat shim resolving the promoted ModelManager inside "
                "__getattr__ only, so nothing is imported until an external "
                "caller touches the deprecated path. Removed at 2.0.",
            ),
        ],
    ),
    (
        "evaluation",
        ("symfluence.models",),
        [
            (
                "analysis_manager.py",
                "symfluence.models",
                "Imports model packages at call time solely to trigger analyzer "
                "registration (IoC seam).",
            ),
        ],
    ),
    (
        # ModelManager was promoted to project/model_manager.py (it is
        # registry-driven framework orchestration, not a model adapter), so
        # project/ now has zero references to the models layer.
        "project",
        ("symfluence.models",),
        [],
    ),
    (
        # Process adapters resolve model components via R.runners /
        # R.result_extractors — no direct model imports remain.
        "coupling",
        ("symfluence.models",),
        [],
    ),
    (
        # State management moved to core.modeling.state with the adapter tier.
        "fews",
        ("symfluence.models",),
        [],
    ),
    (
        # State management moved to core.modeling.state with the adapter tier.
        "data_assimilation",
        ("symfluence.models",),
        [],
    ),
    (
        "cli",
        ("symfluence.models",),
        [
            (
                "init_presets.py",
                "symfluence.models",
                "Preset catalogue lists installed model packages at call time.",
            ),
            (
                "preset_registry.py",
                "symfluence.models",
                "Preset registry lists installed model packages at call time.",
            ),
        ],
    ),
    (
        "gui",
        ("symfluence.models",),
        [
            (
                "server.py",
                "symfluence.models",
                "Server triggers model plugin registration at call time.",
            ),
        ],
    ),
    (
        "data",
        ("symfluence.models",),
        [],
    ),
    # ---- Data-handler encapsulation (community-service liftoff prep) ----
    # Handlers are consumed registry-first (R.acquisition_handlers /
    # R.observation_handlers) or through the AcquisitionBackend protocol —
    # never by importing a handler module from outside data/.
    (
        "evaluation",
        ("symfluence.data.acquisition.handlers", "symfluence.data.observation.handlers"),
        [],
    ),
    (
        "models",
        ("symfluence.data.acquisition.handlers", "symfluence.data.observation.handlers"),
        [],
    ),
    (
        "models",
        (
            "symfluence.coupling",
            "symfluence.optimization",
            "symfluence.reporting",
            "symfluence.evaluation",
            "symfluence.data.model_ready",
            "symfluence.data.utils.netcdf_utils",
        ),
        [],
    ),
    (
        "optimization",
        ("symfluence.data.acquisition.handlers", "symfluence.data.observation.handlers"),
        [],
    ),
    (
        "project",
        ("symfluence.data.acquisition.handlers", "symfluence.data.observation.handlers"),
        [],
    ),
    (
        "geospatial",
        ("symfluence.data.acquisition.handlers", "symfluence.data.observation.handlers"),
        [],
    ),
    (
        "cli",
        ("symfluence.data.acquisition.handlers", "symfluence.data.observation.handlers"),
        [],
    ),
    (
        "geospatial",
        ("symfluence.models",),
        [],
    ),
    (
        "reporting",
        ("symfluence.models",),
        [],
    ),
]


def find_boundary_violations(src_root: Path | None = None) -> List[Violation]:
    """Scan package-boundary rules; reuse the core scanner per package."""
    if src_root is None:
        src_root = Path(__file__).resolve().parent.parent / "src" / "symfluence"
    violations: List[Violation] = []
    global UPPER_LAYERS, ALLOWED_DEFERRED
    saved = (UPPER_LAYERS, ALLOWED_DEFERRED)
    try:
        for package, forbidden, allowed in BOUNDARY_RULES:
            pkg_root = src_root / package
            if not pkg_root.exists():
                continue
            # Reuse _scan_file by temporarily retargeting the module filter.
            UPPER_LAYERS = {f.split(".")[1] for f in forbidden}
            ALLOWED_DEFERRED = allowed
            for py_file in sorted(pkg_root.rglob("*.py")):
                for v in _scan_file(py_file, pkg_root):
                    if any(v.module.startswith(f) for f in forbidden):
                        violations.append(
                            Violation(
                                f"{package}/{v.path}", v.lineno, v.module,
                                v.deferred, v.string_built,
                            )
                        )
    finally:
        UPPER_LAYERS, ALLOWED_DEFERRED = saved
    return violations


def main() -> int:
    core_violations = find_violations()
    boundary_violations = find_boundary_violations()
    if not core_violations and not boundary_violations:
        print(
            "core layering OK: no disallowed core -> upper-layer imports; "
            "package boundaries OK (models removable, models -> no interface layers)"
        )
        return 0

    if core_violations:
        print("core layering VIOLATIONS found (RTI review item 19):\n", file=sys.stderr)
        for v in core_violations:
            print(f"  {v.describe()}", file=sys.stderr)
    if boundary_violations:
        print("package-boundary VIOLATIONS found:\n", file=sys.stderr)
        for v in boundary_violations:
            print(f"  {v.describe()}", file=sys.stderr)
    print(
        "\nFix: move shared code down into core, invert the dependency through "
        "the registry, or (for a genuine call-time IoC seam) add it to "
        "ALLOWED_DEFERRED / BOUNDARY_RULES in scripts/check_core_layering.py "
        "with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
