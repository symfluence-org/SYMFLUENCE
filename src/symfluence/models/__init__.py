# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# src/symfluence.models/__init__.py
"""Hydrological model utilities.

This module provides:
- ModelRegistry: Central registry for model runners/preprocessors/postprocessors
- Execution Framework: Unified subprocess/SLURM execution (execution submodule)
- Config Schemas: Type-safe configuration contracts (config submodule)
- Templates: Base classes for new model implementations (templates submodule)
"""

# Import execution framework components
from __future__ import annotations

try:
    from symfluence.core.modeling.execution import (
        ExecutionResult,
        ModelExecutor,
        RoutingConfig,
        SlurmJobConfig,
        SpatialMode,
        SpatialOrchestrator,
    )
except ImportError:
    pass  # Optional - may not be needed by all users

# Import config schema components
try:
    from .config import (
        ModelConfigSchema,
        get_model_schema,
        validate_model_config,
    )
except ImportError:
    pass  # Optional

# Import template components
try:
    from symfluence.core.modeling.templates import (
        ModelRunResult,
        UnifiedModelRunner,
    )
except ImportError:
    pass  # Optional

# Import all models to register them
import logging
import warnings

logger = logging.getLogger(__name__)

# Distribution-level contract declaration. Every entry-point callable under
# ``symfluence.models.*`` inherits these targets during central discovery.
# Keeping this once at the package root avoids 30+ decorators drifting apart.
__symfluence_contracts__ = {
    "models": "0.4.0",
    "calibration": "0.2.0",
    "metrics": "0.1.0",
    "geospatial-utils": "0.1.0",
}

# Suppress experimental module warnings and missing optional dependency warnings
warnings.filterwarnings('ignore', message='.*is an EXPERIMENTAL module.*')
warnings.filterwarnings('ignore', message='.*import failed.*')

# In-tree models register through the `symfluence.plugins` entry points declared
# in pyproject.toml — one `symfluence_<name> = "symfluence.models.<name>:register"`
# per model. At `import symfluence`, the bootstrap (`core/_bootstrap._discover_plugins`)
# loads every entry point and calls its register(), the *same* non-privileged path
# external plugins use. There is deliberately no hard-coded model list and no import
# loop here: adding a model means creating the package with a top-level register() and
# declaring its entry point. `tests/unit/models/test_entry_point_dogfood.py` fails the
# build if the on-disk register() set and the pyproject entry points drift apart.
#
# NOTE: entry points are read from installed dist metadata, so a newly added or pulled
# model only becomes visible after `pip install -e .` regenerates that metadata.
# `_discover_plugins` logs a loud error if zero in-tree models are discovered (the
# stale-install signal).
#
# That register() is also where a model declares the knowledge core used to hold on
# its behalf — calibration bounds (`register_bounds` -> `register_model_bounds`) and
# spatial capabilities (`register_model_spatial_capability`), plus the schema in
# `models/config/model_config_schema.py`. Service decomposition, item 2: declaring at
# plugin-discovery time is what lets a model change its own bounds or capabilities
# without a core release, and is the only path an external plugin has. Stated here
# once rather than restated in each package's register().


#: Optional per-model facilities that register themselves as a decorator side
#: effect when a submodule is imported, mapped to the registry the import
#: populates.  See ``_declare_capability_modules``.
_CAPABILITY_MODULES = {
    "forcing_adapter": "forcing_adapters",
    "init_preset": "presets",
    # Calibration workers had no discovery pass of their own, so the coupled
    # optimizer used to resolve a participant's worker by hardcoding the
    # ``symfluence.models.<name>`` layout — a path no external plugin can ever
    # satisfy.
    "calibration/worker": "workers",
    # Parameter managers DID have a discovery pass, but only inside
    # ``symfluence.optimization.parameter_managers`` — a DEPRECATED shim due for
    # removal at 2.0. Nothing else imported it, so after a plain
    # ``import symfluence`` six models (GR, HYPE, MESH, NGEN, PIHM, RHESSYS)
    # had no parameter manager registered at all, and the registry's only
    # population path was a module scheduled for deletion.
    "calibration/parameter_manager": "parameter_managers",
}


def model_packages_with(submodule: str) -> tuple[str, ...]:
    """Return the in-tree model package names that contain *submodule*``.py``.

    *submodule* may be nested, using ``/`` as the separator:
    ``model_packages_with('forcing_adapter')`` -> ``('fuse', 'gr', ...)``,
    ``model_packages_with('calibration/worker')`` -> ``('clm', 'crhm', ...)``.

    **Internal to this distribution.** This is the models package introspecting
    its *own* directory; the framework must never call it, because it cannot
    see external plugin packages and it assumes the models suite is installed.
    Framework-facing discovery goes through the registries — see
    :func:`_declare_capability_modules`.
    """
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent
    return tuple(sorted(
        p.relative_to(pkg_dir).parts[0]
        for p in pkg_dir.glob(f"*/{submodule}.py")
    ))


def _declare_capability_modules() -> None:
    """Declare the in-tree side-effect capability modules into the registries.

    Forcing adapters and ``symfluence init`` presets register through a
    decorator that only runs when ``<model>/forcing_adapter.py`` /
    ``<model>/init_preset.py`` is imported.  The framework used to find those
    modules by globbing this package's source tree from ``core``/``cli`` — an
    upward dependency on a distribution that may not be installed, and one that
    structurally cannot see external plugin packages.

    Instead this distribution *declares* the modules it owns (``add_module``)
    when it is imported, and the framework merely drains the declarations
    (``Registry.load_modules()``).  A model package — in-tree or external —
    can equivalently declare its own with
    ``model_manifest(forcing_adapter_module=..., init_preset_module=...)``;
    declarations are idempotent, so the two coexist while in-tree models
    migrate to declaring for themselves.
    """
    from symfluence.core.registries import R

    for submodule, registry_name in _CAPABILITY_MODULES.items():
        registry = getattr(R, registry_name)
        dotted = submodule.replace("/", ".")
        for package in model_packages_with(submodule):
            registry.add_module(f"{__name__}.{package}.{dotted}")


_declare_capability_modules()


__all__ = [
    "model_packages_with",
    # Execution Framework
    "ModelExecutor",
    "ExecutionResult",
    "SlurmJobConfig",
    "SpatialOrchestrator",
    "SpatialMode",
    "RoutingConfig",
    # Config Schemas
    "ModelConfigSchema",
    "get_model_schema",
    "validate_model_config",
    # Templates
    "UnifiedModelRunner",
    "ModelRunResult",
]
