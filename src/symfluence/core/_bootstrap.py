# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""One-time bootstrap for static registrations.

Called once from ``symfluence/__init__.py`` to populate:

* Delineation strategy aliases
* BMI adapter lazy imports and aliases
* Metric registry entries with aliases
* External plugins discovered via ``importlib.metadata`` entry points

This module should be kept lightweight — no heavy dependencies.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_bootstrapped = False

#: Entry-point group that external packages use to register plugins.
PLUGIN_ENTRY_POINT_GROUP = "symfluence.plugins"


def bootstrap() -> None:
    """Populate static registrations.  Safe to call multiple times."""
    global _bootstrapped  # noqa: PLW0603
    if _bootstrapped:
        return
    _bootstrapped = True

    from symfluence.core.registries import R

    _bootstrap_delineation_aliases(R)
    _bootstrap_bmi_adapters(R)
    _bootstrap_metrics(R)
    _discover_plugins()


def _bootstrap_delineation_aliases(R: type) -> None:  # noqa: N803
    """Register canonical delineation aliases."""
    aliases = {
        "delineate": "semidistributed",
        "distribute": "distributed",
        "subset": "semidistributed",
        "discretized": "semidistributed",
    }
    for alias, canonical in aliases.items():
        R.delineation_strategies.alias(alias, canonical)


def _bootstrap_bmi_adapters(R: type) -> None:  # noqa: N803
    """Register BMI/dCoupler adapters as lazy imports + aliases."""

    process_models = {
        "SUMMA": "symfluence.coupling.adapters.process_adapters.SUMMAProcessComponent",
        "MIZUROUTE": "symfluence.coupling.adapters.process_adapters.MizuRouteProcessComponent",
        "TROUTE": "symfluence.coupling.adapters.process_adapters.TRouteProcessComponent",
        "PARFLOW": "symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent",
        "MODFLOW": "symfluence.coupling.adapters.process_adapters.MODFLOWProcessComponent",
        "MESH": "symfluence.coupling.adapters.process_adapters.MESHProcessComponent",
        "CLM": "symfluence.coupling.adapters.process_adapters.CLMProcessComponent",
    }
    jax_models = {
        "SNOW17": "symfluence.coupling.adapters.jax_adapters.Snow17JAXComponent",
        "XAJ": "symfluence.coupling.adapters.jax_adapters.XAJJAXComponent",
        "SACSMA": "symfluence.coupling.adapters.jax_adapters.SacSmaJAXComponent",
        "HBV": "symfluence.coupling.adapters.jax_adapters.HBVJAXComponent",
        "HECHMS": "symfluence.coupling.adapters.jax_adapters.HecHmsJAXComponent",
        "TOPMODEL": "symfluence.coupling.adapters.jax_adapters.TopmodelJAXComponent",
    }

    for name, path in process_models.items():
        R.bmi_adapters.add_lazy(name, path)
    for name, path in jax_models.items():
        R.bmi_adapters.add_lazy(name, path)

    # Aliases for common alternate names
    R.bmi_adapters.alias("XINANJIANG", "XAJ")
    R.bmi_adapters.alias("SAC-SMA", "SACSMA")
    R.bmi_adapters.alias("HEC-HMS", "HECHMS")


def _bootstrap_metrics(R: type) -> None:  # noqa: N803
    """Seed the unified metrics registry from the existing METRIC_REGISTRY.

    We import the existing metric registry dict and re-register each entry
    into ``R.metrics`` so that both old and new consumers see the same data.
    """
    try:
        from symfluence.evaluation.metrics_registry import METRIC_REGISTRY
    except ImportError:
        logger.debug("metrics_registry not available; skipping metric bootstrap")
        return

    # Primary entries (use exact casing from the dict keys)
    _primary_names = {
        "NSE", "logNSE", "KGE", "KGEp", "KGEnp", "VE",
        "RMSE", "NRMSE", "MAE", "MARE", "bias", "PBIAS",
        "correlation", "R2",
    }

    # Use identity normalization for metrics — preserve original casing
    # (metrics registry has mixed case keys like "logNSE", "KGEp", etc.)
    R.metrics._normalize = lambda s: s  # noqa: E731

    for name in _primary_names:
        if name in METRIC_REGISTRY:
            R.metrics.add(name, METRIC_REGISTRY[name])

    # Aliases (lowercase and alternative names)
    _aliases = {
        "kge": "KGE",
        "nse": "NSE",
        "kge_prime": "KGEp",
        "kge_np": "KGEnp",
        "r_squared": "R2",
        "log_nse": "logNSE",
    }
    for alias, canonical in _aliases.items():
        R.metrics.alias(alias, canonical)


# ======================================================================
# External plugin discovery via entry points
# ======================================================================


def _discover_plugins() -> None:
    """Load external plugins registered under the ``symfluence.plugins`` group.

    Each entry point should reference a callable (typically a function)
    that performs its own registrations using ``R.*.add()``,
    ``model_manifest()``, or any other registry API.  The callable is
    invoked with no arguments.

    A failing plugin is logged and skipped — it never takes down the
    framework.

    **How to write a plugin** (in the external package's ``pyproject.toml``)::

        [project.entry-points."symfluence.plugins"]
        my_model = "my_package:register"

    Where ``my_package.register`` is a zero-arg function::

        # my_package/__init__.py
        def register():
            from symfluence.core.registries import R
            from .runner import MyRunner
            R.runners.add("MY_MODEL", MyRunner)
    """
    import sys

    if sys.version_info >= (3, 12):
        from importlib.metadata import entry_points
    else:
        # Python 3.9-3.11: entry_points() accepts the *group* keyword
        # starting from 3.9, but the return type changed in 3.12.
        from importlib.metadata import entry_points

    try:
        eps = entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    except TypeError:
        # Very old importlib_metadata fallback (shouldn't happen on 3.11+)
        eps = entry_points().get(PLUGIN_ENTRY_POINT_GROUP, [])  # type: ignore[assignment]

    for ep in eps:
        try:
            plugin_fn = ep.load()
            plugin_fn()
            logger.debug("Loaded plugin %r from %s", ep.name, ep.value)
        except Exception:  # noqa: BLE001 — never let a broken plugin crash the framework
            logger.warning(
                "Failed to load symfluence plugin %r (%s); skipping.",
                ep.name,
                ep.value,
                exc_info=True,
            )
