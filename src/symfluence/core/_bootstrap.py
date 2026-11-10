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

#: Entry points that failed to load, as ``(name, value)`` pairs. A plugin whose
#: ``register()`` never ran contributes nothing to the registries — including
#: its build instructions, which is how a model that cannot import its Python
#: components also loses the ability to BUILD its binary, the very thing that
#: would fix it. Consumers that can degrade gracefully read this to attempt a
#: narrower recovery; see ``cli.external_tools_config``.
_FAILED_PLUGIN_ENTRY_POINTS: list[tuple[str, str]] = []


def failed_plugin_entry_points() -> tuple[tuple[str, str], ...]:
    """Entry points that raised during discovery, as ``(name, value)`` pairs."""
    return tuple(_FAILED_PLUGIN_ENTRY_POINTS)


def bootstrap() -> None:
    """Populate static registrations.  Safe to call multiple times."""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    from symfluence.core.registries import R

    _bootstrap_delineation_aliases(R)
    _bootstrap_bmi_adapters(R)
    _bootstrap_model_aliases(R)
    # Deferred: seeding R.metrics imports the evaluation stack (~1 s of
    # pandas/scipy/geospatial), which most CLI invocations never read.
    R.metrics.set_seeder(lambda: _bootstrap_metrics(R))
    # Deferred: delineation strategies self-register via decorators when the
    # delineation machinery is imported; importing it eagerly costs ~1 s of
    # raster stack. First strategy lookup triggers the import instead.
    R.delineation_strategies.set_seeder(_seed_delineation_strategies)
    # Deferred: evaluators self-register via decorators in
    # symfluence.evaluation.evaluators, which pulls the observation stack.
    R.evaluators.set_seeder(_seed_evaluators)
    # Deferred: the in-tree model optimizers/workers/parameter managers
    # register when optimization.model_optimizers is imported (~0.6 s).
    R.optimizers.set_seeder(_seed_model_optimizers)
    R.workers.set_seeder(_seed_model_optimizers)
    R.parameter_managers.set_seeder(_seed_model_optimizers)
    # Deferred: data handlers self-register via decorators when their package
    # imports; consumers resolve them registry-first (never by importing a
    # handler module directly — the backend/registry seam for the community-
    # service liftoff), so first lookup triggers the import.
    R.acquisition_handlers.set_seeder(_seed_acquisition_handlers)
    R.observation_handlers.set_seeder(_seed_observation_handlers)
    # Deferred: build instructions register by decorator when a package's
    # build_instructions module is imported, and packages DECLARE that module
    # rather than the framework hunting for it on disk. Without a seeder the
    # entries appear only once some consumer happens to drain the declarations,
    # which made `R.build_instructions.get(...)` depend on what ran first.
    R.build_instructions.set_seeder(_seed_build_instructions)
    _discover_plugins()


def _seed_acquisition_handlers() -> None:
    """Import the acquisition package so its handler decorators register."""
    import importlib

    importlib.import_module("symfluence.data.acquisition")


def _seed_observation_handlers() -> None:
    """Import the observation package so its handler decorators register."""
    import importlib

    importlib.import_module("symfluence.data.observation")


def _seed_build_instructions() -> None:
    """Drain the build-instruction modules packages have declared."""
    from symfluence.core.registries import R

    R.build_instructions.load_modules()


def _seed_delineation_strategies() -> None:
    """Import the delineation machinery so its strategy decorators register."""
    import importlib
    importlib.import_module("symfluence.geospatial.delineation")


def _seed_evaluators() -> None:
    """Import the evaluators package so its decorators and aliases register."""
    import importlib
    importlib.import_module("symfluence.evaluation.evaluators")


def _seed_model_optimizers() -> None:
    """Import the in-tree model optimizers so their decorators register."""
    import importlib
    importlib.import_module("symfluence.optimization.model_optimizers")

    # COUPLED is framework composition machinery living in core, but its
    # registration used to fire only when ``optimization._autodiscover``
    # pkgutil-scanned ``symfluence.models.*`` and happened to import the
    # back-compat shim. That made a core capability hostage to the models
    # distribution: an install carrying only external model plugins would hold
    # the code while ``R.optimizers.get('COUPLED')`` returned None and
    # ``optimization_manager`` routed to it regardless. Seed it here instead.
    #
    # Idempotent: the shim import still happens during autodiscovery, but
    # ``sys.modules`` caching means each decorator fires exactly once.
    for module in (
        "symfluence.core.calibration.coupled.optimizer",
        "symfluence.core.calibration.coupled.parameter_manager",
        "symfluence.core.calibration.coupled.worker",
    ):
        importlib.import_module(module)

    # Parameter managers register by decorator when their module is imported,
    # and importing model_optimizers above does not reach them: that pass
    # discovers only calibration/optimizer modules. The one pass that DID reach
    # them lives in symfluence.optimization.parameter_managers, a deprecated
    # shim due for removal at 2.0 — so six models had no parameter manager
    # registered after a plain `import symfluence`, and the registry's only
    # population path was a module scheduled for deletion. Drain the packages'
    # own declarations instead, which works for external plugins too.
    from symfluence.core.registries import R

    R.parameter_managers.load_modules()


def _bootstrap_delineation_aliases(R: type) -> None:
    """Register canonical delineation aliases."""
    aliases = {
        "delineate": "semidistributed",
        "distribute": "distributed",
        "subset": "semidistributed",
        "discretized": "semidistributed",
    }
    for alias, canonical in aliases.items():
        R.delineation_strategies.alias(alias, canonical)


def _bootstrap_bmi_adapters(R: type) -> None:
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


def _bootstrap_model_aliases(R: type) -> None:
    """Alias hyphenated model names to their canonical registry keys.

    Some models register their components under a hyphen-free canonical name
    (e.g. ``jhechms`` registers ``HECHMS``, ``jsacsma`` registers ``SACSMA``,
    and the coupled land-surface/subsurface runner registers ``CLMPARFLOW``).
    A config using the conventional hyphenated spelling
    (``HYDROLOGICAL_MODEL: HEC-HMS`` or ``CLM-ParFlow``) would otherwise fail
    to resolve a runner.  Aliases are resolved lazily at lookup time, so they
    may be declared here before the plugin entry points register the canonical
    keys.

    Only aliases a package cannot declare for itself remain here — a package
    owning its canonical key uses ``model_manifest(aliases=[...])`` instead.
    Two kinds qualify: hyphenated spellings whose canonical key belongs to an
    EXTERNAL package (``HEC-HMS`` -> ``HECHMS``, owned by ``jhechms``), and an
    alias whose target is not a model at all (the SUMMA+MODFLOW coupling ->
    the ``COUPLED_GW`` calibration pipeline, which no single manifest owns).

    Note this differs from the BMI-adapter aliases above: e.g. the BMI adapter
    is registered as ``XAJ`` whereas the standalone runner is registered as
    ``XINANJIANG``, so no runner-level alias is added for it. The guard below
    additionally refuses to shadow a real registration with an alias.
    """
    # alias -> canonical, applied across every model-component registry.
    #
    # A package that owns its canonical key declares its own alternate
    # spellings with ``model_manifest(aliases=[...])`` — RHESSYS ("RHESS") and
    # CLMPARFLOW ("CLM-ParFlow") now do. What remains here is what a package
    # cannot declare for itself:
    #
    #   * HEC-HMS / SAC-SMA — the canonical keys belong to the external jhechms
    #     and jsacsma packages. They are compatibility entries until those
    #     packages adopt the manifest field; keeping them means a config using
    #     the conventional hyphenated spelling keeps resolving meanwhile.
    #
    # (A former "SUMMA-MODFLOW" -> COUPLED_GW alias was removed: COUPLED_GW is
    # a calibration pipeline with no runner/preprocessor/postprocessor, so the
    # alias resolved to nothing at run time. HYDROLOGICAL_MODEL validation now
    # rejects it with a pointer to GROUNDWATER_MODEL: MODFLOW.)
    model_aliases = {
        "HEC-HMS": "HECHMS",
        "SAC-SMA": "SACSMA",
    }
    component_registries = (
        R.runners,
        R.preprocessors,
        R.postprocessors,
        R.optimizers,
        R.workers,
    )
    for alias_key, canonical in model_aliases.items():
        for registry in component_registries:
            # Never let an alias shadow a real registration of the same name.
            if alias_key.upper() in registry.keys():
                continue
            registry.alias(alias_key, canonical)


def _bootstrap_metrics(R: type) -> None:
    """Seed the unified metrics registry from the existing METRIC_REGISTRY.

    We import the existing metric registry dict and re-register each entry
    into ``R.metrics`` so that both old and new consumers see the same data.
    """
    try:
        from symfluence.core.metrics.metrics_registry import METRIC_REGISTRY
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
    R.metrics._normalize = lambda s: s

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

    from symfluence.core.contracts import (
        ContractCompatibilityError,
        assert_plugin_compatible,
    )

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

    in_tree_loaded = 0
    for ep in eps:
        try:
            plugin_fn = ep.load()
            assert_plugin_compatible(plugin_fn)
            plugin_fn()
            logger.debug("Loaded plugin %r from %s", ep.name, ep.value)
            if ep.value.startswith("symfluence.models."):
                in_tree_loaded += 1
        except ContractCompatibilityError as exc:
            logger.warning(
                "Plugin %r (%s) targets an incompatible SYMFLUENCE contract: "
                "%s. Registration was not attempted.",
                ep.name,
                ep.value,
                exc,
            )
            _FAILED_PLUGIN_ENTRY_POINTS.append((ep.name, ep.value))
        except ImportError as exc:
            from symfluence.core.exceptions import OptionalDependencyError
            missing_module = getattr(exc, "name", None) or ""
            if (
                isinstance(exc, OptionalDependencyError)
                or (
                    isinstance(exc, ModuleNotFoundError)
                    and missing_module.partition(".")[0] not in ("", "symfluence")
                )
            ):
                # A missing *third-party* module almost always means an optional
                # dependency isn't installed (e.g. an MPI/GPU model on a
                # laptop). Keep this quiet — the same models were debug-logged
                # by the old import loop — so it doesn't drown the logs on
                # every `import symfluence`.
                logger.debug(
                    "Plugin %r (%s) not loaded — optional dependency missing: %s",
                    ep.name,
                    ep.value,
                    exc,
                )
                _FAILED_PLUGIN_ENTRY_POINTS.append((ep.name, ep.value))
            else:
                # Any other ImportError (e.g. "cannot import name ... from
                # symfluence...") means the installed plugin was built against
                # a different SYMFLUENCE API. Burying this at DEBUG as an
                # "optional dependency" silently removes the plugin's models
                # from the registry, so calibration runs fail later with an
                # unhelpful "unknown model" error. Warn loudly instead.
                logger.warning(
                    "Plugin %r (%s) is incompatible with this SYMFLUENCE "
                    "version (%s). Its models will be unavailable — upgrade "
                    "the plugin package to a compatible release.",
                    ep.name,
                    ep.value,
                    exc,
                )
                _FAILED_PLUGIN_ENTRY_POINTS.append((ep.name, ep.value))
        except Exception:  # noqa: BLE001 — never let a broken plugin crash the framework
            logger.warning(
                "Failed to load symfluence plugin %r (%s); skipping.",
                ep.name,
                ep.value,
                exc_info=True,
            )
            _FAILED_PLUGIN_ENTRY_POINTS.append((ep.name, ep.value))

    # In-tree models register through these same entry points (declared in
    # SYMFLUENCE's own pyproject.toml). Discovering zero of them means the
    # installed dist metadata predates the entry-point declarations — almost
    # always a stale editable install. Without this signal the framework would
    # silently come up with no runnable models. We warn loudly rather than raise
    # so unusual-but-valid setups (e.g. partial vendoring) are not hard-blocked.
    if in_tree_loaded == 0:
        logger.error(
            "No in-tree SYMFLUENCE models were discovered via entry points. The "
            "installed package metadata is likely stale — reinstall with "
            "`pip install -e .` to regenerate it. Model runs will fail until then."
        )
