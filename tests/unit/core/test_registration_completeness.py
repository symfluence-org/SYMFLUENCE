# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""A model whose ``register()`` fails must not disappear quietly.

``core._bootstrap._discover_plugins`` deliberately never lets a broken plugin
take down the framework: any exception out of a model package's ``register()``
becomes ``logger.warning(...)`` and an absent model. That is the right runtime
behaviour and a terrible test posture — a package that loses its runner,
preprocessor, postprocessor, config schema, optimizer, worker AND parameter
manager all at once produces one log line and a green suite. A total WFLOW
registration failure was caught only by accident, via a 16-model spatial
capability table that happened to list it.

This module closes that hole two ways:

* **Completeness.** Every in-tree model package (derived from the installed
  ``symfluence.plugins`` entry points, not a hardcoded list) must resolve the
  execution-tier components ``Registries.validate_model`` calls required, plus
  its config schema. The models that legitimately register fewer are enumerated
  exactly, so both losing a component and quietly gaining one are red.
* **Silence.** A clean ``import symfluence`` in a fresh interpreter must emit
  no warning or error from the bootstrap logger, and must record no failed
  in-tree entry point. This is the guard that catches a registration failure
  in a package this file has no expectations about at all.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys

import pytest

import symfluence  # noqa: F401 — triggers plugin discovery -> model register()
from symfluence.core._bootstrap import PLUGIN_ENTRY_POINT_GROUP
from symfluence.core.registries import R

pytestmark = [pytest.mark.unit]


def _in_tree_model_packages() -> dict[str, str]:
    """Model registry key -> in-tree package name, from the entry points.

    Derived from what is installed rather than enumerated, so a new model
    package is covered the day it ships and a package that loses its entry
    point declaration disappears from the expectations loudly (see
    :func:`test_the_in_tree_model_set_is_not_empty`).
    """
    from importlib.metadata import entry_points

    packages = {}
    for ep in entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        if not ep.value.startswith("symfluence.models."):
            continue
        package = ep.value.split(":")[0].rsplit(".", 1)[-1]
        packages[package.upper()] = package
    return packages


_IN_TREE_MODELS = _in_tree_model_packages()

#: The components a model package must contribute to be runnable end to end.
#: ``runners``/``preprocessors``/``postprocessors`` are what
#: ``Registries.validate_model`` treats as required; ``config_schemas`` is added
#: because routing, parallel calibration and spatial-mode resolution all read
#: the registered ``ModelConfigSchema`` and a model without one silently falls
#: back to core defaults.
_REQUIRED_COMPONENTS = ("runners", "preprocessors", "postprocessors", "config_schemas")

#: In-tree packages that deliberately register fewer than the full set, with
#: the exact components each omits. Stated as an exact expectation in both
#: directions: a package losing a component fails, and a package *gaining* one
#: fails until the entry is removed here, so the exemption cannot outlive the
#: reason for it.
_KNOWN_PARTIAL_REGISTRATIONS = {
    # (GNN was exempt here until its runner/preprocessor/postprocessor were
    # registered. It had never registered them — `git log -S` back to the
    # registry unification — so ~1275 lines of importable, contract-conformant
    # code were unreachable and the model could not run at all. LSTM, equally
    # self_training=True, had always registered all three.)
    # Routing model: nothing post-processes a mizuRoute run through the model
    # postprocessor tier — the routed output is consumed by the analysis stack.
    "MIZUROUTE": {"postprocessors"},
    # A fire side-model coupled into RHESSYS rather than a standalone
    # hydrological model: it contributes a postprocessor and a schema, and is
    # driven by the RHESSYS runner rather than one of its own.
    "WMFIRE": {"runners", "preprocessors"},
}


def test_the_in_tree_model_set_is_not_empty():
    """Everything below is parametrized over this set — it may not be empty.

    Zero in-tree entry points means stale dist metadata (``pip install -e .``),
    which ``_discover_plugins`` already logs as an error. Without this, every
    completeness test in this file would pass vacuously in exactly the
    situation they exist to catch.
    """
    assert len(_IN_TREE_MODELS) >= 30, (
        f"only {len(_IN_TREE_MODELS)} in-tree model entry points discovered "
        f"({sorted(_IN_TREE_MODELS)}); the installed package metadata is "
        "probably stale — reinstall with `pip install -e .`"
    )


@pytest.mark.parametrize("model", sorted(_IN_TREE_MODELS))
def test_every_in_tree_model_registers_its_execution_components(model):
    """A model package's ``register()`` reached every seam it should.

    This is the assertion a swallowed ``register()`` exception has to trip: the
    package contributes nothing, so every component below resolves to None at
    once.
    """
    missing = {
        component for component in _REQUIRED_COMPONENTS
        if getattr(R, component).get(model) is None
    }
    expected_missing = _KNOWN_PARTIAL_REGISTRATIONS.get(model, set())

    assert missing == expected_missing, (
        f"{model} (symfluence.models.{_IN_TREE_MODELS[model]}) resolves "
        f"{sorted(_REQUIRED_COMPONENTS)} minus {sorted(missing)}; expected it "
        f"to be missing exactly {sorted(expected_missing)}. Either its "
        "register() raised and _discover_plugins swallowed it into a log line, "
        "or a deliberate registration change needs recording in "
        "_KNOWN_PARTIAL_REGISTRATIONS."
    )


def test_validate_model_agrees_the_registrations_are_complete():
    """The framework's own notion of completeness, over the same set.

    ``Registries.validate_model`` is what a consumer calls to decide a model is
    usable; the per-component test above would still pass if that helper's
    required tuple drifted away from what it checks.
    """
    invalid = {
        model: R.validate_model(model)["missing"]
        for model in sorted(_IN_TREE_MODELS)
        if not R.validate_model(model)["valid"]
        and model not in _KNOWN_PARTIAL_REGISTRATIONS
    }
    assert invalid == {}, (
        f"validate_model reports incomplete in-tree models: {invalid}"
    )


# ---------------------------------------------------------------------------
# Calibration tier
# ---------------------------------------------------------------------------

#: The in-tree models that are calibratable, frozen. Not every model
#: calibrates — routing models (MIZUROUTE, TROUTE), self-training models
#: (LSTM, GNN), the coupled-only side models (MODFLOW, WMFIRE) — so this is a
#: pinned set rather than a rule. Its job is to make a *loss* loud, which the
#: execution-tier test above cannot do for components it does not require.
_CALIBRATABLE_MODELS = {
    "CLM", "CLMPARFLOW", "CRHM", "CWATM", "FUSE", "GR", "GSFLOW", "HYPE",
    "IGNACIO", "LISFLOOD", "MESH", "MHM", "NGEN", "NOAHMP", "PARFLOW",
    "PCRGLOBWB", "PIHM", "PRMS", "RAVEN", "RHESSYS", "SUMMA", "SWAT", "VIC",
    "WATFLOOD", "WFLOW", "WRFHYDRO",
}

_CALIBRATION_REGISTRIES = ("workers", "optimizers", "parameter_managers")


@pytest.fixture(scope="module")
def drained_calibration_registries():
    """Fire every path that contributes a calibration component.

    Doing it explicitly makes this test independent of whatever else the
    session happened to import first, which is the difference between a frozen
    set and a flaky one.

    Note this fixture cannot detect the seeding gap it used to describe — it
    imports the deprecated ``optimization.parameter_managers`` shim itself, so
    the six models whose parameter managers only that shim reached would look
    registered either way. ``test_parameter_managers_register_without_the_
    deprecated_shim`` is the guard for that; keep them separate.
    """
    import importlib

    for module in (
        "symfluence.optimization.model_optimizers",
        "symfluence.optimization.parameter_managers",
        "symfluence.optimization.workers",
    ):
        importlib.import_module(module)
    for registry_name in _CALIBRATION_REGISTRIES:
        getattr(R, registry_name).load_modules()


@pytest.mark.parametrize("registry_name", _CALIBRATION_REGISTRIES)
def test_calibration_components_registered_for_the_same_models_as_ever(
        registry_name, drained_calibration_registries):
    """A model may not silently stop being calibratable.

    A failed registration here surfaces only as ``No optimizer registered for
    model: X`` at calibration time — long after the run that needed it was
    launched — because ``_autodiscover`` and ``_discover_plugins`` both
    downgrade the failure to a log line.
    """
    registry = getattr(R, registry_name)
    actual = {model for model in _IN_TREE_MODELS if registry.get(model) is not None}

    lost = sorted(_CALIBRATABLE_MODELS - actual)
    gained = sorted(actual - _CALIBRATABLE_MODELS)
    assert not lost, (
        f"in-tree models lost their {registry_name} registration: {lost}. "
        "A swallowed register()/import exception looks exactly like this."
    )
    assert not gained, (
        f"in-tree models newly register a {registry_name} entry: {gained} — "
        "record it in _CALIBRATABLE_MODELS so the set keeps its teeth"
    )


# ---------------------------------------------------------------------------
# A clean import must be silent
# ---------------------------------------------------------------------------

_CLEAN_IMPORT_PROBE = r'''
import json
import logging

_records = []


class _Collect(logging.Handler):
    def emit(self, record):
        _records.append((record.name, record.levelno, record.getMessage()))


logging.getLogger().addHandler(_Collect())

import symfluence  # noqa: F401 — the thing under test

from symfluence.core._bootstrap import failed_plugin_entry_points

print("PROBE" + json.dumps({
    "noisy": [
        [name, level, message] for name, level, message in _records
        if name.startswith("symfluence.core._bootstrap")
        and level >= logging.WARNING
    ],
    "failed": [list(pair) for pair in failed_plugin_entry_points()],
}))
'''


@pytest.fixture(scope="module")
def clean_import_probe() -> dict:
    """Result of importing ``symfluence`` in a fresh interpreter.

    A subprocess is required: ``bootstrap()`` is a once-per-process global, so
    by the time any test runs, plugin discovery has already happened and its
    log records are gone.
    """
    result = subprocess.run(
        [sys.executable, "-c", _CLEAN_IMPORT_PROBE],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"importing symfluence failed.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    payload = [ln for ln in result.stdout.splitlines() if ln.startswith("PROBE")]
    assert payload, f"probe produced no result.\nstdout:\n{result.stdout}"
    return json.loads(payload[-1][len("PROBE"):])


def test_plugin_discovery_logs_no_registration_warnings(clean_import_probe):
    """``import symfluence`` must not warn about a plugin on a healthy install.

    Every registration failure — a raising ``register()``, a contract-
    incompatible plugin, stale dist metadata — funnels through the
    ``core._bootstrap`` logger at WARNING or ERROR and is then discarded. This
    turns that log line back into a failure, for any plugin, whether or not
    this file knows the model exists.
    """
    noisy = [
        (name, logging.getLevelName(level), message)
        for name, level, message in clean_import_probe["noisy"]
    ]
    assert noisy == [], (
        "plugin discovery logged warnings on a clean import — a model is "
        "silently unavailable:\n" + "\n".join(f"  {n} {lvl}: {m}" for n, lvl, m in noisy)
    )


def test_no_in_tree_plugin_entry_point_failed(clean_import_probe):
    """The structured record of the same failure, independent of log capture.

    ``failed_plugin_entry_points()`` is consulted by ``cli.external_tools_config``
    to attempt a narrower recovery, so a non-empty in-tree entry means a model
    lost not just its Python components but its build instructions — the very
    thing that would let a user fix it.
    """
    failed_in_tree = sorted(
        (name, value) for name, value in clean_import_probe["failed"]
        if value.startswith("symfluence.models.")
    )
    assert failed_in_tree == [], (
        f"in-tree model entry points failed to load: {failed_in_tree}"
    )


_SHIM_FREE_SEEDING = r'''
import sys

import symfluence  # noqa: F401  — bootstrap runs plugin discovery
from symfluence.core.registries import R

# Reading the registry fires its deferred seeder. Nothing here imports the
# deprecated optimization.parameter_managers shim.
missing = [
    name for name in ("GR", "HYPE", "MESH", "NGEN", "PIHM", "RHESSYS")
    if R.parameter_managers.get(name) is None
]
assert not missing, f"no parameter manager registered for {missing}"

shim = "symfluence.optimization.parameter_managers"
assert shim not in sys.modules, (
    f"{shim} was imported — the registry is still being populated through the "
    "deprecated path rather than by the packages' own declarations"
)
print("SHIM-FREE-SEEDING-OK")
'''


def test_parameter_managers_register_without_the_deprecated_shim():
    """Six models' parameter managers must not depend on a 2.0-removal path.

    ``_seed_model_optimizers`` imports ``optimization.model_optimizers``, which
    auto-discovers only each package's ``calibration/optimizer`` module. The one
    pass that reached ``calibration/parameter_manager`` lived inside
    ``optimization.parameter_managers`` — a deprecated shim — so GR, HYPE, MESH,
    NGEN, PIHM and RHESSYS had no parameter manager after a plain
    ``import symfluence``, and ``component_factory.create_parameter_manager``
    raised for them. They are declared and drained now; this fails if that
    regresses, or if the shim quietly becomes load-bearing again.

    Runs in a subprocess because the assertion is about what a FRESH
    interpreter registers — in-process, any earlier test that touched the shim
    would mask it.
    """
    result = subprocess.run(
        [sys.executable, "-c", _SHIM_FREE_SEEDING],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"parameter-manager seeding regressed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "SHIM-FREE-SEEDING-OK" in result.stdout
