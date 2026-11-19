# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Registry-driven contract sweep — every registered component, implicitly.

The framework's source of truth for "what components exist" is the unified
registry (``symfluence.core.registries.R``), not any hand-maintained list.  This
module enumerates that registry at collection time and asserts a *contract* for
every entry, so a malformed or drifted registration fails the build the moment
it lands — without anyone writing a per-component test.

What it guarantees, for each class-based component registry:

* **Resolvable** — a lazy registration imports cleanly (optional-dependency
  models that aren't installed are *skipped*, not failed — the jFUSE/cFUSE
  pattern, generalized).
* **Is a class** — the registry stores classes; the caller instantiates.
* **Concrete** — no leftover ``@abstractmethod``, i.e. the class can actually be
  instantiated.  (This is what catches a model that subclasses a base but
  forgets to implement a required hook.)
* **Subclasses its base** — or is on a small, documented allow-list of
  standalone duck-typed classes, so *new* drift is blocked while existing
  intentional exceptions are recorded.
* **Exposes the expected methods** — a duck-typed safety net that also covers
  the allow-listed standalone classes.

Plus two model-level completeness guards driven by ``R.registered_models()`` and
``R.validate_model()``:

* every model that has a *runner* also has a *preprocessor* (a real invariant);
* the set of *incomplete* models matches a documented baseline, so a model can
  never silently lose a registered component.

Design note: parametrize IDs are built from ``reg.keys()``, which does **not**
resolve lazy imports — collection stays cheap and import-safe.  Resolution (and
the optional-dependency skip) happens inside each test body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Type

import pytest

import symfluence  # noqa: F401 — import triggers bootstrap / entry-point registration
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.modeling.base.base_config import ModelConfigAdapter
from symfluence.core.modeling.base.base_extractor import ModelResultExtractor
from symfluence.core.modeling.base.base_postprocessor import BaseModelPostProcessor
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.modeling.base.base_runner import BaseModelRunner
from symfluence.core.modeling.output_variables import OutputVariableAdapter
from symfluence.core.registries import R

# Base classes that registered components are expected to subclass.
from symfluence.data.acquisition.base import BaseAcquisitionHandler
from symfluence.data.observation.base import BaseObservationHandler

pytestmark = pytest.mark.unit


# ======================================================================
# Contract specification — one entry per class-based component registry
# ======================================================================


@dataclass(frozen=True)
class Contract:
    """The contract a single registry's entries must satisfy."""

    base: Type
    #: Methods every entry must expose as callables (duck-typed safety net).
    methods: Tuple[str, ...] = ()
    #: Non-empty string class attributes every entry must define.
    attrs: Tuple[str, ...] = ()
    #: Keys allowed to NOT subclass ``base`` (intentional standalone classes).
    #: Each value documents *why* it is exempt.  New drift is blocked.
    standalone: dict = field(default_factory=dict)


CONTRACTS: dict[str, Contract] = {
    "runners": Contract(BaseModelRunner, methods=("run",)),
    "preprocessors": Contract(BaseModelPreProcessor, methods=("run_preprocessing",)),
    "postprocessors": Contract(
        BaseModelPostProcessor,
        standalone={
            # WMFIRE is a wildfire model with no streamflow output; it does not
            # subclass the (streamflow-oriented) postprocessor base.
            "WMFIRE": "wildfire model, standalone postprocessor",
        },
    ),
    "workers": Contract(
        BaseWorker, methods=("apply_parameters", "run_model", "calculate_metrics")
    ),
    "parameter_managers": Contract(
        BaseParameterManager,
        methods=(
            "update_model_files",
            "get_initial_parameters",
            "normalize_parameters",
            "denormalize_parameters",
        ),
    ),
    "optimizers": Contract(BaseModelOptimizer),
    "result_extractors": Contract(
        ModelResultExtractor, methods=("get_output_file_patterns", "extract_variable")
    ),
    "output_adapters": Contract(
        OutputVariableAdapter, methods=("extract", "supported_variables"), attrs=("MODEL_NAME",)
    ),
    "config_adapters": Contract(
        ModelConfigAdapter,
        methods=("get_config_schema", "get_defaults", "get_field_transformers"),
    ),
    "acquisition_handlers": Contract(BaseAcquisitionHandler, methods=("download",)),
    "observation_handlers": Contract(
        BaseObservationHandler,
        methods=("acquire", "process"),
        attrs=("obs_type", "source_name"),
    ),
}


def _all_entries() -> List[Tuple[str, str]]:
    """(registry_name, key) for every entry — import-free (keys() is lazy-safe)."""
    pairs: List[Tuple[str, str]] = []
    for reg_name in CONTRACTS:
        reg = getattr(R, reg_name)
        for key in reg.keys():
            pairs.append((reg_name, key))
    return pairs


ENTRIES = _all_entries()


def _resolve_or_skip(reg_name: str, key: str):
    """Resolve a registry entry, skipping if its optional dependency is absent."""
    reg = getattr(R, reg_name)
    try:
        return reg.get(key)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"{reg_name}[{key}] optional dependency not installed: {exc}")


# ======================================================================
# Per-entry contract sweep
# ======================================================================


@pytest.mark.parametrize("reg_name,key", ENTRIES, ids=[f"{r}:{k}" for r, k in ENTRIES])
def test_entry_resolves_to_a_class(reg_name, key):
    """Every registration resolves to a class (lazy imports included)."""
    obj = _resolve_or_skip(reg_name, key)
    assert isinstance(obj, type), f"{reg_name}[{key}] resolved to {type(obj)!r}, not a class"


@pytest.mark.parametrize("reg_name,key", ENTRIES, ids=[f"{r}:{k}" for r, k in ENTRIES])
def test_entry_is_concrete(reg_name, key):
    """Every registered class is instantiable — no leftover abstract methods."""
    obj = _resolve_or_skip(reg_name, key)
    abstract = sorted(getattr(obj, "__abstractmethods__", ()))
    assert not abstract, (
        f"{reg_name}[{key}] = {obj.__module__}.{obj.__name__} is registered but "
        f"still abstract; unimplemented: {abstract}. It cannot be instantiated."
    )


@pytest.mark.parametrize("reg_name,key", ENTRIES, ids=[f"{r}:{k}" for r, k in ENTRIES])
def test_entry_subclasses_base_or_is_allowlisted(reg_name, key):
    """Every entry subclasses its base, or is a documented standalone class."""
    contract = CONTRACTS[reg_name]
    obj = _resolve_or_skip(reg_name, key)
    if issubclass(obj, contract.base):
        return
    assert key in contract.standalone, (
        f"{reg_name}[{key}] = {obj.__module__}.{obj.__name__} does not subclass "
        f"{contract.base.__name__} and is not in the documented standalone "
        f"allow-list. Either subclass the base, or add it to CONTRACTS['{reg_name}']"
        ".standalone with a reason."
    )


_ENTRIES_WITH_METHODS = [(r, k) for r, k in ENTRIES if CONTRACTS[r].methods]


@pytest.mark.parametrize(
    "reg_name,key",
    _ENTRIES_WITH_METHODS,
    ids=[f"{r}:{k}" for r, k in _ENTRIES_WITH_METHODS],
)
def test_entry_exposes_required_methods(reg_name, key):
    """Every entry exposes its registry's required methods as callables.

    Only registries that declare required methods are parametrized here;
    registries without a shared method contract (postprocessors, optimizers) are
    still covered by the resolves/concrete/subclass sweeps above.
    """
    contract = CONTRACTS[reg_name]
    obj = _resolve_or_skip(reg_name, key)
    missing = [m for m in contract.methods if not callable(getattr(obj, m, None))]
    assert not missing, (
        f"{reg_name}[{key}] = {obj.__module__}.{obj.__name__} is missing required "
        f"callables: {missing}"
    )


@pytest.mark.parametrize(
    "reg_name,key",
    [(r, k) for r, k in ENTRIES if CONTRACTS[r].attrs],
    ids=[f"{r}:{k}" for r, k in ENTRIES if CONTRACTS[r].attrs],
)
def test_entry_defines_required_attrs(reg_name, key):
    """Entries with required class attrs define them as non-empty strings."""
    contract = CONTRACTS[reg_name]
    obj = _resolve_or_skip(reg_name, key)
    for attr in contract.attrs:
        val = getattr(obj, attr, "")
        assert isinstance(val, str) and val.strip(), (
            f"{reg_name}[{key}] = {obj.__module__}.{obj.__name__} must set a "
            f"non-empty '{attr}' class attribute (got {val!r})"
        )


# ======================================================================
# Population sanity — guard against an empty/broken registry sweep
# ======================================================================


def test_sweep_covers_a_realistic_population():
    """The sweep is non-trivial — a broken bootstrap would collapse these counts."""
    counts = {name: len(getattr(R, name)) for name in CONTRACTS}
    assert counts["runners"] >= 30, counts
    assert counts["acquisition_handlers"] >= 50, counts
    assert counts["observation_handlers"] >= 30, counts
    assert len(ENTRIES) >= 300, f"only {len(ENTRIES)} entries swept"


# ======================================================================
# Model-level completeness — driven by R.validate_model()
# ======================================================================

# Models that intentionally do NOT carry the full runner+preprocessor+
# postprocessor triple, with the reason. validate_model() reports these as
# "invalid"; this baseline records why each is legitimately partial, so a model
# silently losing a component (or a new partial model appearing) fails the test.
INCOMPLETE_MODELS: dict[str, str] = {
    "CFUSE_ROUTED": "routed variant; reuses CFUSE runner/preprocessor",
    "HBV_ROUTED": "routed variant; reuses HBV runner/preprocessor",
    "JFUSE_ROUTED": "routed variant; reuses JFUSE runner/preprocessor",
    # GNN was here, described as running "via adapter". It did not run at all:
    # it had never registered a runner, so R.runners.get('GNN') was None while
    # its runner/preprocessor/postprocessor sat importable and unreachable.
    # Registered now, matching LSTM, which is equally self-training.
    "SNOW17": "conceptual/BMI model; runs via NGEN/BMI coupling, not standalone",
    "MIZUROUTE": "routing model; output handled by coupled postprocessing",
    "WMFIRE": "wildfire model; postprocessor-only registration",
}


# Models that are always present regardless of which optional plugins (jfuse,
# cfuse, …) are installed — the count of registered models varies by install
# extras, so assert on this stable core instead of a brittle absolute count.
_CORE_MODELS = {"SUMMA", "FUSE", "GR", "HYPE", "MIZUROUTE", "MESH", "VIC"}


def test_registered_models_is_nonempty():
    models = set(R.registered_models())
    assert len(models) >= 30, f"only {len(models)} models registered"
    missing_core = _CORE_MODELS - models
    assert not missing_core, f"core models missing from registry: {sorted(missing_core)}"


@pytest.mark.parametrize("model", sorted(R.registered_models()))
def test_validate_model_returns_wellformed_report(model):
    """validate_model() yields a structured, self-consistent report for each model."""
    report = R.validate_model(model)
    assert set(report) >= {"valid", "registered", "missing", "optional_missing"}
    assert report["valid"] == (len(report["missing"]) == 0)


_RUNNER_MODELS = sorted(m for m in R.registered_models() if R.runners.get(m) is not None)


@pytest.mark.parametrize("model", _RUNNER_MODELS)
def test_runner_implies_preprocessor(model):
    """A model that can run must also be able to preprocess its inputs."""
    assert R.preprocessors.get(model) is not None, (
        f"{model} has a runner but no preprocessor — it cannot prepare inputs."
    )


def test_incomplete_models_match_baseline():
    """The set of incomplete models matches the documented baseline (anti-drift).

    If this fails, a model either lost a registered component (fix the model) or a
    new legitimately-partial model was added (add it to INCOMPLETE_MODELS with a
    reason).
    """
    registered = set(R.registered_models())
    actual = {m for m in registered if not R.validate_model(m)["valid"]}
    # Only compare against baseline entries that are actually registered here:
    # optional plugins (jfuse/cfuse and their *_ROUTED variants) are absent in
    # some CI installs, and a baseline model that isn't installed simply can't
    # be incomplete. This still catches both real regressions (a registered
    # model becoming incomplete) and stale baseline entries.
    expected = {m for m in INCOMPLETE_MODELS if m in registered}
    newly_incomplete = sorted(actual - expected)
    newly_complete = sorted(expected - actual)
    assert actual == expected, (
        f"Model completeness drifted.\n"
        f"  Newly incomplete (lost a component, or new partial model): {newly_incomplete}\n"
        f"  Now complete (remove from INCOMPLETE_MODELS): {newly_complete}"
    )
