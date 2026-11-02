# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unified component registry for SYMFLUENCE.

Provides ``Registry[T]``, a single generic class that replaces all custom
registry implementations with a consistent, Pythonic API.  Also provides
``model_manifest()`` — a declarative one-liner that replaces the 15-25 line
boilerplate in each model's ``__init__.py``.

Design choices
--------------
* **UPPERCASE key normalization** by default (matches 14/18 existing registries).
  Configurable via the *normalize* constructor kwarg.
* **``get()`` returns ``None``; ``[]`` raises ``KeyError``** — dict-like API.
* **Always stores classes** — the caller instantiates.
* **Metadata per entry** — handles ``runner_method`` and future extensibility.
* **Lazy imports** — native ``add_lazy`` for the BMI-registry pattern.
* **Declared side-effect modules** — ``add_module()`` records a module whose
  *import* performs the registration (decorator-style), so a package can
  declare "my presets live here" without the framework globbing a source tree.
* **Aliases** — native ``alias()`` for the delineation-registry pattern.
* **Advisory protocol validation** — ``warnings.warn`` on registration when
  a class doesn't match the declared protocol; never blocks.
"""

from __future__ import annotations

import importlib
import logging
import types
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _LazyEntry:
    """Sentinel wrapping an import path for deferred resolution."""

    __slots__ = ("import_path",)

    def __init__(self, import_path: str) -> None:
        self.import_path = import_path

    def resolve(self) -> Any:
        module_path, class_name = self.import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        try:
            attr = getattr(module, class_name)
        except AttributeError:
            # The import path may refer to a module (not a class/attribute).
            # Fall back to importing the full dotted path as a module.
            attr = importlib.import_module(self.import_path)

        if isinstance(attr, types.ModuleType):
            # A module is not a registry value. This used to fall back to
            # calling every module-level callable with no arguments, swallowing
            # exceptions, and returning the first result that happened to be a
            # dict -- so an unrelated helper returning a dict would silently
            # become the registered entry, and a module with no such callable
            # was cached as the value itself. Modules whose *import* performs
            # the registration belong in add_module()/load_modules(), which is
            # what build_instructions (the only user of that fallback) now uses.
            raise TypeError(
                f"{self.import_path!r} resolves to a module, not a registry "
                f"value. If importing it performs the registration, declare it "
                f"with add_module() instead of add_lazy()."
            )
        return attr


class Registry(Generic[T]):
    """A generic, dict-like registry for SYMFLUENCE components.

    Parameters
    ----------
    name : str
        Human-readable name (used in ``__repr__`` and error messages).
    normalize : callable, optional
        Key normalization function.  Defaults to ``str.upper``.
    protocol : type or None, optional
        If given, newly-registered values are advisory-checked against this
        protocol (via ``isinstance`` for ``@runtime_checkable`` protocols,
        or ``hasattr`` probing otherwise).
    doc : str, optional
        One-line description shown in :meth:`summary`.
    """

    def __init__(
        self,
        name: str,
        *,
        normalize: Callable[[str], str] = str.upper,
        protocol: Optional[Type] = None,
        doc: str = "",
    ) -> None:
        self._name = name
        self._normalize = normalize
        self._protocol = protocol
        self._doc = doc
        self._entries: Dict[str, Any] = {}       # key -> value (class) or _LazyEntry
        self._meta: Dict[str, Dict[str, Any]] = {}  # key -> metadata dict
        self._aliases: Dict[str, str] = {}        # alias_key -> canonical_key
        self._frozen = False
        self._seeder: Optional[Callable[[], None]] = None  # deferred population hook
        self._modules: List[str] = []             # declared side-effect modules
        self._loaded_modules: set[str] = set()    # already imported by load_modules()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(
        self,
        key: str,
        value: Optional[T] = None,
        **meta: Any,
    ) -> Union[T, Callable[[T], T]]:
        """Register *value* under *key*, or return a decorator if *value* is ``None``.

        Examples
        --------
        Direct form::

            R.runners.add("SUMMA", SummaRunner, runner_method="run")

        Decorator form::

            @R.runners.add("SUMMA", runner_method="run")
            class SummaRunner: ...
        """
        if value is not None:
            self._set(key, value, meta)
            return value  # type: ignore[return-value]

        # Decorator form
        def decorator(cls: T) -> T:
            self._set(key, cls, meta)
            return cls

        return decorator

    def add_lazy(self, key: str, import_path: str, **meta: Any) -> None:
        """Register a lazy import — the class is imported on first access.

        Parameters
        ----------
        key : str
            Registry key.
        import_path : str
            Fully-qualified ``"package.module.ClassName"`` string.
        """
        self._check_frozen()
        nkey = self._normalize(key)
        self._entries[nkey] = _LazyEntry(import_path)
        if meta:
            self._meta[nkey] = meta

    def add_module(self, module_path: str) -> None:
        """Declare a module whose *import* registers entries in this registry.

        Some components register themselves as a decorator side effect when a
        per-package submodule is imported (``@R.presets.add('fuse-basic')`` in
        ``<pkg>.init_preset``).  The registry cannot see them until something
        imports that module, and the framework must not go looking for it on
        disk — that is a filesystem dependency on a package that may live in a
        separate distribution (or not be installed at all).

        Instead the owning package *declares* the module here, and the
        consumer drains the declarations with :meth:`load_modules`.

        Declarations are idempotent; importing happens at most once per
        module path.
        """
        self._check_frozen()
        if module_path not in self._modules:
            self._modules.append(module_path)

    def declared_modules(self) -> Tuple[str, ...]:
        """Return the module paths declared via :meth:`add_module`."""
        return tuple(self._modules)

    def load_modules(self) -> None:
        """Import every declared module not yet imported, for its side effects.

        Deliberately *not* wired into read access (``get``/``keys``/...): a
        consumer opts in by calling this, which keeps the set of entries a
        plain registry read returns exactly what it returns today.  Modules
        that cannot be imported are skipped with a debug log, matching the
        tolerance the per-consumer import loops had.
        """
        for module_path in list(self._modules):
            if module_path in self._loaded_modules:
                continue
            self._loaded_modules.add(module_path)
            try:
                importlib.import_module(module_path)
            except ImportError:
                logger.debug(
                    "%s: declared module %r is not importable", self._name, module_path
                )

    def alias(self, alias_key: str, canonical_key: str) -> None:
        """Create *alias_key* as an alias for *canonical_key*.

        Both keys are normalized.  The canonical key need not be registered
        yet (it will be resolved at lookup time).
        """
        self._check_frozen()
        nalias = self._normalize(alias_key)
        ncanon = self._normalize(canonical_key)
        self._aliases[nalias] = ncanon

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def set_seeder(self, seeder: Callable[[], None]) -> None:
        """Install a deferred population hook, invoked on first read access.

        Lets startup code declare *how* to populate a registry without paying
        the import cost until something actually reads it (e.g. the metrics
        registry pulls the evaluation stack, which ``--version`` never needs).
        """
        self._check_frozen()
        self._seeder = seeder

    def _ensure_seeded(self) -> None:
        if self._seeder is not None:
            # Clear before running so a seeder that reads the registry
            # cannot recurse.
            seeder, self._seeder = self._seeder, None
            seeder()

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Return the registered value for *key*, or *default* on miss."""
        self._ensure_seeded()
        nkey = self._resolve_alias(self._normalize(key))
        entry = self._entries.get(nkey)
        if entry is None:
            return default
        return self._unwrap(nkey, entry)

    def __getitem__(self, key: str) -> T:
        """Return the registered value for *key*; raise ``KeyError`` on miss."""
        self._ensure_seeded()
        nkey = self._resolve_alias(self._normalize(key))
        entry = self._entries.get(nkey)
        if entry is None:
            available = sorted(self._entries.keys())
            raise KeyError(
                f"{self._name}: unknown key {key!r}. "
                f"Available: {available}"
            )
        return self._unwrap(nkey, entry)

    def __contains__(self, key: str) -> bool:
        self._ensure_seeded()
        nkey = self._resolve_alias(self._normalize(key))
        return nkey in self._entries

    def meta(self, key: str) -> Dict[str, Any]:
        """Return the metadata dict for *key* (empty dict if none)."""
        self._ensure_seeded()
        nkey = self._resolve_alias(self._normalize(key))
        return self._meta.get(nkey, {})

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """Return sorted list of canonical (non-alias) keys."""
        self._ensure_seeded()
        return sorted(self._entries.keys())

    def items(self) -> List[Tuple[str, T]]:
        """Return sorted list of ``(key, value)`` pairs, resolving lazy entries."""
        self._ensure_seeded()
        return [(k, self._unwrap(k, v)) for k, v in sorted(self._entries.items())]

    def __len__(self) -> int:
        self._ensure_seeded()
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        self._ensure_seeded()
        return iter(sorted(self._entries.keys()))

    def __repr__(self) -> str:
        return f"<Registry {self._name!r} ({len(self)} entries)>"

    def __bool__(self) -> bool:
        return True  # a Registry instance is always truthy

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarizing this registry."""
        return {
            "name": self._name,
            "doc": self._doc,
            "entries": len(self._entries),
            "aliases": len(self._aliases),
            "keys": self.keys(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries, aliases, and metadata.  Unfreezes."""
        self._entries.clear()
        self._meta.clear()
        self._aliases.clear()
        self._frozen = False

    def freeze(self) -> None:
        """Prevent further mutations (advisory; for post-bootstrap safety)."""
        # Run any pending seeder first — a frozen registry can no longer
        # accept the seeder's registrations.
        self._ensure_seeded()
        self._frozen = True

    def remove(self, key: str) -> None:
        """Remove *key* (and its metadata).  Does **not** remove aliases pointing to it."""
        self._check_frozen()
        nkey = self._normalize(key)
        self._entries.pop(nkey, None)
        self._meta.pop(nkey, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set(self, key: str, value: Any, meta: Dict[str, Any]) -> None:
        self._check_frozen()
        nkey = self._normalize(key)
        self._validate_protocol(value, nkey)
        self._entries[nkey] = value
        if meta:
            self._meta[nkey] = meta

    def _resolve_alias(self, nkey: str) -> str:
        """Follow one level of aliasing.

        A real registration always wins over an alias of the same name,
        whichever was declared first.
        """
        if nkey in self._entries:
            return nkey
        return self._aliases.get(nkey, nkey)

    def _unwrap(self, nkey: str, entry: Any) -> T:
        """Resolve a ``_LazyEntry`` on first access."""
        if isinstance(entry, _LazyEntry):
            resolved = entry.resolve()
            self._entries[nkey] = resolved  # cache
            self._validate_protocol(resolved, nkey)
            return resolved  # type: ignore[return-value]
        return entry  # type: ignore[return-value]

    def _check_frozen(self) -> None:
        if self._frozen:
            raise RuntimeError(
                f"Registry {self._name!r} is frozen; mutations are not allowed."
            )

    def _validate_protocol(self, value: Any, nkey: str) -> None:
        """Advisory protocol check — warns but never blocks."""
        if self._protocol is None:
            return
        try:
            if isinstance(value, type):
                # Collect expected attributes from protocol annotations,
                # abstract methods, and __protocol_attrs__ (if present).
                attrs: set[str] = set()
                if hasattr(self._protocol, "__protocol_attrs__"):
                    attrs.update(self._protocol.__protocol_attrs__)  # type: ignore[attr-defined]
                if hasattr(self._protocol, "__abstractmethods__"):
                    attrs.update(self._protocol.__abstractmethods__)
                # Protocol-defined annotations (e.g. MODEL_NAME: str)
                for cls in self._protocol.__mro__:
                    if cls is object:
                        continue
                    for attr in getattr(cls, "__annotations__", {}):
                        if not attr.startswith("_"):
                            attrs.add(attr)
                # Protocol-defined methods (non-dunder, non-private)
                for attr in vars(self._protocol):
                    if not attr.startswith("_") and callable(
                        getattr(self._protocol, attr, None)
                    ):
                        attrs.add(attr)
                if not attrs:
                    return
                missing = [a for a in sorted(attrs) if not hasattr(value, a)]
                if missing:
                    warnings.warn(
                        f"{self._name}: {value!r} registered under "
                        f"{nkey!r} may not satisfy {self._protocol.__name__}; "
                        f"missing: {missing}",
                        stacklevel=4,
                    )
        except Exception:  # noqa: BLE001 — advisory only, never fail
            pass


# ======================================================================
# model_manifest() — declarative per-model registration
# ======================================================================

#: Models whose calibration is internal training rather than an external
#: parameter search, declared via ``model_manifest(self_training=True)``.
#: Owned by the declaring package: core cannot know which models an installed
#: plugin trains internally, and the hardcoded list this replaced could only
#: ever describe the in-tree suite.
_SELF_TRAINING_MODELS: set[str] = set()


def self_training_models() -> frozenset[str]:
    """Canonical keys of models that train internally during the run step."""
    return frozenset(_SELF_TRAINING_MODELS)


def model_manifest(
    model_name: str,
    *,
    preprocessor: Optional[Type] = None,
    runner: Optional[Type] = None,
    runner_method: Optional[str] = None,
    postprocessor: Optional[Type] = None,
    visualizer: Optional[Any] = None,
    config_adapter: Optional[Type] = None,
    config_schema: Optional[Type] = None,
    config_defaults: Optional[Dict[str, Any]] = None,
    config_transformers: Optional[Dict] = None,
    config_validator: Optional[Any] = None,
    result_extractor: Optional[Type] = None,
    optimizer: Optional[Type] = None,
    worker: Optional[Type] = None,
    parameter_manager: Optional[Type] = None,
    decision_analyzer: Optional[Type] = None,
    sensitivity_analyzer: Optional[Type] = None,
    koopman_analyzer: Optional[Type] = None,
    plotter: Optional[Type] = None,
    forcing_adapter: Optional[Type] = None,
    forcing_adapter_module: Optional[str] = None,
    init_preset_module: Optional[str] = None,
    build_instructions_module: Optional[str] = None,
    aliases: Optional[Sequence[str]] = None,
    self_training: bool = False,
) -> None:
    """Declaratively register all components for a single model.

    Replaces the 15-25 line boilerplate in each model's ``__init__.py``
    with a single call.

    Parameters
    ----------
    model_name : str
        Canonical model identifier (e.g. ``"SUMMA"``).
    preprocessor, runner, postprocessor, visualizer : type, optional
        Execution-layer component classes.
    runner_method : str, optional
        Name of the run method on *runner* (default ``"run"``).
    config_adapter, config_schema, config_defaults,
    config_transformers, config_validator : optional
        Configuration-layer components.
    result_extractor : type, optional
        Result-extraction class.
    optimizer, worker, parameter_manager : type, optional
        Optimization-layer classes.
    decision_analyzer, sensitivity_analyzer, koopman_analyzer : type, optional
        Analysis-layer classes.
    plotter : type, optional
        Plotter class.
    forcing_adapter : type, optional
        Forcing adapter class.
    forcing_adapter_module : str, optional
        Dotted import path to a module whose import registers this model's
        forcing adapter (``@R.forcing_adapters.add(...)``).  Declared into
        ``R.forcing_adapters`` via ``add_module``; imported when a consumer
        drains the declarations.  Use this *instead of* ``forcing_adapter``
        when the adapter class pulls heavy dependencies that must not load at
        plugin-discovery time.
    init_preset_module : str, optional
        Dotted import path to a module whose import registers this model's
        ``symfluence init`` presets (``@R.presets.add(...)``).  Declared into
        ``R.presets`` via ``add_module``.
    build_instructions_module : str, optional
        Dotted import path to the build instructions module — will be
        registered as a lazy import in ``R.build_instructions``.
    aliases : sequence of str, optional
        Alternate spellings that should resolve to *model_name* across every
        component registry — hyphenated forms (``"HEC-HMS"`` for ``HECHMS``),
        short names (``"RHESS"`` for ``RHESSYS``), or a legacy key. Declaring
        them here keeps the mapping with the package that owns the canonical
        name; core previously carried a hardcoded table it could not know was
        complete.
    self_training : bool, default False
        True for models whose "calibration" is internal training during the run
        step (gradient descent) rather than an external DDS/PSO parameter
        search. They register no optimizer or worker and have no calibrated
        parameters, so the calibration and sensitivity-analysis paths skip them
        instead of reporting a failure. Read back via
        ``SupportedModels.SELF_TRAINING``.
    """
    # Deferred import to avoid circular dependency at module-parse time.
    from symfluence.core.registries import Registries as R

    runner_meta: Dict[str, Any] = {}
    if runner_method:
        runner_meta["runner_method"] = runner_method

    # Bridge an adapter-provided config schema into R.config_schemas when no
    # explicit schema is given. The ModelConfig validator resolves typed
    # model configs from R.config_schemas; plugins that register only a
    # config_adapter (e.g. the JAX models) would otherwise be absent there, so
    # config.model.<model> is never built and the plugin silently runs on schema
    # defaults instead of the user's settings.
    if config_schema is None and config_adapter is not None:
        # Best-effort: a malformed adapter must not break model registration
        # (which runs at import time). Narrow to realistic adapter failures.
        try:
            config_schema = config_adapter(model_name).get_config_schema()
        except (TypeError, AttributeError, ValueError, RuntimeError, ImportError):
            config_schema = None

    _pairs: list[tuple[Registry, str, Any, Dict[str, Any]]] = [
        (R.preprocessors,          model_name, preprocessor,          {}),
        (R.runners,                model_name, runner,                runner_meta),
        (R.postprocessors,         model_name, postprocessor,         {}),
        (R.visualizers,            model_name, visualizer,            {}),
        (R.config_adapters,        model_name, config_adapter,        {}),
        (R.config_schemas,         model_name, config_schema,         {}),
        (R.config_defaults,        model_name, config_defaults,       {}),
        (R.config_transformers,    model_name, config_transformers,   {}),
        (R.config_validators,      model_name, config_validator,      {}),
        (R.result_extractors,      model_name, result_extractor,      {}),
        (R.optimizers,             model_name, optimizer,             {}),
        (R.workers,                model_name, worker,                {}),
        (R.parameter_managers,     model_name, parameter_manager,     {}),
        (R.decision_analyzers,     model_name, decision_analyzer,     {}),
        (R.sensitivity_analyzers,  model_name, sensitivity_analyzer,  {}),
        (R.koopman_analyzers,      model_name, koopman_analyzer,      {}),
        (R.plotters,               model_name, plotter,               {}),
        (R.forcing_adapters,       model_name, forcing_adapter,       {}),
    ]

    for registry, key, value, meta in _pairs:
        if value is not None:
            registry.add(key, value, **meta)

    # Capability modules: the model declares *where* a decorator-registered
    # capability lives; the framework imports it only when a consumer asks.
    if forcing_adapter_module is not None:
        R.forcing_adapters.add_module(forcing_adapter_module)

    if init_preset_module is not None:
        R.presets.add_module(init_preset_module)

    if build_instructions_module is not None:
        # Declared, not lazily keyed. Every build-instructions module registers
        # itself with @R.build_instructions.add('<tool>'), so importing it is
        # what produces the entry -- and the tool name is not always the model
        # name. Keying a lazy sentinel under model_name instead used to collide
        # with the decorator's entry on the same (lower-cased) key, and resolving
        # that sentinel invoked a heuristic that called every module-level
        # callable until one returned a dict.
        R.build_instructions.add_module(build_instructions_module)

    # Alternate spellings resolve to the canonical key in the five registries a
    # model is looked up BY NAME in -- not in all eighteen a manifest populates,
    # so e.g. R.result_extractors and R.config_schemas are reachable only under
    # the canonical key. That matches what core/_bootstrap did before aliases
    # became declarable; widening it is a deliberate change, not a tidy-up.
    # Aliases resolve lazily at lookup time, so declaring one before the
    # canonical registration is fine.
    #
    # No "is the alias already a real key?" check here: real registrations
    # take precedence at lookup (Registry._resolve_alias), regardless of
    # order. Checking via registry.keys() at registration time forced the
    # lazy optimizer/worker seeders, which imported every model's calibration
    # stack (and GR's embedded R) on every `import symfluence`.
    for alias in aliases or ():
        for registry in (R.runners, R.preprocessors, R.postprocessors,
                         R.optimizers, R.workers):
            registry.alias(alias, model_name)

    if self_training:
        _SELF_TRAINING_MODELS.add(model_name.upper())
