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
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _RegistryProxy(dict):
    """Dict-like proxy that reads from a :class:`Registry` instance.

    Used as a backward-compatibility shim so that code which accesses
    ``OldRegistry._internal_dict`` (e.g. ``'SUMMA' in ComponentRegistry._preprocessors``)
    continues to work after Phase 4 removed the real dicts.

    The proxy is read-only from the ``dict`` interface — writes go nowhere.
    All reads are forwarded to the underlying ``Registry``.
    """

    def __init__(self, registry: "Registry") -> None:
        # Do NOT call super().__init__() with data — keep the real dict empty.
        super().__init__()
        object.__setattr__(self, "_registry", registry)

    # --- read operations proxied to Registry ---
    def __contains__(self, key: object) -> bool:
        return key in self._registry  # type: ignore[operator]

    def __getitem__(self, key: str) -> Any:
        return self._registry[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._registry.get(key, default)

    def keys(self):  # type: ignore[override]
        return self._registry.keys()

    def values(self):  # type: ignore[override]
        return [v for _, v in self._registry.items()]

    def items(self):  # type: ignore[override]
        return self._registry.items()

    def __iter__(self) -> Iterator:
        return iter(self._registry)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return dict(self._registry.items()).__repr__()

    def __bool__(self) -> bool:
        return len(self._registry) > 0

    # --- write operations are no-ops (R.* is the source of truth) ---
    def __setitem__(self, key: str, value: Any) -> None:
        pass  # silently ignored — use R.* directly

    def __delitem__(self, key: str) -> None:
        pass


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

        # If the resolved attribute is a sub-module (e.g. build_instructions_module
        # pointed to "pkg.build_instructions" rather than "pkg.build_instructions.func"),
        # search inside it for a single callable provider and invoke it.
        if isinstance(attr, types.ModuleType):
            for obj in vars(attr).values():
                if callable(obj) and not isinstance(obj, type):
                    try:
                        result = obj()
                        if isinstance(result, dict):
                            return result
                    except Exception:  # noqa: BLE001
                        continue
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

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Return the registered value for *key*, or *default* on miss."""
        nkey = self._resolve_alias(self._normalize(key))
        entry = self._entries.get(nkey)
        if entry is None:
            return default
        return self._unwrap(nkey, entry)

    def __getitem__(self, key: str) -> T:
        """Return the registered value for *key*; raise ``KeyError`` on miss."""
        nkey = self._resolve_alias(self._normalize(key))
        entry = self._entries.get(nkey)
        if entry is None:
            available = sorted(self._entries.keys())
            raise KeyError(
                f"{self._name}: unknown key {key!r}. "
                f"Available: {available}"
            )
        return self._unwrap(nkey, entry)

    def __contains__(self, key: str) -> bool:  # noqa: D105
        nkey = self._resolve_alias(self._normalize(key))
        return nkey in self._entries

    def meta(self, key: str) -> Dict[str, Any]:
        """Return the metadata dict for *key* (empty dict if none)."""
        nkey = self._resolve_alias(self._normalize(key))
        return self._meta.get(nkey, {})

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """Return sorted list of canonical (non-alias) keys."""
        return sorted(self._entries.keys())

    def items(self) -> List[Tuple[str, T]]:
        """Return sorted list of ``(key, value)`` pairs, resolving lazy entries."""
        return [(k, self._unwrap(k, v)) for k, v in sorted(self._entries.items())]

    def __len__(self) -> int:  # noqa: D105
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:  # noqa: D105
        return iter(sorted(self._entries.keys()))

    def __repr__(self) -> str:  # noqa: D105
        return f"<Registry {self._name!r} ({len(self)} entries)>"

    def __bool__(self) -> bool:  # noqa: D105
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
        """Follow one level of aliasing."""
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
    build_instructions_module: Optional[str] = None,
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
    build_instructions_module : str, optional
        Dotted import path to the build instructions module — will be
        registered as a lazy import in ``R.build_instructions``.
    """
    # Deferred import to avoid circular dependency at module-parse time.
    from symfluence.core.registries import Registries as R

    runner_meta: Dict[str, Any] = {}
    if runner_method:
        runner_meta["runner_method"] = runner_method

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

    if build_instructions_module is not None:
        R.build_instructions.add_lazy(model_name, build_instructions_module)
