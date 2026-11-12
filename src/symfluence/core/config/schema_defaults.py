# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Schema defaults by flat key.

The Pydantic config models are the one place a default is declared. A
validated config carries those defaults; a hand-built flat dict (a test, a
script, a partially populated worker task) does not. ``schema_default`` lets
a reader of such a dict fall back to the declared default instead of
restating it, which is how inline defaults came to disagree with the schema.
"""
from __future__ import annotations

import threading
import typing
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

__all__ = ['NO_SCHEMA_DEFAULT', 'schema_default']

#: Returned for a key no schema declares, or whose field has no plain default.
NO_SCHEMA_DEFAULT: Any = object()

_LOCK = threading.Lock()
_DEFAULTS: Dict[str, Any] = {}
_BUILT_FOR: Optional[int] = None


def _nested_model(annotation: Any) -> Optional[Type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        found = _nested_model(arg)
        if found is not None:
            return found
    return None


def _collect(model: Type[BaseModel], into: Dict[str, Any], seen: set) -> None:
    if model in seen:
        return
    seen.add(model)
    for field in model.model_fields.values():
        nested = _nested_model(field.annotation)
        if nested is not None:
            _collect(nested, into, seen)
            continue
        alias = field.alias
        if not (isinstance(alias, str) and alias.isupper()):
            continue
        if field.default is PydanticUndefined:
            # Required, or built by a factory: nothing a reader can borrow.
            continue
        default = field.default
        into.setdefault(alias, getattr(default, 'value', default))


def _schemas() -> Tuple[Type[BaseModel], ...]:
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.core.registries import R

    models = [SymfluenceConfig]
    for name in sorted(R.config_schemas.keys()):
        schema = R.config_schemas.get(name)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            models.append(schema)
    return tuple(models)


def schema_default(flat_key: str) -> Any:
    """The default the config schema declares for ``flat_key``.

    Covers the framework sections and every registered model schema (plugins
    included, once they have registered). Returns :data:`NO_SCHEMA_DEFAULT`
    when no schema declares the key or the field is required or
    factory-built; ``None`` is a real answer and means the schema's default
    is "not set".
    """
    global _BUILT_FOR
    from symfluence.core.registries import R

    # Model schemas register as their packages (and plugins) are imported;
    # a changed count is the cue to collect again.
    registered = len(R.config_schemas)
    if registered != _BUILT_FOR:
        with _LOCK:
            if registered != _BUILT_FOR:
                collected: Dict[str, Any] = {}
                seen: set = set()
                for schema in _schemas():
                    _collect(schema, collected, seen)
                _DEFAULTS.clear()
                _DEFAULTS.update(collected)
                _BUILT_FOR = registered
    return _DEFAULTS.get(flat_key, NO_SCHEMA_DEFAULT)
