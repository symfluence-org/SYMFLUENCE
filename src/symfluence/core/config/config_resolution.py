# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model configuration resolution helpers.

Free functions that resolve a model's configuration components (schema,
defaults, field transformers, validator) from the unified ``R`` registry,
preferring a registered config adapter when one exists and falling back to
directly-registered values otherwise.

These replace the resolution methods that previously lived on the deprecated
``ConfigRegistry`` / ``ModelRegistry`` facade classes. They depend only on the
unified ``R`` registry (no model imports), so they live in ``core`` and the
config system imports them directly. (The re-export shim that used to sit at
``symfluence.models.config_resolution`` was removed under ADR-0011.)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Type

from symfluence.core.registries import R


def get_config_adapter(model_name: str) -> Optional[Any]:
    """Return an instantiated config adapter for *model_name*, or ``None``."""
    adapter_cls = R.config_adapters.get(model_name.upper())
    return adapter_cls(model_name) if adapter_cls else None


def get_config_schema(model_name: str) -> Optional[Type]:
    """Return the Pydantic config schema for *model_name* (adapter first)."""
    adapter = get_config_adapter(model_name)
    if adapter:
        return adapter.get_config_schema()
    return R.config_schemas.get(model_name.upper())


def get_config_defaults(model_name: str) -> Dict[str, Any]:
    """Return default config values for *model_name* (adapter first)."""
    adapter = get_config_adapter(model_name)
    if adapter:
        return adapter.get_defaults()
    return R.config_defaults.get(model_name.upper()) or {}


def get_config_transformers(model_name: str) -> Dict[str, Tuple[str, ...]]:
    """Return flat-to-nested field transformers for *model_name* (adapter first)."""
    adapter = get_config_adapter(model_name)
    if adapter:
        return adapter.get_field_transformers()
    return R.config_transformers.get(model_name.upper()) or {}


def get_config_validator(model_name: str) -> Optional[Callable]:
    """Return the config validator for *model_name* (adapter first)."""
    adapter = get_config_adapter(model_name)
    if adapter:
        return adapter.validate
    return R.config_validators.get(model_name.upper())


def validate_model_config(model_name: str, config: Dict[str, Any]) -> None:
    """Validate *config* for *model_name* using its registered validator (if any)."""
    validator = get_config_validator(model_name)
    if validator:
        validator(config)
