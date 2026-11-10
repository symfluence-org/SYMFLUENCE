# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tiered validation of unrecognized flat configuration keys.

SYMFLUENCE configs are flat, uppercase YAML (``DOMAIN_NAME: ...``). The base
config models keep ``extra='allow'`` so that plugins can inject their own keys,
which means a typo such as ``HYDROLOGICAL_MDOEL`` is otherwise accepted
silently (RTI architectural review, open question 3 / Tier 3 item 21).

Rather than flipping ``extra='forbid'`` on the Pydantic models — which would
also reject legitimate plugin keys and acts at the wrong (post flat->nested)
layer — this module validates raw flat keys at ingestion against an allowlist
that already unions:

* every core config alias,
* every registered plugin schema (``R.config_schemas``), and
* every legacy alias.

The *response* is tiered, not the allowlist:

* **warn by default** — log a warning naming each unknown key with a
  "did you mean?" suggestion, so existing configs keep loading unchanged;
* **strict mode** — raise :class:`ConfigValidationError` instead. Strict is
  enabled per-config via a ``STRICT_CONFIG`` key, or globally via the existing
  ``SYMFLUENCE_STRICT_CONFIG`` environment variable (shared with
  :mod:`symfluence.core.config.coercion`).

Escape hatch: list genuinely freeform keys under ``ALLOW_UNKNOWN_KEYS`` to
suppress the warning/error for those keys without registering a schema.
"""
from __future__ import annotations

import logging
import os
from difflib import get_close_matches
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Control keys consumed by this validator itself — never reported as unknown.
RESERVED_CONTROL_KEYS = frozenset({'STRICT_CONFIG', 'ALLOW_UNKNOWN_KEYS'})

# Truthy spellings shared with coercion's SYMFLUENCE_STRICT_CONFIG handling.
_TRUTHY = ('true', '1', 'yes')


def _env_strict() -> bool:
    """Return True if strict config mode is requested via the environment."""
    return os.environ.get('SYMFLUENCE_STRICT_CONFIG', '').strip().lower() in _TRUTHY


def is_strict_config_mode(flat_config: Optional[Dict[str, Any]] = None) -> bool:
    """Resolve whether unknown keys should be a hard error.

    Precedence: an explicit per-config ``STRICT_CONFIG`` key (truthy) wins over
    the ``SYMFLUENCE_STRICT_CONFIG`` environment variable, which in turn wins
    over the default (warn-only).
    """
    if flat_config:
        for key in ('STRICT_CONFIG', 'strict_config'):
            if key in flat_config:
                value = flat_config[key]
                if isinstance(value, str):
                    return value.strip().lower() in _TRUTHY
                return bool(value)
    return _env_strict()


def _allowed_unknown(flat_config: Dict[str, Any]) -> Set[str]:
    """Extract the user-declared ``ALLOW_UNKNOWN_KEYS`` allowlist (uppercased)."""
    raw = flat_config.get('ALLOW_UNKNOWN_KEYS') or flat_config.get('allow_unknown_keys')
    if not raw:
        return set()
    if isinstance(raw, str):
        items: Iterable[str] = (part.strip() for part in raw.split(','))
    elif isinstance(raw, (list, tuple, set)):
        items = (str(part).strip() for part in raw)
    else:
        return set()
    return {item.upper() for item in items if item}


def find_unknown_keys(
    flat_config: Dict[str, Any],
    known_keys: Iterable[str],
    *,
    allow_unknown: Iterable[str] = (),
) -> List[str]:
    """Return sorted flat keys not present in *known_keys*.

    Reserved control keys and any key listed in ``ALLOW_UNKNOWN_KEYS`` (merged
    with the explicit *allow_unknown* argument) are excluded. Comparison is
    case-insensitive on the canonical uppercase form.
    """
    known = {str(k).upper() for k in known_keys}
    allowed = {str(k).upper() for k in allow_unknown} | _allowed_unknown(flat_config)
    allowed |= RESERVED_CONTROL_KEYS

    unknown: Set[str] = set()
    for key in flat_config:
        if not isinstance(key, str):
            continue
        upper = key.upper()
        if upper in known or upper in allowed:
            continue
        unknown.add(upper)
    return sorted(unknown)


def _resolve_basemodel_annotation(annotation: Any) -> Optional[type]:
    """Return the BaseModel subclass a field annotation resolves to, if any.

    Unwraps ``Optional``/``Union``/``Annotated`` wrappers. Mapping (``Dict``)
    and sequence (``List``) annotations deliberately resolve to ``None`` — their
    contents are opaque to nested-key validation.
    """
    import types
    import typing

    from pydantic import BaseModel

    for _ in range(10):  # defensive bound against pathological nesting
        origin = typing.get_origin(annotation)
        if origin is None:
            break
        args = typing.get_args(annotation)
        if origin is typing.Annotated:  # Annotated[X, ...] -> X
            annotation = args[0]
            continue
        if origin is typing.Union or origin is types.UnionType:
            for arg in args:
                resolved = _resolve_basemodel_annotation(arg)
                if resolved is not None:
                    return resolved
            return None
        # Dict[...], List[...], and other generics: opaque.
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _field_for_key(model: type, key: str) -> Optional[Any]:
    """Return the FieldInfo that input *key* populates, or None.

    With ``populate_by_name=True`` both spellings are legal input, so a key is
    known if it matches the field name, the field ``alias``, or any string
    choice in a ``validation_alias`` (plain string or ``AliasChoices``).
    """
    from pydantic import AliasChoices

    for name, field in model.model_fields.items():
        if key == name or key == field.alias:
            return field
        validation_alias = field.validation_alias
        if isinstance(validation_alias, str) and key == validation_alias:
            return field
        if isinstance(validation_alias, AliasChoices) and key in [
            c for c in validation_alias.choices if isinstance(c, str)
        ]:
            return field
    return None


def _known_flat_universe(data: Dict[str, Any]) -> Set[str]:
    """The recognized flat-key universe (core ∪ plugins ∪ legacy ∪ model transformers).

    Mirrors the allowlist used by the strict shipped-configs guard. The
    hydrological model (needed for model-specific transformer keys) is looked
    up in both the flat and nested spellings.
    """
    from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS
    from symfluence.core.config.transformers import build_combined_flat_to_nested_map

    hydrological_model = data.get('HYDROLOGICAL_MODEL')
    if not hydrological_model:
        model_section = data.get('model')
        if isinstance(model_section, dict):
            hydrological_model = model_section.get('hydrological_model') or model_section.get('HYDROLOGICAL_MODEL')
    if isinstance(hydrological_model, list):
        hydrological_model = ','.join(str(m) for m in hydrological_model)

    return (
        set(build_combined_flat_to_nested_map(hydrological_model))
        | set(RECOGNIZED_FLAT_KEYS)
        | set(RESERVED_CONTROL_KEYS)
    )


def _walk_nested_keys(data: Dict[str, Any], model: type, prefix: str, unknown: List[str]) -> None:
    """Recursively collect unknown keys in *data* against Pydantic *model*."""
    from symfluence.core.config.models.model_configs import ModelConfig

    for key, value in data.items():
        if not isinstance(key, str) or key == '_extra':
            continue
        field = _field_for_key(model, key)
        if field is None:
            if model is ModelConfig:
                # model.<model_name> subsections are registry-typed, not fields.
                # Resolve via R.config_schemas; skip unresolvable names (e.g.
                # external-plugin models absent from this environment) so a
                # missing plugin never produces a false positive.
                from symfluence.core.registries import R

                schema_cls = R.config_schemas.get(key.upper())
                if schema_cls is not None and isinstance(value, dict):
                    _walk_nested_keys(value, schema_cls, f"{prefix}{key}.", unknown)
                continue
            unknown.append(f"{prefix}{key}")
            continue
        if isinstance(value, dict):
            child_model = _resolve_basemodel_annotation(field.annotation)
            if child_model is not None:
                _walk_nested_keys(value, child_model, f"{prefix}{key}.", unknown)


def find_unknown_nested_keys(data: Dict[str, Any], model: Optional[type] = None) -> List[str]:
    """Return dotted paths of unknown keys in a *nested* config dict.

    Walks *data* against the Pydantic model tree rooted at *model* (default:
    :class:`SymfluenceConfig`). At each dict level backed by a ``BaseModel``, a
    key is known if it matches a field name, a field ``alias``, or any string
    choice in a ``validation_alias`` ``AliasChoices`` (``populate_by_name=True``
    makes both spellings legal). Anything else is reported as a dotted path,
    e.g. ``"evaluation.snotel.station_id"``.

    Descends only into fields whose annotation (after unwrapping
    ``Optional``/``Union``/``Annotated``) is a ``BaseModel`` subclass; mapping
    (``Dict[str, X]``) and list fields are opaque. The ``model.<name>``
    subsections resolve through ``R.config_schemas``; unresolvable subsections
    (absent plugins, ``Any``-typed) are skipped rather than flagged.

    Top-level UPPERCASE keys are flat-style keys mixed into a nested config and
    are checked against the recognized flat-key universe instead (same
    allowlist as strict flat validation), honoring legacy normalization
    aliases. The reserved ``_extra`` key is always skipped.
    """
    if model is None:
        from symfluence.core.config.models import SymfluenceConfig
        model = SymfluenceConfig

    unknown: List[str] = []
    known_flat: Optional[Set[str]] = None
    section_names = set(model.model_fields)
    nested: Dict[str, Any] = {}

    for key, value in data.items():
        if not isinstance(key, str) or key == '_extra':
            continue
        # Section keys are matched case-insensitively at the top level
        # (mirrors factories._normalize_nested_config, which accepts SYSTEM:).
        if key.lower() in section_names and key not in model.model_fields:
            nested[key.lower()] = value
            continue
        if key == key.upper() and _field_for_key(model, key) is None:
            # Flat-style uppercase key mixed into a nested config: check it
            # against the flat universe (with legacy normalization aliases).
            from symfluence.core.config.legacy_aliases import NORMALIZATION_ALIASES

            if known_flat is None:
                known_flat = _known_flat_universe(data)
            if NORMALIZATION_ALIASES.get(key, key) not in known_flat:
                unknown.append(key)
            continue
        nested[key] = value

    _walk_nested_keys(nested, model, "", unknown)
    return sorted(unknown)


def _sibling_names(path: str, model: Optional[type] = None) -> Set[str]:
    """Field names/aliases valid next to the dotted *path* (for suggestions)."""
    if model is None:
        from symfluence.core.config.models import SymfluenceConfig
        model = SymfluenceConfig
    current: Optional[type] = model
    for part in path.split('.')[:-1]:
        field = _field_for_key(current, part) if current is not None else None
        current = _resolve_basemodel_annotation(field.annotation) if field is not None else None
        if current is None:
            return set()
    return set(current.model_fields)


def validate_known_nested_keys(
    nested_config: Dict[str, Any],
    *,
    strict: Optional[bool] = None,
    source: Optional[str] = None,
) -> List[str]:
    """Nested-format counterpart of :func:`validate_known_flat_keys`.

    Unknown keys are reported as dotted paths (``domain.download_land_cover``)
    with a did-you-mean suggestion from the same section. Warns by default;
    raises :class:`ConfigValidationError` in strict mode (top-level
    ``STRICT_CONFIG`` / ``system.strict_config`` or ``SYMFLUENCE_STRICT_CONFIG``).
    ``ALLOW_UNKNOWN_KEYS`` entries may name a dotted path or a leaf key.
    """
    unknown = find_unknown_nested_keys(nested_config)
    if not unknown:
        return []

    control: Dict[str, Any] = {
        k: v for k, v in nested_config.items()
        if isinstance(k, str) and k.upper() in RESERVED_CONTROL_KEYS
    }
    system = nested_config.get('system')
    if isinstance(system, dict):
        for k, v in system.items():
            if isinstance(k, str) and k.upper() in RESERVED_CONTROL_KEYS:
                control.setdefault(k.upper(), v)
    allowed = _allowed_unknown(control)
    unknown = [
        path for path in unknown
        if path.upper() not in allowed and path.rsplit('.', 1)[-1].upper() not in allowed
        and path.rsplit('.', 1)[-1].upper() not in RESERVED_CONTROL_KEYS
    ]
    if not unknown:
        return []

    where = f" in {source}" if source else ""
    lines = [f"Unrecognized configuration key(s){where}:"]
    for path in unknown:
        leaf = path.rsplit('.', 1)[-1]
        matches = get_close_matches(leaf, _sibling_names(path), n=1, cutoff=0.6) if '.' in path else []
        if matches:
            parent = path.rsplit('.', 1)[0]
            lines.append(f"  - {path}  (did you mean '{parent}.{matches[0]}'?)")
        else:
            lines.append(f"  - {path}")
    lines.append(
        "Fix the spelling/section, or list intentional custom keys under "
        "ALLOW_UNKNOWN_KEYS to silence this."
    )
    message = "\n".join(lines)

    if strict is None:
        strict = is_strict_config_mode(control)
    if strict:
        from symfluence.core.exceptions import ConfigValidationError
        raise ConfigValidationError(message)
    logger.warning(message)
    return unknown


def _format_message(unknown: List[str], known_keys: Set[str], source: Optional[str]) -> str:
    """Build a user-facing message with 'did you mean?' suggestions."""
    where = f" in {source}" if source else ""
    lines = [f"Unrecognized configuration key(s){where}:"]
    for key in unknown:
        matches = get_close_matches(key, known_keys, n=1, cutoff=0.6)
        if matches:
            lines.append(f"  - {key}  (did you mean '{matches[0]}'?)")
        else:
            lines.append(f"  - {key}")
    lines.append(
        "Fix the spelling, or list intentional custom keys under "
        "ALLOW_UNKNOWN_KEYS to silence this."
    )
    return "\n".join(lines)


def validate_known_flat_keys(
    flat_config: Dict[str, Any],
    known_keys: Iterable[str],
    *,
    strict: Optional[bool] = None,
    source: Optional[str] = None,
    allow_unknown: Iterable[str] = (),
) -> List[str]:
    """Validate that every flat key is recognized; warn or raise on unknowns.

    Args:
        flat_config: Flat (uppercase) user configuration dictionary.
        known_keys: The allowlist (core ∪ plugins ∪ legacy ∪ model transformers).
        strict: Force strict (raise) or lenient (warn). When ``None`` (default),
            resolved from the ``STRICT_CONFIG`` key / ``SYMFLUENCE_STRICT_CONFIG``
            environment variable via :func:`is_strict_config_mode`.
        source: Optional origin (e.g. file path) included in the message.
        allow_unknown: Additional keys to treat as recognized.

    Returns:
        The sorted list of unknown keys found (empty when the config is clean).

    Raises:
        ConfigValidationError: In strict mode when unknown keys are present.
    """
    known = {str(k).upper() for k in known_keys}
    unknown = find_unknown_keys(flat_config, known, allow_unknown=allow_unknown)
    if not unknown:
        return []

    message = _format_message(unknown, known, source)

    if strict is None:
        strict = is_strict_config_mode(flat_config)

    if strict:
        from symfluence.core.exceptions import ConfigValidationError
        raise ConfigValidationError(message)

    logger.warning(message)
    return unknown
