# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Spatial Mode Definitions for SYMFLUENCE Models.

Provides centralized spatial mode validation across all model runners.
This module defines the spatial-mode vocabulary, the *shape* of a model's
spatial capability declaration, and the validation logic that reads it.

Phase 3 Addition: Centralizes spatial mode validation that was previously scattered
across individual model runners.

Amended during service-decomposition prep (July 2026): the per-model VALUES are
no longer a table in this file. ``MODEL_SPATIAL_CAPABILITIES`` was a hardcoded
dict of 16 per-model declarations living in ``core``, which meant

* a model package could not change its own spatial capabilities without a core
  edit and a core release, and
* an out-of-tree plugin package could not declare them **at all** — it silently
  fell into the "unknown model" branch of :func:`validate_spatial_mode` and got
  no validation whatsoever.

:func:`register_model_spatial_capability` is the seam. It mirrors
``register_model_bounds()`` in
``core.calibration.parameters.parameter_bounds_registry``: core owns the record
type and the read side, the owning package contributes the values at
registration time, and the read side is unchanged for every consumer.

Why this shape rather than the alternatives considered:

* ``R.base_settings`` (a plain per-model string registry) carries untyped
  values and no notion of a structured declaration; a capability is a record
  with four fields and a merge policy, not a settings string.
* ``ModelConfigSchema`` / ``register_model_schema`` is the closest fit and is
  where ``spatial_mode_key`` already lives — but that schema is about *config
  keys* (installation/execution/input/output are all required fields), so a
  model would have to invent a full config contract just to say "I support
  lumped only". Several capability-declaring models (MHM, CRHM, PCRGLOBWB,
  WATFLOOD) have no such schema today. A dedicated registration function keeps
  the two declarations independent, exactly as ``register_model_bounds`` is
  independent of the schema.

The migration is complete: every declaration now lives in the ``register()`` of
the package that owns the model, and this module holds no per-model values at
all. A model contributes its capability at plugin-discovery time, exactly as an
external package does, so core can ship without the models distribution and a
capability change never needs a core release.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

_logger = logging.getLogger(__name__)


class SpatialMode(Enum):
    """
    Spatial modeling mode enumeration.

    Attributes:
        LUMPED: Single unit domain (point-scale, no spatial heterogeneity)
        SEMI_DISTRIBUTED: Multiple units with partial spatial representation
        DISTRIBUTED: Full spatial discretization (HRUs/grid cells with routing)
    """
    LUMPED = "lumped"
    SEMI_DISTRIBUTED = "semi_distributed"
    DISTRIBUTED = "distributed"

    @classmethod
    def from_string(cls, value: str) -> 'SpatialMode':
        """Normalize aliases and parse spatial mode from string.

        Handles common variations like 'point' -> LUMPED,
        'delineate' -> DISTRIBUTED, 'semidistributed' -> SEMI_DISTRIBUTED.
        """
        normalized = value.lower().replace('-', '_').replace(' ', '_')
        mapping = {
            'lumped': cls.LUMPED,
            'point': cls.LUMPED,
            'semi_distributed': cls.SEMI_DISTRIBUTED,
            'semidistributed': cls.SEMI_DISTRIBUTED,
            'distributed': cls.DISTRIBUTED,
            'delineate': cls.DISTRIBUTED,
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown spatial mode: {value}. Valid: {list(mapping.keys())}")
        return mapping[normalized]

    def __eq__(self, other):
        if isinstance(other, Enum) and hasattr(other, 'value'):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value


@dataclass
class ModelSpatialCapability:
    """
    Defines spatial mode capabilities for a specific model.

    Attributes:
        supported_modes: Set of SpatialMode values the model supports
        default_mode: The default spatial mode if none specified
        requires_routing: Dict mapping SpatialMode to whether routing is required
        warning_message: Optional warning message for suboptimal configurations
    """
    supported_modes: Set[SpatialMode]
    default_mode: SpatialMode
    requires_routing: Dict[SpatialMode, bool] = field(default_factory=dict)
    warning_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Capability registry (the extension seam)
# ---------------------------------------------------------------------------

class _CapabilityRegistry(Dict[str, "ModelSpatialCapability"]):
    """Registry storage that is also the legacy public mapping.

    ``MODEL_SPATIAL_CAPABILITIES`` is a PUBLIC name on the models contract
    surface: external model packages import it directly, and before the
    registration seam existed, assigning into it was the *only* way such a
    package could declare its spatial capabilities. The ``models`` contract is
    pre-1.0 and additive-only, so neither the name nor any behaviour it had may
    be withdrawn — which rules out both replacing it with a static snapshot
    (would go stale the moment a package registers) and wrapping it in a
    read-only proxy (a write that worked at 0.3.0 would start raising).

    Making the registry storage a ``dict`` subclass satisfies every constraint
    at once: the exported name IS the live registry (so old importers and the
    new seam can never disagree), it is a genuine ``dict`` (so
    ``isinstance(..., dict)`` in third-party code still holds), and legacy
    item assignment is forwarded to :func:`register_model_spatial_capability`
    so it picks up key normalisation and double-registration logging.
    """

    def __setitem__(self, key: str, value: "ModelSpatialCapability") -> None:
        # Route legacy `MODEL_SPATIAL_CAPABILITIES['X'] = cap` writes through
        # the seam. The seam writes back with dict.__setitem__ to avoid
        # recursing through this override.
        register_model_spatial_capability(key, value)


#: Runtime registry: model key (uppercase) -> declared capability. Populated
#: exclusively by :func:`register_model_spatial_capability` — core contributes
#: nothing — and exported as ``MODEL_SPATIAL_CAPABILITIES``.
_SPATIAL_CAPABILITIES: _CapabilityRegistry = _CapabilityRegistry()


def register_model_spatial_capability(
    model: str,
    capability: ModelSpatialCapability,
) -> None:
    """Declare a model's spatial-mode capabilities.

    The extension seam for model packages (in-tree or external): call this from
    the package's ``register()`` so :func:`validate_spatial_mode` and
    :func:`get_model_capabilities` serve the model's own declaration without
    editing core.

    A model that never registers stays *unknown* to validation, which
    deliberately permits any spatial mode — declaring nothing must never be
    more restrictive than declaring something. That is also what a model whose
    package is simply not installed degrades to.

    Re-registering an *equal* declaration is silent: ``register()`` is
    idempotent and may run more than once (entry-point discovery plus an
    explicit call). Re-registering a *different* one is logged, because that
    means two owners are competing for one model key.

    Args:
        model: Model key, case-insensitive (e.g. ``'FUSE'``, ``'MYMODEL'``).
        capability: The :class:`ModelSpatialCapability` record to serve.
    """
    key = model.upper()
    existing = _SPATIAL_CAPABILITIES.get(key)
    if existing is not None and existing != capability:
        _logger.warning(
            "spatial capability for '%s' re-registered with a different "
            "declaration; the later one wins. Two packages should not own one "
            "model's capabilities.",
            key,
        )
    dict.__setitem__(_SPATIAL_CAPABILITIES, key, capability)


def registered_spatial_capability_models() -> List[str]:
    """Model keys with a declared spatial capability."""
    return sorted(_SPATIAL_CAPABILITIES)


def spatial_capabilities() -> Dict[str, ModelSpatialCapability]:
    """Snapshot of every declared spatial capability.

    A detached copy: mutating it does not change the registry. Use
    :func:`register_model_spatial_capability` to contribute a declaration.
    """
    return dict(_SPATIAL_CAPABILITIES)


def __getattr__(name: str):
    """Public module attribute ``MODEL_SPATIAL_CAPABILITIES`` (PEP 562).

    Part of the models contract surface — external model packages import this
    name directly, so it must keep resolving with its historical mapping
    semantics. It returns the LIVE registry (a real ``dict``; see
    :class:`_CapabilityRegistry`), not a snapshot, so old importers and the
    registration seam always see the same data.

    New code should prefer :func:`get_model_capabilities` /
    :func:`spatial_capabilities` for reads and
    :func:`register_model_spatial_capability` for writes.
    """
    if name == "MODEL_SPATIAL_CAPABILITIES":
        return _SPATIAL_CAPABILITIES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_spatial_mode_from_config(config_dict) -> SpatialMode:
    """
    Determine spatial mode from configuration dictionary or typed config.

    Uses DOMAIN_DEFINITION_METHOD and ROUTING_DELINEATION to infer the spatial mode.

    Args:
        config_dict: Configuration dictionary or typed config with domain settings

    Returns:
        Inferred SpatialMode
    """
    from symfluence.core.mixins.config import resolve_config_value

    domain_method = resolve_config_value(
        config_dict, lambda: config_dict.domain.definition_method, 'lumped', 'DOMAIN_DEFINITION_METHOD')
    # (The typed branch used to read model.mizuroute.routing_delineation,
    # which doesn't exist, so typed configs always got 'lumped'.)
    routing_delineation = resolve_config_value(
        config_dict, lambda: config_dict.domain.delineation.routing, 'lumped', 'ROUTING_DELINEATION')

    # Map domain method to spatial mode
    if domain_method in ('point', 'lumped'):
        if routing_delineation == 'river_network':
            # Lumped domain but with network routing = semi-distributed behavior
            return SpatialMode.SEMI_DISTRIBUTED
        return SpatialMode.LUMPED

    elif domain_method in ('subset', 'semi_distributed'):
        return SpatialMode.SEMI_DISTRIBUTED

    elif domain_method in ('delineate', 'distributed'):
        return SpatialMode.DISTRIBUTED

    # Default to lumped if unknown
    return SpatialMode.LUMPED


def validate_spatial_mode(
    model_name: str,
    spatial_mode: SpatialMode,
    has_routing_configured: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Validate spatial mode for a specific model.

    Args:
        model_name: Name of the model (uppercase)
        spatial_mode: The spatial mode to validate
        has_routing_configured: Whether routing model is configured

    Returns:
        Tuple of (is_valid, warning_message)
    """
    model_name = model_name.upper()

    capability = _SPATIAL_CAPABILITIES.get(model_name)
    if capability is None:
        # Model declared no spatial capability - allow any mode. Declaring
        # nothing must never be more restrictive than declaring something,
        # otherwise adding the seam would break every model that has not
        # migrated (and every external plugin).
        return True, None

    # Check if mode is supported
    if spatial_mode not in capability.supported_modes:
        return False, (
            f"{model_name} does not support '{spatial_mode.value}' mode. "
            f"Supported modes: {[m.value for m in capability.supported_modes]}"
        )

    # Check routing requirements
    if capability.requires_routing.get(spatial_mode, False) and not has_routing_configured:
        warning = (
            f"{model_name} in {spatial_mode.value} mode typically requires a routing model "
            f"(e.g., mizuRoute). Consider adding ROUTING_MODEL to configuration."
        )
        return True, warning

    # Return any general warning for the model/mode combination
    if capability.warning_message and spatial_mode != capability.default_mode:
        return True, capability.warning_message

    return True, None


def get_model_capabilities(model_name: str) -> Optional[ModelSpatialCapability]:
    """
    Get the declared spatial capabilities for a model.

    Args:
        model_name: Name of the model

    Returns:
        ModelSpatialCapability if the model declared one, None otherwise.
        ``None`` means "no declaration", not "no capabilities" — see
        :func:`validate_spatial_mode`.
    """
    return _SPATIAL_CAPABILITIES.get(model_name.upper())
