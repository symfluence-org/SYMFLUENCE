# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Spatial Mode Definitions for SYMFLUENCE Models.

Provides centralized spatial mode validation across all model runners.
This module defines supported spatial modes, model capabilities, and validation logic.

Phase 3 Addition: Centralizes spatial mode validation that was previously scattered
across individual model runners.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set


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


# Model spatial capabilities registry
MODEL_SPATIAL_CAPABILITIES: Dict[str, ModelSpatialCapability] = {
    'SUMMA': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'FUSE': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'GR': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'LSTM': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # LSTM handles routing internally
            SpatialMode.SEMI_DISTRIBUTED: False,
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "LSTM works best in lumped mode for streamflow prediction. "
            "Consider using GNN for spatially-distributed graph-based modeling."
        )
    ),

    'GNN': ModelSpatialCapability(
        supported_modes={SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False  # GNN has internal graph-based routing
        },
        warning_message=(
            "GNN requires distributed domain with graph structure. "
            "Use LSTM for lumped modeling."
        )
    ),

    'HYPE': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.SEMI_DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # HYPE has internal routing
            SpatialMode.SEMI_DISTRIBUTED: False,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'MESH': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # MESH has internal routing (WATFLOOD/PDMROF)
            SpatialMode.SEMI_DISTRIBUTED: False,
            SpatialMode.LUMPED: False  # Uses noroute mode (RFF+DRAINSOL proxy)
        },
        warning_message=None  # Lumped mode now fully supported
    ),

    'NGEN': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,  # Uses t-route for routing
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'RHESSYS': ModelSpatialCapability(
        # RHESSys is inherently hierarchical/distributed but can operate with a
        # single aggregate hillslope/patch for lumped experiments.
        supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # Internal hillslope routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "RHESSys performs best with distributed landscape hierarchy. "
            "Lumped mode is supported when world/flow files are pre-aggregated."
        )
    ),

    'VIC': ModelSpatialCapability(
        # VIC is designed for grid-based distributed modeling but can operate
        # with a single-cell domain for lumped experiments.
        supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,  # VIC outputs cell runoff, needs external routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "VIC is designed for distributed grid-based modeling. "
            "For lumped mode, a single-cell domain will be created."
        )
    ),

    'SWAT': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={SpatialMode.LUMPED: False},
        warning_message=(
            "SWAT is a semi-distributed model. Lumped mode uses "
            "a single-HRU/subbasin configuration."
        )
    ),

    'MHM': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={SpatialMode.LUMPED: False},
        warning_message=(
            "mHM is a mesoscale hydrological model. Lumped mode uses "
            "a single-cell domain with multiscale parameter regionalization."
        )
    ),

    'CRHM': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={SpatialMode.LUMPED: False},
        warning_message=(
            "CRHM is a cold-region hydrological model. Lumped mode uses "
            "a single-HRU configuration with blowing snow and frozen soil."
        )
    ),

    'GSFLOW': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED},
        default_mode=SpatialMode.SEMI_DISTRIBUTED,
        requires_routing={
            SpatialMode.SEMI_DISTRIBUTED: False,  # Internal SFR routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "GSFLOW couples PRMS surface processes with MODFLOW-NWT groundwater. "
            "Internal SFR/UZF packages handle GW-SW exchange."
        )
    ),

    'WATFLOOD': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # Internal channel routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "WATFLOOD uses GRU-grid distributed structure with internal "
            "channel routing. Lumped mode uses a single-GRU configuration."
        )
    ),
}


def get_spatial_mode_from_config(config_dict) -> SpatialMode:
    """
    Determine spatial mode from configuration dictionary or typed config.

    Uses DOMAIN_DEFINITION_METHOD and ROUTING_DELINEATION to infer the spatial mode.

    Args:
        config_dict: Configuration dictionary or typed config with domain settings

    Returns:
        Inferred SpatialMode
    """
    try:
        domain_method = config_dict.domain.definition_method or 'lumped'
    except (AttributeError, TypeError):
        domain_method = config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped') if isinstance(config_dict, dict) else 'lumped'
    try:
        routing_delineation = config_dict.model.mizuroute.routing_delineation or 'lumped'
    except (AttributeError, TypeError):
        routing_delineation = config_dict.get('ROUTING_DELINEATION', 'lumped') if isinstance(config_dict, dict) else 'lumped'

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

    if model_name not in MODEL_SPATIAL_CAPABILITIES:
        # Unknown model - allow any mode
        return True, None

    capability = MODEL_SPATIAL_CAPABILITIES[model_name]

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
    Get spatial capabilities for a model.

    Args:
        model_name: Name of the model

    Returns:
        ModelSpatialCapability if found, None otherwise
    """
    return MODEL_SPATIAL_CAPABILITIES.get(model_name.upper())
