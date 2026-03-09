# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Routing Decider

Unified routing decision logic for all hydrological models.
Consolidates duplicate needs_routing() implementations from workers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RoutingDecider:
    """
    Unified routing decision logic for all models.

    Determines whether mizuRoute routing is needed based on:
    - Calibration variable (streamflow only)
    - Explicit routing model configuration
    - Model-specific routing integration settings
    - Spatial mode configuration
    - Domain definition method
    - Routing delineation settings
    - Existence of mizuRoute control files
    """

    # Model-specific config keys for spatial mode
    SPATIAL_MODE_KEYS: Dict[str, str] = {
        'SUMMA': 'DOMAIN_DEFINITION_METHOD',
        'FUSE': 'FUSE_SPATIAL_MODE',
        'HYPE': 'HYPE_SPATIAL_MODE',
        'GR': 'GR_SPATIAL_MODE',
        'MESH': 'MESH_SPATIAL_MODE',
        'NGEN': 'NGEN_SPATIAL_MODE',
    }

    # Model-specific routing integration config keys
    ROUTING_INTEGRATION_KEYS: Dict[str, str] = {
        'FUSE': 'FUSE_ROUTING_INTEGRATION',
    }

    def needs_routing(
        self,
        config: Dict[str, Any],
        model: str,
        settings_dir: Optional[Path] = None
    ) -> bool:
        """
        Determine if routing (mizuRoute) is needed for a model run.

        Decision hierarchy:
        1. If CALIBRATION_VARIABLE != 'streamflow': False
        2. If ROUTING_MODEL == 'mizuRoute' or 'default': True
        3. If model-specific routing integration requests mizuRoute: True
        4. If spatial_mode in ['semi_distributed', 'distributed']: True
        5. If domain_method not in ['point', 'lumped']: True
        6. If lumped but ROUTING_DELINEATION == 'river_network': True
        7. If mizuRoute control file exists in settings_dir: True
        8. Otherwise: False

        Args:
            config: Configuration dictionary
            model: Model name (e.g., 'SUMMA', 'FUSE', 'HYPE')
            settings_dir: Optional settings directory to check for mizuRoute control files

        Returns:
            True if routing is needed
        """
        result, _ = self._evaluate_routing(config, model, settings_dir)
        return result

    def needs_routing_verbose(
        self,
        config: Dict[str, Any],
        model: str,
        settings_dir: Optional[Path] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if routing is needed with diagnostic information.

        Args:
            config: Configuration dictionary
            model: Model name
            settings_dir: Optional settings directory

        Returns:
            Tuple of (needs_routing, diagnostics_dict)
            diagnostics_dict contains 'reason' and 'checks' keys
        """
        return self._evaluate_routing(config, model, settings_dir)

    def _evaluate_routing(
        self,
        config: Dict[str, Any],
        model: str,
        settings_dir: Optional[Path] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Internal method that evaluates routing need with diagnostics.
        Based primarily on spatial configuration as requested by project standards.
        """
        model = model.upper()
        diagnostics: Dict[str, Any] = {
            'model': model,
            'reason': None,
            'checks': {}
        }

        # 1. Explicit routing model check
        # Only use this if explicitly set - don't short-circuit on default value
        routing_model = config.get('ROUTING_MODEL', '').lower()
        diagnostics['checks']['routing_model'] = routing_model

        # If explicitly set to 'none', disable routing
        if 'ROUTING_MODEL' in config and routing_model == 'none':
            diagnostics['reason'] = 'routing_model_explicitly_none'
            return False, diagnostics

        # Check for point mode BEFORE enabling routing based on ROUTING_MODEL
        # Point-scale simulations never need routing regardless of ROUTING_MODEL setting
        domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped').lower()
        if domain_method == 'point':
            diagnostics['reason'] = 'point_scale_no_routing_needed'
            diagnostics['checks']['domain_method'] = domain_method
            return False, diagnostics

        # Check model-specific spatial mode BEFORE ROUTING_MODEL.
        # ROUTING_MODEL may be a global template setting, but if the model
        # explicitly declares itself lumped, routing is not meaningful.
        if model in self.SPATIAL_MODE_KEYS:
            spatial_key = self.SPATIAL_MODE_KEYS[model]
            model_spatial = config.get(spatial_key, '').lower()
            if model_spatial == 'lumped':
                routing_delineation = config.get('ROUTING_DELINEATION', 'lumped').lower()
                if routing_delineation == 'lumped':
                    diagnostics['reason'] = 'model_spatial_mode_lumped'
                    diagnostics['checks']['model_spatial_mode'] = model_spatial
                    diagnostics['checks']['routing_delineation'] = routing_delineation
                    return False, diagnostics

        # If explicitly set to a routing model, enable routing (for non-point domains)
        if routing_model in ['mizuroute', 'mizu_route', 'mizu']:
            diagnostics['reason'] = 'routing_model_is_mizuroute'
            return True, diagnostics

        if routing_model in ['troute', 't-route', 't_route']:
            diagnostics['reason'] = 'routing_model_is_troute'
            return True, diagnostics

        # 2. Model-specific routing integration (e.g., FUSE)
        # Check this BEFORE spatial config so it can override lumped settings
        if model in self.ROUTING_INTEGRATION_KEYS:
            integration_key = self.ROUTING_INTEGRATION_KEYS[model]
            integration = config.get(integration_key, 'none')
            diagnostics['checks']['routing_integration'] = integration
            if integration == 'mizuRoute':
                diagnostics['reason'] = f'{model.lower()}_routing_integration_mizuroute'
                return True, diagnostics

        # 3. Spatial configuration check (The primary driver)
        # Use model-specific spatial mode key when available (e.g., FUSE_SPATIAL_MODE)
        # to override the generic DOMAIN_DEFINITION_METHOD.
        spatial_mode = domain_method
        if model in self.SPATIAL_MODE_KEYS:
            spatial_key = self.SPATIAL_MODE_KEYS[model]
            model_spatial = config.get(spatial_key, '').lower()
            if model_spatial:
                spatial_mode = model_spatial
                diagnostics['checks']['model_spatial_mode'] = model_spatial

        routing_delineation = config.get('ROUTING_DELINEATION', 'lumped').lower()

        diagnostics['checks']['domain_method'] = domain_method
        diagnostics['checks']['spatial_mode'] = spatial_mode
        diagnostics['checks']['routing_delineation'] = routing_delineation

        # If it's a lumped domain AND lumped routing, mizuRoute is NOT needed
        # (point mode already handled above)
        if spatial_mode == 'lumped' and routing_delineation == 'lumped':
            diagnostics['reason'] = 'lumped_domain_with_lumped_routing'
            return False, diagnostics

        # 4. Distributed/Network routing triggers
        if spatial_mode in ['semi_distributed', 'semidistributed', 'distributed', 'hru', 'gru']:
            diagnostics['reason'] = f'distributed_domain_method_{domain_method}'
            return True, diagnostics

        if routing_delineation in ['river_network', 'reach', 'vector']:
            diagnostics['reason'] = f'network_routing_delineation_{routing_delineation}'
            return True, diagnostics

        # 5. Filesystem check for existing mizuRoute setup as fallback
        if settings_dir:
            settings_dir = Path(settings_dir)
            control_exists = self._check_mizuroute_control_exists(settings_dir, model)
            diagnostics['checks']['mizuroute_control_exists'] = control_exists
            if control_exists:
                diagnostics['reason'] = 'mizuroute_control_file_exists'
                return True, diagnostics

        diagnostics['reason'] = 'no_spatial_routing_conditions_met'
        return False, diagnostics

    def _check_mizuroute_control_exists(
        self,
        settings_dir: Path,
        model: str
    ) -> bool:
        """
        Check if mizuRoute control file exists.

        Handles directory structure variations for different models.

        Args:
            settings_dir: Settings directory path
            model: Model name

        Returns:
            True if mizuRoute control file exists
        """
        # Standard location
        mizu_control = settings_dir / 'mizuRoute' / 'mizuroute.control'
        if mizu_control.exists():
            logger.debug(f"Found mizuRoute control file at {mizu_control}")
            return True

        # Handle model-specific subdirectory cases (e.g., FUSE settings in subdirectory)
        if settings_dir.name == model.upper():
            parent_mizu = settings_dir.parent / 'mizuRoute' / 'mizuroute.control'
            if parent_mizu.exists():
                logger.debug(f"Found mizuRoute control file at {parent_mizu}")
                return True

        return False


# Module-level instance for convenience
_routing_decider = RoutingDecider()


def needs_routing(
    config: Dict[str, Any],
    model: str,
    settings_dir: Optional[Path] = None
) -> bool:
    """
    Convenience function for routing decision.

    See RoutingDecider.needs_routing for full documentation.
    """
    return _routing_decider.needs_routing(config, model, settings_dir)


def needs_routing_verbose(
    config: Dict[str, Any],
    model: str,
    settings_dir: Optional[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function for routing decision with diagnostics.

    See RoutingDecider.needs_routing_verbose for full documentation.
    """
    return _routing_decider.needs_routing_verbose(config, model, settings_dir)
