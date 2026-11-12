# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MizuRoute-specific configuration mixins.

Provides standardized access to mizuRoute configuration values via properties,
replacing scattered config_dict.get() calls with typed accessors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


class MizuRouteConfigMixin:
    """
    Mixin for mizuRoute configuration access.

    Provides properties for accessing mizuRoute-specific configuration values
    from the typed config, with sensible defaults.

    Requires the class to have:
    - self.config: SymfluenceConfig instance
    - self._get_config_value(): method from ConfigMixin
    """

    # =========================================================================
    # Core Routing Configuration
    # =========================================================================

    @property
    def mizu_routing_var(self) -> str:
        """Routing output variable name from config.model.mizuroute.routing_var."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.routing_var,
            dict_key='SETTINGS_MIZU_ROUTING_VAR',
        )

    @property
    def mizu_routing_units(self) -> str:
        """Routing output units from config.model.mizuroute.routing_units."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.routing_units,
            dict_key='SETTINGS_MIZU_ROUTING_UNITS',
        )

    @property
    def mizu_routing_dt(self) -> int:
        """Routing time step in seconds from config.model.mizuroute.routing_dt."""
        return int(self._get_config_value(
            lambda: self.config.model.mizuroute.routing_dt,
            dict_key='SETTINGS_MIZU_ROUTING_DT',
        ))

    @property
    def mizu_within_basin(self) -> bool:
        """Within-basin routing flag from config.model.mizuroute.within_basin."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.within_basin,
            dict_key='SETTINGS_MIZU_WITHIN_BASIN',
        )

    # =========================================================================
    # File Configuration
    # =========================================================================

    @property
    def mizu_topology_file(self) -> Optional[str]:
        """Topology file name from config.model.mizuroute.topology."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.topology,
            default=None
        )

    @property
    def mizu_control_file(self) -> Optional[str]:
        """Control file name from config.model.mizuroute.control_file."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.control_file,
            default=None
        )

    @property
    def mizu_parameters_file(self) -> Optional[str]:
        """Parameters file name from config.model.mizuroute.parameters."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.parameters,
            default=None
        )

    @property
    def mizu_remap_file(self) -> Optional[str]:
        """Remap file name from config.model.mizuroute.remap."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.remap,
            default=None
        )

    # =========================================================================
    # Integration Configuration
    # =========================================================================

    @property
    def mizu_from_model(self) -> str:
        """Source model for routing input from config.model.mizuroute.from_model."""
        return self._get_config_value(lambda: self.config.model.mizuroute.from_model, dict_key='MIZU_FROM_MODEL')

    @property
    def mizu_needs_remap(self) -> bool:
        """Remap needed flag from config.model.mizuroute.needs_remap."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.needs_remap,
            dict_key='SETTINGS_MIZU_NEEDS_REMAP',
        )

    @property
    def mizu_make_outlet(self) -> Optional[str]:
        """Make outlet segment IDs (comma-separated) from config.model.mizuroute.make_outlet."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.make_outlet,
            default=None
        )

    # =========================================================================
    # Output Configuration
    # =========================================================================

    @property
    def mizu_output_freq(self) -> str:
        """Output frequency from config.model.mizuroute.output_freq."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.output_freq,
            dict_key='SETTINGS_MIZU_OUTPUT_FREQ',
        )

    @property
    def mizu_output_vars(self) -> Optional[str]:
        """Output variables from config.model.mizuroute.output_vars."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.output_vars,
            default=None
        )

    # =========================================================================
    # Path Configuration
    # =========================================================================

    @property
    def mizu_settings_path(self) -> Optional[str]:
        """Settings path from config.model.mizuroute.settings_path."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.settings_path,
            default=None
        )

    @property
    def mizu_install_path(self) -> Optional[str]:
        """Install path from config.model.mizuroute.install_path."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.install_path,
            default=None
        )

    @property
    def mizu_exe(self) -> str:
        """Executable name from config.model.mizuroute.exe."""
        return self._get_config_value(lambda: self.config.model.mizuroute.exe, dict_key='MIZUROUTE_EXE')

    @property
    def mizu_experiment_output(self) -> Optional[str]:
        """Experiment output path from config.model.mizuroute.experiment_output."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.experiment_output,
            default=None
        )

    @property
    def mizu_experiment_log(self) -> Optional[str]:
        """Experiment log path from config.model.mizuroute.experiment_log."""
        return self._get_config_value(
            lambda: self.config.model.mizuroute.experiment_log,
            default=None
        )


__all__ = ['MizuRouteConfigMixin']
