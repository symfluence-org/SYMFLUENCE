# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TRoute configuration mixin.

Provides typed property accessors for TRouteConfig fields via _get_config_value.
"""
from __future__ import annotations


class TRouteConfigMixin:
    """Typed config accessors for t-route settings."""

    @property
    def troute_install_path(self) -> str:
        return self._get_config_value(lambda: self.config.model.troute.install_path, dict_key='TROUTE_INSTALL_PATH')

    @property
    def troute_settings_path(self) -> str:
        return self._get_config_value(lambda: self.config.model.troute.settings_path, dict_key='SETTINGS_TROUTE_PATH')

    @property
    def troute_topology_file(self) -> str:
        return self._get_config_value(
            lambda: self.config.model.troute.topology_file,
            dict_key='SETTINGS_TROUTE_TOPOLOGY',
        )

    @property
    def troute_config_file(self) -> str:
        return self._get_config_value(
            lambda: self.config.model.troute.config_file,
            dict_key='SETTINGS_TROUTE_CONFIG_FILE',
        )

    @property
    def troute_dt_seconds(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.troute.dt_seconds,
            dict_key='SETTINGS_TROUTE_DT_SECONDS',
        )

    @property
    def troute_routing_method(self) -> str:
        return self._get_config_value(
            lambda: self.config.model.troute.routing_method,
            dict_key='TROUTE_ROUTING_METHOD',
        )

    @property
    def troute_from_model(self) -> str:
        return self._get_config_value(lambda: self.config.model.troute.from_model, dict_key='TROUTE_FROM_MODEL')

    @property
    def troute_mannings_n(self) -> float:
        return self._get_config_value(lambda: self.config.model.troute.mannings_n, dict_key='TROUTE_MANNINGS_N')

    @property
    def troute_experiment_output(self) -> str:
        return self._get_config_value(
            lambda: self.config.model.troute.experiment_output,
            dict_key='EXPERIMENT_OUTPUT_TROUTE',
        )

    @property
    def troute_experiment_log(self) -> str:
        return self._get_config_value(
            lambda: self.config.model.troute.experiment_log,
            dict_key='EXPERIMENT_LOG_TROUTE',
        )

    @property
    def troute_hg_width_coeff(self) -> float:
        return self._get_config_value(
            lambda: self.config.model.troute.hg_width_coeff,
            dict_key='TROUTE_HG_WIDTH_COEFF',
        )

    @property
    def troute_hg_width_exp(self) -> float:
        return self._get_config_value(lambda: self.config.model.troute.hg_width_exp, dict_key='TROUTE_HG_WIDTH_EXP')

    @property
    def troute_make_outlet(self) -> str:
        return self._get_config_value(
            lambda: None,
            default='n/a',
            dict_key='TROUTE_MAKE_OUTLET',
        )

    @property
    def troute_needs_remap(self) -> bool:
        return self._get_config_value(
            lambda: None,
            default=False,
            dict_key='TROUTE_NEEDS_REMAP',
        )
