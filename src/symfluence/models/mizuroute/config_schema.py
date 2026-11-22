# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for MIZUROUTE.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from symfluence.core.config.models.base import FROZEN_CONFIG


class MizuRouteConfig(BaseModel):
    """mizuRoute routing model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MIZUROUTE_INSTALL_PATH')
    exe: str = Field(default='mizuRoute.exe', alias='MIZUROUTE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MIZU_PATH')
    within_basin: int = Field(default=0, alias='SETTINGS_MIZU_WITHIN_BASIN')
    routing_dt: int = Field(default=3600, alias='SETTINGS_MIZU_ROUTING_DT')
    routing_units: str = Field(default='m/s', alias='SETTINGS_MIZU_ROUTING_UNITS')
    routing_var: str = Field(default='averageRoutedRunoff', alias='SETTINGS_MIZU_ROUTING_VAR')
    output_freq: str = Field(default='single', alias='SETTINGS_MIZU_OUTPUT_FREQ')
    output_vars: str = Field(default='1', alias='SETTINGS_MIZU_OUTPUT_VARS')
    make_outlet: str = Field(default='n/a', alias='SETTINGS_MIZU_MAKE_OUTLET')
    needs_remap: bool = Field(default=False, alias='SETTINGS_MIZU_NEEDS_REMAP')
    topology: str = Field(default='topology.nc', alias='SETTINGS_MIZU_TOPOLOGY')
    parameters: str = Field(default='param.nml.default', alias='SETTINGS_MIZU_PARAMETERS')
    control_file: str = Field(default='mizuroute.control', alias='SETTINGS_MIZU_CONTROL_FILE')
    remap: str = Field(default='routing_remap.nc', alias='SETTINGS_MIZU_REMAP')
    from_model: str = Field(default='default', alias='MIZU_FROM_MODEL')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_MIZUROUTE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MIZUROUTE')
    # Additional mizuRoute settings
    output_var: str = Field(default='IRFroutedRunoff', alias='SETTINGS_MIZU_OUTPUT_VAR')
    parameter_file: str = Field(default='param.nml.default', alias='SETTINGS_MIZU_PARAMETER_FILE')
    remap_file: str = Field(default='routing_remap.nc', alias='SETTINGS_MIZU_REMAP_FILE')
    topology_file: str = Field(default='topology.nc', alias='SETTINGS_MIZU_TOPOLOGY_FILE')
    params_to_calibrate: str = Field(
        default='velo,diff',
        alias='MIZUROUTE_PARAMS_TO_CALIBRATE'
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_MIZUROUTE')
    timeout: int = Field(default=3600, alias='MIZUROUTE_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    num_threads: Optional[int] = Field(
        default=1, alias='MIZUROUTE_NUM_THREADS', ge=1,
        description='OpenMP thread count for the mizuRoute subprocess'
    )
    time_rounding_freq: str = Field(
        default='h',
        alias='MIZUROUTE_TIME_ROUNDING_FREQ',
        description='Frequency for rounding time values (e.g., "h" for hour, "min" for minute, "none" to disable)'
    )

    @field_validator('output_vars', mode='before')
    @classmethod
    def normalize_output_vars(cls, v):
        """Convert list or other types to string for output_vars"""
        if isinstance(v, list):
            return ' '.join(str(item).strip() for item in v)
        return str(v)


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
MizuRouteConfig.LEGACY_FLAT_ALIASES = {
    'INSTALL_PATH_MIZUROUTE': 'install_path',
    'EXE_NAME_MIZUROUTE': 'exe',
    'SETTINGS_MIZU_PATH': 'settings_path',
    'SETTINGS_MIZU_WITHIN_BASIN': 'within_basin',
    'SETTINGS_MIZU_ROUTING_DT': 'routing_dt',
    'SETTINGS_MIZU_ROUTING_UNITS': 'routing_units',
    'SETTINGS_MIZU_ROUTING_VAR': 'routing_var',
    'SETTINGS_MIZU_OUTPUT_FREQ': 'output_freq',
    'SETTINGS_MIZU_OUTPUT_VARS': 'output_vars',
    'SETTINGS_MIZU_MAKE_OUTLET': 'make_outlet',
    'SETTINGS_MIZU_NEEDS_REMAP': 'needs_remap',
    'SETTINGS_MIZU_TOPOLOGY': 'topology',
    'SETTINGS_MIZU_PARAMETERS': 'parameters',
    'SETTINGS_MIZU_CONTROL_FILE': 'control_file',
    'SETTINGS_MIZU_REMAP': 'remap',
    'SETTINGS_MIZU_OUTPUT_VAR': 'output_var',
    'SETTINGS_MIZU_PARAMETER_FILE': 'parameter_file',
    'SETTINGS_MIZU_REMAP_FILE': 'remap_file',
    'SETTINGS_MIZU_TOPOLOGY_FILE': 'topology_file',
}
