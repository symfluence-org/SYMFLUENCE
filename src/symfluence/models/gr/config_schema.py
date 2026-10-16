# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for GR.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class GRConfig(BaseModel):
    """GR (GR4J/GR5J) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GR_INSTALL_PATH')
    exe: str = Field(default='GR.r', alias='GR_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='GR_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='GR_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_GR_PATH')
    control: str = Field(default='default', alias='SETTINGS_GR_CONTROL')
    params_to_calibrate: str = Field(
        default='X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff',
        alias='GR_PARAMS_TO_CALIBRATE'
    )
    # Fallback behavior control - default to False to prevent silent data corruption
    allow_dummy_observations: bool = Field(
        default=False,
        alias='GR_ALLOW_DUMMY_OBSERVATIONS',
        description='If True, use zero-filled dummy observations when no streamflow data found'
    )
    allow_default_area: bool = Field(
        default=False,
        alias='GR_ALLOW_DEFAULT_AREA',
        description='If True, use 1.0 km² default area when basin shapefile not found'
    )
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='GR_PARAM_BOUNDS')
    gr4j_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='GR4J_PARAM_BOUNDS')
    initial_params: str = Field(default='default', alias='GR_INITIAL_PARAMS')
    default_params: Optional[List[float]] = Field(default=None, alias='GR_DEFAULT_PARAMS')
    model_type: Optional[str] = Field(
        default=None, alias='GR_MODEL_TYPE',
        description="GR model variant (e.g. 'GR4J', 'GR5J'); default GR4J"
    )
    # Number of equal-area CemaNeige elevation layers (airGR NLayers).
    snow_layers: int = Field(default=5, ge=1, alias='GR_SNOW_LAYERS')
    # Lumped glacier mode: CemaNeige and GR4J run separately and degree-day
    # ice melt from the outline's share of each layer joins GR4J's input.
    glacier_enabled: bool = Field(default=False, alias='GR_GLACIER_ENABLED')
    glacier_outline: Optional[str] = Field(default=None, alias='GR_GLACIER_OUTLINE')
    # Observed mean annual net glacier mass balance (m w.e./yr) over the
    # calibration water years. When set, the ice degree-day factor is solved
    # in every run to reproduce it; otherwise ICE_DDF is a parameter.
    glacier_mb_target: Optional[float] = Field(default=None, alias='GR_GLACIER_MB_TARGET')
    glacier_ice_ddf_bounds: List[float] = Field(default=[1.0, 20.0], alias='GR_GLACIER_ICE_DDF_BOUNDS')
    glacier_snowfree_swe: float = Field(default=1.0, ge=0.0, alias='GR_GLACIER_SNOWFREE_SWE')
    # LamaH-Ice style glacier_timeseries.csv; scales the outline area to the
    # observed area of each water year.
    glacier_area_file: Optional[str] = Field(default=None, alias='GR_GLACIER_AREA_FILE')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
GRConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_GR_PATH': 'settings_path',
    'SETTINGS_GR_CONTROL': 'control',
}
