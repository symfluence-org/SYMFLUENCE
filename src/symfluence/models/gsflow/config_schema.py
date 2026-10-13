# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for GSFLOW.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class GSFLOWConfig(BaseModel):
    """GSFLOW (coupled PRMS + MODFLOW-NWT) configuration.

    GSFLOW is a USGS coupled groundwater–surface-water model that integrates
    PRMS (surface/soil) with MODFLOW-NWT (saturated zone) via SFR and UZF
    packages for bidirectional exchange.

    Reference:
        Markstrom, S.L., et al. (2008): GSFLOW—Coupled Ground-Water and
        Surface-Water Flow Model Based on the Integration of the
        Precipitation-Runoff Modeling System (PRMS) and the Modular
        Ground-Water Flow Model (MODFLOW-2005). USGS Techniques and
        Methods 6-D1.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GSFLOW_INSTALL_PATH')
    exe: str = Field(default='gsflow', alias='GSFLOW_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_GSFLOW_PATH')
    control_file: str = Field(default='gsflow.control', alias='GSFLOW_CONTROL_FILE')
    parameter_file: str = Field(default='params.dat', alias='GSFLOW_PARAMETER_FILE')
    modflow_nam_file: str = Field(default='modflow.nam', alias='GSFLOW_MODFLOW_NAM_FILE')
    spatial_mode: SpatialModeType = Field(default='semi_distributed', alias='GSFLOW_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_GSFLOW')
    params_to_calibrate: str = Field(
        default='soil_moist_max,ssr2gw_rate,K,SY,slowcoef_lin,carea_max,smidx_coef,jh_coef,tmax_allsnow,rain_adj,snow_adj',
        alias='GSFLOW_PARAMS_TO_CALIBRATE'
    )
    param_bounds: dict[str, dict[str, float | str]] = Field(default_factory=dict, alias='GSFLOW_PARAM_BOUNDS')
    gsflow_mode: str = Field(default='COUPLED', alias='GSFLOW_MODE')
    distributed_gw: bool = Field(default=False, alias='GSFLOW_DISTRIBUTED_GW')
    gw_grid_n: int = Field(default=10, alias='GSFLOW_GW_GRID_N', ge=2, le=50)
    initial_recharge: float = Field(default=1.e-8, alias='GSFLOW_INITIAL_RECHARGE', ge=0.,
                                    description='Steady-state initialization recharge, m/day; estimate from spin-up, not evaluation discharge.')
    uzf_saturated_water_content: float = Field(default=0.30, alias='GSFLOW_UZF_SATURATED_WATER_CONTENT', gt=0., lt=1.)
    timeout: int = Field(default=7200, alias='GSFLOW_TIMEOUT', ge=60, le=86400)


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
GSFLOWConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_GSFLOW_PATH': 'settings_path',
}
