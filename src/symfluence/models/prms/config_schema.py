# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for PRMS.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class PRMSConfig(BaseModel):
    """PRMS (Precipitation-Runoff Modeling System) configuration.

    PRMS is a deterministic, distributed-parameter, physical-process
    watershed model developed by the USGS for simulating the effects
    of precipitation, climate, and land use on streamflow.

    Reference:
        Markstrom, S.L., et al. (2015): PRMS-IV, the Precipitation-Runoff
        Modeling System, Version 4. USGS Techniques and Methods 6-B7.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='PRMS_INSTALL_PATH')
    exe: str = Field(default='prms', alias='PRMS_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_PRMS_PATH')
    control_file: str = Field(default='control.dat', alias='PRMS_CONTROL_FILE')
    parameter_file: str = Field(default='params.dat', alias='PRMS_PARAMETER_FILE')
    data_file: str = Field(default='data.dat', alias='PRMS_DATA_FILE')
    spatial_mode: SpatialModeType = Field(default='semi_distributed', alias='PRMS_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PRMS')
    params_to_calibrate: str = Field(
        default='soil_moist_max,soil_rechr_max,tmax_allrain,tmax_allsnow,hru_percent_imperv,carea_max,smidx_coef,slowcoef_lin,gwflow_coef,ssr2gw_rate',
        alias='PRMS_PARAMS_TO_CALIBRATE'
    )
    model_mode: str = Field(default='DAILY', alias='PRMS_MODEL_MODE')
    streamflow_module: Literal['auto', 'strmflow', 'muskingum'] = Field(
        default='auto', alias='PRMS_STREAMFLOW_MODULE',
        description='auto uses Muskingum when glaciers are enabled so glacier discharge reaches the outlet.'
    )
    timeout: int = Field(default=3600, alias='PRMS_TIMEOUT', ge=60, le=86400)
    use_obs_solar: bool = Field(default=False, alias='PRMS_USE_OBS_SOLAR')

    nhru: int = Field(default=1, alias='PRMS_NHRU', ge=1)
    glacier_flag: bool = Field(default=False, alias='PRMS_GLACIER_FLAG')
    glacier_dynamics: bool = Field(default=True, alias='PRMS_GLACIER_DYNAMICS',
                                   description='False requires the patched fixed-geometry native build and separate pure glacier/land HRUs.')
    water_balance_tolerance_mm: float | None = Field(default=None, alias='PRMS_WATER_BALANCE_TOLERANCE_MM', gt=0.)
    cryosphere_constraints: bool = Field(default=False, alias='PRMS_CRYOSPHERE_CONSTRAINTS')
    snow_cover_error_scale: float = Field(default=0.2, alias='PRMS_SNOW_COVER_ERROR_SCALE', gt=0.)
    glacier_mb_error_scale: float = Field(default=1., alias='PRMS_GLACIER_MB_ERROR_SCALE', gt=0.)
    forcing_mode: Literal['station', 'climate_hru'] = Field(
        default='station', alias='PRMS_FORCING_MODE',
        description='station: one basin-mean series lapsed to each HRU (precip_1sta/temp_1sta). '
                    'climate_hru: per-HRU CBH files from forcing remapped to HRUs (no lapse).'
    )
    hru_forcing_path: str | None = Field(
        default=None, alias='PRMS_HRU_FORCING_PATH',
        description='Directory of canonical forcing NetCDF with an hru dimension (climate_hru mode).'
    )
    hru_forcing_map: list[int] | None = Field(
        default=None, alias='PRMS_HRU_FORCING_MAP',
        description='Source hruId for each PRMS HRU, in PRMS order (e.g. split land/glacier '
                    'HRUs sharing one band). Default: source hruIds 1..nhru.'
    )
    routing_segments: int = Field(
        default=1, alias='PRMS_ROUTING_SEGMENTS', ge=1,
        description='Muskingum reaches in series. Native K_coef is capped at 24 h per reach, '
                    'so longer basin lags need several reaches.'
    )
    separate_glacier_depletion_curve: bool = Field(
        default=False, alias='PRMS_SEPARATE_GLACIER_DEPLETION_CURVE',
        description='Assign land and glacier HRUs separate snow-cover depletion curves. The land '
                    'curve maximum is the calibratable land_sca_max; the glacier curve is unscaled.'
    )
    parameter_overrides: dict[str, dict] = Field(default_factory=dict, alias='PRMS_PARAMETER_OVERRIDES')
    param_bounds: dict[str, dict[str, float]] = Field(default_factory=dict, alias='PRMS_PARAM_BOUNDS')
