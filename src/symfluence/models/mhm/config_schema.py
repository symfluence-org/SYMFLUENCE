# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for MHM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class MHMConfig(BaseModel):
    """mHM (mesoscale Hydrological Model) configuration.

    mHM is a spatially distributed hydrological model developed at the
    Helmholtz Centre for Environmental Research (UFZ). It uses multiscale
    parameter regionalization (MPR) for parameter transfer.

    Reference:
        Samaniego, L., et al. (2010): Multiscale parameter regionalization
        of a grid-based hydrologic model at the mesoscale. Water Resources
        Research, 46, W05523.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MHM_INSTALL_PATH')
    exe: str = Field(default='mhm', alias='MHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MHM_PATH')
    namelist_file: str = Field(default='mhm.nml', alias='MHM_NAMELIST_FILE')
    routing_namelist: str = Field(default='mrm.nml', alias='MHM_ROUTING_NAMELIST')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MHM')
    params_to_calibrate: str = Field(
        default='canopyInterceptionFactor,snowTreshholdTemperature,degreeDayFactor_forest,degreeDayFactor_pervious,PTF_Ks_constant,interflowRecession_slope,rechargeCoefficient,GeoParam(1,:),infiltrationShapeFactor,rootFractionCoefficient_pervious,interflowStorageCapacityFactor,slowInterflowRecession_Ks,muskingumTravelTime_constant,orgMatterContent_forest',
        alias='MHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='MHM_TIMEOUT', ge=60, le=86400)
    distributed_morph: bool = Field(default=False, alias='MHM_DISTRIBUTED')
    grid_res: float = Field(default=0.02, alias='MHM_GRID_RES', gt=0.0)

    match_lumped_area: bool = Field(default=False, alias='MHM_MATCH_LUMPED_AREA')
    # Model time step in hours; mHM accepts divisors of 24.
    timestep: Literal[1, 2, 3, 4, 6, 12, 24] = Field(default=1, alias='MHM_TIMESTEP')

    # Distributed-forcing options (only used with distributed_morph). The
    # basin-mean forcing is taken to represent the basin-mean DEM elevation and
    # temperature is lapsed to each cell's elevation; None keeps it uniform.
    temp_lapse_rate: Optional[float] = Field(default=None, alias='MHM_TEMP_LAPSE_RATE')  # K/km
    pet_method: Literal['hamon', 'oudin'] = Field(default='hamon', alias='MHM_PET_METHOD')
    # Glacier outline (vector). Cells whose glacierised fraction is at least
    # glacier_min_fraction become mHM land class 2 (fully sealed) and receive
    # the external degree-day ice melt calibrated through glacierIceDDF.
    glacier_outline: Optional[str] = Field(default=None, alias='MHM_GLACIER_OUTLINE')
    glacier_min_fraction: float = Field(default=0.5, alias='MHM_GLACIER_MIN_FRACTION', ge=0.0, le=1.0)
    # Scale each glacier cell's ice melt by its glacierised fraction, so the
    # melting area matches the outline instead of whole cells.
    glacier_area_from_outline: bool = Field(default=False, alias='MHM_GLACIER_AREA_FROM_OUTLINE')
    # Observed mean annual net glacier mass balance (m w.e./yr) over the
    # calibration water years. When set, the worker solves for the ice
    # degree-day factor that reproduces it in every evaluation, and
    # glacierIceDDF is no longer calibrated.
    glacier_mb_target: Optional[float] = Field(default=None, alias='MHM_GLACIER_MB_TARGET')
