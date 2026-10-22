# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for WFLOW.

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


class WflowConfig(BaseModel):
    """Wflow (wflow_sbm) distributed hydrological model configuration.

    Wflow is a distributed hydrological model developed by Deltares,
    implemented in Julia. The wflow_sbm concept uses topography-driven
    subsurface flow with a kinematic wave for overland and river routing.

    Reference:
        van Verseveld et al. (2024): Wflow_sbm v0.7.3, a spatially
        distributed hydrological model. Geosci. Model Dev., 17, 3021-3043.
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WFLOW_INSTALL_PATH')
    exe: str = Field(default='wflow_cli', alias='WFLOW_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WFLOW_PATH')
    config_file: str = Field(default='wflow_sbm.toml', alias='WFLOW_CONFIG_FILE')
    staticmaps_file: str = Field(default='wflow_staticmaps.nc', alias='WFLOW_STATICMAPS_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='WFLOW_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WFLOW')
    output_file: str = Field(default='output.nc', alias='WFLOW_OUTPUT_FILE')
    pet_method: str = Field(default='oudin', alias='WFLOW_PET_METHOD')
    params_to_calibrate: str = Field(
        default='KsatVer,f,SoilThickness,InfiltCapPath,RootingDepth,KsatHorFrac,n_river,PathFrac,thetaS,thetaR,Cfmax,TT,TTI,TTM,WHC',
        alias='WFLOW_PARAMS_TO_CALIBRATE',
    )
    hru_file: str = Field(default='', alias='WFLOW_HRU_FILE')
    dem_file: str = Field(default='', alias='WFLOW_DEM_FILE')
    grid_resolution: float = Field(default=2000.0, gt=0, alias='WFLOW_GRID_RESOLUTION')
    river_threshold_km2: float = Field(default=20.0, gt=0, alias='WFLOW_RIVER_THRESHOLD_KM2')
    glacier_fraction: float = Field(default=0.0, ge=0, le=1, alias='WFLOW_GLACIER_FRACTION')
    glacier_ice_mm: float = Field(default=100000.0, gt=0, alias='WFLOW_GLACIER_ICE_MM')
    glacier_outline_file: str = Field(default='', alias='WFLOW_GLACIER_OUTLINE_FILE')
    glacier_thickness_dir: str = Field(default='', alias='WFLOW_GLACIER_THICKNESS_DIR')
    corrected_routing: bool = Field(default=False, alias='WFLOW_CORRECTED_ROUTING')
    outlet_mode: Literal['dem_min', 'gauge'] = Field(default='dem_min', alias='WFLOW_OUTLET_MODE')
    cryosphere_constraints: bool = Field(default=False, alias='WFLOW_CRYOSPHERE_CONSTRAINTS')
    calibration_only_period: bool = Field(default=False, alias='WFLOW_CALIBRATION_ONLY_PERIOD')
    snow_cover_error_scale: float = Field(default=0.2, gt=0, alias='WFLOW_SNOW_COVER_ERROR_SCALE')
    glacier_mb_error_scale: float = Field(default=1.0, gt=0, alias='WFLOW_GLACIER_MB_ERROR_SCALE')
    snow_cover_threshold_mm: float = Field(default=5.0, gt=0, alias='WFLOW_SNOW_COVER_THRESHOLD_MM')
    # Fixed seasonal air-temperature corrections (deg C, keys DJF/MAM/JJA/SON); PET is recomputed.
    temperature_offsets: dict[str, float] = Field(default_factory=dict, alias='WFLOW_TEMPERATURE_OFFSETS')
    frozen_soil: bool = Field(default=False, alias='WFLOW_FROZEN_SOIL')
    parameter_bounds: dict[str, dict[str, float]] = Field(default_factory=dict, alias='WFLOW_PARAMETER_BOUNDS')
    initial_parameters: dict[str, float] = Field(default_factory=dict, alias='WFLOW_INITIAL_PARAMETERS')
    timeout: int = Field(default=7200, alias='WFLOW_TIMEOUT', ge=60, le=86400)
