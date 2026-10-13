# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for CRHM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class CRHMConfig(BaseModel):
    """CRHM (Cold Regions Hydrological Model) configuration.

    CRHM is a physically-based, object-oriented hydrological model
    designed specifically for cold-region processes including blowing
    snow, energy-balance snowmelt, and frozen soil infiltration.

    Reference:
        Pomeroy, J.W., et al. (2007): The Cold Regions Hydrological Model:
        a platform for basing process representation and model structure on
        physical evidence. Hydrological Processes, 21(19), 2650-2667.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='CRHM_INSTALL_PATH')
    exe: str = Field(default='crhm', alias='CRHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_CRHM_PATH')
    project_file: str = Field(default='model.prj', alias='CRHM_PROJECT_FILE')
    observation_file: str = Field(default='forcing.obs', alias='CRHM_OBSERVATION_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CRHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CRHM')
    params_to_calibrate: str = Field(
        default='Ht,Asnow,inhibit_evap,Ksat,soil_rechr_max,soil_moist_max,soil_gw_K,Sdmax,fetch,inhibit_subl,N_S,Kstorage,gw_K,tfactor,delay_melt,gwKstorage,gwLag',
        alias='CRHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='CRHM_TIMEOUT', ge=60, le=86400)
    elevation_bands: bool = Field(default=False, alias='CRHM_ELEVATION_BANDS')
    elevation_band_size: float = Field(
        default=200.0, alias='CRHM_ELEVATION_BAND_SIZE', gt=0.0)
    terrain_radiation: bool = Field(default=False, alias='CRHM_TERRAIN_RADIATION')
    sublimation_from_swe: bool = Field(default=False, alias='CRHM_SUBLIMATION_FROM_SWE',
                                     description='Use the native ebsm explicit SWE sublimation option (Qe_subl_from_SWE=1).')
    blowing_snow: bool = Field(default=True, alias='CRHM_BLOWING_SNOW',
                               description='Enable PBSM wind transport and blowing-snow sublimation; requires defensible HRU transport connectivity.')

    route_groundwater_to_stream: bool = Field(default=False, alias='CRHM_ROUTE_GROUNDWATER_TO_STREAM')
