# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for FUSE.

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


class FUSEConfig(BaseModel):
    """FUSE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='FUSE_INSTALL_PATH')
    exe: str = Field(default='fuse.exe', alias='FUSE_EXE')
    routing_integration: str = Field(default='default', alias='FUSE_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_FUSE_PATH')
    filemanager: str = Field(default='default', alias='SETTINGS_FUSE_FILEMANAGER')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='FUSE_SPATIAL_MODE')
    subcatchment_dim: str = Field(default='longitude', alias='FUSE_SUBCATCHMENT_DIM')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_FUSE')
    params_to_calibrate: str = Field(
        default='MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',
        alias='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
    )
    decision_options: Optional[Dict[str, List[str]]] = Field(default_factory=dict, alias='FUSE_DECISION_OPTIONS')
    # Additional FUSE settings
    file_id: Optional[str] = Field(default=None, alias='FUSE_FILE_ID')
    n_elevation_bands: int = Field(default=1, alias='FUSE_N_ELEVATION_BANDS', ge=1)
    forcing_elevation: Optional[float] = Field(
        default=None, alias='FUSE_FORCING_ELEVATION',
        description=('Elevation (m) the forcing represents (Z_FORCING), e.g. reanalysis orography; '
                     'omitted uses the area-weighted mean elevation of the bands')
    )
    timeout: int = Field(default=3600, alias='FUSE_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    run_internal_calibration: bool = Field(default=True, alias='FUSE_RUN_INTERNAL_CALIBRATION')
    output_timestep_seconds: int = Field(default=86400, alias='FUSE_OUTPUT_TIMESTEP_SECONDS')
    snow_model: Optional[str] = Field(default=None, alias='FUSE_SNOW_MODEL')
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='FUSE_PARAM_BOUNDS')
    parameter_regionalization: str = Field(default='lumped', alias='PARAMETER_REGIONALIZATION')
    use_transfer_functions: bool = Field(default=False, alias='USE_TRANSFER_FUNCTIONS')
    transfer_function_coeff_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='TRANSFER_FUNCTION_COEFF_BOUNDS')
    solution_method: Optional[int] = Field(
        default=None, alias='FUSE_SOLUTION_METHOD',
        description='Numerical solver: 0=explicit Euler (fast), 1=implicit Euler (stable). Default: 0'
    )
    timestep_type: Optional[int] = Field(
        default=None, alias='FUSE_TIMESTEP_TYPE',
        description='Timestep control: 0=fixed, 1=adaptive. Default: 0'
    )
    run_mode: Optional[str] = Field(
        default=None, alias='FUSE_RUN_MODE',
        description="Legacy run-mode override. Calibration always uses "
                    "'run_pre'; 'run_def' only runs once as the runner's "
                    "initial default run (a run_def request during "
                    "calibration is ignored with a warning)."
    )
    template_path: Optional[str] = Field(
        default=None, alias='FUSE_TEMPLATE_PATH',
        description='Path to FUSE settings template directory'
    )


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
FUSEConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_FUSE_PATH': 'settings_path',
    'SETTINGS_FUSE_FILEMANAGER': 'filemanager',
    'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'params_to_calibrate',
}
