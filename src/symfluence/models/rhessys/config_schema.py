# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for RHESSYS.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.models.wmfire.config_schema import WMFireConfig


class RHESSysConfig(BaseModel):
    """RHESSys (Regional Hydro-Ecologic Simulation System) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='RHESSYS_INSTALL_PATH')
    exe: str = Field(default='rhessys', alias='RHESSYS_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_RHESSYS_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_RHESSYS')
    forcing_path: str = Field(default='default', alias='FORCING_RHESSYS_PATH')
    world_template: str = Field(default='world.template', alias='RHESSYS_WORLD_TEMPLATE')
    flow_template: str = Field(default='flow.template', alias='RHESSYS_FLOW_TEMPLATE')
    # LNA/TWI controls (optional; None disables caps)
    lna_area_cap_m2: Optional[float] = Field(default=None, alias='RHESSYS_LNA_AREA_CAP_M2')
    lna_min: Optional[float] = Field(default=None, alias='RHESSYS_LNA_MIN')
    lna_max: Optional[float] = Field(default=None, alias='RHESSYS_LNA_MAX')
    params_to_calibrate: str = Field(
        default=(
            'sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,porosity_decay,'
            'soil_depth,snow_melt_Tcoef,max_snow_temp,min_rain_temp,theta_mean_std_p1'
        ),
        alias='RHESSYS_PARAMS_TO_CALIBRATE'
    )
    skip_calibration: bool = Field(default=True, alias='RHESSYS_SKIP_CALIBRATION')
    # WMFire integration (wildfire spread module)
    use_wmfire: bool = Field(default=False, alias='RHESSYS_USE_WMFIRE')
    wmfire_install_path: str = Field(default='installs/wmfire/lib', alias='WMFIRE_INSTALL_PATH')
    wmfire_lib: str = Field(default='libwmfire.so', alias='WMFIRE_LIB')
    wmfire: Optional[WMFireConfig] = Field(default=None, description='Enhanced WMFire configuration')
    # Legacy VMFire aliases
    use_vmfire: bool = Field(default=False, alias='RHESSYS_USE_VMFIRE')
    vmfire_install_path: str = Field(default='installs/wmfire/lib', alias='VMFIRE_INSTALL_PATH')
    # Execution settings
    timeout: int = Field(default=7200, alias='RHESSYS_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    # Grow mode for Farquhar photosynthesis and transpiration (default True)
    use_grow_mode: bool = Field(default=True, alias='RHESSYS_USE_GROW_MODE')
    # Subgrid variability (-stdev); 0 disables. Read by both runner and calibration worker.
    std_scale: float = Field(default=1.0, ge=0, alias='RHESSYS_STD_SCALE')
    # -subsurfacegw (needs the SYMFLUENCE-patched binary; skipped if unsupported).
    use_subsurface_gw: bool = Field(default=True, alias='RHESSYS_USE_SUBSURFACE_GW')
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='RHESSYS_PARAM_BOUNDS')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
RHESSysConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_RHESSYS_PATH': 'settings_path',
}
