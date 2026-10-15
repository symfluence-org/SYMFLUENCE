# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for MESH.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class MESHFrozenSoilConfig(BaseModel):
    """Explicit native frozen-soil assumptions; no universal basin defaults."""
    model_config = FROZEN_CONFIG
    soil_porosity: float = Field(gt=0, le=1)
    soil_depth: float = Field(gt=0)
    surface_saturation: float = Field(gt=0, le=1)
    ice_lens_temperature: float = Field(ge=-50, le=0)
    infiltration_coefficient: float = Field(gt=0)
    opportunity_time: float = Field(ge=0, description='Hours; zero selects the native empirical estimate')


class MESHConfig(BaseModel):
    """MESH (Modélisation Environnementale-Surface Hydrology) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MESH_INSTALL_PATH')
    exe: str = Field(default='mesh.exe', alias='MESH_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='MESH_SPATIAL_MODE')
    # MESH run mode: 'runrte' (WATROUTE channel routing + wf_lzs lower-zone
    # baseflow store active) or 'noroute' (streamflow taken directly from the
    # basin water balance). Left unset (None) => auto: multi-cell domains route
    # ('runrte'), a single cell has no channel network and falls back to
    # 'noroute'. Set explicitly to 'runrte' to keep the baseflow store (and its
    # FLZ/PWR/RCHARG parameters) live even on a single-cell domain.
    run_mode: Optional[str] = Field(default=None, alias='MESH_RUNMODE')
    settings_path: str = Field(default='default', alias='SETTINGS_MESH_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MESH')
    forcing_path: str = Field(default='default', alias='MESH_FORCING_PATH')
    forcing_vars: str = Field(default='default', alias='MESH_FORCING_VARS')
    forcing_units: str = Field(default='default', alias='MESH_FORCING_UNITS')
    forcing_to_units: str = Field(default='default', alias='MESH_FORCING_TO_UNITS')
    landcover_stats_path: str = Field(default='default', alias='MESH_LANDCOVER_STATS_PATH')
    landcover_stats_dir: str = Field(default='default', alias='MESH_LANDCOVER_STATS_DIR')
    landcover_stats_file: str = Field(default='default', alias='MESH_LANDCOVER_STATS_FILE')
    main_id: str = Field(default='default', alias='MESH_MAIN_ID')
    ds_main_id: str = Field(default='default', alias='MESH_DS_MAIN_ID')
    landcover_classes: str = Field(default='default', alias='MESH_LANDCOVER_CLASSES')
    ddb_vars: str = Field(default='default', alias='MESH_DDB_VARS')
    ddb_units: str = Field(default='default', alias='MESH_DDB_UNITS')
    ddb_to_units: str = Field(default='default', alias='MESH_DDB_TO_UNITS')
    ddb_min_values: str = Field(default='default', alias='MESH_DDB_MIN_VALUES')
    gru_dim: str = Field(default='default', alias='MESH_GRU_DIM')
    hru_dim: str = Field(default='default', alias='MESH_HRU_DIM')
    outlet_value: str = Field(default='default', alias='MESH_OUTLET_VALUE')
    # Additional MESH settings
    input_file: str = Field(default='default', alias='SETTINGS_MESH_INPUT')
    params_to_calibrate: str = Field(
        default='ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR',
        alias='MESH_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='MESH_SPINUP_DAYS')
    gru_min_total: float = Field(default=0.0, alias='MESH_GRU_MIN_TOTAL')
    # Lumped mode enforcement settings
    force_single_gru: bool = Field(default=True, alias='MESH_FORCE_SINGLE_GRU')
    apply_params_all_grus: bool = Field(default=True, alias='MESH_APPLY_PARAMS_ALL_GRUS')
    use_landcover_multipliers: bool = Field(default=True, alias='MESH_USE_LANDCOVER_MULTIPLIERS')
    enable_frozen_soil: bool = Field(default=True, alias='MESH_ENABLE_FROZEN_SOIL')
    frozen_soil_parameters: Optional[MESHFrozenSoilConfig] = Field(default=None, alias='MESH_FROZEN_SOIL_PARAMETERS')
    daily_tolerance_days: int = Field(default=1, alias='MESH_DAILY_TOLERANCE_DAYS')
    glacier_mode: Literal['none', 'native_ice_surface', 'finite_ice_guard'] = Field(default='none', alias='MESH_GLACIER_MODE')
    spatial_units: Optional[str] = Field(default=None, alias='MESH_SPATIAL_UNITS')
    spatial_forcing: Optional[str] = Field(default=None, alias='MESH_SPATIAL_FORCING')
    spatial_diagnostics: bool = Field(default=False, alias='MESH_SPATIAL_DIAGNOSTICS')
    spatial_routing_kernel: Optional[str] = Field(default=None, alias='MESH_SPATIAL_ROUTING_KERNEL')
    cryosphere_constraints: bool = Field(default=False, alias='MESH_CRYOSPHERE_CONSTRAINTS')
    cryosphere_observations: Optional[str] = Field(default=None, alias='MESH_CRYOSPHERE_OBSERVATIONS')
    snow_cover_error_scale: float = Field(default=0.2, gt=0, alias='MESH_SNOW_COVER_ERROR_SCALE')
    glacier_mb_error_scale: float = Field(default=1.0, gt=0, alias='MESH_GLACIER_MB_ERROR_SCALE')
    snow_cover_max_rmse: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False, alias='MESH_SNOW_COVER_MAX_RMSE', description='Optional hard calibration ceiling for fractional snow-cover RMSE.')
    glacier_mb_max_rmse: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False, alias='MESH_GLACIER_MB_MAX_RMSE', description='Optional hard calibration ceiling for glacier balance RMSE in m water equivalent.')
    glacier_manifest: Optional[str] = Field(default=None, alias='MESH_GLACIER_MANIFEST')
    glacier_unit_ids: Optional[List[str]] = Field(default=None, alias='MESH_GLACIER_UNIT_IDS')
    glacier_albedo_visible: Optional[float] = Field(default=None, gt=0, lt=1, alias='MESH_GLACIER_ALBEDO_VISIBLE')
    glacier_albedo_infrared: Optional[float] = Field(default=None, gt=0, lt=1, alias='MESH_GLACIER_ALBEDO_INFRARED')
    glacier_temperature_c: Optional[float] = Field(default=None, gt=-100, le=0, alias='MESH_GLACIER_TEMPERATURE_C')

    @model_validator(mode='after')
    def validate_glacier_inputs(self):
        if (self.snow_cover_max_rmse is not None or self.glacier_mb_max_rmse is not None) and not self.cryosphere_constraints:
            raise ValueError('Cryosphere RMSE ceilings require cryosphere constraints')
        if bool(self.spatial_units) != bool(self.spatial_forcing):
            raise ValueError('Spatial units and spatial forcing must be supplied together')
        if self.spatial_diagnostics and not self.spatial_units:
            raise ValueError('Spatial diagnostics require explicit spatial units')
        if self.spatial_routing_kernel and not self.spatial_diagnostics:
            raise ValueError('Spatial routing requires per-unit diagnostics')
        if self.cryosphere_constraints and (not self.spatial_diagnostics or self.glacier_mode != 'finite_ice_guard' or not self.cryosphere_observations):
            raise ValueError('Cryosphere constraints require spatial diagnostics, finite ice and observations')
        if self.glacier_mode != 'none':
            if not self.glacier_manifest or not self.glacier_unit_ids or any(value is None for value in (
                self.glacier_albedo_visible, self.glacier_albedo_infrared, self.glacier_temperature_c
            )):
                raise ValueError('Native glacier mode requires a geometry manifest, ordered unit IDs, albedos and temperature')
        return self

    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='MESH_PARAM_BOUNDS')

    # ------------------------------------------------------------------
    # CLASS / hydrology field overrides
    # ------------------------------------------------------------------
    # meshflow hard-derives these regime-determining fields from the input
    # data (soil texture, drainage density, initial states, veg params). When
    # set, these keys OVERRIDE the meshflow-derived values so the surface-runoff
    # regime can be controlled from config alone (no shipping of hand-tuned
    # files). Left as None => keep whatever meshflow produced.
    soil_sand: Optional[List[float]] = Field(default=None, alias='MESH_SOIL_SAND')
    soil_clay: Optional[List[float]] = Field(default=None, alias='MESH_SOIL_CLAY')
    soil_orgm: Optional[List[float]] = Field(default=None, alias='MESH_SOIL_ORGM')
    drainage_density: Optional[float] = Field(default=None, alias='MESH_DD')
    mid: Optional[int] = Field(default=None, alias='MESH_MID')
    init_tbar: Optional[List[float]] = Field(default=None, alias='MESH_INIT_TBAR')
    init_thlq: Optional[List[float]] = Field(default=None, alias='MESH_INIT_THLQ')
    veg_cmas: Optional[float] = Field(default=None, alias='MESH_VEG_CMAS')
    veg_qa50: Optional[float] = Field(default=None, alias='MESH_VEG_QA50')
    veg_vpda: Optional[float] = Field(default=None, alias='MESH_VEG_VPDA')
    veg_vpdb: Optional[float] = Field(default=None, alias='MESH_VEG_VPDB')
    iwf: Optional[int] = Field(default=None, alias='MESH_IWF')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
MESHConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_MESH_PATH': 'settings_path',
    'SETTINGS_MESH_INPUT': 'input_file',
}
