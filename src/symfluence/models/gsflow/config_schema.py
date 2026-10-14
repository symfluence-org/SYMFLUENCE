# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for GSFLOW.

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

    # --- Real-basin geometry (gw_grid_mode: basin) -------------------------
    gw_grid_mode: Literal['square', 'basin'] = Field(
        default='square', alias='GSFLOW_GW_GRID_MODE',
        description='square: schematic grid (distributed_gw / 3x3). basin: MODFLOW grid clipped to the '
                    'delineated subbasins, SFR along the river network, HRUs = subbasin x elevation band '
                    'x land/glacier mapped to cells by true overlap.')
    gw_cell_size_m: float = Field(default=2000., alias='GSFLOW_GW_CELL_SIZE_M', gt=0.)
    grid_crs: int | None = Field(default=None, alias='GSFLOW_GRID_CRS',
                                 description='Projected EPSG code for the grid; default: UTM zone of the basin.')
    hru_band_size_m: float = Field(default=200., alias='GSFLOW_HRU_BAND_SIZE_M', gt=0.)
    min_hru_area_km2: float = Field(default=2., alias='GSFLOW_MIN_HRU_AREA_KM2', ge=0.)
    min_glacier_hru_area_km2: float = Field(default=0.25, alias='GSFLOW_MIN_GLACIER_HRU_AREA_KM2', ge=0.)
    subbasin_shapefile: str | None = Field(default=None, alias='GSFLOW_SUBBASIN_SHAPEFILE',
                                           description='Delineated subbasins (GRU_ID, gru_to_seg).')
    river_network_shapefile: str | None = Field(default=None, alias='GSFLOW_RIVER_NETWORK_SHAPEFILE',
                                                description='TauDEM river network (LINKNO, DSLINKNO).')
    dem_path: str | None = Field(default=None, alias='GSFLOW_DEM_PATH')
    glacier_outlines: str | None = Field(default=None, alias='GSFLOW_GLACIER_OUTLINES')
    glacier_outline_query: str | None = Field(default=None, alias='GSFLOW_GLACIER_OUTLINE_QUERY',
                                              description='pandas query selecting outlines, e.g. "id == 86".')
    channel_depth_m: float = Field(default=1., alias='GSFLOW_CHANNEL_DEPTH_M', ge=0.)
    uzf_ntrail: int = Field(default=15, alias='GSFLOW_UZF_NTRAIL', ge=1,
                            description='UZF NTRAIL2 (trailing waves); fewer is faster (basin mode).')
    uzf_nsets: int = Field(default=20, alias='GSFLOW_UZF_NSETS', ge=1, description='UZF NSETS2 (basin mode).')
    aquifer_thickness_m: float | None = Field(default=None, alias='GSFLOW_AQUIFER_THICKNESS_M', gt=0.,
                                              description='Default: max(100 m, 5-95% relief).')
    cov_type: int = Field(default=3, alias='GSFLOW_COV_TYPE', ge=0, le=4,
                          description='PRMS cover type for land HRUs (0 bare, 1 grass, 2 shrub, 3 trees).')

    # --- Glacier and forcing (basin mode) ------------------------------------
    glacier_flag: bool = Field(default=False, alias='GSFLOW_GLACIER_FLAG')
    glacier_dynamics: bool = Field(
        default=True, alias='GSFLOW_GLACIER_DYNAMICS',
        description='False = fixed glacier geometry; requires a `binary install gsflow --patched` build.')
    forcing_mode: Literal['station', 'climate_hru'] = Field(default='station', alias='GSFLOW_FORCING_MODE')
    hru_forcing_path: str | None = Field(default=None, alias='GSFLOW_HRU_FORCING_PATH',
                                         description='Canonical forcing NetCDF with an hru dimension.')
    hru_forcing_band_ids: dict[int, int] | None = Field(
        default=None, alias='GSFLOW_HRU_FORCING_BAND_IDS',
        description='Elevation-band lower bound (m) -> source hruId; each HRU uses its band (nearest band '
                    'outside the mapped range).')
    separate_glacier_depletion_curve: bool = Field(default=False, alias='GSFLOW_SEPARATE_GLACIER_DEPLETION_CURVE')
    cryosphere_constraints: bool = Field(default=False, alias='GSFLOW_CRYOSPHERE_CONSTRAINTS',
                                         description='Add snow-cover and glacier mass-balance scores (PRMS scheme).')
    snow_cover_error_scale: float = Field(default=0.2, alias='GSFLOW_SNOW_COVER_ERROR_SCALE', gt=0.)
    glacier_mb_error_scale: float = Field(default=1., alias='GSFLOW_GLACIER_MB_ERROR_SCALE', gt=0.)
    parameter_overrides: dict[str, dict] = Field(default_factory=dict, alias='GSFLOW_PARAMETER_OVERRIDES')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
GSFLOWConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_GSFLOW_PATH': 'settings_path',
}
