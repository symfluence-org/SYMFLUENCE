# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for MODFLOW.

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


class MODFLOWConfig(BaseModel):
    """MODFLOW 6 (USGS modular groundwater flow model) configuration.

    MODFLOW 6 simulates three-dimensional groundwater flow using the
    finite-difference method. In SYMFLUENCE it is used as a lumped
    single-cell groundwater model coupled with land surface models
    (e.g., SUMMA) to separate baseflow from surface runoff.

    Reference:
        Langevin, C.D., et al. (2017): Documentation for the MODFLOW 6
        Groundwater Flow Model. USGS Techniques and Methods 6-A55.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='MODFLOW_INSTALL_PATH')
    exe: str = Field(default='mf6', alias='MODFLOW_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_MODFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MODFLOW_SPATIAL_MODE')

    # Grid: 'lumped' (single cell, default) or 'basin' (real-basin grid from the
    # delineated subbasins, river network and DEM; see basin_model.py)
    grid_mode: Literal['lumped', 'basin'] = Field(default='lumped', alias='MODFLOW_GRID_MODE')
    subbasin_shapefile: Optional[str] = Field(default=None, alias='MODFLOW_SUBBASIN_SHAPEFILE')
    river_network_shapefile: Optional[str] = Field(default=None, alias='MODFLOW_RIVER_NETWORK_SHAPEFILE')
    dem_path: Optional[str] = Field(default=None, alias='MODFLOW_DEM_PATH')
    grid_crs: Optional[int] = Field(default=None, alias='MODFLOW_GRID_CRS',
                                    description='EPSG code of the (metric) grid CRS; default: UTM zone')
    channel_depth: float = Field(default=1.0, alias='MODFLOW_CHANNEL_DEPTH', ge=0)
    aquifer_thickness: float = Field(default=100.0, alias='MODFLOW_AQUIFER_THICKNESS', gt=0)
    river_leakance: float = Field(default=10.0, alias='MODFLOW_RIVER_LEAKANCE', gt=0,
                                  description='River drain conductance per reach length (m2/d per m)')
    seepage_leakance: float = Field(default=0.1, alias='MODFLOW_SEEPAGE_LEAKANCE', gt=0,
                                    description='Land-surface seepage drain conductance per area (1/d)')
    save_heads: Literal['none', 'last', 'all'] = Field(default='last', alias='MODFLOW_SAVE_HEADS')
    # Land-model recharge zones (basin mode): shapefile whose id field matches
    # the land model's spatial units (HYPE subid, SUMMA hruId)
    land_zone_shapefile: Optional[str] = Field(default=None, alias='MODFLOW_LAND_ZONE_SHAPEFILE')
    land_zone_id_field: str = Field(default='HRU_ID', alias='MODFLOW_LAND_ZONE_ID_FIELD')
    land_outlet_id: Optional[int] = Field(default=None, alias='MODFLOW_LAND_OUTLET_ID')

    # Grid discretization (lumped mode)
    grid_type: str = Field(default='dis', alias='MODFLOW_GRID_TYPE')
    nlay: int = Field(default=1, alias='MODFLOW_NLAY', ge=1, le=100)
    nrow: int = Field(default=1, alias='MODFLOW_NROW', ge=1, le=10000)
    ncol: int = Field(default=1, alias='MODFLOW_NCOL', ge=1, le=10000)
    cell_size: Optional[float] = Field(default=None, alias='MODFLOW_CELL_SIZE', gt=0)

    # Aquifer properties
    k: float = Field(default=5.0, alias='MODFLOW_K', gt=0)
    sy: float = Field(default=0.15, alias='MODFLOW_SY', gt=0, le=0.5)
    ss: float = Field(default=1e-5, alias='MODFLOW_SS', gt=0, le=0.1)
    strt: Optional[float] = Field(default=None, alias='MODFLOW_STRT')
    top: float = Field(default=1500.0, alias='MODFLOW_TOP')
    bot: float = Field(default=1400.0, alias='MODFLOW_BOT')

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='MODFLOW_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='MODFLOW_RECHARGE_VARIABLE')

    # Drain package
    drain_elevation: Optional[float] = Field(default=None, alias='MODFLOW_DRAIN_ELEVATION')
    drain_conductance: float = Field(default=50.0, alias='MODFLOW_DRAIN_CONDUCTANCE', gt=0)

    # Stress period
    stress_period_length: float = Field(default=1.0, alias='MODFLOW_STRESS_PERIOD_LENGTH', gt=0)
    nstp: int = Field(default=1, alias='MODFLOW_NSTP', ge=1, le=1000)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MODFLOW')

    # Calibration
    params_to_calibrate: str = Field(
        default='K,SY,DRAIN_CONDUCTANCE',
        description='Lumped: K,SY,DRAIN_CONDUCTANCE. Basin: any of K,SY,AQUIFER_THICKNESS,RIVER_LEAKANCE,SEEPAGE_LEAKANCE',
        alias='MODFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='MODFLOW_TIMEOUT', ge=60, le=86400)
