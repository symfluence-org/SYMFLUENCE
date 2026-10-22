# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for CLMPARFLOW.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class CLMParFlowConfig(BaseModel):
    """ParFlow with tightly-coupled CLM (Common Land Model) configuration.

    ParFlow-CLM is built from the ParFlow source with -DPARFLOW_HAVE_CLM=ON.
    CLM is embedded as Fortran modules inside ParFlow and handles land surface
    energy balance, evapotranspiration, snow dynamics, and vegetation processes.
    This is a different binary from standalone ParFlow (-DPARFLOW_HAVE_CLM=OFF).

    Reference:
        Kollet, S.J. & Maxwell, R.M. (2008): Capturing the influence of
        groundwater dynamics on land surface processes using an integrated,
        distributed watershed model. Water Resources Research 44(2).

        Dai, Y. et al. (2003): The Common Land Model. Bulletin of the
        American Meteorological Society 84(8).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='CLMPARFLOW_INSTALL_PATH')
    exe: str = Field(default='parflow', alias='CLMPARFLOW_EXE')
    parflow_dir: str = Field(default='default', alias='CLMPARFLOW_DIR')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_CLMPARFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CLMPARFLOW_SPATIAL_MODE')

    # Grid discretization (same as ParFlow)
    nx: int = Field(default=3, alias='CLMPARFLOW_NX', ge=1, le=10000)
    ny: int = Field(default=1, alias='CLMPARFLOW_NY', ge=1, le=10000)
    nz: int = Field(default=1, alias='CLMPARFLOW_NZ', ge=1, le=100)
    dx: float = Field(default=1000.0, alias='CLMPARFLOW_DX', gt=0)
    dy: float = Field(default=1000.0, alias='CLMPARFLOW_DY', gt=0)
    dz: float = Field(default=2.0, alias='CLMPARFLOW_DZ', gt=0)

    # Domain geometry
    top: float = Field(default=2.0, alias='CLMPARFLOW_TOP')
    bot: float = Field(default=0.0, alias='CLMPARFLOW_BOT')

    # Subsurface properties
    k_sat: float = Field(default=5.0, alias='CLMPARFLOW_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='CLMPARFLOW_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='CLMPARFLOW_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='CLMPARFLOW_VG_N', gt=1.0)
    s_res: float = Field(default=0.1, alias='CLMPARFLOW_S_RES', ge=0, lt=1.0)
    s_sat: float = Field(default=1.0, alias='CLMPARFLOW_S_SAT', gt=0, le=1.0)
    ss: float = Field(default=1e-5, alias='CLMPARFLOW_SS', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='CLMPARFLOW_MANNINGS_N', gt=0)
    slope_x: float = Field(default=0.01, alias='CLMPARFLOW_SLOPE_X')

    # Initial conditions
    initial_pressure: Optional[float] = Field(default=None, alias='CLMPARFLOW_INITIAL_PRESSURE')

    # CLM-specific files
    vegm_file: str = Field(default='drv_vegm.dat', alias='CLMPARFLOW_VEGM_FILE')
    vegp_file: str = Field(default='drv_vegp.dat', alias='CLMPARFLOW_VEGP_FILE')
    drv_clmin_file: str = Field(default='drv_clmin.dat', alias='CLMPARFLOW_DRV_CLMIN_FILE')

    # CLM land surface settings
    istep_start: int = Field(default=1, alias='CLMPARFLOW_ISTEP_START', ge=1)
    clm_metfile: str = Field(default='forcing.1d', alias='CLMPARFLOW_METFILE')
    clm_metpath: str = Field(default='default', alias='CLMPARFLOW_METPATH')

    # Solver
    solver: str = Field(default='Richards', alias='CLMPARFLOW_SOLVER')
    timestep_hours: float = Field(default=1.0, alias='CLMPARFLOW_TIMESTEP_HOURS', gt=0)

    # Parallel execution
    num_procs: int = Field(default=1, alias='CLMPARFLOW_NUM_PROCS', ge=1, le=1024)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CLMPARFLOW')

    # Calibration — combined subsurface + CLM land surface + Snow-17 + routing params
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,S_RES,MANNINGS_N,SNOW17_SCF,SNOW17_MFMAX,SNOW17_MFMIN,SNOW17_PXTEMP,SNOW_LAPSE_RATE,ROUTE_ALPHA,ROUTE_K_SLOW,ROUTE_BASEFLOW',
        alias='CLMPARFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=7200, alias='CLMPARFLOW_TIMEOUT', ge=60, le=86400)

    # Basin mode (opt-in): regular grid over the real basin, water-year segments with spin-up (basin/ package).
    # Glacier water and the ice albedo need the patched build: symfluence binary install clmparflow --patched.
    basin_mode: bool = Field(default=False, alias='CLMPARFLOW_BASIN_MODE')
    basin_outline: str = Field(default='default', alias='CLMPARFLOW_BASIN_OUTLINE')
    basin_units: str = Field(default='default', alias='CLMPARFLOW_BASIN_UNITS')
    basin_unit_field: str = Field(default='HRU_ID', alias='CLMPARFLOW_BASIN_UNIT_FIELD')
    basin_ice_field: Optional[str] = Field(default=None, alias='CLMPARFLOW_BASIN_ICE_FIELD')
    basin_ice_value: str = Field(default='ice', alias='CLMPARFLOW_BASIN_ICE_VALUE')
    basin_dem: str = Field(default='default', alias='CLMPARFLOW_BASIN_DEM')
    basin_pour_point: str = Field(default='default', alias='CLMPARFLOW_BASIN_POUR_POINT')
    basin_crs: Optional[str] = Field(default=None, alias='CLMPARFLOW_BASIN_CRS')
    basin_dx: float = Field(default=1000.0, alias='CLMPARFLOW_BASIN_DX', gt=0)
    basin_min_slope: float = Field(default=1e-4, alias='CLMPARFLOW_BASIN_MIN_SLOPE', gt=0)
    basin_forcing: str = Field(default='default', alias='CLMPARFLOW_BASIN_FORCING')
    basin_forcing_unit_variable: str = Field(default='hruId', alias='CLMPARFLOW_BASIN_FORCING_UNIT_VARIABLE')
    basin_forcing_variables: Optional[Dict[str, str]] = Field(default=None, alias='CLMPARFLOW_BASIN_FORCING_VARIABLES')
    basin_precip_scale: float = Field(default=1.0, alias='CLMPARFLOW_BASIN_PRECIP_SCALE', gt=0)
    basin_ice_albedo: Optional[List[float]] = Field(default=None, alias='CLMPARFLOW_BASIN_ICE_ALBEDO')
    basin_glacier_bed_k: Optional[float] = Field(default=None, alias='CLMPARFLOW_BASIN_GLACIER_BED_K', gt=0)
    basin_soil_k: float = Field(default=0.1, alias='CLMPARFLOW_BASIN_SOIL_K', gt=0)
    basin_aquifer_k: float = Field(default=1.0, alias='CLMPARFLOW_BASIN_AQUIFER_K', gt=0)
    basin_manning: float = Field(default=3e-6, alias='CLMPARFLOW_BASIN_MANNING', gt=0)
    basin_initial_water_table_depth: float = Field(default=3.0, alias='CLMPARFLOW_BASIN_INITIAL_WATER_TABLE_DEPTH', ge=0)
    basin_spinup_years: Optional[List[int]] = Field(default=None, alias='CLMPARFLOW_BASIN_SPINUP_YEARS')
    basin_first_water_year: Optional[int] = Field(default=None, alias='CLMPARFLOW_BASIN_FIRST_WATER_YEAR')
    basin_last_water_year: Optional[int] = Field(default=None, alias='CLMPARFLOW_BASIN_LAST_WATER_YEAR')
    basin_drift_tol: float = Field(default=0.05, alias='CLMPARFLOW_BASIN_DRIFT_TOL', gt=0)
    basin_max_spinup_cycles: int = Field(default=3, alias='CLMPARFLOW_BASIN_MAX_SPINUP_CYCLES', ge=1)
    basin_initial_state: Optional[str] = Field(default=None, alias='CLMPARFLOW_BASIN_INITIAL_STATE')
