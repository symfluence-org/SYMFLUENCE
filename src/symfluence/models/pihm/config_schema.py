# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for PIHM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class PIHMConfig(BaseModel):
    """PIHM (Penn State Integrated Hydrologic Model) configuration.

    PIHM is a finite-volume, unstructured-mesh, fully-coupled
    surface-subsurface model solving Richards equation + diffusion wave
    overland flow + 1D channel routing. Uses SUNDIALS CVODE solver.

    Reference:
        Qu, Y. & Duffy, C.J. (2007): A semidiscrete finite volume
        formulation for multiprocess watershed simulation.
        Water Resources Research 43(8).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='PIHM_INSTALL_PATH')
    # Flux-PIHM (Noah-LSM build) is the canonical PIHM SYMFLUENCE targets: the
    # preprocessor writes the Noah .ic/.lsm format and the installer builds the
    # flux-pihm binary. A plain 'pihm' build rejects the .ic ("size does not match").
    exe: str = Field(default='flux-pihm', alias='PIHM_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_PIHM_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='PIHM_SPATIAL_MODE')
    # Number of hillslope bands per bank for a semi-distributed mesh (1 = lumped
    # two-element mesh). >1 discretises each hillslope into a cascade so
    # subsurface water traverses several elements before reaching the channel,
    # supplying the slow hillslope storage-routing a lumped element cannot.
    hillslope_bands: int = Field(default=1, alias='PIHM_HILLSLOPE_BANDS', ge=1)

    # Subsurface properties
    k_sat: float = Field(default=1e-5, alias='PIHM_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='PIHM_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='PIHM_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='PIHM_VG_N', gt=1.0)
    macropore_k: float = Field(default=1e-4, alias='PIHM_MACROPORE_K', gt=0)
    macropore_depth: float = Field(default=0.5, alias='PIHM_MACROPORE_DEPTH', ge=0)
    soil_depth: float = Field(default=2.0, alias='PIHM_SOIL_DEPTH', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='PIHM_MANNINGS_N', gt=0)

    # Initial conditions
    init_gw_depth: float = Field(default=1.0, alias='PIHM_INIT_GW_DEPTH', ge=0)

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='PIHM_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='PIHM_RECHARGE_VARIABLE')

    # Solver
    solver_reltol: float = Field(default=1e-3, alias='PIHM_SOLVER_RELTOL', gt=0)
    solver_abstol: float = Field(default=1e-4, alias='PIHM_SOLVER_ABSTOL', gt=0)
    timestep_seconds: int = Field(default=60, alias='PIHM_TIMESTEP_SECONDS', ge=1, le=86400)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PIHM')
    outlet_segment: int = Field(default=1, alias='PIHM_OUTLET_SEGMENT', ge=1)

    # Calibration
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,MACROPORE_K,MANNINGS_N,SOIL_DEPTH',
        alias='PIHM_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='PIHM_TIMEOUT', ge=60, le=86400)
