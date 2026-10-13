# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM (mesoscale Hydrological Model) Parameter Definitions.

This module provides parameter bounds and default values for the mHM model,
as well as the full parameter specification for generating ``mhm_parameter.nml``.

mHM uses Multiscale Parameter Regionalization (MPR) to transfer parameters
across scales. The parameters defined here represent the key process controls
that can be adjusted during calibration.

The mHM binary expects three namelist files at runtime:
  - ``mhm.nml``           -- main simulation configuration
  - ``mrm.nml``           -- routing configuration
  - ``mhm_parameter.nml`` -- parameter values, bounds, flags, and scaling

Each entry in ``mhm_parameter.nml`` follows the Fortran namelist format::

    parameterName = lower_bound, upper_bound, value, FLAG, SCALING

where FLAG=1 enables optimisation and SCALING is typically 1.

References:
    Samaniego, L., et al. (2010): Multiscale parameter regionalization of a
    grid-based hydrologic model at the mesoscale. Water Resources Research,
    46, W05523.

    Kumar, R., et al. (2013): Toward computationally efficient large-scale
    hydrologic predictions with a multiscale regionalization scheme. Water
    Resources Research, 49, 5700-5714.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# =============================================================================
# CALIBRATION PARAMETER BOUNDS  (used by the optimizer / parameter manager)
# =============================================================================

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    'canopyInterceptionFactor': {'min': 0.15, 'max': 0.4},
    'snowTreshholdTemperature': {'min': -2.0, 'max': 2.0},         # deg C (mHM spelling)
    'degreeDayFactor_forest': {'min': 0.0001, 'max': 4.0},         # mm/deg C/hour
    'degreeDayFactor_pervious': {'min': 0.0, 'max': 2.0},          # mm/deg C/hour
    'PTF_Ks_constant': {'min': -1.2, 'max': -0.285},               # PTF constant
    'interflowRecession_slope': {'min': 0.0, 'max': 10.0},         # unitless
    'rechargeCoefficient': {'min': 0.0, 'max': 50.0},              # unitless
    'GeoParam(1,:)': {'min': 10.0, 'max': 500.0},                   # baseflow recession (days)
    'infiltrationShapeFactor': {'min': 1.0, 'max': 4.0},           # infiltration curve shape
    'rootFractionCoefficient_pervious': {'min': 0.001, 'max': 0.09},  # root water uptake depth
    'interflowStorageCapacityFactor': {'min': 75.0, 'max': 200.0}, # interflow storage volume
    'slowInterflowRecession_Ks': {'min': 1.0, 'max': 30.0},       # slow interflow routing speed
    'muskingumTravelTime_constant': {'min': 0.31, 'max': 0.35},   # channel routing travel time
    'orgMatterContent_forest': {'min': 5.0, 'max': 10.0},         # soil organic matter
    'minCorrectionFactorPET': {'min': 0.30, 'max': 1.30},        # direct PET multiplier (fAsp)
    'maxCorrectionFactorPET': {'min': 0.01, 'max': 0.20},        # PET correction range
    'rootFractionCoefficient_forest': {'min': 0.80, 'max': 0.999},  # forest root depth fraction
    'fastInterflowRecession_forest': {'min': 1.0, 'max': 3.0},  # fast interflow recession
    'exponentSlowInterflow': {'min': 0.05, 'max': 0.30},        # slow interflow recession shape
    'imperviousStorageCapacity': {'min': 0.0, 'max': 5.0},      # impervious surface retention
    'rechargeFactor_karstic': {'min': -5.0, 'max': 5.0},        # deep groundwater loss adjustment
    'degreeDayFactor_impervious': {'min': 0.0, 'max': 1.0},
    'increaseDegreeDayFactorByPrecip': {'min': 0.1, 'max': 0.9},
    'maxDegreeDayFactor_pervious': {'min': 0.0, 'max': 8.0},
    'maxDegreeDayFactor_impervious': {'min': 0.0, 'max': 8.0},
    # Forcing-side pseudo-parameters (FORCING_PARAMS): applied by the
    # calibration worker to the precipitation input, not written to
    # mhm_parameter.nml.
    'precipCorrectionFactor': {'min': 0.8, 'max': 1.8},         # multiplier on precipitation
    'glacierIceDDF': {'min': 2.0, 'max': 12.0},                 # mm/deg C/day, external ice melt
}

# Pseudo-parameters with no mHM counterpart. mHM v5.13 has neither a
# precipitation correction nor a glacier store, so the worker rescales the
# precipitation grid and adds degree-day ice melt on glacier cells instead.
FORCING_PARAMS = ('precipCorrectionFactor', 'glacierIceDDF')


# =============================================================================
# DEFAULT PARAMETERS  (simple name -> value mapping for quick access)
# =============================================================================

DEFAULT_PARAMS: Dict[str, float] = {
    'canopyInterceptionFactor': 0.15,
    'snowTreshholdTemperature': 1.0,
    'degreeDayFactor_forest': 1.5,
    'degreeDayFactor_pervious': 0.5,
    'PTF_Ks_constant': -0.585,
    'interflowRecession_slope': 7.0,
    'rechargeCoefficient': 35.0,
    'GeoParam(1,:)': 100.0,
    'infiltrationShapeFactor': 1.75,
    'rootFractionCoefficient_pervious': 0.02,
    'interflowStorageCapacityFactor': 85.0,
    'slowInterflowRecession_Ks': 15.0,
    'muskingumTravelTime_constant': 0.325,
    'orgMatterContent_forest': 7.0,
    'minCorrectionFactorPET': 0.90,
    'maxCorrectionFactorPET': 0.10,
    'rootFractionCoefficient_forest': 0.97,
    'fastInterflowRecession_forest': 1.5,
    'exponentSlowInterflow': 0.125,
    'imperviousStorageCapacity': 0.5,
    'rechargeFactor_karstic': -1.0,
    'degreeDayFactor_impervious': 0.5,
    'increaseDegreeDayFactorByPrecip': 0.5,
    'maxDegreeDayFactor_pervious': 4.0,
    'maxDegreeDayFactor_impervious': 3.5,
    'precipCorrectionFactor': 1.0,
    'glacierIceDDF': 6.0,
}


# =============================================================================
# FULL mhm_parameter.nml SPECIFICATION
#
# Each entry is (lower, upper, value, flag, scaling).
# These are organised into namelist blocks exactly as mHM expects.
# Parameter names and default ranges are derived from the reference
# mhm_parameter.nml shipped with mHM v5.12+.
# =============================================================================

# Type alias for a single parameter row
ParamRow = Tuple[float, float, float, int, int]

# Ordered list of (block_name, [(param_name, lower, upper, value, flag, scaling), ...])
MHM_PARAMETER_NML_SPEC: List[Tuple[str, List[Tuple[str, float, float, float, int, int]]]] = [
    # -- Interception ---------------------------------------------------------
    ('interception1', [
        ('canopyInterceptionFactor', 0.15, 0.40, 0.15, 1, 1),
    ]),

    # -- Snow -----------------------------------------------------------------
    ('snow1', [
        ('snowTreshholdTemperature',        -2.0,   2.0,  1.0, 1, 1),
        ('degreeDayFactor_forest',           0.0001, 4.0,  1.5, 1, 1),
        ('degreeDayFactor_impervious',       0.0,    1.0,  0.5, 1, 1),
        ('degreeDayFactor_pervious',         0.0,    2.0,  0.5, 1, 1),
        ('increaseDegreeDayFactorByPrecip',  0.1,    0.9,  0.5, 1, 1),
        ('maxDegreeDayFactor_forest',        0.0,    8.0,  3.0, 1, 1),
        ('maxDegreeDayFactor_impervious',    0.0,    8.0,  3.5, 1, 1),
        ('maxDegreeDayFactor_pervious',      0.0,    8.0,  4.0, 1, 1),
    ]),

    # -- Soil moisture (process option 1) -------------------------------------
    ('soilmoisture1', [
        ('orgMatterContent_forest',            5.0,    10.0,     7.0,      1, 1),
        ('orgMatterContent_impervious',        0.0,     1.0,     0.5,      1, 1),
        ('orgMatterContent_pervious',          1.0,     5.0,     2.5,      1, 1),
        ('PTF_lower66_5_constant',             0.75,    0.80,    0.788,    1, 1),
        ('PTF_lower66_5_clay',                 0.0008,  0.0012,  0.001,    1, 1),
        ('PTF_lower66_5_Db',                  -0.27,   -0.25,   -0.263,   1, 1),
        ('PTF_higher66_5_constant',            0.80,    0.90,    0.8907,   1, 1),
        ('PTF_higher66_5_clay',               -0.0012, -0.0008, -0.001,   1, 1),
        ('PTF_higher66_5_Db',                 -0.35,   -0.30,   -0.322,   1, 1),
        ('PTF_Ks_constant',                   -1.20,   -0.285,  -0.585,   1, 1),
        ('PTF_Ks_sand',                        0.006,   0.026,   0.0125,  1, 1),
        ('PTF_Ks_clay',                        0.003,   0.013,   0.0063,  1, 1),
        ('PTF_Ks_curveSlope',                 60.96,   60.96,   60.96,    0, 1),
        ('rootFractionCoefficient_forest',     0.90,    0.999,   0.97,    1, 1),
        ('rootFractionCoefficient_impervious', 0.90,    0.95,    0.93,    1, 1),
        ('rootFractionCoefficient_pervious',   0.001,   0.09,    0.02,    1, 1),
        ('infiltrationShapeFactor',            1.0,     4.0,     1.75,    1, 1),
    ]),

    # -- Direct runoff --------------------------------------------------------
    ('directRunoff1', [
        ('imperviousStorageCapacity', 0.0, 5.0, 0.5, 1, 1),
    ]),

    # -- PET (option 0: aspect-based correction) ------------------------------
    ('PET0', [
        ('minCorrectionFactorPET', 0.30, 1.30, 0.90, 1, 1),
        ('maxCorrectionFactorPET', 0.01, 0.20, 0.10, 1, 1),
        ('aspectTresholdPET',    160.0, 200.0, 180.0, 1, 1),
    ]),

    # -- PET (option 1: Hargreaves-Samani) ------------------------------------
    ('PET1', [
        ('minCorrectionFactorPET', 0.30,   1.30,  0.93,   1, 1),
        ('maxCorrectionFactorPET', 0.01,   0.20,  0.19,   1, 1),
        ('aspectTresholdPET',    160.0,   200.0,  171.0,  1, 1),
        ('HargreavesSamaniCoeff',  0.0016,  0.003, 0.0023, 1, 1),
    ]),

    # -- PET (option 2: Priestley-Taylor) -------------------------------------
    ('PET2', [
        ('PriestleyTaylorCoeff',    0.75, 1.75, 1.19,  1, 1),
        ('PriestleyTaylorLAIcorr', -0.50, 0.20, 0.058, 1, 1),
    ]),

    # -- PET (option 3: Penman-Monteith) --------------------------------------
    ('PET3', [
        ('canopyheigth_forest',             15.0,  40.0,  15.0,   1, 1),
        ('canopyheigth_impervious',          0.01,  0.50,  0.02,  1, 1),
        ('canopyheigth_pervious',            0.10,  5.00,  0.11,  1, 1),
        ('displacementheight_coeff',         0.50,  0.85,  0.64,  1, 1),
        ('roughnesslength_momentum_coeff',   0.09,  0.16,  0.095, 1, 1),
        ('roughnesslength_heat_coeff',       0.07,  0.13,  0.075, 1, 1),
        ('stomatal_resistance',             10.0, 200.0,  56.0,   1, 1),
    ]),

    # -- Interflow ------------------------------------------------------------
    ('interflow1', [
        ('interflowStorageCapacityFactor', 75.0, 200.0, 85.0,  1, 1),
        ('interflowRecession_slope',        0.0,  10.0,  7.0,  1, 1),
        ('fastInterflowRecession_forest',   1.0,   3.0,  1.5,  1, 1),
        ('slowInterflowRecession_Ks',       1.0,  30.0, 15.0,  1, 1),
        ('exponentSlowInterflow',           0.05,  0.30, 0.125, 1, 1),
    ]),

    # -- Percolation ----------------------------------------------------------
    ('percolation1', [
        ('rechargeCoefficient',              0.0,  50.0, 35.0, 1, 1),
        ('rechargeFactor_karstic',          -5.0,   5.0, -1.0, 1, 1),
        ('gain_loss_GWreservoir_karstic',    1.0,   1.0,  1.0, 0, 1),
    ]),

    # -- Routing (Muskingum) --------------------------------------------------
    ('routing1', [
        ('muskingumTravelTime_constant',     0.31,  0.35, 0.325, 1, 1),
        ('muskingumTravelTime_riverLength',  0.07,  0.08, 0.075, 1, 1),
        ('muskingumTravelTime_riverSlope',   1.95,  2.10, 2.0,   1, 1),
        ('muskingumTravelTime_impervious',   0.09,  0.11, 0.1,   1, 1),
        ('muskingumAttenuation_riverSlope',  0.01,  0.50, 0.3,   1, 1),
    ]),

    # -- Routing (adaptive time step / celerity) ------------------------------
    ('routing2', [
        ('streamflow_celerity', 0.1, 15.0, 1.5, 0, 1),
    ]),

    # -- Routing (distributed) ------------------------------------------------
    ('routing3', [
        ('slope_factor', 0.1, 100.0, 30.0, 0, 1),
    ]),

    # -- Geological / baseflow parameters -------------------------------------
    # Must have exactly as many GeoParam entries as nGeo_Formations in
    # geology_classdefinition.txt. For a lumped setup with 1 geology class,
    # only GeoParam(1,:) is needed.
    ('geoparameter', [
        ('GeoParam(1,:)',  10.0, 500.0, 100.0, 1, 1),
    ]),
]
