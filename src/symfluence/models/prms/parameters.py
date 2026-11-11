# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS (Precipitation-Runoff Modeling System) Parameter Definitions.

This module provides parameter bounds and default values for the PRMS model,
used during calibration with the SYMFLUENCE optimization framework.

PRMS uses an HRU-based semi-distributed structure. The calibration parameters
control key hydrological processes: soil moisture storage, snow partitioning,
surface runoff generation, subsurface flow, and groundwater discharge.

Each parameter is written to the PRMS parameter file (params.dat) in the
standard PRMS format::

    parameter_name
    ndim
    dimension_name
    dimension_size
    type_code
    value(s)

References:
    Markstrom, S.L., et al. (2015): PRMS-IV, the Precipitation-Runoff
    Modeling System, Version 4. USGS Techniques and Methods 6-B7.

    Hay, L.E., et al. (2006): Sensitivity Analysis for Calibration of a
    Conceptual Rainfall-Runoff Model.
"""
from __future__ import annotations

from typing import Dict, List

# =============================================================================
# CALIBRATION PARAMETER BOUNDS  (used by the optimizer / parameter manager)
#
# Units follow the parameter file the preprocessor writes: temp_units = 1, so
# temperature parameters (tmax_allrain, tmax_allrain_offset, tmax_allsnow) are
# in degC. precip_units = 1 only declares the units of the precipitation
# INPUT (mm); PRMS storage-depth parameters (soil_moist_max, soil_rechr_max,
# soil2gw_max, ...) are in inches whatever precip_units says.
# =============================================================================

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    'soil_moist_max': {
        'min': 1.0, 'max': 20.0,                   # inches, maximum soil moisture storage
    },
    'soil_rechr_max': {
        'min': 0.5, 'max': 10.0,                   # inches, maximum recharge zone storage (not read in PRMS5 mode)
    },
    'soil_rechr_max_frac': {
        'min': 0.05, 'max': 1.0,                   # fraction of soil_moist_max (PRMS5 mode)
    },
    'tmax_allrain': {
        'min': 1.0, 'max': 7.0,                    # degC, temp above which all precip is rain (not read in PRMS5 mode)
    },
    'tmax_allrain_offset': {
        'min': 0.5, 'max': 7.0,                    # degC above tmax_allsnow at which all precip is rain (PRMS5 mode)
    },
    'tmax_allsnow': {
        'min': -3.0, 'max': 2.0,                   # degC, temp below which all precip is snow
    },
    'hru_percent_imperv': {
        'min': 0.0, 'max': 0.5,                    # fraction, impervious area per HRU
    },
    'carea_max': {
        'min': 0.1, 'max': 1.0,                    # fraction, maximum contributing area
    },
    'smidx_coef': {
        'min': 0.001, 'max': 0.10,                 # coefficient in surface runoff equation
    },
    'slowcoef_lin': {
        'min': 0.001, 'max': 0.5,                  # linear gravity drainage coefficient
    },
    'gwflow_coef': {
        'min': 0.001, 'max': 0.5,                  # groundwater outflow coefficient
    },
    'ssr2gw_rate': {
        'min': 0.001, 'max': 1.0,                  # gravity reservoir to GW rate
    },
    'soil2gw_max': {
        'min': 0.01, 'max': 5.0,                   # inches/day, max soil-to-GW percolation
    },
    'K_coef': {
        'min': 1.0, 'max': 24.0,                   # hours per Muskingum reach (native cap 24)
    },
    'x_coef': {
        'min': 0.0, 'max': 0.5,                    # Muskingum weighting factor
    },
    'rain_cbh_adj': {
        'min': 0.7, 'max': 1.5,                    # climate_hru rain multiplier
    },
    'snow_cbh_adj': {
        'min': 0.7, 'max': 2.5,                    # climate_hru snow multiplier
    },
    'land_sca_max': {
        'min': 0.4, 'max': 1.0,                    # max snow-covered fraction of land HRUs (derived)
    },
}


# =============================================================================
# DEFAULT PARAMETERS  (simple name -> value mapping for quick access)
# =============================================================================

DEFAULT_PARAMS: Dict[str, float] = {
    'soil_moist_max': 6.0,        # inches
    'soil_rechr_max': 2.0,        # inches
    'soil_rechr_max_frac': 1.0 / 3.0,
    'tmax_allrain': 3.3,          # degC (the parameter file declares temp_units = 1)
    'tmax_allrain_offset': 3.3,   # degC above tmax_allsnow
    'tmax_allsnow': 0.0,          # degC
    'hru_percent_imperv': 0.01,
    'carea_max': 0.6,
    'smidx_coef': 0.01,
    'slowcoef_lin': 0.015,
    'gwflow_coef': 0.015,
    'ssr2gw_rate': 0.1,
    'soil2gw_max': 0.5,
}


# =============================================================================
# PARAMETER FILE SPECIFICATION
#
# Maps parameter names to their PRMS parameter file attributes.
# Format: param_name -> (dimension, type_code, nhru_or_nmonths)
#   type_code: 1=int, 2=float, 3=double, 4=string
# =============================================================================

PRMS_PARAM_SPEC: Dict[str, Dict] = {
    'soil_moist_max':      {'dimension': 'nhru', 'type_code': 2},
    'soil_rechr_max':      {'dimension': 'nhru', 'type_code': 2},
    'soil_rechr_max_frac': {'dimension': 'nhru', 'type_code': 2},
    'tmax_allrain':        {'dimension': 'nmonths', 'type_code': 2},
    'tmax_allrain_offset': {'dimension': 'nmonths', 'type_code': 2},
    'tmax_allsnow':        {'dimension': 'nmonths', 'type_code': 2},
    'hru_percent_imperv':  {'dimension': 'nhru', 'type_code': 2},
    'carea_max':           {'dimension': 'nhru', 'type_code': 2},
    'smidx_coef':          {'dimension': 'nhru', 'type_code': 2},
    'slowcoef_lin':        {'dimension': 'nhru', 'type_code': 2},
    'gwflow_coef':         {'dimension': 'nhru', 'type_code': 2},
    'ssr2gw_rate':         {'dimension': 'nhru', 'type_code': 2},
    'soil2gw_max':         {'dimension': 'nhru', 'type_code': 2},
}


# =============================================================================
# MODE-DEPENDENT PARAMETER NAMES
#
# With model_mode PRMS5 (PRMS4_flag off) PRMS does not read soil_rechr_max or
# tmax_allrain: the recharge-zone capacity is soil_rechr_max_frac x
# soil_moist_max and the all-rain threshold is tmax_allsnow +
# tmax_allrain_offset. The preprocessor writes whichever set the mode reads.
# =============================================================================

DEFAULT_CALIBRATION_PARAMS = (
    'soil_moist_max,soil_rechr_max,tmax_allrain,tmax_allsnow,hru_percent_imperv,'
    'carea_max,smidx_coef,slowcoef_lin,gwflow_coef,ssr2gw_rate'
)

PRMS5_PARAMETER_NAMES: Dict[str, str] = {
    'soil_rechr_max': 'soil_rechr_max_frac',
    'tmax_allrain': 'tmax_allrain_offset',
}


def is_prms5_mode(model_mode: object) -> bool:
    """True when ``model_mode`` makes PRMS read the PRMS5 parameter set."""
    return str(model_mode).strip().upper().startswith('PRMS5')


def default_calibration_parameters(model_mode: object) -> List[str]:
    """Default calibration list, using the parameter names ``model_mode`` reads."""
    names = [name.strip() for name in DEFAULT_CALIBRATION_PARAMS.split(',')]
    if is_prms5_mode(model_mode):
        names = [PRMS5_PARAMETER_NAMES.get(name, name) for name in names]
    return names
