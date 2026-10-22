# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Parameter Definitions

Default parameter values and calibration bounds for the
Cold Regions Hydrological Model (CRHM).

CRHM parameters are organized by process module.  The preprocessor
writes *all* parameters into the .prj file (Shared + module-specific).
Only the subset listed in PARAM_BOUNDS is exposed for calibration.

Module parameter categories:
  Shared   - broadcast to every module (basin_area, hru_area, hru_elev, ...)
  albedo   - snow/bare-ground albedo
  basin    - basin metadata (names, state, run control)
  ebsm     - energy-balance snowmelt
  evap     - evapotranspiration
  global   - time offset
  Netroute - routing (Muskingum K, lag, order, whereto)
  obs      - observation bridge (lapse rates, climate change, HRU mapping)
  pbsm     - blowing-snow (Prairie Blowing Snow Model)
  Soil     - soil moisture, groundwater, depression storage
"""

# ---------------------------------------------------------------------------
# Parameter bounds for calibration
# ---------------------------------------------------------------------------
# Only parameters that should be varied during calibration are listed.
# Keys match the CRHM .prj parameter names.
from __future__ import annotations

PARAM_BOUNDS = {
    # -- Shared --
    'basin_area':       {'min': 1.0,   'max': 10000.0},   # km2
    'Ht':               {'min': 0.01,  'max': 5.0},       # vegetation height [m]
    'inhibit_evap':     {'min': 0.0,   'max': 5.0},       # evap inhibition factor
    'soil_rechr_max':   {'min': 10.0,  'max': 200.0},     # max recharge zone [mm]
    'Sdmax':            {'min': 0.0,   'max': 100.0},     # max depression storage [mm]
    'fetch':            {'min': 300.0, 'max': 10000.0},   # blowing snow fetch [m]
    # -- albedo --
    'Albedo_bare':      {'min': 0.05,  'max': 0.40},      # bare-ground albedo
    'Albedo_snow':      {'min': 0.5,   'max': 0.95},      # fresh-snow albedo
    'Asnow':            {'min': 0.5,   'max': 2.5},       # albedo modifier for new snow
    # -- evap --
    'F_Qg':             {'min': 0.0,   'max': 0.5},       # ground heat flux fraction
    'Zwind':            {'min': 1.0,   'max': 30.0},      # wind measurement height [m]
    # -- ebsm snowmelt timing --
    'tfactor':          {'min': 0.0,   'max': 10.0},      # melt temperature factor
    'nfactor':          {'min': 0.0,   'max': 10.0},      # net-radiation melt factor
    'delay_melt':       {'min': 0.0,   'max': 60.0},      # melt onset delay [day-of-year offset]
    # -- Netroute --
    'Kstorage':         {'min': 0.0,   'max': 200.0},     # Muskingum K (channel)
    'Lag':              {'min': 0.0,   'max': 100.0},     # Muskingum lag (channel)
    'gwKstorage':       {'min': 0.0,   'max': 200.0},     # gw storage coeff
    'gwLag':            {'min': 0.0,   'max': 1000.0},    # gw routing lag
    # -- obs --
    'lapse_rate':       {'min': 0.3,   'max': 1.5},       # temp lapse rate [C/100m]
    'precip_elev_adj':  {'min': 0.0,   'max': 1.0},       # orographic precip gradient
    'ClimChng_precip':  {'min': 0.8,   'max': 1.5},       # precipitation multiplier; document forcing uncertainty
    'tmax_allrain':     {'min': 0.0,   'max': 8.0},       # all-rain threshold [C]
    'tmax_allsnow':     {'min': -5.0,  'max': 2.0},       # all-snow threshold [C]
    # -- pbsm / snow sublimation --
    'A_S':              {'min': 0.001, 'max': 0.01},      # snow age decay
    'N_S':              {'min': 50.0,  'max': 500.0},     # vegetation density
    'inhibit_subl':     {'min': 0.0,   'max': 1.0},       # 0/1 switch: PBSM blowing-snow sublimation off
    'Qe_subl_from_SWE': {'min': 0.0,   'max': 1.0},       # native integer switch, not a continuous fraction
    # -- Soil --
    'Ksat':             {'min': 0.01,  'max': 100.0},     # saturated hydraulic conductivity [mm/hr]
    'gw_K':             {'min': 0.0001, 'max': 100.0},    # maximum gw drainage [mm/d]
    'gw_max':           {'min': 50.0,  'max': 500.0},     # max gw storage [mm]
    'soil_gw_K':        {'min': 0.0001, 'max': 100.0},    # maximum excess-soil-to-gw transfer [mm/d]
    'soil_moist_max':   {'min': 50.0,  'max': 500.0},     # max soil moisture [mm]
    'rechr_ssr_K':      {'min': 0.0001, 'max': 100.0},    # maximum recharge-zone drainage [mm/d]
    'lower_ssr_K':      {'min': 0.0001, 'max': 100.0},    # maximum lower-soil drainage [mm/d]
}

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
# These are the values written to the .prj when no calibration override
# is supplied.  They represent physically reasonable mid-range values
# for a cold-region catchment.
#
# NOTE: The preprocessor now writes comprehensive module-specific
# parameters directly.  This dictionary is kept for backward
# compatibility and for the calibration parameter_manager to reference
# default values when constructing trial parameter sets.
DEFAULT_PARAMS = {
    # -- Shared --
    'basin_area': 100.0,           # km2
    'Ht': 0.3,                     # vegetation height [m]
    'inhibit_evap': 1.0,           # evap inhibition factor
    'soil_rechr_max': 60.0,        # max recharge zone storage [mm]
    'Sdmax': 10.0,                 # max depression storage [mm]
    'fetch': 1000.0,               # blowing snow fetch [m]
    # -- albedo --
    'Albedo_bare': 0.17,
    'Albedo_snow': 0.85,
    'Asnow': 1.0,                     # albedo modifier for new snow
    # -- evap --
    'evap_type': 0,
    'F_Qg': 0.05,
    'Zwind': 10.0,
    # -- Netroute --
    'Kstorage': 1.0,
    'Lag': 3.0,
    'gwKstorage': 0.0,
    'gwLag': 0.0,
    # -- obs --
    'lapse_rate': 0.75,
    'tmax_allrain': 4.0,
    'tmax_allsnow': 0.0,
    # -- pbsm --
    'A_S': 0.003,
    'N_S': 320.0,
    # -- Soil --
    'Ksat': 10.0,                     # saturated hydraulic conductivity [mm/hr]
    'cov_type': 1,
    'gw_init': 75.0,
    'gw_K': 0.001,
    'gw_max': 150.0,
    'lower_ssr_K': 0.001,
    'rechr_ssr_K': 0.001,
    'soil_gw_K': 0.001,
    'soil_moist_init': 125.0,
    'soil_moist_max': 250.0,
    'soil_rechr_init': 30.0,
}
