# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Parameter Definitions.

GSFLOW couples PRMS surface processes with MODFLOW-NWT groundwater.
Calibration parameters span both PRMS (soil, runoff) and MODFLOW-NWT (K, SY).

Note: In COUPLED mode (GSFLOW v2.4.0+), PRMS6 ignores several legacy
parameters that are superseded by MODFLOW-NWT or replaced by fractional
variants:
  - soil_rechr_max  → replaced by soil_rechr_max_frac (internal)
  - gwflow_coef     → GW flow handled by MODFLOW-NWT
  - gw_seep_coef    → GW seepage handled by MODFLOW-NWT
  - tmax_allrain    → replaced by tmax_allrain_offset (internal)
These are excluded from calibration to avoid wasting optimization budget
on inert dimensions.

Bounds are NOT defined here. ``PARAM_BOUNDS`` is re-exported from
:mod:`symfluence.models.gsflow.parameter_bounds`, which is the single source of
truth: the same definitions this package contributes to the central bounds
catalogue via ``register_model_bounds()``. It used to be an independent literal
dict, which is how ``K`` came to be calibrated against ``0.1..5000`` linear here
while the catalogue said ``0.001..100`` log.

References:
    Markstrom, S.L., et al. (2008): GSFLOW—Coupled Ground-Water and
    Surface-Water Flow Model. USGS Techniques and Methods 6-D1.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

from .parameter_bounds import CALIBRATION_BOUNDS

#: Calibration bounds, keyed by served (unprefixed) name. A live view, not a
#: dict: it resolves the centrally-owned half at lookup time rather than at
#: import, when registration may not have run.
#: Single source of truth: :mod:`symfluence.models.gsflow.parameter_bounds`.
PARAM_BOUNDS: Mapping = CALIBRATION_BOUNDS

DEFAULT_PARAMS: Dict[str, float] = {
    'soil_moist_max': 6.0,
    'ssr2gw_rate': 0.1,
    'K': 1.0,
    'SY': 0.15,
    'slowcoef_lin': 0.015,
    'carea_max': 0.6,
    'smidx_coef': 0.01,
    'jh_coef': 0.014,
    'tmax_allsnow': 0.0,
    'rain_adj': 1.0,
    'snow_adj': 1.0,
}

# PRMS parameter file specification (for ####-delimited blocks)
PRMS_PARAM_SPEC: Dict[str, Dict] = {
    'soil_moist_max': {'dimension': 'nhru', 'type_code': 2},
    'ssr2gw_rate': {'dimension': 'nhru', 'type_code': 2},
    'slowcoef_lin': {'dimension': 'nhru', 'type_code': 2},
    'carea_max': {'dimension': 'nhru', 'type_code': 2},
    'smidx_coef': {'dimension': 'nhru', 'type_code': 2},
    'jh_coef': {'dimension': 'nmonths', 'type_code': 2},
    'tmax_allsnow': {'dimension': 'nmonths', 'type_code': 2},
    'rain_adj': {'dimension': 'nmonths', 'type_code': 2},
    'snow_adj': {'dimension': 'nmonths', 'type_code': 2},
}

# MODFLOW-NWT UPW package parameter specification
MODFLOW_PARAM_SPEC: Dict[str, Dict] = {
    'K': {'package': 'UPW', 'layer': 1},
    'SY': {'package': 'UPW', 'layer': 1},
}
