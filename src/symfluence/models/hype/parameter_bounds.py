# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing HYPE's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters HYPE calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  HYPE resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for HYPE: ``lp``.

:func:`register_bounds` is called from this package's ``register()``, i.e. at
plugin-discovery time, which runs on ``import symfluence`` -- before any
calibration code can read bounds.
"""
from __future__ import annotations

from typing import Dict, List

from symfluence.core.calibration.parameters.parameter_bounds_registry import (
    ParameterInfo,
    register_model_bounds,
)

#: Tier B -- definitions only HYPE resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'ttmp': ParameterInfo(-5.0, 5.0, '°C', 'Snowmelt threshold temperature', 'snow'),
    'cmlt': ParameterInfo(0.5, 20.0, 'mm/°C/day', 'Snowmelt degree-day coefficient', 'snow'),
    'ttpi': ParameterInfo(0.5, 4.0, '°C', 'Temperature interval for mixed precipitation', 'snow'),
    'cmrefr': ParameterInfo(0.0, 0.5, '-', 'Snow refreeze capacity', 'snow'),
    'sdnsnew': ParameterInfo(0.05, 0.25, 'kg/dm³', 'Fresh snow density', 'snow'),
    'snowdensdt': ParameterInfo(0.0005, 0.005, '1/day', 'Snow densification parameter', 'snow'),
    'fsceff': ParameterInfo(0.5, 1.0, '-', 'Fractional snow cover efficiency', 'snow'),
    'cevp': ParameterInfo(0.1, 2.0, '-', 'Evapotranspiration coefficient (expanded for alpine)', 'et'),
    'epotdist': ParameterInfo(1.0, 15.0, '-', 'PET depth dependency coefficient', 'et'),
    'fepotsnow': ParameterInfo(0.0, 1.0, '-', 'Fraction of PET for snow sublimation', 'et'),
    'ttrig': ParameterInfo(-5.0, 5.0, '°C', 'Soil temperature threshold for transpiration', 'et'),
    'treda': ParameterInfo(0.5, 1.0, '-', 'Soil temp response coefficient A', 'et'),
    'tredb': ParameterInfo(0.1, 0.8, '-', 'Soil temp response coefficient B', 'et'),
    'rrcs1': ParameterInfo(0.001, 1.0, '1/day', 'Recession coefficient upper layer', 'soil'),
    'rrcs2': ParameterInfo(0.0001, 0.5, '1/day', 'Recession coefficient lower layer', 'soil'),
    'rrcs3': ParameterInfo(0.0, 0.3, '1/°', 'Recession slope dependence', 'soil'),
    'wcwp': ParameterInfo(0.01, 0.3, '-', 'Wilting point water content', 'soil'),
    'wcfc': ParameterInfo(0.1, 0.6, '-', 'Field capacity', 'soil'),
    'wcep': ParameterInfo(0.2, 0.7, '-', 'Effective porosity', 'soil'),
    'srrcs': ParameterInfo(0.0, 0.5, '1/day', 'Surface runoff coefficient', 'soil'),
    'bfroznsoil': ParameterInfo(1.0, 10.0, '-', 'Frozen soil infiltration parameter', 'soil'),
    'logsatmp': ParameterInfo(0.5, 3.0, 'log(cm)', 'Saturated matric potential', 'soil'),
    'bcosby': ParameterInfo(4.0, 15.0, '-', 'Cosby B parameter', 'soil'),
    'sfrost': ParameterInfo(0.5, 3.0, 'cm/°C', 'Frost depth parameter', 'soil'),
    'rcgrw': ParameterInfo(1e-05, 1.0, '1/day', 'Regional groundwater recession coefficient', 'baseflow'),
    'deepperc': ParameterInfo(0.0, 0.5, 'mm/day', 'Deep percolation loss rate', 'baseflow'),
    'deepmem': ParameterInfo(100.0, 2000.0, 'days', 'Deep soil temperature memory', 'soil'),
    'surfmem': ParameterInfo(5.0, 50.0, 'days', 'Upper soil temperature memory', 'soil'),
    'depthrel': ParameterInfo(0.5, 3.0, '-', 'Depth relation for soil temperature', 'soil'),
    'rivvel': ParameterInfo(0.2, 30.0, 'm/s', 'River flow velocity', 'routing'),
    'damp': ParameterInfo(0.0, 1.0, '-', 'River damping fraction', 'routing'),
    'qmean': ParameterInfo(10.0, 1000.0, 'mm/yr', 'Initial mean flow', 'routing'),
    'ilratk': ParameterInfo(0.1, 1000.0, '-', 'Internal lake rating curve coefficient', 'routing'),
    'ilratp': ParameterInfo(1.0, 10.0, '-', 'Internal lake rating curve exponent', 'routing'),
    'illdepth': ParameterInfo(0.1, 2.0, 'm', 'Internal lake depth', 'routing'),
    'cmrad': ParameterInfo(0.0, 1.5, 'mm/(MJ/m2)', 'Snow radiation melt coefficient (snowmeltmodel 2)', 'snow'),
    'fscdist0': ParameterInfo(0.3, 0.8, '-', 'Snow distribution coefficient (fractional snow cover)', 'snow'),
    'pcusnow': ParameterInfo(0.0, 0.6, '-', 'Snowfall undercatch correction', 'precipitation'),
    'pcurain': ParameterInfo(0.0, 0.3, '-', 'Rainfall undercatch correction', 'precipitation'),
    'pcelevadd': ParameterInfo(0.0, 0.15, '1/100 m', 'Precipitation increase per 100 m above pcelevth', 'precipitation'),
    'mperc1': ParameterInfo(0.1, 20.0, 'mm/day', 'Maximum percolation, soil layer 1 to 2', 'soil', transform='log'),
    'mperc2': ParameterInfo(0.1, 20.0, 'mm/day', 'Maximum percolation, soil layer 2 to 3', 'soil', transform='log'),
    'glaccmlt': ParameterInfo(2.0, 12.0, 'mm/°C/day', 'Glacier ice degree-day melt coefficient', 'glacier'),
    'glacttmp': ParameterInfo(-2.0, 2.0, '°C', 'Glacier ice melt threshold temperature', 'glacier'),
    'glaccmrad': ParameterInfo(0.0, 1.5, 'mm/(MJ/m2)', 'Glacier radiation melt coefficient (snowmeltmodel 2)', 'glacier'),
}

#: Tier A -- the catalogue names composing HYPE's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
BOUND_SET: List[str] = [
    'ttmp',
    'cmlt',
    'ttpi',
    'cmrefr',
    'sdnsnew',
    'snowdensdt',
    'fsceff',
    'cevp',
    'lp',
    'epotdist',
    'fepotsnow',
    'ttrig',
    'treda',
    'tredb',
    'rrcs1',
    'rrcs2',
    'rrcs3',
    'wcwp',
    'wcfc',
    'wcep',
    'srrcs',
    'bfroznsoil',
    'logsatmp',
    'bcosby',
    'sfrost',
    'rcgrw',
    'deepperc',
    'deepmem',
    'surfmem',
    'depthrel',
    'rivvel',
    'damp',
    'qmean',
    'ilratk',
    'ilratp',
    'illdepth',
    'cmrad',
    'fscdist0',
    'pcusnow',
    'pcurain',
    'pcelevadd',
    'mperc1',
    'mperc2',
    'glaccmlt',
    'glacttmp',
    'glaccmrad',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute HYPE's bounds to the central catalogue."""
    register_model_bounds(
        'HYPE',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
