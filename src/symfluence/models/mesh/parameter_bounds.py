# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MESH (CLASS + WATROUTE) calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing MESH's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters MESH calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  MESH resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for MESH: ``PWR``, ``R2N``.

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

#: Tier B -- definitions only MESH resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'KSAT': ParameterInfo(1.0, 500.0, 'mm/hr', 'Saturated hydraulic conductivity', 'soil', 'log'),
    'DRN': ParameterInfo(0.5, 5.0, '-', 'Drainage parameter', 'soil'),
    'SDEP': ParameterInfo(0.5, 1.5, 'm', 'Soil depth', 'soil'),
    'DD': ParameterInfo(1.0, 100.0, '-', 'CLASS drainage density (line 12)', 'soil'),
    'XSLP': ParameterInfo(0.01, 0.3, '-', 'Slope for overland flow', 'surface'),
    'XDRAINH': ParameterInfo(0.01, 1.0, '-', 'Horizontal drainage coefficient', 'soil'),
    'MANN_CLASS': ParameterInfo(0.01, 0.5, '-', 'Manning coefficient for overland flow', 'surface'),
    'LAMX': ParameterInfo(0.3, 6.0, 'm²/m²', 'Maximum LAI for primary vegetation class', 'et'),
    'LAMN': ParameterInfo(0.1, 1.5, 'm²/m²', 'Minimum LAI for primary vegetation class (seasonal ET cycle)', 'et'),
    'ROOT': ParameterInfo(0.1, 2.0, 'm', 'Root depth for primary vegetation class', 'et'),
    'CMAS': ParameterInfo(1.0, 10.0, 'kg/m²', 'Annual maximum canopy mass (controls interception)', 'et'),
    'RSMIN': ParameterInfo(100.0, 800.0, 's/m', 'Minimum stomatal resistance (controls max transpiration rate)', 'et'),
    'QA50': ParameterInfo(10.0, 100.0, 'Pa', 'Reference VPD for half-maximum stomatal conductance', 'et'),
    'VPDA': ParameterInfo(0.3, 1.5, '-', 'VPD slope parameter for stomatal conductance', 'et'),
    'PSGA': ParameterInfo(0.3, 2.0, '-', 'Soil moisture stress parameter A for stomatal conductance', 'et'),
    'ZSNL': ParameterInfo(0.001, 0.1, 'm', 'Limiting snow depth', 'snow'),
    'ZPLG': ParameterInfo(0.0, 0.5, 'm', 'Maximum ponding depth (ground)', 'soil'),
    'ZPLS': ParameterInfo(0.0, 0.5, 'm', 'Maximum ponding depth (snow)', 'snow'),
    'FRZTH': ParameterInfo(0.0, 5.0, 'm', 'Frozen soil infiltration threshold', 'soil'),
    'MANN': ParameterInfo(0.01, 0.3, '-', 'Manning roughness coefficient', 'routing'),
    'R1N': ParameterInfo(0.0, 2.0, '-', 'River routing parameter', 'routing'),
    'FLZ': ParameterInfo(0.001, 0.1, '-', 'Baseflow recession coefficient', 'baseflow', 'log'),
    'RCHARG': ParameterInfo(0.0, 1.0, '-', 'Recharge fraction to groundwater', 'baseflow'),
    'DRAINFRAC': ParameterInfo(0.0, 1.0, '-', 'Drainage fraction', 'soil'),
    'BASEFLW': ParameterInfo(0.001, 0.1, 'm/day', 'Baseflow rate', 'baseflow'),
    'WF_R2': ParameterInfo(0.1, 0.5, '-', 'Channel roughness coefficient for WATFLOOD routing', 'routing'),
    'DTMINUSR': ParameterInfo(60.0, 600.0, 's', 'Routing time-step', 'routing'),
    'SAND': ParameterInfo(20.0, 95.0, '%', 'Sand fraction of land GRU soil layers', 'soil'),
    'CLAY': ParameterInfo(1.0, 30.0, '%', 'Clay fraction of land GRU soil layers', 'soil'),
    'PRECIP_MULT': ParameterInfo(0.8, 1.6, '-', 'Total precipitation multiplier on run forcing', 'forcing'),
    'TEMP_OFFSET': ParameterInfo(-2.0, 2.0, 'K', 'Air temperature offset on run forcing', 'forcing'),
}

#: Tier A -- the catalogue names composing MESH's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
BOUND_SET: List[str] = [
    'KSAT',
    'DRN',
    'SDEP',
    'DD',
    'XSLP',
    'XDRAINH',
    'MANN_CLASS',
    'LAMX',
    'LAMN',
    'ROOT',
    'CMAS',
    'RSMIN',
    'QA50',
    'VPDA',
    'PSGA',
    'ZSNL',
    'ZPLG',
    'ZPLS',
    'FRZTH',
    'MANN',
    'R2N',
    'R1N',
    'FLZ',
    'PWR',
    'RCHARG',
    'DRAINFRAC',
    'BASEFLW',
    'WF_R2',
    'DTMINUSR',
    'SAND',
    'CLAY',
    'PRECIP_MULT',
    'TEMP_OFFSET',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute MESH's bounds to the central catalogue."""
    register_model_bounds(
        'MESH',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
