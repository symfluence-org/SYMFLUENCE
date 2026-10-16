# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GR4J + CemaNeige calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing GR's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters GR calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  GR resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Every GR parameter is solo, so the whole set lives here.

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

#: Tier B -- definitions only GR resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'X1': ParameterInfo(1.0, 5000.0, 'mm', 'Production store capacity', 'soil'),
    'X2': ParameterInfo(-10.0, 10.0, 'mm/day', 'Groundwater exchange coefficient', 'baseflow'),
    'X3': ParameterInfo(1.0, 500.0, 'mm', 'Routing store capacity', 'soil'),
    'X4': ParameterInfo(0.5, 5.0, 'days', 'Unit hydrograph time constant', 'routing'),
    'CTG': ParameterInfo(0.0, 1.0, '-', 'Snow process parameter', 'snow'),
    'Kf': ParameterInfo(0.0, 20.0, 'mm/°C/day', 'Melt factor', 'snow'),
    'Gratio': ParameterInfo(0.01, 200.0, '-', 'Thermal coefficient for snow pack thermal state', 'snow'),
    'Albedo_diff': ParameterInfo(0.001, 1.0, '-', 'Albedo diffusion coefficient', 'snow'),
    'PCORR': ParameterInfo(0.5, 2.0, '-', 'Total precipitation multiplier', 'forcing'),
    'TCORR': ParameterInfo(-3.0, 3.0, 'K', 'Additive temperature offset', 'forcing'),
    'ICE_DDF': ParameterInfo(1.0, 20.0, 'mm/°C/day', 'Glacier ice degree-day factor', 'glacier'),
}

#: Tier A -- the catalogue names composing GR's bound set, in served order.
BOUND_SET: List[str] = [
    'X1',
    'X2',
    'X3',
    'X4',
    'CTG',
    'Kf',
    'Gratio',
    'Albedo_diff',
    'PCORR',
    'TCORR',
    'ICE_DDF',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute GR's bounds to the central catalogue."""
    register_model_bounds(
        'GR',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
