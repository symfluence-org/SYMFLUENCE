# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to
``symfluence.core.calibration.parameters.parameter_bounds_registry``.

Until now this path held a full 1185-line *copy* of the bounds catalogue rather
than a shim, and the copy predated #368. Consumers importing from here therefore
silently got pre-#368 values:

* ``get_fuse_bounds()`` served Snow-17's ``MBASE``/``MFMAX``/``MFMIN`` melt
  bounds to FUSE (the copy lacked the ``fuse_``-namespaced entries), i.e. the
  exact collision #368 fixed.
* ``get_ngen_bounds()`` / ``get_ngen_cfe_bounds()`` served ``soil_depth``
  ``2..15`` instead of the ``1..5`` chosen to avoid a CFE segfault.

Collapsing the copy into a real shim is therefore an intended behaviour change:
external consumers of those three getters now receive the corrected values.
Every other name this module exported is unchanged, and the canonical module is
a strict superset of the copy's public surface (it additionally exports the
``register_model_bounds`` / ``get_model_bounds`` seam).
"""
from __future__ import annotations

import warnings as _warnings

import symfluence.core.calibration.parameters.parameter_bounds_registry as _impl
from symfluence.core.calibration.parameters.parameter_bounds_registry import *  # noqa: F401,F403

_warnings.warn(
    "symfluence.optimization.core.parameter_bounds_registry is deprecated; import from symfluence.core.calibration.parameters.parameter_bounds_registry instead. "
    "This back-compat shim will be removed before SYMFLUENCE 1.0 (see docs/adr/0011-retire-unused-back-compat-shims.md).",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return dir(_impl)
