# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.geometry_utils``.

Promoted to core as part of the geospatial contract surface: model
preprocessors mix in these utilities, so they must be importable from core
alone (models may not depend on the geospatial capability package).
"""
from __future__ import annotations

import warnings as _warnings

import symfluence.core.geometry_utils as _impl
from symfluence.core.geometry_utils import *  # noqa: F401,F403

_warnings.warn(
    "symfluence.geospatial.geometry_utils is deprecated; import from symfluence.core.geometry_utils instead. "
    "This back-compat shim will be removed before SYMFLUENCE 1.0 (see docs/adr/0011-retire-unused-back-compat-shims.md).",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
