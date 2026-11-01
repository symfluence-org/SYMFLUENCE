# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.modeling.mixins`` (adapter contract tier)."""
from __future__ import annotations

import warnings as _warnings

import symfluence.core.modeling.mixins as _impl
from symfluence.core.modeling.mixins import *  # noqa: F401,F403

_warnings.warn(
    "symfluence.models.mixins is deprecated; import from symfluence.core.modeling.mixins instead. "
    "This back-compat shim will be removed before SYMFLUENCE 1.0 (see docs/adr/0011-retire-unused-back-compat-shims.md).",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
