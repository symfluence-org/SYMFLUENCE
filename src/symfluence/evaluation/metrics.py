# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: this module moved to ``symfluence.core.metrics.metrics``.

Kept so external plugins and downstream code importing the old
``symfluence.evaluation.metrics`` path keep working. New code should import from ``symfluence.core.metrics.metrics``.
"""
from __future__ import annotations

import warnings as _warnings

import symfluence.core.metrics.metrics as _impl
from symfluence.core.metrics.metrics import *  # noqa: F401,F403

_warnings.warn(
    "symfluence.evaluation.metrics is deprecated; import from symfluence.core.metrics.metrics instead. "
    "This back-compat shim will be removed before SYMFLUENCE 1.0 (see docs/adr/0011-retire-unused-back-compat-shims.md).",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
