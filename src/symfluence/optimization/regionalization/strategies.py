# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the core calibration implementation."""
from __future__ import annotations

import warnings as _warnings

import symfluence.core.calibration.regionalization.strategies as _impl
from symfluence.core.calibration.regionalization.strategies import *  # noqa: F401,F403

_warnings.warn(
    "symfluence.optimization.regionalization.strategies is deprecated; import from symfluence.core.calibration.regionalization.strategies instead. "
    "This back-compat shim will be removed before SYMFLUENCE 1.0 (see docs/adr/0011-retire-unused-back-compat-shims.md).",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_impl, name)
