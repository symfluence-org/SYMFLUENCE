# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the promoted NetCDF encoding helpers."""
from __future__ import annotations

import warnings as _warnings

from symfluence.core.modeling.netcdf_utils import create_minimal_encoding, create_netcdf_encoding

_warnings.warn(
    "symfluence.data.utils.netcdf_utils is deprecated; import from symfluence.core.modeling.netcdf_utils instead. "
    "This back-compat shim will be removed before SYMFLUENCE 1.0 (see docs/adr/0011-retire-unused-back-compat-shims.md).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["create_minimal_encoding", "create_netcdf_encoding"]
