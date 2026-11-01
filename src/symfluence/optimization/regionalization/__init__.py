# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model-Agnostic Parameter Regionalization Framework.

Provides strategies for handling spatially distributed parameters during
calibration.  The framework is shared across all models (FUSE, SUMMA, HYPE,
NGEN, ...); each model supplies a thin adapter that provides:

1. Spatial unit discovery - how many units and their IDs
2. Parameter bounds - from the model's bounds registry
3. Default param-to-attribute mapping - param_config dict
4. Catchment/unit attributes - a DataFrame of physical descriptors
5. Config-file writer - how to write per-unit values to disk

Strategies:
- lumped - single parameter set broadcast to all units
- transfer_function - MPR-style param = a + b * attr_norm
- zones - group units into zones with shared parameters
- distributed - independent parameter per unit (needs regularization)
"""
from __future__ import annotations

from symfluence.core.calibration.regionalization.strategies import (
    DistributedRegionalization,
    LumpedRegionalization,
    ParameterRegionalization,
    RegionalizationFactory,
    TransferFunctionRegionalization,
    ZoneRegionalization,
    get_regionalization_info,
)
from symfluence.core.calibration.regionalization.transfer_functions import (
    ConstantTF,
    ExponentialTF,
    FlexiblePowerTF,
    LinearTF,
    PowerTF,
    TransferFunction,
)

__all__ = [
    "ParameterRegionalization",
    "LumpedRegionalization",
    "TransferFunctionRegionalization",
    "ZoneRegionalization",
    "DistributedRegionalization",
    "RegionalizationFactory",
    "get_regionalization_info",
    "TransferFunction",
    "LinearTF",
    "PowerTF",
    "ExponentialTF",
    "ConstantTF",
    "FlexiblePowerTF",
]
