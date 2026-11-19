# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility exports for the core model evaluator contracts."""
from __future__ import annotations

from symfluence.core.modeling.evaluators import (
    BasinETEvaluator,
    BasinSCAEvaluator,
    BasinSWEEvaluator,
    ETEvaluator,
    GlacierMassBalanceEvaluator,
    GroundwaterEvaluator,
    ModelEvaluator,
    SnowEvaluator,
    SoilMoistureEvaluator,
    StreamflowEvaluator,
    TWSEvaluator,
)

__all__ = [
    "ETEvaluator", "GroundwaterEvaluator", "ModelEvaluator", "SnowEvaluator",
    "SoilMoistureEvaluator", "StreamflowEvaluator", "TWSEvaluator",
    "GlacierMassBalanceEvaluator", "BasinSWEEvaluator", "BasinSCAEvaluator", "BasinETEvaluator",
]
