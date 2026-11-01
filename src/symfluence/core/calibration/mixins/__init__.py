# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Optimization Mixins

Reusable mixin classes that provide common functionality for optimizers:
- ParallelExecutionMixin: Parallel processing infrastructure
- ResultsTrackingMixin: Results persistence and tracking
- RetryExecutionMixin: Retry logic with exponential backoff
- GradientOptimizationMixin: ADAM/LBFGS gradient-based optimization
- SUMMAOptimizerMixin: SUMMA-specific functionality (extracted from legacy BaseOptimizer)
"""
from __future__ import annotations

from .gradient_optimization import GradientOptimizationMixin
from .parallel_execution import ParallelExecutionMixin
from .results_tracking import ResultsTrackingMixin
from .retry_execution import RetryExecutionMixin

# SUMMAOptimizerMixin deliberately does NOT live here: it wraps the SUMMA model
# package, which core must not depend on. Import it from
# ``symfluence.models.summa.calibration.optimizer_mixin``.

__all__ = [
    'ParallelExecutionMixin',
    'ResultsTrackingMixin',
    'RetryExecutionMixin',
    'GradientOptimizationMixin',
]
