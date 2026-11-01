# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Optimization module for SYMFLUENCE.

This module provides optimization infrastructure for hydrological model calibration,
including support for multiple models (SUMMA, FUSE, NGEN) and various optimization
algorithms (DDS, PSO, SCE-UA, DE, ADAM, LBFGS).

Main Components:
    - OptimizerRegistry: Central registry for model-specific optimizers and workers
    - BaseModelOptimizer: Abstract base class for model-specific optimizers
    - BaseWorker: Abstract base class for parallel worker implementations
    - ObjectiveRegistry: Registry for objective functions and metrics

Model Optimizers:
    - SUMMAModelOptimizer: Optimizer for SUMMA model
    - FUSEModelOptimizer: Optimizer for FUSE model
    - NgenModelOptimizer: Optimizer for NextGen model

Usage:
    >>> from symfluence.core.registries import R
    >>> optimizer_cls = R.optimizers.get('FUSE')
    >>> optimizer = optimizer_cls(config, logger)
    >>> results = optimizer.run_pso()
"""

# Trigger objective registration
from __future__ import annotations

from typing import TYPE_CHECKING

from . import objectives
from .objectives import ObjectiveRegistry

try:
    from .objectives import multivariate
except ImportError:
    pass

# The in-tree model optimizers register with R.optimizers/R.workers/
# R.parameter_managers when `.model_optimizers` is imported. That import-all
# costs ~0.6 s, so it is deferred: the registry bootstrap installs seeders
# that trigger it on first registry access. The class re-exports below are
# likewise lazy (PEP 562) — plugin config adapters import this package at
# `import symfluence` and must not pay for the optimizer stack.
_LAZY_IMPORTS = {
    'BaseModelOptimizer': ('symfluence.core.calibration.optimizers.base_model_optimizer', 'BaseModelOptimizer'),
    'BaseWorker': ('symfluence.core.calibration.workers.base_worker', 'BaseWorker'),
    'WorkerResult': ('symfluence.core.calibration.workers.base_worker', 'WorkerResult'),
    'WorkerTask': ('symfluence.core.calibration.workers.base_worker', 'WorkerTask'),
    'EMA': ('.gradient', 'EMA'),
    'AdamW': ('.gradient', 'AdamW'),
    'CosineAnnealingWarmRestarts': ('.gradient', 'CosineAnnealingWarmRestarts'),
    'CosineDecay': ('.gradient', 'CosineDecay'),
}


def __getattr__(name: str):
    """Lazy import handler for optimization re-exports."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        value = getattr(import_module(module_path, package=__name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['ObjectiveRegistry', 'objectives'])


if TYPE_CHECKING:
    from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
    from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerResult, WorkerTask

    from .gradient import EMA, AdamW, CosineAnnealingWarmRestarts, CosineDecay

__all__ = [
    # Registries
    "ObjectiveRegistry",
    # Base classes
    "BaseModelOptimizer",
    "BaseWorker",
    "WorkerTask",
    "WorkerResult",
    # Gradient-based optimization utilities
    "AdamW",
    "CosineAnnealingWarmRestarts",
    "CosineDecay",
    "EMA",
]
