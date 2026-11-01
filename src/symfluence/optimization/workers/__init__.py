# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Optimization Workers

Worker classes that handle the evaluation of parameter sets during optimization.
Each worker is responsible for:
- Applying parameters to model configuration files
- Running model simulations
- Calculating objective metrics from outputs

Available workers:
- BaseWorker: Abstract base class for all workers
- InMemoryModelWorker: Base class for in-memory model workers (jFUSE, cFUSE, HBV)
- WorkerTask: Data structure for worker inputs
- WorkerResult: Data structure for worker outputs

Model-specific workers are available via:
1. Direct import: from symfluence.models.{model}.calibration.worker import {Model}Worker
2. Registry pattern: OptimizerRegistry.get_worker('{MODEL}')

Note: Model-specific workers have been moved to their canonical locations under
symfluence.models/{model}/calibration/worker.py. Use the registry or direct
import from the canonical location.
"""
from __future__ import annotations

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerResult, WorkerTask
from symfluence.core.calibration.workers.inmemory_worker import InMemoryModelWorker


# Lazy import of worker classes from canonical locations to avoid circular dependencies
def __getattr__(name):
    """Lazy import of worker classes from canonical model locations."""
    # Map worker class names to their canonical module paths
    worker_mapping = {
        'FUSEWorker': 'symfluence.models.fuse.calibration.worker',
        'JFUSEWorker': 'symfluence.models.jfuse.calibration.worker',
        'CFUSEWorker': 'symfluence.models.cfuse.calibration.worker',
        'GRWorker': 'symfluence.models.gr.calibration.worker',
        'HBVWorker': 'jhbv.calibration.worker',
        'HYPEWorker': 'symfluence.models.hype.calibration.worker',
        'MESHWorker': 'symfluence.models.mesh.calibration.worker',
        'NgenWorker': 'symfluence.models.ngen.calibration.worker',
        'RHESSysWorker': 'symfluence.models.rhessys.calibration.worker',
        'SUMMAWorker': 'symfluence.models.summa.calibration.worker',
        'GNNWorker': 'symfluence.models.gnn.calibration.worker',
        'LSTMWorker': 'symfluence.models.lstm.calibration.worker',
        'SacSmaWorker': 'jsacsma.calibration.worker',
        'XinanjiangWorker': 'jxaj.calibration.worker',
    }

    if name in worker_mapping:
        from importlib import import_module
        try:
            module = import_module(worker_mapping[name])
            return getattr(module, name)
        except ImportError as exc:
            # Surface the real cause (e.g. a missing optional dependency) rather
            # than collapsing every failure into a generic "not found".
            raise AttributeError(
                f"Worker '{name}' could not be imported from '{worker_mapping[name]}' "
                f"({type(exc).__name__}: {exc}). Ensure the model package and its "
                f"dependencies are installed."
            ) from exc
        except AttributeError as exc:
            raise AttributeError(
                f"Worker class '{name}' is not defined in '{worker_mapping[name]}'."
            ) from exc

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseWorker',
    'InMemoryModelWorker',
    'WorkerTask',
    'WorkerResult',
]
