# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parallel Processing Module

Provides infrastructure for parallel execution of optimization tasks,
including HPC-specific utilities for scratch storage management.
"""

from .config_updater import ConfigurationUpdater
from .directory_manager import DirectoryManager
from .execution_strategies import (
    ExecutionStrategy,
    MPIExecutionStrategy,
    PersistentMPIExecutionStrategy,
    ProcessPoolExecutionStrategy,
    SequentialExecutionStrategy,
)
from .local_scratch_manager import LocalScratchManager
from .task_distributor import TaskDistributor
from .worker_environment import WorkerEnvironmentConfig

__all__ = [
    'DirectoryManager',
    'ConfigurationUpdater',
    'TaskDistributor',
    'WorkerEnvironmentConfig',
    'LocalScratchManager',
    'ExecutionStrategy',
    'SequentialExecutionStrategy',
    'ProcessPoolExecutionStrategy',
    'MPIExecutionStrategy',
    'PersistentMPIExecutionStrategy',
]
