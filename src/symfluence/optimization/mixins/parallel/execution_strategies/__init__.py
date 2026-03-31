# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Execution Strategies

Different strategies for parallel task execution.
"""

from .base import ExecutionStrategy
from .mpi import MPIExecutionStrategy
from .mpi_persistent import PersistentMPIExecutionStrategy
from .process_pool import ProcessPoolExecutionStrategy
from .sequential import SequentialExecutionStrategy

__all__ = [
    'ExecutionStrategy',
    'SequentialExecutionStrategy',
    'ProcessPoolExecutionStrategy',
    'MPIExecutionStrategy',
    'PersistentMPIExecutionStrategy',
]
