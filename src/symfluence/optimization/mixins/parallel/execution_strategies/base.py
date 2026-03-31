# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base Execution Strategy

Abstract base class for parallel execution strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class ExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies.

    Defines the interface for different parallel execution approaches
    (sequential, process pool, MPI, etc.).
    """

    @abstractmethod
    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results from task execution
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name for logging."""
        pass

    def startup(self) -> None:
        """Called once before the first execute() call.

        Override in subclasses that need one-time initialization
        (e.g., launching persistent worker processes).
        """

    def shutdown(self) -> None:
        """Called once after the last execute() call.

        Override in subclasses that need cleanup
        (e.g., terminating persistent worker processes).
        """

    @property
    def is_persistent(self) -> bool:
        """Whether this strategy keeps workers alive across execute() calls."""
        return False
