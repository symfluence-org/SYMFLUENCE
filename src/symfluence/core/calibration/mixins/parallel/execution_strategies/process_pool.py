# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Process Pool Execution Strategy

Executes tasks using Python's ProcessPoolExecutor.
"""
from __future__ import annotations

import concurrent.futures
import logging
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List

from .base import ExecutionStrategy


class ProcessPoolExecutionStrategy(ExecutionStrategy):
    """
    Process pool execution strategy.

    Uses ProcessPoolExecutor to run tasks in parallel while
    preserving result order.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize process pool execution strategy.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Strategy identifier for logging and selection."""
        return "process_pool"

    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks using ProcessPoolExecutor.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results in the same order as input tasks
        """
        if max_workers == 1 or len(tasks) == 1:
            # Fall back to sequential for single worker/task
            return [worker_func(task) for task in tasks]

        results: List[Dict[str, Any]] = [{} for _ in tasks]

        try:
            # Use ProcessPoolExecutor.map to preserve order
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                try:
                    results = []
                    for wave in self._waves_without_shared_dirs(tasks):
                        results.extend(executor.map(worker_func, wave))
                except BrokenExecutor as e:
                    self.logger.warning(f"Process pool was broken: {str(e)}. Falling back to sequential execution.")
                    # Fallback to sequential execution
                    results = [worker_func(task) for task in tasks]
        except (ValueError, RuntimeError, concurrent.futures.TimeoutError) as e:
            self.logger.error(f"Error in process pool execution: {str(e)}. Falling back to sequential execution.")
            # Fallback to sequential execution for any other errors
            results = [worker_func(task) for task in tasks]

        return results

    @staticmethod
    def _waves_without_shared_dirs(tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split tasks into consecutive waves in which no two tasks share a ``proc_id``.

        Tasks with the same ``proc_id`` use the same per-process run directory
        (settings, forcing, output). When a batch holds more tasks than workers,
        ``executor.map`` would hand a freed worker the next task even though the
        task occupying that directory is still running.
        """
        waves: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        seen: set = set()
        for task in tasks:
            proc_id = task.get('proc_id') if isinstance(task, dict) else None
            if proc_id is not None and proc_id in seen:
                waves.append(current)
                current, seen = [], set()
            current.append(task)
            if proc_id is not None:
                seen.add(proc_id)
        if current:
            waves.append(current)
        return waves
