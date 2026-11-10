# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Wrapper around SYMFLUENCE core for workflow execution from the TUI.

Provides config loading, step enumeration, status queries, and background
execution support.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowService:
    """High-level interface for loading configs and running workflows."""

    def __init__(self):
        self._sf = None
        self._config_path: Optional[Path] = None
        self._last_error: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._sf is not None

    @property
    def config_path(self) -> Optional[Path]:
        return self._config_path

    @property
    def last_error(self) -> str:
        return self._last_error

    def load_config(self, config_path: str) -> bool:
        """Load a SYMFLUENCE config and initialize the system.

        Returns True on success, False on failure.
        """
        try:
            from symfluence import SYMFLUENCE
            self._config_path = Path(config_path)
            self._sf = SYMFLUENCE(config_input=self._config_path)
            self._last_error = ""
            return True
        except Exception as exc:  # noqa: BLE001 — UI resilience
            self._last_error = str(exc)
            logger.error("Failed to load config: %s", exc)
            self._sf = None
            return False

    def get_status(self) -> Dict[str, Any]:
        """Query workflow step completion status."""
        if not self._sf:
            return {}
        try:
            return self._sf.get_workflow_status()
        except Exception as exc:  # noqa: BLE001 — UI resilience
            self._last_error = str(exc)
            logger.error("Failed to get status: %s", exc)
            return {}

    def get_step_names(self) -> List[str]:
        """Return list of workflow step CLI names."""
        if not self._sf:
            return []
        try:
            steps = self._sf.workflow_orchestrator.define_workflow_steps()
            return [s.cli_name for s in steps]
        except Exception:  # noqa: BLE001 — UI resilience
            return []

    def get_domain_name(self) -> str:
        """Return the configured domain name."""
        if not self._sf:
            return ""
        try:
            return self._sf.config.domain.name
        except Exception:  # noqa: BLE001 — UI resilience
            return ""

    def get_experiment_id(self) -> str:
        """Return the configured experiment ID."""
        if not self._sf:
            return ""
        try:
            return self._sf.config.domain.experiment_id
        except Exception:  # noqa: BLE001 — UI resilience
            return ""

    def run_workflow(self, force_rerun: bool = False) -> None:
        """Run the full workflow (blocking). Call from a worker thread."""
        if not self._sf:
            raise RuntimeError("No config loaded")
        # None (not False) lets the config FORCE_RUN_ALL_STEPS decide.
        self._sf.run_workflow(force_run=True if force_rerun else None)

    def run_steps(self, step_names: List[str]) -> None:
        """Run specific workflow steps (blocking). Call from a worker thread."""
        if not self._sf:
            raise RuntimeError("No config loaded")
        self._sf.run_individual_steps(step_names)

    def invalidate(self) -> None:
        """Force re-creation of the SYMFLUENCE instance on next load."""
        self._sf = None
        self._config_path = None
