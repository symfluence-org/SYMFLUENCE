# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the top-level SYMFLUENCE system facade (review item 17).

Scoped to construction wiring and delegation to the workflow orchestrator (the
managers, logging, provenance, and orchestrator are mocked so no real workflow,
filesystem, or subprocess work runs). Deep step-execution behaviour is integration
territory and out of scope here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from symfluence import SYMFLUENCE
from symfluence.core.config.models import SymfluenceConfig
from symfluence.project import system


@pytest.fixture
def facade():
    cfg = SymfluenceConfig.from_minimal(
        domain_name="test_basin", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
    )
    with patch.object(system, "LoggingManager"), \
         patch.object(system, "LazyManagerDict"), \
         patch.object(system, "WorkflowOrchestrator"), \
         patch.object(system, "capture_provenance", return_value=None), \
         patch.object(system, "finalize_provenance"):
        sym = SYMFLUENCE(cfg)
        yield sym  # keep patches active for the test body


def test_construction_wires_components(facade):
    assert isinstance(facade.typed_config, SymfluenceConfig)
    assert isinstance(facade.config, dict)  # flattened backward-compat view
    assert facade.workflow_orchestrator is not None
    assert facade.managers is not None
    assert facade.provenance is None  # capture_provenance patched to None


def test_run_workflow_delegates_to_orchestrator(facade):
    facade.workflow_orchestrator.get_workflow_status.return_value = {
        "step_details": [], "total_steps": 0, "completed_steps": 0,
    }
    facade.run_workflow(force_run=True)
    facade.workflow_orchestrator.run_workflow.assert_called_once_with(force_run=True)


def test_run_workflow_reraises_on_failure(facade):
    facade.workflow_orchestrator.run_workflow.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        facade.run_workflow(force_run=True)
    # the run-summary is still written in the finally block
    facade.logging_manager.create_run_summary.assert_called_once()


def test_run_individual_steps_delegates(facade):
    facade.workflow_orchestrator.run_individual_steps.return_value = [
        {"success": True, "cli": "setup", "fn": "setup_project"},
    ]
    facade.run_individual_steps(["setup_project"])
    facade.workflow_orchestrator.run_individual_steps.assert_called_once_with(
        ["setup_project"], False
    )


def test_get_workflow_status_delegates(facade):
    sentinel = {"total_steps": 3, "completed_steps": 1, "step_details": []}
    facade.workflow_orchestrator.get_workflow_status.return_value = sentinel
    assert facade.get_workflow_status() == sentinel


def test_run_with_failed_steps_is_not_recorded_as_completed(facade):
    """STOP_ON_ERROR=False: a run with failed steps is summarised as partial/failed."""
    orch = facade.workflow_orchestrator
    orch.last_step_results = [
        {"cli_name": "define_domain", "status": "completed"},
        {"cli_name": "run_model", "status": "failed", "error": "boom"},
    ]
    orch.failed_step_names.return_value = ["run_model"]
    facade.run_workflow(force_run=False)
    kwargs = facade.logging_manager.create_run_summary.call_args.kwargs
    assert kwargs["status"] == "partial"
    assert kwargs["errors"] == [{"where": "run_model", "error": "boom"}]
    assert facade.get_failed_steps() == ["run_model"]


def test_run_without_cli_flag_uses_config_force_run(facade):
    orch = facade.workflow_orchestrator
    orch.last_step_results = []
    orch.failed_step_names.return_value = []
    facade.typed_config = facade.typed_config.model_copy(
        update={"system": facade.typed_config.system.model_copy(update={"force_run_all_steps": True})}
    )
    facade.run_workflow(force_run=None)
    orch.run_workflow.assert_called_once_with(force_run=True)
