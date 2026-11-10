# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Randomized walks through the workflow over many "workflow trees".

Rather than asserting one hand-picked workflow run, these property-based tests
generate many *trees* — a random registered model plus a random config mutation —
and walk the orchestrator/stage-marker machinery for each, asserting the
invariants that must hold for every tree. Managers are mocked, so no model
binaries, downloads, or real data are touched; the orchestrator's planning,
step-list construction, and config-hash skip/resume logic are what get exercised.

Invariants under test:
  * the step list is config-invariant and in canonical order (conditionals are
    applied at execution time, not by dropping steps from the plan);
  * every step exposes a callable func and a check_func that returns a bool
    without raising;
  * a full mocked walk writes a stage marker for every stage;
  * changing a config section invalidates *exactly* the stages that depend on
    that section — the property the resume/skip system relies on;
  * a random set of "already-completed" stages reports as current, and a section
    change flips exactly the dependent ones stale (a random partial walk).

Hypothesis is an optional dependency; the module skips cleanly without it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import symfluence  # noqa: F401,E402 — triggers bootstrap / registry population
from symfluence.core.config.models import SymfluenceConfig  # noqa: E402
from symfluence.core.registries import R  # noqa: E402
from symfluence.core.stage_marker import (  # noqa: E402
    STAGE_CONFIG_SECTIONS,
    compute_config_hash,
)
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator  # noqa: E402
from symfluence.workflow_steps import WORKFLOW_STEP_ITEMS  # noqa: E402

pytestmark = [pytest.mark.unit]

CANONICAL_ORDER = [name for name, _ in WORKFLOW_STEP_ITEMS]
ALL_STAGES = sorted(STAGE_CONFIG_SECTIONS)
ALL_SECTIONS = sorted({s for secs in STAGE_CONFIG_SECTIONS.values() for s in secs})


def _build_config(domain: str, model: str, data_dir: str = tempfile.gettempdir(), **extra) -> SymfluenceConfig:
    return SymfluenceConfig.from_minimal(
        domain_name=domain,
        model=model,
        time_start="2010-01-01 00:00",
        time_end="2010-12-31 23:00",
        SYMFLUENCE_DATA_DIR=data_dir,
        **extra,
    )


def _safe_models() -> list[str]:
    """Registered runner models that from_minimal can construct (registry-driven)."""
    ok: list[str] = []
    for name in R.runners.keys():
        try:
            _build_config("probe", name)
            ok.append(name)
        except Exception:  # noqa: BLE001 — only keep models that build cleanly
            continue
    return ok


SAFE_MODELS = _safe_models()

# Config mutations, each intended to perturb a different section. The tests do
# not assume which section each touches — they derive the changed set empirically
# and assert the resulting stale-stage partition, so this list only needs variety.
MUTATIONS: list[dict] = [
    {"FORCING_DATASET": "RDRS"},
    {"OPTIMIZATION_METHODS": ["iteration"]},
    {"DOMAIN_DEFINITION_METHOD": "delineate"},
    {"SUB_GRID_DISCRETIZATION": "elevation"},
    {"domain": "a_different_domain"},
]


def _mock_managers() -> dict:
    return {k: MagicMock() for k in ("project", "domain", "data", "model", "analysis", "optimization")}


def _changed_sections(a: SymfluenceConfig, b: SymfluenceConfig) -> set[str]:
    return {s for s in ALL_SECTIONS if compute_config_hash(a, [s]) != compute_config_hash(b, [s])}


def _stale_stages(a: SymfluenceConfig, b: SymfluenceConfig) -> set[str]:
    return {
        stage
        for stage in ALL_STAGES
        if compute_config_hash(a, STAGE_CONFIG_SECTIONS[stage])
        != compute_config_hash(b, STAGE_CONFIG_SECTIONS[stage])
    }


def test_safe_models_population_is_realistic():
    """The registry-driven model pool is non-trivial — guards a broken bootstrap."""
    assert len(SAFE_MODELS) >= 10, SAFE_MODELS


# ----------------------------------------------------------------------
# Tree shape: the step plan is config-invariant and canonically ordered
# ----------------------------------------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(model=st.sampled_from(SAFE_MODELS or ["SUMMA"]), mutation=st.sampled_from(MUTATIONS))
def test_step_plan_is_config_invariant_and_ordered(model, mutation):
    extra = {k: v for k, v in mutation.items() if k != "domain"}
    orch = WorkflowOrchestrator(_mock_managers(), _build_config("tree", model, **extra), MagicMock())

    steps = orch.define_workflow_steps()

    assert [s.cli_name for s in steps] == CANONICAL_ORDER
    for step in steps:
        assert callable(step.func), f"{step.cli_name}.func not callable"
        verdict = step.check_func()
        assert isinstance(verdict, bool), f"{step.cli_name}.check_func returned {type(verdict)}"


# ----------------------------------------------------------------------
# Full mocked walk: every stage gets a marker, idempotently
# ----------------------------------------------------------------------


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(model=st.sampled_from(SAFE_MODELS or ["SUMMA"]))
def test_full_walk_writes_a_marker_for_every_stage(model):
    with tempfile.TemporaryDirectory() as td:
        config = _build_config("walk", model, data_dir=td)
        orch = WorkflowOrchestrator(_mock_managers(), config, MagicMock())

        orch.run_workflow()

        marker_dir = Path(orch.project_dir) / ".symfluence" / "stage_markers"
        # Experiment-scoped stages write per-experiment markers in a subdirectory.
        written = {p.stem for p in marker_dir.rglob("*.json")}
        missing = set(ALL_STAGES) - written
        assert not missing, f"stages with no marker after a full walk: {sorted(missing)}"

        # Idempotent: a second walk neither errors nor drops markers.
        orch.run_workflow()
        assert {p.stem for p in marker_dir.rglob("*.json")} >= set(ALL_STAGES)


# ----------------------------------------------------------------------
# Core property: a config-section change invalidates exactly its dependents
# ----------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    model=st.sampled_from(SAFE_MODELS or ["SUMMA"]),
    mutation=st.sampled_from(MUTATIONS),
)
def test_section_change_invalidates_exactly_dependent_stages(model, mutation):
    base = _build_config("inv", model)
    if "domain" in mutation:
        other = _build_config(mutation["domain"], model)
    else:
        other = _build_config("inv", model, **mutation)

    changed = _changed_sections(base, other)
    hypothesis.assume(changed)  # skip no-op mutations for this model

    stale = _stale_stages(base, other)
    for stage in ALL_STAGES:
        depends = bool(set(STAGE_CONFIG_SECTIONS[stage]) & changed)
        assert (stage in stale) == depends, (
            f"stage {stage!r} sections={STAGE_CONFIG_SECTIONS[stage]} changed={sorted(changed)}: "
            f"stale={stage in stale} but depends={depends}"
        )


# ----------------------------------------------------------------------
# Random partial walk: completed stages are current; a change flips dependents
# ----------------------------------------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    model=st.sampled_from(SAFE_MODELS or ["SUMMA"]),
    completed=st.lists(st.sampled_from(ALL_STAGES), unique=True),
    mutation=st.sampled_from(MUTATIONS),
)
def test_random_partial_walk_marker_currency(model, completed, mutation):
    from symfluence.core.stage_marker import is_stage_current, write_marker

    with tempfile.TemporaryDirectory() as td:
        base = _build_config("partial", model, data_dir=td)
        project_dir = Path(td) / "domain_partial"

        # Mark a random subset of stages complete with their current hash.
        for stage in completed:
            h = compute_config_hash(base, STAGE_CONFIG_SECTIONS[stage])
            write_marker(project_dir, stage, h)

        # Each completed stage reads as current against the unchanged config.
        for stage in completed:
            h = compute_config_hash(base, STAGE_CONFIG_SECTIONS[stage])
            assert is_stage_current(project_dir, stage, h), f"{stage} should be current"

        # After a section change, exactly the dependent completed stages go stale.
        if "domain" in mutation:
            other = _build_config(mutation["domain"], model, data_dir=td)
        else:
            other = _build_config("partial", model, data_dir=td, **mutation)
        changed = _changed_sections(base, other)

        for stage in completed:
            new_hash = compute_config_hash(other, STAGE_CONFIG_SECTIONS[stage])
            depends = bool(set(STAGE_CONFIG_SECTIONS[stage]) & changed)
            still_current = is_stage_current(project_dir, stage, new_hash)
            assert still_current == (not depends), (
                f"{stage}: depends={depends} but still_current={still_current}"
            )
