# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for the HBV decision-analysis skip marker.

Co-author SH read the previous decision-analysis log for HBV as a
bug:

    No decision analyzer registered for model: HBV.
    Available analyzers: ['SUMMA']

SH's interpretation was "HBV isn't wired up"; the actual meaning is
"HBV is a fixed-structure conceptual model with no physics decisions
to analyse — this step is not applicable". The step still reported
✓ Complete, reinforcing the misread.

These tests pin that AnalysisManager.run_decision_analysis now:

1. Records an explicit ``skipped=True`` entry for models without a
   registered decision analyzer, naming "no decision structure by
   design" as the reason.
2. Emits a summary log naming the skipped models so a reviewer
   skimming the log can tell analysed-vs-skipped at a glance.
3. SUMMA (which does have a registered analyzer) still runs
   normally and is recorded with ``skipped`` unset.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_manager(models: str):
    """Build a bare AnalysisManager bypassing its real __init__ so
    the test doesn't depend on all of SYMFLUENCE's config
    infrastructure."""
    from symfluence.evaluation.analysis_manager import AnalysisManager

    cfg = MagicMock()
    cfg.model.hydrological_model = models
    cfg.analysis.run_decision_analysis = True

    mgr = AnalysisManager.__new__(AnalysisManager)
    mgr.config = cfg
    mgr.logger = logging.getLogger("test_decision_skip")
    mgr.reporting_manager = MagicMock()
    mgr._get_config_value = lambda getter, default=None, **_: (
        getter() if callable(getter) else default
    )
    mgr._import_model_analyzers = lambda: None
    return mgr


def test_hbv_is_marked_skipped_with_design_reason(caplog):
    """HBV must be recorded as skipped-by-design, not absent or
    failed, and the log must say "no decision structure by design"
    so a reader doesn't mistake it for a wiring bug."""
    mgr = _make_manager("HBV")

    with patch(
        "symfluence.evaluation.analysis_manager.R.decision_analyzers"
    ) as registry:
        registry.get.return_value = None  # HBV has no analyzer
        registry.keys.return_value = ["SUMMA"]
        caplog.set_level(logging.INFO)
        result = mgr.run_decision_analysis()

    assert result is not None, "skip markers must be returned, not swallowed"
    assert "HBV" in result
    assert result["HBV"].get("skipped") is True
    assert "no decision structure by design" in result["HBV"].get("skipped_reason", "")

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "HBV" in log_text
    assert "skipping decision analysis" in log_text
    assert "no decision structure by design" in log_text


def test_summary_log_names_analysed_and_skipped(caplog):
    """The per-run summary log must name both the analysed and the
    skipped models so reviewers skimming the log can tell at a
    glance what happened — no need to re-read per-model lines."""
    mgr = _make_manager("SUMMA,HBV,GR4J")

    summa_analyzer_cls = MagicMock()
    summa_analyzer_cls.return_value.run_full_analysis.return_value = (
        "/tmp/fake_results.csv",
        {"KGE": {"score": 0.87}},
    )

    def fake_get(name):
        return summa_analyzer_cls if name == "SUMMA" else None

    with patch(
        "symfluence.evaluation.analysis_manager.R.decision_analyzers"
    ) as registry:
        registry.get.side_effect = fake_get
        registry.keys.return_value = ["SUMMA"]
        caplog.set_level(logging.INFO)
        result = mgr.run_decision_analysis()

    assert result is not None
    assert result["SUMMA"].get("skipped") is not True
    assert result["HBV"].get("skipped") is True
    assert result["GR4J"].get("skipped") is True

    # The summary line should be unambiguous about what ran vs skipped.
    summary_lines = [
        r.getMessage() for r in caplog.records
        if "Decision analysis summary" in r.getMessage()
    ]
    assert summary_lines, "summary line must be emitted"
    line = summary_lines[0]
    assert "1/3 models analysed" in line
    assert "HBV" in line and "GR4J" in line
    assert "skipped by design" in line
