# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for Sobol SA loud-fail on malformed inputs.

A co-author reported that the sensitivity_analysis workflow step
consistently reported '✓ Complete' with empty result panels. Root
cause was that when the calibration-results CSV contained stringified
parameter bounds (YAML round-trip artefact), SALib's sobol.analyze
raised ``ValueError("Bounds are not legal")`` — and that exception
was silently swallowed by a surrounding error-handler context with
``reraise=False``.

These tests pin two behaviours:

1. ``SensitivityAnalyzer`` never lets un-analysable columns crash the
   step: stringified/non-numeric columns and zero-variance (constant)
   columns are dropped at source by the parameter selector, so Sobol
   runs on the parameters that actually varied instead of failing the
   whole analysis.
2. ``AnalysisManager.run_sensitivity_analysis`` raises
   ``EvaluationError`` when no configured model produces results,
   so the workflow step is marked FAILED instead of silently
   'complete with no output'.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from symfluence.core.exceptions import EvaluationError
from symfluence.evaluation.sensitivity_analysis import SensitivityAnalyzer

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_analyzer():
    """Build an analyzer without running __init__.

    SensitivityAnalyzer's ConfigMixin base computes a project directory
    from config.system.data_dir at construction time. With a MagicMock
    config those attribute accesses return child MagicMocks whose repr
    (e.g. "<MagicMock name='mock.system.data_dir' ...>") contains '<'
    and '>'. POSIX Path() silently accepts that string; Windows rejects
    it with WinError 123. Bypassing __init__ avoids the path-construction
    code entirely — the bounds validation under test doesn't touch self
    attributes beyond self.logger.
    """
    analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
    analyzer.config = MagicMock()
    analyzer.logger = logging.getLogger("test_sa_loud_fail")
    analyzer.reporting_manager = MagicMock()
    return analyzer


def test_stringified_columns_are_filtered_not_crashed_on():
    """If the samples DataFrame has stringified / non-numeric
    columns (YAML round-trip artefact, ADAM's ``current_params``
    list, etc.), ``_parameter_columns_for_sa`` drops them at source.
    Sobol then runs cleanly on only the genuine numeric parameters.

    This used to be a raise-loudly assertion (bounds-are-not-legal
    with an actionable message), but the dtype-based filter
    introduced alongside the ADAM/L-BFGS fix makes the crash path
    unreachable for this case — stringified columns never reach the
    validator. The two remaining bad-data cases that still warrant
    a loud raise (numeric-but-degenerate bounds, numeric-but-constant
    parameter) are covered below.
    """
    samples = pd.DataFrame({
        "param_a": ["0.1", "0.5", "2.0"],  # stringified → filtered
        "param_b": [1.0, 2.0, 3.0],
        "RMSE": [0.1, 0.2, 0.3],
    })
    analyzer = _make_analyzer()
    # Must not raise — param_a is filtered out, SA runs on param_b only.
    result = analyzer.perform_sobol_analysis(samples, metric="RMSE")
    assert list(result.index) == ["param_b"], (
        f"stringified column leaked through filter; "
        f"parameters SA saw: {list(result.index)}"
    )


def test_constant_parameter_is_dropped_not_raised():
    """A parameter column with zero variance (min == max) carries no
    attributable sensitivity. Rather than failing the whole analysis
    (which discards the perfectly good indices for every other
    parameter), the constant column is dropped and Sobol runs on the
    parameters that actually varied. This keeps a run's results usable
    even when the optimiser pinned one parameter at a bound — the
    behaviour the calibration ensemble needs across all 17 algorithms.

    Previously this raised an actionable ValueError; the parameter
    selector now excludes zero-variance columns at source, so the
    degenerate-bounds validator is never reached for this case.
    """
    samples = pd.DataFrame({
        "const_param": [0.5, 0.5, 0.5],
        "other_param": [1.0, 2.0, 3.0],
        "RMSE": [0.1, 0.2, 0.3],
    })
    analyzer = _make_analyzer()
    result = analyzer.perform_sobol_analysis(samples, metric="RMSE")
    assert list(result.index) == ["other_param"], (
        f"constant column should be dropped, leaving only the varying "
        f"parameter; SA saw: {list(result.index)}"
    )


def test_analysis_manager_raises_when_no_model_produces_results(monkeypatch, tmp_path):
    """When every configured model's sensitivity analysis fails or
    returns nothing, AnalysisManager must raise EvaluationError so
    the workflow runner marks the step FAILED — not silently
    'complete with no output' like it did before."""
    from symfluence.evaluation.analysis_manager import AnalysisManager

    cfg = MagicMock()
    cfg.model.hydrological_model = "SUMMA,FUSE"
    logger = logging.getLogger("test_sa_manager_loud")
    reporting = MagicMock()

    mgr = AnalysisManager.__new__(AnalysisManager)
    mgr.config = cfg
    mgr.logger = logger
    mgr.reporting_manager = reporting
    mgr._get_config_value = lambda getter, default=None, **_: (
        default if getter() is None else getter()
    )

    def fail_generic(model):
        raise ValueError(
            "Sobol analysis cannot run: parameter 'x' has non-numeric bounds"
        )

    monkeypatch.setattr(mgr, "_run_generic_sensitivity_analysis", fail_generic)

    with pytest.raises(EvaluationError) as excinfo:
        mgr.run_sensitivity_analysis()

    msg = str(excinfo.value)
    assert "no results" in msg.lower()
    assert "SUMMA" in msg
    assert "FUSE" in msg
    assert "non-numeric bounds" in msg


def _make_manager(hydrological_model):
    """Build an AnalysisManager wired to a fake config, bypassing __init__."""
    from symfluence.evaluation.analysis_manager import AnalysisManager

    cfg = MagicMock()
    cfg.model.hydrological_model = hydrological_model

    mgr = AnalysisManager.__new__(AnalysisManager)
    mgr.config = cfg
    mgr.logger = logging.getLogger("test_sa_self_training")
    mgr.reporting_manager = MagicMock()
    mgr._get_config_value = lambda getter, default=None, **_: (
        default if getter() is None else getter()
    )
    return mgr


def test_self_training_only_config_skips_without_failing(monkeypatch):
    """A config of only self-training models (LSTM/GNN) has no parameters to
    perturb. Sensitivity analysis must return None (clean skip), NOT raise
    EvaluationError — these models would otherwise drag the whole step to
    'no results for any configured model'."""
    mgr = _make_manager("LSTM,GNN")

    # Guard: the analyzer paths must never be reached for self-training models.
    def explode(model):
        raise AssertionError(f"sensitivity analysis should not run for {model}")

    monkeypatch.setattr(mgr, "_run_generic_sensitivity_analysis", explode)

    assert mgr.run_sensitivity_analysis() is None


def test_self_training_models_excluded_from_mixed_config(monkeypatch):
    """In a mixed config, LSTM/GNN are dropped and only the calibratable
    models are analyzed — and the 'no results' aggregation never mentions the
    skipped self-training models."""
    mgr = _make_manager("LSTM,SUMMA")

    analyzed = []

    def record(model):
        analyzed.append(model)
        return {"param_a": {"S1": 0.5}}

    monkeypatch.setattr(mgr, "_run_generic_sensitivity_analysis", record)

    results = mgr.run_sensitivity_analysis()

    assert analyzed == ["SUMMA"], f"LSTM should have been skipped, saw: {analyzed}"
    assert results is not None and "LSTM" not in results
