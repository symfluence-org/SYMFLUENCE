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

1. ``SensitivityAnalyzer.perform_sobol_analysis`` validates bounds
   up front and raises a named, actionable ``ValueError`` (naming the
   offending parameter and its bogus value) rather than letting
   SALib's opaque message propagate.
2. ``AnalysisManager.run_sensitivity_analysis`` raises
   ``EvaluationError`` when no configured model produces results,
   so the workflow step is marked FAILED instead of silently
   'complete with no output'.
"""

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


def test_constant_parameter_raises_actionable_value_error():
    """A parameter column with zero variance (min == max) has no
    attributable sensitivity; SALib rejects it, but the current
    silent-swallow made this look like success. Validate here and
    point the user at the fix."""
    samples = pd.DataFrame({
        "const_param": [0.5, 0.5, 0.5],
        "other_param": [1.0, 2.0, 3.0],
        "RMSE": [0.1, 0.2, 0.3],
    })
    analyzer = _make_analyzer()
    with pytest.raises(ValueError) as excinfo:
        analyzer.perform_sobol_analysis(samples, metric="RMSE")
    msg = str(excinfo.value)
    assert "const_param" in msg
    assert "degenerate" in msg.lower() or "constant" in msg.lower() or "variance" in msg.lower()


def test_analysis_manager_raises_when_no_model_produces_results(monkeypatch, tmp_path):
    """When every configured model's sensitivity analysis fails or
    returns nothing, AnalysisManager must raise EvaluationError so
    the workflow runner marks the step FAILED — not silently
    'complete with no output' like it did before."""
    from symfluence.evaluation.analysis_manager import AnalysisManager

    cfg = MagicMock()
    cfg.model.hydrological_model = "SUMMA,FUSE"
    cfg.analysis.run_sensitivity_analysis = True
    logger = logging.getLogger("test_sa_manager_loud")
    reporting = MagicMock()

    mgr = AnalysisManager.__new__(AnalysisManager)
    mgr.config = cfg
    mgr.logger = logger
    mgr.reporting_manager = reporting
    mgr._get_config_value = lambda getter, default=None, **_: (
        getter() if callable(getter) else default
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
