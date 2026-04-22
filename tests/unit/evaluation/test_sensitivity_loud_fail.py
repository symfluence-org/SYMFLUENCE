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
    cfg = MagicMock()
    logger = logging.getLogger("test_sa_loud_fail")
    reporting = MagicMock()
    return SensitivityAnalyzer(cfg, logger, reporting)


def test_stringified_bounds_raise_actionable_value_error():
    """If the samples DataFrame was round-tripped through YAML and
    parameter columns ended up as strings, perform_sobol_analysis
    must refuse up-front with a message that names the parameter
    and the offending values — not SALib's generic 'Bounds are not
    legal'."""
    samples = pd.DataFrame({
        "param_a": ["0.1", "0.5", "2.0"],
        "param_b": [1.0, 2.0, 3.0],
        "RMSE": [0.1, 0.2, 0.3],
    })
    analyzer = _make_analyzer()
    with pytest.raises(ValueError) as excinfo:
        analyzer.perform_sobol_analysis(samples, metric="RMSE")
    msg = str(excinfo.value)
    assert "param_a" in msg
    assert "non-numeric" in msg
    assert "stringified" in msg.lower() or "yaml" in msg.lower()


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
