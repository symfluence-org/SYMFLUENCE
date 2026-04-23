# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for the VISCOUS / calibration-sample handling in
``SensitivityAnalyzer``.

Co-author PW reported (07_sensitivity_analysis):

  "After removing the brackets, the analysis ran for 3 of the 4
  methods. VISCOUS gave NaN, presumably due to the sampling being
  based on the calibration runs."

Two failure modes were masked end-to-end:

1. Crash-regime sentinels (``<= -900``) and non-finite metric values
   leaked through ``preprocess_data`` straight into ``x``/``y`` for
   every method, poisoning VISCOUS's copula fit.
2. When VISCOUS's GMCM fit didn't converge (common on highly-clustered
   DDS samples) the library could still return a non-finite result.
   The wrapper appended that NaN directly to the output Series — the
   CSV and plot then silently showed NaN for that parameter with no
   indication the result was invalid.

Pin the fixes:
  * ``preprocess_data`` now drops rows whose metric is NaN/inf or a
    SYMFLUENCE failure sentinel before any SA method sees them.
  * ``perform_sensitivity_analysis`` checks the VISCOUS result for
    non-finiteness, logs a specific WARNING naming the parameter and
    likely cause, and records -999 instead of NaN.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from symfluence.evaluation.sensitivity_analysis import SensitivityAnalyzer

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_analyzer():
    """Build without running ConfigMixin __init__ (same trick as the
    Sobol loud-fail tests)."""
    analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
    analyzer.config = MagicMock()
    analyzer.logger = logging.getLogger("test_viscous_nan")
    analyzer.reporting_manager = MagicMock()
    return analyzer


def test_preprocess_drops_failure_sentinel_rows(caplog):
    """Rows with metric = -999 (SYMFLUENCE crash marker) must be
    removed before VISCOUS/Sobol/RBD-FAST see them."""
    analyzer = _make_analyzer()
    df = pd.DataFrame({
        'Iteration': list(range(6)),
        'p1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'p2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Calib_RMSE': [0.8, -999.0, 0.6, -9999.0, 0.4, 0.3],
    })
    caplog.set_level(logging.INFO)
    out = analyzer.preprocess_data(df, metric='Calib_RMSE')
    assert len(out) == 4, "sentinel rows must be dropped"
    assert out['Calib_RMSE'].min() > -900
    assert "failure-sentinel" in "\n".join(r.getMessage() for r in caplog.records)


def test_preprocess_drops_nan_metric_rows():
    """NaN metric values (from crashed iterations) must be dropped."""
    analyzer = _make_analyzer()
    df = pd.DataFrame({
        'Iteration': [0, 1, 2, 3],
        'p1': [0.1, 0.2, 0.3, 0.4],
        'Calib_RMSE': [0.5, float('nan'), 0.7, float('inf')],
    })
    out = analyzer.preprocess_data(df, metric='Calib_RMSE')
    assert len(out) == 2
    assert np.isfinite(out['Calib_RMSE']).all()


def test_preprocess_handles_missing_metric_column_gracefully():
    """If the chosen metric column isn't in the DataFrame, don't
    crash — just fall through to deduplication. Some optimisers use
    non-standard metric names."""
    analyzer = _make_analyzer()
    df = pd.DataFrame({
        'Iteration': [0, 1, 2],
        'p1': [0.1, 0.2, 0.3],
        'score': [0.5, 0.6, 0.7],
    })
    out = analyzer.preprocess_data(df, metric='Calib_RMSE')
    assert len(out) == 3


def test_viscous_nan_result_recorded_as_sentinel(caplog):
    """When pyviscous returns a non-finite sensitivity (GMCM did not
    converge for any component count), the wrapper must log
    specifically AND record -999 — not silently pass NaN through to
    the output CSV."""
    analyzer = _make_analyzer()
    df = pd.DataFrame({
        'p1': np.linspace(0.1, 0.9, 80),
        'p2': np.linspace(1.0, 9.0, 80),
        'Calib_KGEnp': np.random.default_rng(0).uniform(0.2, 0.9, 80),
    })
    # Patch viscous to return NaN (simulating a GMCM non-convergence).
    with patch(
        'symfluence.evaluation.sensitivity_analysis._pyviscous',
        side_effect=lambda x, y, i, sensType='total': (float('nan'), None),
    ):
        caplog.set_level(logging.WARNING)
        result = analyzer.perform_sensitivity_analysis(
            df, metric='Calib_KGEnp', min_samples=10,
        )
    assert (result == -999).all(), f"expected -999 sentinels, got {result.tolist()}"
    warn_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "VISCOUS returned a non-finite index" in warn_text
    assert "p1" in warn_text or "p2" in warn_text


def test_viscous_normal_result_passes_through():
    """Sanity — when VISCOUS returns a finite number, that number
    (not a sentinel) reaches the output Series."""
    analyzer = _make_analyzer()
    df = pd.DataFrame({
        'p1': np.linspace(0.1, 0.9, 80),
        'p2': np.linspace(1.0, 9.0, 80),
        'Calib_KGEnp': np.random.default_rng(0).uniform(0.2, 0.9, 80),
    })
    with patch(
        'symfluence.evaluation.sensitivity_analysis._pyviscous',
        side_effect=lambda x, y, i, sensType='total': (0.42 + 0.01 * i, None),
    ):
        result = analyzer.perform_sensitivity_analysis(
            df, metric='Calib_KGEnp', min_samples=10,
        )
    assert result['p1'] == pytest.approx(0.42)
    assert result['p2'] == pytest.approx(0.43)
