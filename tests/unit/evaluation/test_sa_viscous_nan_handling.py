# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for the calibration-sample handling in
``SensitivityAnalyzer`` (trajectory-correlation diagnostic).

The VISCOUS / trajectory-surrogate Sobol methods these tests originally
guarded were removed (their samples are an optimiser trajectory, not a
design); the preprocessing they pinned still feeds the trajectory
correlation.

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
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

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


def test_all_crashed_calibration_skips_without_salib_error(tmp_path, caplog):
    """When a model crashed on every calibration trial, every score is a failure
    sentinel (<= -900). preprocess_data drops them all, leaving an empty sample
    set. The raw len>=10 guard passes (there are 20 rows), so without a post-clean
    re-check this empty array reached SALib and raised the cryptic
    'array of sample points is empty'. The analyzer must instead skip gracefully
    with an actionable message (this is what MESH/WFLOW hit in the ensemble run).
    """
    analyzer = _make_analyzer()
    analyzer.output_folder = tmp_path
    df = pd.DataFrame({
        'iteration': np.arange(20),
        'score': np.full(20, -999.0),          # every trial crashed
        'p1': np.linspace(0.1, 0.9, 20),
        'p2': np.linspace(1.0, 9.0, 20),
    })
    results_file = tmp_path / "all_crashed.csv"
    df.to_csv(results_file, index=False)

    with caplog.at_level(logging.WARNING):
        result = analyzer.run_sensitivity_analysis(results_file)

    assert result is None, "all-crashed calibration must skip, not run SA"
    warn_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "usable calibration samples" in warn_text
    assert "array of sample points is empty" not in warn_text
