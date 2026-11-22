# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for sensitivity analysis on gradient-optimiser output.

Co-author SH (04_calibration_ensemble, iter-3) reported:

  The sensitivity analysis step fails entirely for gradient-based
  algorithms — all parameters throw a type error caused by
  ADAM-specific columns (grad_norm, lr, current_params) in the
  results file that the sensitivity analyzer cannot handle.

Root cause: SA built its parameter list by subtracting
``_NON_PARAM_COLS``. That set didn't include the diagnostic columns
written by ADAM / L-BFGS, so SA treated them as calibration
parameters. ``current_params`` is a list that pandas writes as a
string literal, so SALib immediately rejects the "bounds".

Pin the fix: the parameter-column selector (a) excludes the known
gradient-optimiser diagnostic columns by name, (b) as belt-and-
suspenders also drops any non-numeric column so a future optimiser
adding its own string-valued diagnostic doesn't regress this path.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from symfluence.evaluation.sensitivity_analysis import (
    _NON_PARAM_COLS,
    SensitivityAnalyzer,
    _constant_parameter_columns,
    _parameter_columns_for_sa,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def test_non_param_cols_includes_gradient_optimiser_diagnostics():
    """The four ADAM/L-BFGS diagnostic columns must be in the
    exclusion set so SA doesn't treat them as parameters."""
    for name in ('grad_norm', 'lr', 'current_params', 'step_size'):
        assert name in _NON_PARAM_COLS, f"{name} missing from _NON_PARAM_COLS"


def test_parameter_selector_drops_gradient_diagnostics():
    """An ADAM-shaped results DataFrame should only expose the real
    hydrological parameter columns to SA."""
    df = pd.DataFrame({
        'iteration': [0, 1, 2],
        'score': [0.5, 0.6, 0.7],
        'timestamp': ['t0', 't1', 't2'],
        'p1': [0.1, 0.2, 0.3],
        'p2': [1.0, 2.0, 3.0],
        # ADAM diagnostics
        'grad_norm': [0.5, 0.3, 0.1],
        'lr': [0.01, 0.01, 0.01],
        'current_params': ['[0.1, 1.0]', '[0.2, 2.0]', '[0.3, 3.0]'],
    })
    cols = _parameter_columns_for_sa(df)
    assert set(cols) == {'p1', 'p2'}, f"unexpected param columns: {cols}"


def test_parameter_selector_drops_non_numeric_columns():
    """Even if a new optimiser adds a diagnostic column we haven't
    yet named, the selector must drop non-numeric columns so they
    don't poison SA."""
    df = pd.DataFrame({
        'p1': [0.1, 0.2],
        'p2': [1.0, 2.0],
        'mystery_string_col': ['a', 'b'],
        'mystery_list_col': ['[1,2]', '[3,4]'],
    })
    cols = _parameter_columns_for_sa(df)
    assert set(cols) == {'p1', 'p2'}


def test_adam_shaped_results_dont_break_parameter_selection(caplog):
    """End-to-end: give trajectory_correlation an ADAM-shaped frame
    and confirm the previous 'Bounds are not legal' path no longer
    fires because SA only sees the real numeric parameter columns."""
    analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
    analyzer.config = MagicMock()
    analyzer.logger = logging.getLogger("test_sa_adam")
    analyzer.reporting_manager = MagicMock()

    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        'iteration': range(n),
        'score': rng.uniform(0.3, 0.9, n),
        'p1': rng.uniform(0.1, 0.9, n),
        'p2': rng.uniform(1.0, 9.0, n),
        'RMSE': rng.uniform(0.05, 0.3, n),
        'grad_norm': rng.uniform(0.01, 1.0, n),
        'lr': [0.01] * n,
        # ADAM writes current_params as a Python list; pandas serialises
        # it as a string — exactly the shape that previously crashed SA.
        'current_params': [f"[{a:.2f}, {b:.2f}]" for a, b
                           in zip(rng.uniform(0.1, 0.9, n), rng.uniform(1, 9, n))],
    })
    # trajectory_correlation should not raise; it'll run SALib on
    # just p1/p2. We don't care about the numeric indices here —
    # only that we no longer crash on non-numeric columns.
    caplog.set_level(logging.WARNING)
    result = analyzer.trajectory_correlation(df, metric='RMSE')
    # Must return a Series indexed by the two real parameters only.
    assert list(result.index) == ['p1', 'p2'], \
        f"analysis ran on unexpected columns: {list(result.index)}"


def test_non_param_cols_includes_sampler_diagnostics():
    """DREAM-style sampler diagnostics must be excluded so they are not
    treated as calibration parameters."""
    for name in ('acceptance_rate', 'n_chains'):
        assert name in _NON_PARAM_COLS, f"{name} missing from _NON_PARAM_COLS"


def test_parameter_selector_drops_constant_columns():
    """A zero-variance numeric column (a pinned model parameter, or an
    optimiser-internal like DREAM's n_chains) must be excluded — it has no
    sensitivity to attribute and would otherwise make SALib's Sobol sampler
    raise 'Bounds are not legal'."""
    df = pd.DataFrame({
        'p1': [0.1, 0.2, 0.3],
        'p2': [1.0, 2.0, 3.0],
        'px_temp': [1.0, 1.0, 1.0],   # real parameter pinned by the optimiser
        'n_chains': [29, 29, 29],     # sampler-internal, constant
    })
    assert set(_parameter_columns_for_sa(df)) == {'p1', 'p2'}
    # px_temp is named as a dropped-constant; n_chains is in the denylist.
    assert 'px_temp' in _constant_parameter_columns(df)


def test_dream_shaped_results_drop_internals_and_run(caplog):
    """End-to-end on a DREAM-shaped frame (constant n_chains + varying
    acceptance_rate + real params): the trajectory correlation runs on the real parameters only,
    with no crash on the constant internal column."""
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        'iteration': range(n),
        'score': rng.uniform(0.7, 0.9, n),
        'p1': rng.uniform(0.1, 0.9, n),
        'p2': rng.uniform(1.0, 9.0, n),
        'RMSE': rng.uniform(0.05, 0.3, n),
        'acceptance_rate': rng.uniform(0.6, 1.0, n),  # varying internal
        'n_chains': [29] * n,                          # constant internal
    })
    analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
    analyzer.config = MagicMock()
    analyzer.logger = logging.getLogger("test_sa_dream")
    analyzer.reporting_manager = MagicMock()

    result = analyzer.trajectory_correlation(df, metric='RMSE')
    assert list(result.index) == ['p1', 'p2'], \
        f"analysis ran on unexpected columns: {list(result.index)}"
