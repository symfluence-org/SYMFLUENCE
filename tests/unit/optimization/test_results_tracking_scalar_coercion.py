# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for scalar-coercion in optimization results recording.

Co-author PW reported that the 07_sensitivity_analysis pipeline failed
because parameter VALUES in the calibration-results CSV were stored
as bracketed strings ("[0.1]", "[273.16]") instead of plain floats.
Root cause: SUMMA's parameter manager returns ``np.array([x])`` from
``_format_parameter_value`` so downstream model wrappers get the
array shape they expect, but those arrays leaked into the DataFrame
that gets serialised with ``to_csv`` — the CSV then carried string
reprs of numpy arrays that SALib's ``sobol.analyze`` couldn't read.

Pin the fix: ``record_iteration`` and ``update_best`` coerce
array-like parameter values to scalars before storing them, so the
serialised history only contains numeric values. The model path
(where the arrays are needed) is untouched.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from symfluence.optimization.mixins.results_tracking import (
    ResultsTrackingMixin,
    _scalar_for_csv,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


class _Tracker(ResultsTrackingMixin):
    """Minimal concrete subclass for direct mixin testing."""

    def __init__(self, results_dir):
        self.config = {}
        self.logger = MagicMock()
        self.results_dir = Path(results_dir)
        self.__init_results_tracking__()


def test_scalar_for_csv_unwraps_length_1_numpy_array():
    assert _scalar_for_csv(np.array([0.1])) == pytest.approx(0.1)
    assert _scalar_for_csv(np.array([273.16])) == pytest.approx(273.16)


def test_scalar_for_csv_preserves_homogeneous_per_hru_broadcast():
    """A per-HRU array where every HRU holds the same value reduces
    to that value — the repr is still a single physical quantity."""
    assert _scalar_for_csv(np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_scalar_for_csv_reduces_varying_per_hru_to_mean():
    """A truly varying per-HRU array collapses to the mean for CSV
    (callers needing per-HRU must log separately). This is a lossy
    summary but doesn't poison the CSV with array reprs."""
    assert _scalar_for_csv(np.array([0.2, 0.4, 0.6, 0.8])) == pytest.approx(0.5)


def test_scalar_for_csv_passes_scalars_through():
    assert _scalar_for_csv(0.1) == 0.1
    assert _scalar_for_csv(np.float64(0.1)) == pytest.approx(0.1)
    assert _scalar_for_csv(None) is None


def test_record_iteration_writes_numeric_csv(tmp_path):
    """End-to-end: record_iteration with SUMMA-shaped array params
    produces a DataFrame whose to_csv output is re-readable as
    numeric — the shape the sensitivity analyzer expects."""
    tracker = _Tracker(tmp_path)
    # Mimic SUMMA's _format_parameter_value output: mix of length-1
    # arrays (basin/depth params) and per-HRU arrays (local params).
    params = {
        'tempCritRain': np.array([273.16]),           # length-1 array
        'Fcapil': np.array([0.06]),                   # length-1 array
        'albedoDecayRate': np.array([1.5e6, 1.5e6, 1.5e6]),  # per-HRU homogeneous
    }
    tracker.record_iteration(iteration=0, score=0.8, params=params)
    tracker.record_iteration(iteration=1, score=0.82, params=params)

    df = pd.DataFrame(tracker._iteration_history)
    out = tmp_path / "iter.csv"
    df.to_csv(out, index=False)

    # Read back and verify every param column is numeric.
    roundtrip = pd.read_csv(out)
    for col in ['tempCritRain', 'Fcapil', 'albedoDecayRate']:
        assert pd.api.types.is_numeric_dtype(roundtrip[col]), \
            f"column {col} is {roundtrip[col].dtype} — expected numeric. " \
            f"First value: {roundtrip[col].iloc[0]!r}"
    # And the values are what we expect
    assert roundtrip['tempCritRain'].iloc[0] == pytest.approx(273.16)
    assert roundtrip['Fcapil'].iloc[0] == pytest.approx(0.06)
    assert roundtrip['albedoDecayRate'].iloc[0] == pytest.approx(1.5e6)


def test_update_best_stores_scalar_params(tmp_path):
    """best_params dumped to JSON must be plain numbers, not numpy
    arrays (JSON doesn't even serialise those without a custom
    encoder — users have seen this break the summary-write step)."""
    tracker = _Tracker(tmp_path)
    params = {
        'tempCritRain': np.array([273.16]),
        'albedoDecayRate': np.array([1.5e6, 1.5e6, 1.5e6]),
    }
    tracker.update_best(score=0.9, params=params, iteration=5)
    assert tracker._best_params is not None
    for k, v in tracker._best_params.items():
        assert isinstance(v, (int, float)), \
            f"_best_params[{k!r}] = {v!r} ({type(v).__name__}) — expected scalar"


def test_sensitivity_analyzer_can_read_roundtripped_csv(tmp_path):
    """The real consumer — ``perform_sobol_analysis`` — reads the
    CSV via samples[col].min()/.max(). Confirm those return numeric
    values after our coercion rather than strings."""
    tracker = _Tracker(tmp_path)
    for i, x in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        tracker.record_iteration(
            iteration=i,
            score=0.5 + 0.1 * i,
            params={'p1': np.array([x]), 'p2': np.array([x * 10])},
        )
    df = pd.DataFrame(tracker._iteration_history)
    out = tmp_path / "hist.csv"
    df.to_csv(out, index=False)
    reread = pd.read_csv(out)
    # The check SALib-friendly consumers rely on:
    assert float(reread['p1'].min()) == pytest.approx(0.1)
    assert float(reread['p1'].max()) == pytest.approx(0.9)
    assert float(reread['p2'].min()) == pytest.approx(1.0)
    assert float(reread['p2'].max()) == pytest.approx(9.0)
