"""NGEN postprocessor KGE/NSE use the shared core metric definitions."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from symfluence.core import metrics as core_metrics
from symfluence.models.ngen.postprocessor import NgenPostProcessor


def _bare():
    # The metric helpers only use their arguments; skip the heavy __init__.
    return NgenPostProcessor.__new__(NgenPostProcessor)


def test_matches_core_metrics():
    rng = np.random.default_rng(0)
    obs = rng.gamma(2.0, 3.0, 200)
    sim = obs * 0.9 + rng.normal(0, 1, 200)
    sim[5] = np.nan
    pp = _bare()
    assert pp._calculate_kge(obs, sim) == pytest.approx(core_metrics.kge(obs, sim))
    assert pp._calculate_nse(obs, sim) == pytest.approx(core_metrics.nse(obs, sim))


def test_constant_obs_returns_nan_without_division_warning():
    obs = np.full(30, 4.0)
    sim = np.linspace(3, 5, 30)
    pp = _bare()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(pp._calculate_kge(obs, sim))
        assert np.isnan(pp._calculate_nse(obs, sim))
