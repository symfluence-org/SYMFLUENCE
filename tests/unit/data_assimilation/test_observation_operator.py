"""Tests for observation operators."""
from __future__ import annotations

import numpy as np
import pytest

from symfluence.data_assimilation.enkf.observation_operator import StreamflowObservationOperator


class TestStreamflowObservationOperator:
    """Tests for StreamflowObservationOperator."""

    def test_single_obs(self):
        op = StreamflowObservationOperator(n_obs=1)
        state = np.array([10.0, 150.0, 5.0, 3.0, 42.0])

        y = op.apply(state)
        assert y.shape == (1,)
        assert y[0] == 42.0

    def test_matrix_form(self):
        op = StreamflowObservationOperator(n_obs=1)
        H = op.get_matrix(n_state=5)
        assert H.shape == (1, 5)
        assert H[0, 4] == 1.0
        assert np.sum(H) == 1.0

    def test_consistency(self):
        """H @ x should equal apply(x)."""
        op = StreamflowObservationOperator(n_obs=1)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = op.get_matrix(5)

        np.testing.assert_allclose(H @ state, op.apply(state))
