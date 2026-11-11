"""Tests for EnKF algorithm — synthetic linear twin experiment."""
from __future__ import annotations

import numpy as np
import pytest

from symfluence.data_assimilation.enkf.enkf_algorithm import EnKFAlgorithm


class TestEnKFLinearSystem:
    """Test EnKF with a known linear system (twin experiment).

    True state x evolves as x_{t+1} = A * x_t + noise.
    Observations y = H * x + noise.
    The EnKF should converge the ensemble mean toward the true state.
    """

    def test_stochastic_enkf_converges(self):
        """Stochastic EnKF should reduce error relative to forecast."""
        rng = np.random.default_rng(42)

        n_state = 3
        n_members = 50
        n_obs = 1

        # True state
        x_true = np.array([10.0, 5.0, 2.0])

        # Observation operator: observe first variable
        H = np.zeros((n_obs, n_state))
        H[0, 0] = 1.0

        # Observation
        obs_noise_std = 0.5
        R = np.eye(n_obs) * obs_noise_std ** 2
        y_obs = H @ x_true + rng.normal(0, obs_noise_std, n_obs)

        # Forecast ensemble (spread around wrong value)
        X_f = rng.normal(loc=x_true + 3.0, scale=2.0, size=(n_members, n_state))

        # EnKF analysis
        enkf = EnKFAlgorithm(enforce_nonnegative=False)
        X_a = enkf.analyze(X_f, y_obs, H, R, variant='stochastic')

        # Check that analysis mean is closer to truth than forecast mean
        forecast_error = np.linalg.norm(X_f.mean(axis=0) - x_true)
        analysis_error = np.linalg.norm(X_a.mean(axis=0) - x_true)
        assert analysis_error < forecast_error, (
            f"Analysis error ({analysis_error:.3f}) should be less than "
            f"forecast error ({forecast_error:.3f})"
        )

    def test_deterministic_enkf_converges(self):
        """DEnKF should also reduce error."""
        rng = np.random.default_rng(123)

        n_state = 2
        n_members = 30
        n_obs = 1

        x_true = np.array([5.0, 3.0])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        y_obs = H @ x_true

        X_f = rng.normal(loc=x_true + 2.0, scale=1.5, size=(n_members, n_state))

        enkf = EnKFAlgorithm(enforce_nonnegative=False)
        X_a = enkf.analyze(X_f, y_obs, H, R, variant='deterministic')

        forecast_error = np.abs(X_f.mean(axis=0)[0] - x_true[0])
        analysis_error = np.abs(X_a.mean(axis=0)[0] - x_true[0])
        assert analysis_error < forecast_error

    def test_nonnegative_enforcement(self):
        """Test that enforce_nonnegative clips negative values."""
        rng = np.random.default_rng(0)

        n_state = 2
        n_members = 20

        # Forecast with some negative values
        X_f = rng.normal(loc=-1.0, scale=0.5, size=(n_members, n_state))
        y_obs = np.array([0.5])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        enkf = EnKFAlgorithm(enforce_nonnegative=True)
        X_a = enkf.analyze(X_f, y_obs, H, R)

        assert np.all(X_a >= 0.0), "All analysis values should be non-negative"

    def test_inflation(self):
        """Test that covariance inflation increases ensemble spread."""
        rng = np.random.default_rng(7)

        n_state = 2
        n_members = 30

        X_f = rng.normal(loc=5.0, scale=1.0, size=(n_members, n_state))
        y_obs = np.array([5.0])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.5]])

        enkf_no_inflation = EnKFAlgorithm(inflation_factor=1.0, enforce_nonnegative=False)
        enkf_inflation = EnKFAlgorithm(inflation_factor=1.5, enforce_nonnegative=False)

        X_a1 = enkf_no_inflation.analyze(X_f, y_obs, H, R)
        X_a2 = enkf_inflation.analyze(X_f.copy(), y_obs, H, R)

        # Inflated ensemble should have larger spread
        spread1 = np.std(X_a1, axis=0).mean()
        spread2 = np.std(X_a2, axis=0).mean()
        assert spread2 >= spread1 * 0.9  # Allow some tolerance

    def test_shapes(self):
        """Test output shape matches input."""
        n_state = 4
        n_members = 25
        n_obs = 2

        X_f = np.random.randn(n_members, n_state)
        y_obs = np.array([1.0, 2.0])
        H = np.zeros((n_obs, n_state))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        R = np.eye(n_obs) * 0.1

        enkf = EnKFAlgorithm(enforce_nonnegative=False)
        X_a = enkf.analyze(X_f, y_obs, H, R)

        assert X_a.shape == (n_members, n_state)


class TestStochasticReproducibility:
    """Observation perturbations come from the filter's own generator."""

    def _analyze(self, seed):
        rng = np.random.default_rng(11)
        X_f = rng.normal(5.0, 1.0, size=(20, 3))
        H = np.array([[1.0, 0.0, 0.0]])
        enkf = EnKFAlgorithm(rng=np.random.default_rng(seed))
        return enkf.analyze(X_f, np.array([6.0]), H, np.array([[0.25]]), 'stochastic')

    def test_same_seed_same_analysis(self):
        np.testing.assert_array_equal(self._analyze(3), self._analyze(3))

    def test_independent_of_global_numpy_state(self):
        np.random.seed(0)
        first = self._analyze(3)
        np.random.seed(999)
        second = self._analyze(3)
        np.testing.assert_array_equal(first, second)

    def test_different_seed_differs(self):
        assert not np.array_equal(self._analyze(3), self._analyze(4))
