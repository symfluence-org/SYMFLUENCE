# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Ensemble Kalman Filter algorithm.

Implements the stochastic and deterministic (DEnKF) variants of the
Ensemble Kalman Filter for hydrological state updating.

References:
    Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation
    and practical implementation. Ocean Dynamics, 53, 343-367.

    Sakov, P. & Oke, P.R. (2008). A deterministic formulation of the
    ensemble Kalman filter: an alternative to ensemble square root filters.
    Tellus A, 60, 361-371.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EnKFAlgorithm:
    """Ensemble Kalman Filter (EnKF) analysis step.

    Supports stochastic EnKF and deterministic EnKF (DEnKF) variants.

    Args:
        inflation_factor: Covariance inflation multiplier (>=1.0, 1.0=none).
        enforce_nonnegative: Clip negative state values to zero after update.
        rng: Random generator for the stochastic variant's observation
            perturbations. Pass a seeded generator for reproducible runs;
            a fresh unseeded one is created otherwise.
    """

    def __init__(
        self,
        inflation_factor: float = 1.0,
        enforce_nonnegative: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.inflation_factor = inflation_factor
        self.enforce_nonnegative = enforce_nonnegative
        self.rng = rng if rng is not None else np.random.default_rng()

    def analyze(
        self,
        X_f: np.ndarray,
        y_obs: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        variant: str = 'stochastic',
    ) -> np.ndarray:
        """Perform the EnKF analysis (update) step.

        Args:
            X_f: Forecast ensemble matrix, shape (n_members, n_state).
            y_obs: Observation vector, shape (n_obs,).
            H: Observation operator matrix, shape (n_obs, n_state).
            R: Observation error covariance matrix, shape (n_obs, n_obs).
            variant: 'stochastic' or 'deterministic'.

        Returns:
            X_a: Analysis (updated) ensemble matrix, shape (n_members, n_state).
        """
        n_members, n_state = X_f.shape

        # 1. Ensemble mean and anomalies
        x_bar = X_f.mean(axis=0)  # (n_state,)
        A = X_f - x_bar  # (n_members, n_state)

        # Optional covariance inflation
        if self.inflation_factor > 1.0:
            A = A * self.inflation_factor
            X_f = x_bar + A

        # 2. Predicted observations per member: Y_f = X_f @ H^T
        Y_f = X_f @ H.T  # (n_members, n_obs)
        y_bar = Y_f.mean(axis=0)  # (n_obs,)

        # 3. Anomalies in observation space
        Y_anom = Y_f - y_bar  # (n_members, n_obs)

        # 4. Cross-covariance P_xy = A^T @ Y_anom / (N-1)
        P_xy = A.T @ Y_anom / (n_members - 1)  # (n_state, n_obs)

        # 5. Innovation covariance P_yy = Y_anom^T @ Y_anom / (N-1) + R
        P_yy = Y_anom.T @ Y_anom / (n_members - 1) + R  # (n_obs, n_obs)

        # 6. Kalman gain K = P_xy @ inv(P_yy)
        try:
            K = P_xy @ np.linalg.inv(P_yy)  # (n_state, n_obs)
        except np.linalg.LinAlgError:
            logger.warning("Singular P_yy matrix, using pseudo-inverse")
            K = P_xy @ np.linalg.pinv(P_yy)

        # 7. Update
        if variant == 'deterministic':
            X_a = self._denkf_update(X_f, K, Y_f, y_obs, A, H, R, n_members)
        else:
            X_a = self._stochastic_update(X_f, K, Y_f, y_obs, R, n_members)

        # 8. Enforce physical constraints
        if self.enforce_nonnegative:
            X_a = np.maximum(X_a, 0.0)

        return X_a

    def _stochastic_update(
        self,
        X_f: np.ndarray,
        K: np.ndarray,
        Y_f: np.ndarray,
        y_obs: np.ndarray,
        R: np.ndarray,
        n_members: int,
    ) -> np.ndarray:
        """Stochastic EnKF: perturb observations per member."""
        n_obs = y_obs.shape[0]

        # Perturb observations: D = y_obs + N(0, R) per member
        # (own generator, not the global numpy state: reproducible when seeded)
        obs_noise = self.rng.multivariate_normal(
            np.zeros(n_obs), R, size=n_members
        )  # (n_members, n_obs)
        D = y_obs + obs_noise  # (n_members, n_obs)

        # Innovation per member
        innovation = D - Y_f  # (n_members, n_obs)

        # Update: X_a = X_f + innovation @ K^T
        X_a = X_f + innovation @ K.T

        return X_a

    def _denkf_update(
        self,
        X_f: np.ndarray,
        K: np.ndarray,
        Y_f: np.ndarray,
        y_obs: np.ndarray,
        A: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        n_members: int,
    ) -> np.ndarray:
        """Deterministic EnKF (DEnKF): no observation perturbation.

        Updates the mean with the full Kalman gain and the anomalies
        with half the Kalman gain to preserve ensemble spread.
        """
        x_bar = X_f.mean(axis=0)
        y_bar = Y_f.mean(axis=0)

        # Update mean
        x_bar_a = x_bar + K @ (y_obs - y_bar)

        # Update anomalies: A_a = A - 0.5 * K @ H @ A^T
        A_a = A - 0.5 * (A @ H.T) @ K.T

        # Reconstruct
        X_a = x_bar_a + A_a

        return X_a
