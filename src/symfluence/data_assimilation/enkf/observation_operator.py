# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Observation operators for the EnKF.

Maps from model state space to observation space (the H matrix).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ObservationOperator(ABC):
    """Abstract base class for observation operators.

    The observation operator maps from the model state vector to the
    predicted observations: y_pred = H(x).
    """

    @abstractmethod
    def apply(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply the observation operator to a state vector.

        Args:
            state_vector: State vector of shape (n_state,).

        Returns:
            Predicted observations of shape (n_obs,).
        """
        ...

    @abstractmethod
    def get_matrix(self, n_state: int) -> np.ndarray:
        """Return the explicit observation operator matrix H.

        Args:
            n_state: Total length of the state vector.

        Returns:
            H matrix of shape (n_obs, n_state).
        """
        ...


class StreamflowObservationOperator(ObservationOperator):
    """Extracts streamflow prediction from an augmented state vector.

    When the state vector is augmented with predicted streamflow as
    the last element(s), this operator simply selects those elements.

    Args:
        n_obs: Number of streamflow observation variables (default: 1).
    """

    def __init__(self, n_obs: int = 1):
        self.n_obs = n_obs

    def apply(self, state_vector: np.ndarray) -> np.ndarray:
        return state_vector[-self.n_obs:]

    def get_matrix(self, n_state: int) -> np.ndarray:
        H = np.zeros((self.n_obs, n_state))
        for i in range(self.n_obs):
            H[i, n_state - self.n_obs + i] = 1.0
        return H
