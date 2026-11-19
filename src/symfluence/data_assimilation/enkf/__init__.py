# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Ensemble Kalman Filter (EnKF) implementation.
"""
from __future__ import annotations

from .enkf_algorithm import EnKFAlgorithm
from .ensemble_manager import (
    DAModel,
    EnsembleManager,
    HBVEnsembleManager,
    XinanjiangEnsembleManager,
)
from .observation_error import StreamflowObservationError
from .observation_operator import ObservationOperator, StreamflowObservationOperator
from .perturbation import GaussianPerturbation, PerturbationStrategy
from .state_vector import StateVariableSpec, StateVector

__all__ = [
    "EnKFAlgorithm",
    "DAModel",
    "EnsembleManager",
    "HBVEnsembleManager",
    "XinanjiangEnsembleManager",
    "StreamflowObservationError",
    "StateVector",
    "StateVariableSpec",
    "ObservationOperator",
    "StreamflowObservationOperator",
    "PerturbationStrategy",
    "GaussianPerturbation",
]
