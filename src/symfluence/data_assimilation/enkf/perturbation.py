# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Perturbation strategies for ensemble generation.

Provides methods for perturbing model parameters, forcing data, and
state variables to create diverse ensemble members.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class PerturbationStrategy(ABC):
    """Abstract base class for perturbation strategies."""

    @abstractmethod
    def perturb_parameters(
        self,
        base_params: Dict[str, float],
        bounds: Dict[str, tuple],
        member_id: int,
    ) -> Dict[str, float]:
        """Perturb model parameters for an ensemble member.

        Args:
            base_params: Base parameter values.
            bounds: Parameter bounds {name: (lower, upper)}.
            member_id: Ensemble member index.

        Returns:
            Perturbed parameter dictionary.
        """
        ...

    @abstractmethod
    def perturb_forcing(
        self,
        forcing_data: np.ndarray,
        member_id: int,
        variable: str = 'precip',
    ) -> np.ndarray:
        """Perturb forcing data for an ensemble member.

        Args:
            forcing_data: Forcing time series.
            member_id: Ensemble member index.
            variable: Variable type ('precip', 'temp', 'pet').

        Returns:
            Perturbed forcing data.
        """
        ...

    @abstractmethod
    def perturb_state(
        self,
        state: Dict[str, np.ndarray],
        member_id: int,
    ) -> Dict[str, np.ndarray]:
        """Perturb model state for an ensemble member.

        Args:
            state: State variable arrays.
            member_id: Ensemble member index.

        Returns:
            Perturbed state dictionary.
        """
        ...


class GaussianPerturbation(PerturbationStrategy):
    """Gaussian perturbation strategy.

    - Parameters: Gaussian noise proportional to parameter range
    - Precipitation: Multiplicative lognormal perturbation
    - Temperature: Additive Gaussian perturbation
    - State: Gaussian noise proportional to variable magnitude

    Args:
        param_std: Std as fraction of parameter range (default: 0.05).
        precip_std: Multiplicative std for precipitation (default: 0.3).
        temp_std: Additive std for temperature in K (default: 1.0).
        state_std: Std as fraction of state value (default: 0.1).
        rng: Optional numpy random generator for reproducibility.
    """

    def __init__(
        self,
        param_std: float = 0.05,
        precip_std: float = 0.3,
        temp_std: float = 1.0,
        state_std: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.param_std = param_std
        self.precip_std = precip_std
        self.temp_std = temp_std
        self.state_std = state_std
        self.rng = rng or np.random.default_rng()

    def perturb_parameters(
        self,
        base_params: Dict[str, float],
        bounds: Dict[str, tuple],
        member_id: int,
    ) -> Dict[str, float]:
        perturbed = {}
        for name, value in base_params.items():
            if name in bounds:
                lo, hi = bounds[name]
                param_range = hi - lo
                noise = self.rng.normal(0, self.param_std * param_range)
                perturbed[name] = float(np.clip(value + noise, lo, hi))
            else:
                perturbed[name] = value
        return perturbed

    def perturb_forcing(
        self,
        forcing_data: np.ndarray,
        member_id: int,
        variable: str = 'precip',
    ) -> np.ndarray:
        data = forcing_data.copy()

        if variable == 'precip':
            # Multiplicative lognormal: preserves non-negativity. The log-mean
            # is -sigma^2/2 so the multiplier has expectation 1; with a
            # log-mean of 0 it is exp(sigma^2/2) (a 4.6 % wet bias at
            # sigma = 0.3) and every member is systematically too wet.
            sigma = self.precip_std
            multiplier = self.rng.lognormal(
                mean=-0.5 * sigma ** 2,
                sigma=sigma,
                size=data.shape,
            )
            data = data * multiplier
        elif variable in ('temp', 'temperature'):
            # Additive Gaussian
            noise = self.rng.normal(0, self.temp_std, size=data.shape)
            data = data + noise
        elif variable == 'pet':
            # Multiplicative, similar to precip but smaller perturbation
            # (unbiased, as for precipitation)
            sigma = self.precip_std * 0.5
            multiplier = self.rng.lognormal(
                mean=-0.5 * sigma ** 2,
                sigma=sigma,
                size=data.shape,
            )
            data = data * multiplier

        return data

    def perturb_state(
        self,
        state: Dict[str, np.ndarray],
        member_id: int,
    ) -> Dict[str, np.ndarray]:
        perturbed = {}
        for name, value in state.items():
            value = np.asarray(value, dtype=np.float64)
            magnitude = np.maximum(np.abs(value), 1.0)
            noise = self.rng.normal(0, self.state_std * magnitude, size=value.shape)
            perturbed[name] = np.maximum(value + noise, 0.0)
        return perturbed
