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


def ar1_coefficient(correlation_steps: float) -> float:
    """Lag-1 autocorrelation ``exp(-1 / tau)`` for an e-folding time of tau steps."""
    if correlation_steps <= 0:
        return 0.0
    return float(np.exp(-1.0 / float(correlation_steps)))


def ar1_standard_normal(
    rng: np.random.Generator,
    n_members: int,
    n_steps: int,
    correlation_steps: float = 0.0,
) -> np.ndarray:
    """Stationary AR(1) series with N(0, 1) marginals, shape (n_members, n_steps)."""
    phi = ar1_coefficient(correlation_steps)
    eps = rng.standard_normal((n_members, n_steps))
    if phi == 0.0 or n_steps == 0:
        return eps
    z = np.empty_like(eps)
    z[:, 0] = eps[:, 0]
    innovation_scale = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n_steps):
        z[:, t] = phi * z[:, t - 1] + innovation_scale * eps[:, t]
    return z


def lognormal_multiplier(z: np.ndarray, sigma: float) -> np.ndarray:
    """Map N(0, 1) noise to a lognormal multiplier with expectation 1."""
    return np.exp(sigma * z - 0.5 * sigma ** 2)


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

    def forcing_noise(
        self,
        variable: str,
        n_members: int,
        n_steps: int,
        correlation_steps: float = 0.0,
    ) -> np.ndarray:
        """Perturbations for a whole period, correlated in time (AR(1)).

        A standard-normal AR(1) series ``z`` is drawn per member,

            z[0] ~ N(0, 1),  z[t] = phi * z[t-1] + sqrt(1 - phi**2) * eps[t],
            phi = exp(-1 / correlation_steps)

        so every ``z[t]`` is marginally N(0, 1) (the series starts in its
        stationary distribution) and the lag-k autocorrelation is ``phi**k``.
        ``correlation_steps = 0`` gives white noise (phi = 0).

        The series is mapped exactly like :meth:`perturb_forcing` maps its
        white noise, so the marginal statistics, and in particular the
        debiasing, are unchanged:

        - ``precip``: multiplier ``exp(sigma * z - sigma**2 / 2)``
          (lognormal with expectation 1), sigma = ``precip_std``
        - ``pet``: same with sigma = ``precip_std / 2``
        - ``temp``: additive offset ``temp_std * z``

        Args:
            variable: 'precip', 'pet', 'temp' or 'temperature'.
            n_members: Ensemble size.
            n_steps: Number of timesteps.
            correlation_steps: e-folding time of the noise in timesteps.

        Returns:
            Array (n_members, n_steps): multipliers for precip/pet, additive
            offsets for temperature.
        """
        z = ar1_standard_normal(self.rng, n_members, n_steps, correlation_steps)
        if variable == 'precip':
            return lognormal_multiplier(z, self.precip_std)
        if variable == 'pet':
            return lognormal_multiplier(z, 0.5 * self.precip_std)
        if variable in ('temp', 'temperature'):
            return self.temp_std * z
        raise ValueError(f"No forcing perturbation defined for '{variable}'")

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
