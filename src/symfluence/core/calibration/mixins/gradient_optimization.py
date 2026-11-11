# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Gradient Optimization Mixin

Provides finite-difference gradient helpers for gradient-based optimization.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from symfluence.core.mixins import ConfigMixin

logger = logging.getLogger(__name__)


class GradientOptimizationMixin(ConfigMixin):
    """
    Mixin class providing gradient-based optimization via finite differences.

    Requires the following attributes on the class using this mixin:
    - self.config: Dict[str, Any]
    - self.logger: logging.Logger
    - self.param_manager: Parameter manager with normalize/denormalize methods
    - self._evaluate_solution: Method to evaluate a parameter set

    Provides:
    - Finite difference gradient computation
    - Line-search degeneracy reporting

    The Adam and L-BFGS drivers themselves live in
    ``core/calibration/optimizers/algorithms``.
    """

    # =========================================================================
    # Configuration
    # =========================================================================

    @property
    def gradient_epsilon(self) -> float:
        """Epsilon for finite difference gradient computation."""
        return self._get_config_value(
            lambda: self.config.optimization.gradient_epsilon,
            default=1e-4
        )

    # =========================================================================
    # Gradient computation
    # =========================================================================

    def compute_fd_gradients(
        self,
        x: np.ndarray,
        evaluate_func: Callable[[np.ndarray], float],
        epsilon: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute gradients using central finite differences.

        Args:
            x: Current parameter values (normalized)
            evaluate_func: Function to evaluate fitness given parameters
            epsilon: Perturbation size (default: self.gradient_epsilon)

        Returns:
            Tuple of (current fitness, gradient array)
        """
        if epsilon is None:
            epsilon = self.gradient_epsilon

        n_params = len(x)
        gradient = np.zeros(n_params)

        # Evaluate at current point
        f_center = evaluate_func(x)

        # Compute central differences
        for i in range(n_params):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] = min(1.0, x[i] + epsilon)
            x_minus[i] = max(0.0, x[i] - epsilon)

            f_plus = evaluate_func(x_plus)
            f_minus = evaluate_func(x_minus)

            # Divide by the step actually taken, not the requested one. The
            # perturbations above are clamped to the [0, 1] box, so for a
            # parameter sitting on a bound the real span is epsilon rather
            # than 2*epsilon — dividing by 2*epsilon there halves the
            # gradient and points the search along a systematically wrong
            # direction exactly where it is most likely to be stuck.
            step = x_plus[i] - x_minus[i]
            gradient[i] = (f_plus - f_minus) / step if step > 0 else 0.0

        return f_center, gradient

    def compute_fd_gradients_forward(
        self,
        x: np.ndarray,
        f_x: float,
        evaluate_func: Callable[[np.ndarray], float],
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute gradients using forward finite differences (faster, less accurate).

        Args:
            x: Current parameter values (normalized)
            f_x: Function value at x (avoids recomputation)
            evaluate_func: Function to evaluate fitness
            epsilon: Perturbation size

        Returns:
            Gradient array
        """
        if epsilon is None:
            epsilon = self.gradient_epsilon

        n_params = len(x)
        gradient = np.zeros(n_params)

        for i in range(n_params):
            x_plus = x.copy()
            x_plus[i] = min(1.0, x[i] + epsilon)

            f_plus = evaluate_func(x_plus)
            # Step backwards when the forward step is clamped away entirely.
            # A parameter resting on the upper bound previously produced
            # (f_x - f_x) / epsilon == 0 — an exactly zero gradient, telling
            # the optimizer the parameter has no effect and freezing it on
            # the bound for the rest of the run.
            step = x_plus[i] - x[i]
            if step <= 0:
                x_minus = x.copy()
                x_minus[i] = max(0.0, x[i] - epsilon)
                step = x[i] - x_minus[i]
                if step <= 0:
                    gradient[i] = 0.0  # degenerate box: lower == upper
                    continue
                gradient[i] = (f_x - evaluate_func(x_minus)) / step
            else:
                gradient[i] = (f_plus - f_x) / step

        return gradient

    #: Fraction of failed line searches above which the search is reported as
    #: degenerate. Occasional failures are normal; a majority means the
    #: gradient is not a usable descent direction.
    _LINE_SEARCH_FAILURE_ALERT = 0.5
    #: Minimum steps before judging, so short runs do not trip the check.
    _LINE_SEARCH_MIN_STEPS = 20

    def _warn_if_line_search_degenerate(
        self, failures: int, steps: int, epsilon: float
    ) -> None:
        """Report once when line searches fail so often the method degrades.

        Every failure already logs, but a wall of identical warnings reads as
        noise, and the run still reports success — a paper calibration was
        observed taking the steepest-descent fallback on 110 of 125 steps,
        i.e. not running L-BFGS at all, with nothing saying so. The usual
        cause is a gradient dominated by evaluation noise: JAX defaults to
        float32 (~1e-7 relative), and a central difference over ``epsilon``
        divides that noise by ``2*epsilon``, so an epsilon below
        ``sqrt(float32 eps) ~= 3.5e-4`` amplifies it rather than resolving a
        slope.
        """
        if steps < self._LINE_SEARCH_MIN_STEPS:
            return
        if failures / steps < self._LINE_SEARCH_FAILURE_ALERT:
            return
        if getattr(self, '_line_search_degenerate_reported', False):
            return
        self._line_search_degenerate_reported = True
        float32_floor = float(np.sqrt(np.finfo(np.float32).eps))
        hint = ""
        if epsilon < float32_floor:
            hint = (
                f" gradient_epsilon={epsilon:g} is below sqrt(float32 eps)="
                f"{float32_floor:.1e}, so if the model evaluates in float32 "
                f"(the JAX default) the gradient is noise, not slope — raise "
                f"gradient_epsilon or evaluate in float64."
            )
        self.logger.error(
            "Line search has failed on %d of %d steps (%.0f%%); the optimizer "
            "is running as steepest descent, not L-BFGS, and its result should "
            "not be read as a converged quasi-Newton solution.%s",
            failures, steps, 100 * failures / steps, hint,
        )
