#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Adam Gradient-Based Optimization Algorithm.

ADAM (Adaptive Moment Estimation) is a gradient-based optimizer that uses
finite-difference gradients and adaptive learning rates. It combines first-moment
(momentum) and second-moment (RMSprop) estimation for efficient convergence.

Useful for hydrological model calibration when:
- Objective function is relatively smooth (limited noise)
- Parameter space is not highly multi-modal
- Gradient computation via finite differences is feasible
- Number of function evaluations is limited

Note: Uses central finite differences for gradient computation, making it
derivative-free but requiring ~2N function evaluations per iteration (N = n_params).

References:
    Kingma, D.P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization.
    In Proceedings of the 3rd International Conference on Learning Representations (ICLR).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from symfluence.core.metrics.penalty import is_penalty

from .base_algorithm import NativeGradientCallback, OptimizationAlgorithm
from .config_schema import AdamDefaults


class AdamAlgorithm(OptimizationAlgorithm):
    """Adam gradient-based optimization with support for native or finite-difference gradients.

    ADAM maintains first and second moment estimates of the gradient:
    - m = exponential moving average of gradients (momentum)
    - v = exponential moving average of squared gradients (adaptive learning rate)
    - Uses bias correction to account for initialization bias

    Algorithm Overview:
        1. Initialize parameters at normalized space midpoint (0.5)
        2. For each step:
           a. Compute gradients (native autodiff or finite-difference)
           b. Update first moment: m ← β1*m + (1-β1)*∇f
           c. Update second moment: v ← β2*v + (1-β2)*∇f²
           d. Bias correct: m̂ = m / (1 - β1^t), v̂ = v / (1 - β2^t)
           e. Update parameters: x ← x + α * m̂ / (√v̂ + ε)
           f. Clip to [0,1] bounds
        3. Return best solution found

    Gradient Computation:
        Supports two modes controlled by gradient_mode parameter:
        1. Native gradients (when compute_gradient callback provided):
           - Uses autodiff (JAX/PyTorch) for exact gradients
           - Cost: ~2 function evaluations per step (forward + backward)
           - ~N times faster than FD for N parameters
        2. Finite-difference gradients (default fallback):
           - Uses central differences: ∇f_i = (f(x+ε*e_i) - f(x-ε*e_i)) / (2ε)
           - Cost: 2*n_params + 1 function evaluations per step

    Hyperparameters:
        - α (lr): Learning rate (default: 0.01)
        - β1: First moment decay (default: 0.9)
        - β2: Second moment decay (default: 0.999)
        - ε: Numerical stability (default: 1e-8)
        - steps: Maximum iterations (default: uses max_iterations)
        - gradient_mode: 'auto', 'native', or 'finite_difference'
    """

    # Consecutive failed evaluations after which Adam stops early.
    MAX_CONSECUTIVE_FAILURES = 5

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "ADAM"

    def optimize(
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        evaluate_population_objectives: Optional[Callable] = None,
        compute_gradient: Optional[NativeGradientCallback] = None,
        gradient_mode: str = 'auto',
        **kwargs
    ) -> Dict[str, Any]:
        """Run ADAM optimization with native or finite-difference gradients.

        Initializes parameters at normalized space midpoint and iteratively
        improves using adaptive moment estimation and gradient ascent.

        Args:
            n_params: Number of parameters to optimize
            evaluate_solution: Function to evaluate single parameter vector
                              Call: score = evaluate_solution(x_normalized, step_id)
            evaluate_population: Function to evaluate population
                                (unused in ADAM, single-solution method)
            denormalize_params: Function to convert normalized [0,1] to actual parameters
            record_iteration: Function to record iteration results
            update_best: Function to update best solution found
            log_progress: Function to log progress messages
            evaluate_population_objectives: Unused for ADAM
            compute_gradient: Optional native gradient callback.
                             Signature: (x_normalized) -> (loss, gradient_array)
                             When provided, enables ~N times faster gradient computation.
            gradient_mode: How to compute gradients:
                          - 'auto': Use native if compute_gradient provided, else FD
                          - 'native': Require native gradients (error if unavailable)
                          - 'finite_difference': Always use FD (for comparison)
            **kwargs: Optional hyperparameters:
                     - steps: Number of iterations (default: max_iterations from config)
                     - lr: Learning rate (default: 0.01)
                     - beta1: First moment decay (default: 0.9)
                     - beta2: Second moment decay (default: 0.999)
                     - eps: Numerical stability constant (default: 1e-8)

        Returns:
            Dict with keys:
            - best_solution: Best parameter vector found (normalized [0,1])
            - best_score: Highest objective value achieved
            - best_params: Denormalized best parameters (dictionary)
            - gradient_method: 'native' or 'finite_difference' (which was used)
        """
        # Adam hyperparameters from config. Step-count precedence:
        # explicit kwarg > ADAM_STEPS (per-algorithm) > NUMBER_OF_ITERATIONS.
        steps = kwargs.get('steps')
        if steps is None:
            steps = self._get_config_value(
                lambda: self.config.optimization.adam.steps,
                default=None,
                dict_key='ADAM_STEPS'
            )
        if steps is None:
            steps = self._get_config_value(
                lambda: self.config.optimization.iterations,
                default=self.max_iterations,
                dict_key='NUMBER_OF_ITERATIONS'
            )
        lr = kwargs.get('lr', self._get_config_value(
            lambda: self.config.optimization.adam.lr,
            default=AdamDefaults.LR,
            dict_key='ADAM_LR'
        ))
        beta1 = kwargs.get('beta1', self._get_config_value(
            lambda: self.config.optimization.adam.beta1,
            default=AdamDefaults.BETA1,
            dict_key='ADAM_BETA1'
        ))
        beta2 = kwargs.get('beta2', self._get_config_value(
            lambda: self.config.optimization.adam.beta2,
            default=AdamDefaults.BETA2,
            dict_key='ADAM_BETA2'
        ))
        eps = kwargs.get('eps', self._get_config_value(
            lambda: self.config.optimization.adam.eps,
            default=AdamDefaults.EPS,
            dict_key='ADAM_EPS'
        ))

        # Gradient epsilon for finite differences
        gradient_epsilon = self._get_config_value(
            lambda: self.config.optimization.gradient_epsilon,
            default=AdamDefaults.GRADIENT_EPSILON,
            dict_key='GRADIENT_EPSILON'
        )

        # Gradient clipping value
        gradient_clip = self._get_config_value(
            lambda: self.config.optimization.gradient_clip_value,
            default=AdamDefaults.GRADIENT_CLIP_VALUE,
            dict_key='GRADIENT_CLIP_VALUE'
        )

        # Validate beta parameters
        valid, msg = AdamDefaults.validate_betas(beta1, beta2)
        if not valid:
            self.logger.warning(f"Adam validation: {msg}")

        # Determine gradient method and create unified gradient function
        use_native = self._should_use_native_gradients(compute_gradient, gradient_mode)
        gradient_method = 'native' if use_native else 'finite_difference'

        gradient_func = self._create_gradient_function(
            compute_gradient=compute_gradient,
            evaluate_solution=evaluate_solution,
            gradient_mode=gradient_mode,
            epsilon=gradient_epsilon
        )

        self.logger.info(
            f"Starting Adam optimization with {n_params} parameters "
            f"({steps} epochs, gradients: {gradient_method})"
        )
        self.logger.debug(f"Adam hyperparameters: lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}")
        if use_native:
            self.logger.debug("Using native gradients (~2 evals/step)")
        else:
            self.logger.debug(f"Using finite differences ({2*n_params + 1} evals/step)")

        # Initialize from initial_guess if provided, otherwise midpoint
        initial_guess = kwargs.get('initial_guess')
        if initial_guess is not None and len(initial_guess) == n_params:
            x = np.array(initial_guess, dtype=float)
            self.logger.debug("Warm-starting Adam from provided initial guess")
        else:
            x = np.full(n_params, 0.5)

        # Adam state
        m = np.zeros(n_params)  # First moment
        v = np.zeros(n_params)  # Second moment

        # Track best
        best_x = x.copy()
        best_fitness = float('-inf')
        n_improvements = 0
        consecutive_failures = 0

        for step in range(steps):
            # Compute gradients using unified gradient function
            fitness, gradient = gradient_func(x)

            # Failed evaluation (NaN / inf / penalty): the gradient is
            # meaningless, and re-evaluating the same x would fail forever.
            # Move to a perturbed copy of the best point; give up after a run
            # of consecutive failures.
            if is_penalty(fitness, self.penalty_score):
                consecutive_failures += 1
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self.logger.warning(
                        f"ADAM step {step}: {consecutive_failures} consecutive failed "
                        f"evaluations, stopping early"
                    )
                    break
                self.logger.warning(
                    f"ADAM step {step}: failed evaluation (fitness={fitness}), "
                    f"perturbing from best point"
                )
                x = np.clip(
                    best_x + self._rng.normal(0.0, max(float(lr or 0.0), 1e-3), n_params), 0.0, 1.0
                )
                log_progress(
                    self.name, step + 1, best_fitness,
                    n_improved=n_improvements, pop_size=step + 1,
                    unit='epochs', total=steps
                )
                continue
            consecutive_failures = 0

            # Clip gradient (handles NaN elements internally)
            gradient = self._clip_gradient(gradient, gradient_clip)

            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()
                n_improvements += 1

            # Record iteration with enhanced tracking for response surface analysis
            params_dict = denormalize_params(best_x)
            grad_norm = float(np.linalg.norm(gradient))
            record_iteration(step, best_fitness, params_dict, {
                'grad_norm': grad_norm,
                'lr': lr,
                'current_params': x.copy().tolist(),
            })
            update_best(best_fitness, params_dict, step)

            # Adam update
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))

            # Update parameters (gradient ascent for maximization)
            x = x + lr * m_hat / (np.sqrt(v_hat) + eps)

            # Clip to [0, 1]
            x = np.clip(x, 0, 1)

            # Progress line (tracker throttles emission)
            log_progress(
                self.name, step + 1, best_fitness,
                n_improved=n_improvements, pop_size=step + 1,
                unit='epochs', total=steps
            )

        if not np.isfinite(best_fitness):
            best_fitness = self.penalty_score

        return {
            'best_solution': best_x,
            'best_score': best_fitness,
            'best_params': denormalize_params(best_x),
            'gradient_method': gradient_method
        }

    # Note: _compute_gradients and _clip_gradient are inherited from base class
    # The base class provides:
    # - _compute_fd_gradients(): Central finite differences
    # - _clip_gradient(): Gradient norm clipping
    # - _create_gradient_function(): Unified gradient function factory
    # - _should_use_native_gradients(): Gradient mode decision logic
