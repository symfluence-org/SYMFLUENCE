#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""L-BFGS Gradient-Based Optimization Algorithm.

Limited-memory BFGS (L-BFGS) is a quasi-Newton optimization method that
approximates the inverse Hessian matrix using a limited history of past
gradients and position changes. This enables quasi-Newton methods to scale
to high-dimensional problems without storing full Hessian matrix.

Effective for hydrological model calibration when:
- Objective function is relatively smooth with limited noise
- Parameter space has low multi-modality
- Memory-efficient method is needed (Hessian approximation via history only)
- Convergence speed is important (quasi-Newton vs first-order methods)

Note: Uses central finite differences for gradient computation, making it
derivative-free but requiring ~2N function evaluations per iteration (N = n_params).

References:
    Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage.
    Mathematics of Computation, 35(151), 773-782.

    Liu, D.C. and Nocedal, J. (1989). On the limited memory BFGS method for
    large scale optimization. Mathematical Programming, 45, 503-528.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_algorithm import NativeGradientCallback, OptimizationAlgorithm
from .config_schema import LBFGSDefaults


class LBFGSAlgorithm(OptimizationAlgorithm):
    """L-BFGS quasi-Newton optimization with native or finite-difference gradients.

    L-BFGS approximates the Hessian inverse using only a limited history of
    gradients and position changes, enabling efficient quasi-Newton optimization
    without storing the full Hessian matrix.

    Algorithm Overview:
        1. Initialize parameters at normalized space midpoint (0.5)
        2. Compute initial gradient (native autodiff or finite differences)
        3. For each step:
           a. Compute search direction using L-BFGS two-loop recursion
           b. Line search: find step size satisfying Wolfe conditions
           c. Update position
           d. Compute new gradient
           e. Store position and gradient differences in history
           f. Maintain limited history (e.g., last 10 updates)
        4. Terminate when gradient norm < 1e-6 (convergence) or max steps reached
        5. Return best solution found

    Maximization convention:
        The framework maximizes fitness, but L-BFGS is formulated for
        minimization. The algorithm therefore minimizes g = -fitness internally:
        secant pairs, the curvature condition and the line search all live in
        this minimization space, and results are reported back as fitness.

    Two-Loop Recursion:
        Efficiently computes r = H*∇g where H is the approximate inverse Hessian,
        using only stored history (s_k, y_k) pairs without explicitly forming the
        Hessian. The descent direction is -r.

    Line Search (weak Wolfe conditions, minimization space):
        Ensures step size satisfies:
        1. Sufficient decrease (Armijo): g(x_new) ≤ g(x) + c1*α*⟨∇g,d⟩
        2. Curvature condition (weak Wolfe): ⟨∇g(x_new),d⟩ ≥ c2*⟨∇g,d⟩

    Gradient Computation:
        Supports two modes controlled by gradient_mode parameter:
        1. Native gradients (when compute_gradient callback provided):
           - Uses autodiff (JAX/PyTorch) for exact gradients
           - Cost: ~2 function evaluations per gradient call
        2. Finite-difference gradients (default fallback):
           - Uses central differences
           - Cost: 2*n_params + 1 function evaluations per gradient call

    Hyperparameters:
        - α (lr): Initial step size for line search (default: 0.1)
        - history_size: # of (s,y) pairs to retain (default: 10)
        - steps: Maximum iterations (default: uses max_iterations)
        - c1, c2: Wolfe condition parameters (default: 1e-4, 0.9)
        - gradient_mode: 'auto', 'native', or 'finite_difference'
    """

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "LBFGS"

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
        """Run L-BFGS optimization with native or finite-difference gradients.

        Implements L-BFGS with a backtracking line search (Armijo sufficient
        decrease, weak-Wolfe curvature where backtracking can reach it) and
        gradient convergence detection. A step that does not improve on the
        current point is never taken.

        Args:
            n_params: Number of parameters to optimize
            evaluate_solution: Function to evaluate single parameter vector
                              Call: score = evaluate_solution(x_normalized, step_id)
            evaluate_population: Unused in L-BFGS (single-solution method)
            denormalize_params: Function to convert normalized [0,1] to actual parameters
            record_iteration: Function to record iteration results
            update_best: Function to update best solution found
            log_progress: Function to log progress messages
            evaluate_population_objectives: Unused for L-BFGS
            compute_gradient: Optional native gradient callback.
                             Signature: (x_normalized) -> (loss, gradient_array)
                             When provided, enables ~N times faster gradient computation.
            gradient_mode: How to compute gradients:
                          - 'auto': Use native if compute_gradient provided, else FD
                          - 'native': Require native gradients (error if unavailable)
                          - 'finite_difference': Always use FD (for comparison)
            **kwargs: Optional hyperparameters:
                     - steps: Maximum iterations (default: max_iterations from config)
                     - lr: Initial trial step for the first (steepest-descent)
                       step and for the steepest-descent fallback (default: 0.1).
                       Once curvature history exists the quasi-Newton direction is
                       already scaled, so the trial step is 1.0.
                     - history_size: # of (s,y) pairs retained (default: 10)
                     - c1: Armijo parameter (default: 1e-4)
                     - c2: Wolfe curvature parameter (default: 0.9)

        Returns:
            Dict with keys:
            - best_solution: Best parameter vector found (normalized [0,1])
            - best_score: Highest objective value achieved
            - best_params: Denormalized best parameters (dictionary)
            - gradient_method: 'native' or 'finite_difference' (which was used)
        """
        # L-BFGS hyperparameters from config. Step-count precedence:
        # explicit kwarg > LBFGS_STEPS (per-algorithm) > NUMBER_OF_ITERATIONS.
        steps = kwargs.get('steps')
        if steps is None:
            steps = self._get_config_value(
                lambda: self.config.optimization.lbfgs.steps,
                default=None,
                dict_key='LBFGS_STEPS'
            )
        if steps is None:
            steps = self._get_config_value(
                lambda: self.config.optimization.iterations,
                default=self.max_iterations,
                dict_key='NUMBER_OF_ITERATIONS'
            )
        lr = kwargs.get('lr', self._get_config_value(
            lambda: self.config.optimization.lbfgs.lr,
            default=LBFGSDefaults.LR,
            dict_key='LBFGS_LR'
        ))
        history_size = kwargs.get('history_size', self._get_config_value(
            lambda: self.config.optimization.lbfgs.history_size,
            default=LBFGSDefaults.HISTORY_SIZE,
            dict_key='LBFGS_HISTORY_SIZE'
        ))
        c1 = kwargs.get('c1', self._get_config_value(
            lambda: self.config.optimization.lbfgs.c1,
            default=LBFGSDefaults.C1,
            dict_key='LBFGS_C1'
        ))
        c2 = kwargs.get('c2', self._get_config_value(
            lambda: self.config.optimization.lbfgs.c2,
            default=LBFGSDefaults.C2,
            dict_key='LBFGS_C2'
        ))

        # Gradient epsilon for finite differences
        gradient_epsilon = self._get_config_value(
            lambda: self.config.optimization.gradient_epsilon,
            default=LBFGSDefaults.GRADIENT_EPSILON,
            dict_key='GRADIENT_EPSILON'
        )

        # Gradient clipping value
        gradient_clip = self._get_config_value(
            lambda: self.config.optimization.gradient_clip_value,
            default=LBFGSDefaults.GRADIENT_CLIP_VALUE,
            dict_key='GRADIENT_CLIP_VALUE'
        )

        # Validate Wolfe condition parameters
        valid, msg = LBFGSDefaults.validate_wolfe(c1, c2)
        if not valid:
            self.logger.warning(f"L-BFGS validation: {msg}")

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
            f"Starting L-BFGS optimization with {n_params} parameters "
            f"({steps} epochs, gradients: {gradient_method})"
        )
        self.logger.debug(f"L-BFGS hyperparameters: lr={lr}, history_size={history_size}")
        if use_native:
            self.logger.debug("Using native gradients (~2 evals/step)")
        else:
            self.logger.debug(f"Using finite differences ({2*n_params + 1} evals/step)")

        # Initialize at midpoint of normalized space
        x = self._start_point(n_params, kwargs, np.full(n_params, 0.5))

        # L-BFGS theory is stated for MINIMIZATION, but the framework maximizes
        # fitness. We therefore minimize g = -fitness internally: the secant
        # pairs, curvature condition and line search all operate in this
        # minimization space. gradient_func returns (fitness, ascent_grad) where
        # ascent_grad = ∇fitness; the minimization gradient is grad_g = -ascent_grad.
        def eval_min(xi: np.ndarray) -> Tuple[float, np.ndarray]:
            fit, ascent_grad = gradient_func(xi)
            return -fit, -ascent_grad

        # L-BFGS history (minimization space)
        s_history: List[np.ndarray] = []  # Position differences s_k = x_{k+1} - x_k
        y_history: List[np.ndarray] = []  # Min-space gradient differences y_k

        # Track best (reported in fitness / maximization terms)
        best_x = x.copy()
        best_fitness = float('-inf')
        n_improvements = 0

        # Initial minimization-space objective and gradient
        g, grad = eval_min(x)
        grad = self._clip_gradient(grad, gradient_clip)

        for step in range(steps):
            fitness = -g

            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()
                n_improvements += 1

            # Record iteration
            params_dict = denormalize_params(best_x)
            record_iteration(step, best_fitness, params_dict)
            update_best(best_fitness, params_dict, step)

            # Two-loop recursion returns r ≈ H·grad, where H approximates the
            # inverse Hessian of g. The descent direction for g is -r.
            direction = -self._lbfgs_direction(grad, s_history, y_history)

            # Line search in minimization space. The two-loop recursion scales
            # the direction by the secant estimate of the inverse Hessian, so
            # the natural trial step is 1.0 once history exists; the first
            # step is plain steepest descent and starts from lr.
            trial_step = 1.0 if s_history else lr
            step_size, g_new, grad_new = self._line_search_with_gradient_func(
                x, direction, g, grad, eval_min,
                trial_step, c1, c2
            )

            if step_size is None and s_history:
                # The quasi-Newton direction gave no acceptable step: drop the
                # history and try a backtracked steepest-descent step instead.
                self.logger.debug(
                    f"L-BFGS line search failed at step {step}; resetting history "
                    "and trying steepest descent"
                )
                s_history.clear()
                y_history.clear()
                direction = -grad
                step_size, g_new, grad_new = self._line_search_with_gradient_func(
                    x, direction, g, grad, eval_min,
                    lr, c1, c2
                )

            if step_size is None:
                # No step along the negative gradient decreases g either. Taking
                # one anyway would move to a worse point, so stop here.
                self.logger.info(
                    f"L-BFGS stopped at step {step}: no improving step found along the "
                    "descent direction (converged to line-search precision, or the "
                    "gradient is unreliable here)"
                )
                break

            x_new = np.clip(x + step_size * direction, 0, 1)

            grad_new = self._clip_gradient(grad_new, gradient_clip)

            # Secant pair update. The standard curvature condition y·s > 0 holds
            # in the region where g is locally convex — i.e. near a maximum of
            # fitness — so the quasi-Newton history now actually accumulates.
            s = x_new - x
            y = grad_new - grad

            if np.dot(y, s) > 1e-10:  # Curvature condition
                s_history.append(s)
                y_history.append(y)

                if len(s_history) > history_size:
                    s_history.pop(0)
                    y_history.pop(0)

            # Update state
            x = x_new
            g = g_new
            grad = grad_new

            # Progress line (tracker throttles emission)
            log_progress(
                self.name, step + 1, best_fitness,
                n_improved=n_improvements, pop_size=step + 1,
                unit='epochs', total=steps
            )

            # Check convergence
            if np.linalg.norm(grad) < 1e-6:
                self.logger.info(f"L-BFGS converged at step {step}")
                break

        # The loop records the iterate at the top of each step, so the point
        # reached by the last step has not been recorded yet. Record it, or
        # best_params.json / the iteration CSV lag the returned result.
        fitness = -g
        if fitness > best_fitness:
            best_fitness = fitness
            best_x = x.copy()
            n_improvements += 1
        final_iteration = step + 1 if steps > 0 else 0
        params_dict = denormalize_params(best_x)
        record_iteration(final_iteration, best_fitness, params_dict)
        update_best(best_fitness, params_dict, final_iteration)

        return {
            'best_solution': best_x,
            'best_score': best_fitness,
            'best_params': params_dict,
            'gradient_method': gradient_method
        }

    # Note: _compute_fd_gradients and _clip_gradient are inherited from base class

    def _lbfgs_direction(
        self,
        gradient: np.ndarray,
        s_history: List[np.ndarray],
        y_history: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute r = H·gradient via the L-BFGS two-loop recursion.

        H is the implicit approximation to the inverse Hessian built from the
        stored (s, y) pairs. This routine is sign-agnostic: the caller forms the
        descent direction as -r (minimization space). With an empty history it
        returns the (scaled) gradient itself, i.e. plain steepest descent.

        Args:
            gradient: Current minimization-space gradient
            s_history: History of position differences s_k = x_{k+1} - x_k
            y_history: History of minimization-space gradient differences

        Returns:
            r = H·gradient (the caller negates this to get the descent direction)
        """
        q = gradient.copy()
        m = len(s_history)
        alphas = []

        # First loop (backward)
        for i in range(m - 1, -1, -1):
            rho_i = 1.0 / (np.dot(y_history[i], s_history[i]) + 1e-10)
            alpha_i = rho_i * np.dot(s_history[i], q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_history[i]

        alphas.reverse()

        # Initial Hessian approximation
        if m > 0:
            gamma = np.dot(s_history[-1], y_history[-1]) / (
                np.dot(y_history[-1], y_history[-1]) + 1e-10
            )
        else:
            gamma = 1.0

        r = gamma * q

        # Second loop (forward)
        for i in range(m):
            rho_i = 1.0 / (np.dot(y_history[i], s_history[i]) + 1e-10)
            beta_i = rho_i * np.dot(y_history[i], r)
            r = r + (alphas[i] - beta_i) * s_history[i]

        return r  # H·gradient; caller negates for the descent direction

    def _line_search_with_gradient_func(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        g_x: float,
        grad_x: np.ndarray,
        eval_min: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        initial_step: float,
        c1: float,
        c2: float,
        max_iter: int = 20
    ) -> Tuple[Optional[float], float, np.ndarray]:
        """
        Backtracking line search on the minimization objective g = -fitness.

        The (weak) Wolfe conditions for minimization are:
          1. Armijo sufficient decrease:  g(x + α d) ≤ g(x) + c1·α·⟨∇g, d⟩
          2. Weak-Wolfe curvature:        ⟨∇g(x + α d), d⟩ ≥ c2·⟨∇g, d⟩

        The step is halved until Armijo holds. The curvature condition rejects
        steps that are too *short*, and backtracking only ever shrinks the
        step, so once the largest Armijo step fails it no smaller step can
        satisfy it: that Armijo step is accepted as it is (requiring curvature
        as well made the search fail on almost every step and burn a gradient
        evaluation per halving). Pairs that would violate the L-BFGS curvature
        requirement are screened by the y·s > 0 guard at the call site, so the
        secant history stays well-conditioned either way.

        A returned step always satisfies g_new < g_x: the caller never moves
        to a point that is not better than the current one.

        Args:
            x: Current parameter values (normalized [0,1])
            direction: Descent direction (-r from the two-loop recursion)
            g_x: Current minimization objective value g = -fitness
            grad_x: Current minimization-space gradient ∇g
            eval_min: Minimization evaluator: (x) -> (g, grad_g)
            initial_step: Initial step size for line search
            c1: Armijo condition parameter (sufficient decrease)
            c2: Wolfe curvature condition parameter (diagnostic only: logged
                at debug level when the accepted Armijo step does not meet it)
            max_iter: Maximum line search iterations

        Returns:
            Tuple of (step_size, g_new, grad_new); step_size is None on failure.
        """
        step_size = initial_step
        directional_deriv = np.dot(grad_x, direction)

        if directional_deriv >= 0:
            # Not a descent direction for g
            return None, g_x, grad_x

        for _ in range(max_iter):
            x_new = np.clip(x + step_size * direction, 0, 1)
            g_new, grad_new = eval_min(x_new)

            # Armijo condition (sufficient decrease for minimization); the
            # strict g_new < g_x guards against accepting a non-improving step
            # when the Armijo slack is lost to rounding.
            if (
                np.isfinite(g_new)
                and g_new < g_x
                and g_new <= g_x + c1 * step_size * directional_deriv
            ):
                if np.dot(grad_new, direction) < c2 * directional_deriv:
                    self.logger.debug(
                        f"L-BFGS line search: accepting Armijo step {step_size:.3g} "
                        "(curvature condition not met)"
                    )
                return step_size, g_new, grad_new

            step_size *= 0.5

            if step_size < 1e-10:
                break

        return None, g_x, grad_x
