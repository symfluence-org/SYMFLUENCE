#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Basin Hopping Global Optimization

A two-phase global optimization algorithm that combines random perturbations
with local minimization to escape local optima and find the global optimum.
Particularly effective for rugged fitness landscapes with many local minima.

The algorithm alternates between:
1. Random jumps to escape the current basin of attraction
2. Local optimization to find the local minimum
3. Metropolis acceptance to decide whether to accept the new basin

Key Features:
    - Escapes local minima through random perturbations
    - Efficient local search within each basin
    - Temperature-controlled acceptance for exploration/exploitation balance
    - Adaptive step size based on acceptance rate

Reference:
    Wales, D.J. and Doye, J.P.K. (1997). Global Optimization by Basin-Hopping
    and the Lowest Energy Structures of Lennard-Jones Clusters Containing up
    to 110 Atoms. Journal of Physical Chemistry A, 101(28), 5111-5116.

    Li, Z. and Scheraga, H.A. (1987). Monte Carlo-minimization approach to the
    multiple-minima problem in protein folding. Proceedings of the National
    Academy of Sciences, 84(19), 6611-6615.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .base_algorithm import OptimizationAlgorithm
from .config_schema import BasinHoppingDefaults


class BasinHoppingAlgorithm(OptimizationAlgorithm):
    """Basin Hopping global optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "BASIN-HOPPING"

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
        compute_gradient: Optional[Callable] = None,
        gradient_mode: str = 'auto',
        log_initial_population: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Basin Hopping optimization.

        The algorithm works as follows:
        1. Start from initial point x
        2. Perturb: x_new = x + random_step
        3. Local minimize: x_local = local_min(x_new)
        4. Accept/reject based on Metropolis criterion:
           - Always accept if f(x_local) > f(x)
           - Accept with probability exp((f_local - f_current) / T) otherwise
        5. Repeat from step 2

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            log_initial_population: Optional callback to log initial population
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        # Per-run RNG (OptimizationAlgorithm._new_rng): never the process-global stream.
        self._rng = self._new_rng()
        self.logger.info(f"Starting Basin Hopping optimization with {n_params} parameters")

        # Basin Hopping parameters from config
        step_size = self._get_config_value(
            lambda: self.config.optimization.basin_hopping.step_size,
            default=BasinHoppingDefaults.STEP_SIZE,
            dict_key='BH_STEP_SIZE'
        )
        temperature = self._get_config_value(
            lambda: self.config.optimization.basin_hopping.temperature,
            default=BasinHoppingDefaults.TEMPERATURE,
            dict_key='BH_TEMPERATURE'
        )
        local_steps = self._get_config_value(
            lambda: self.config.optimization.basin_hopping.local_steps,
            default=BasinHoppingDefaults.LOCAL_STEPS,
            dict_key='BH_LOCAL_STEPS'
        )
        local_method = self._get_config_value(
            lambda: self.config.optimization.basin_hopping.local_method,
            default=BasinHoppingDefaults.LOCAL_METHOD,
            dict_key='BH_LOCAL_METHOD'
        )
        target_accept_rate = self._get_config_value(
            lambda: self.config.optimization.basin_hopping.target_accept,
            default=BasinHoppingDefaults.TARGET_ACCEPT,
            dict_key='BH_TARGET_ACCEPT'
        )
        adapt_interval = self._get_config_value(
            lambda: self.config.optimization.basin_hopping.adapt_interval,
            default=BasinHoppingDefaults.ADAPT_INTERVAL,
            dict_key='BH_ADAPT_INTERVAL'
        )

        # Validate Basin Hopping configuration
        valid, warning = BasinHoppingDefaults.validate_parameters(step_size, temperature)
        if not valid:
            self.logger.warning(f"Basin Hopping parameter validation: {warning}")

        if local_steps < 20:
            self.logger.warning(
                f"BH_LOCAL_STEPS is set to {local_steps}, which is very low. "
                "Consider increasing to at least 50 for effective local optimization."
            )

        self.logger.info(
            f"Basin Hopping settings: step_size={step_size}, temperature={temperature}, "
            f"local_steps={local_steps}, method={local_method}"
        )

        # Initialize at random point or center
        current_pos = self._start_point(n_params, kwargs, self._rng.uniform(0, 1, n_params))
        current_fit = evaluate_solution(current_pos, 0)

        # Local minimize the starting point
        current_pos, current_fit = self._local_minimize(
            current_pos, evaluate_solution, local_steps, local_method, n_params
        )

        # Track best solution (global best across all basins)
        best_pos = current_pos.copy()
        best_fit = current_fit

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, 1, best_fit)

        # Acceptance tracking for adaptive step size
        accept_history: List[bool] = []

        # Main Basin Hopping loop
        for iteration in range(1, self.max_iterations + 1):
            # Random perturbation (jump to new basin)
            perturbation = self._rng.uniform(-step_size, step_size, n_params)
            new_pos = current_pos + perturbation

            # Reflect at bounds
            new_pos = self._reflect_at_bounds(new_pos)

            # Local minimization in new basin
            new_pos, new_fit = self._local_minimize(
                new_pos, evaluate_solution, local_steps, local_method, n_params
            )

            # Metropolis acceptance criterion
            # We're maximizing, so accept if new_fit > current_fit
            # or with probability exp((new_fit - current_fit) / T)
            accept = False
            if new_fit > current_fit:
                accept = True
            else:
                # Boltzmann probability (for maximization)
                delta = new_fit - current_fit
                if temperature > 0:
                    accept_prob = np.exp(delta / temperature)
                    accept = self._rng.random() < accept_prob

            accept_history.append(accept)

            if accept:
                current_pos = new_pos
                current_fit = new_fit

                # Update global best
                if current_fit > best_fit:
                    best_fit = current_fit
                    best_pos = current_pos.copy()

            # Adaptive step size adjustment
            if iteration % adapt_interval == 0 and len(accept_history) >= adapt_interval:
                recent_accepts = accept_history[-adapt_interval:]
                accept_rate = sum(recent_accepts) / len(recent_accepts)

                # Adjust step size to achieve target acceptance rate
                if accept_rate < target_accept_rate - 0.1:
                    step_size *= 0.9  # Decrease step size if too few accepts
                elif accept_rate > target_accept_rate + 0.1:
                    step_size *= 1.1  # Increase step size if too many accepts

                # Keep step size in reasonable bounds
                step_size = np.clip(step_size, 0.01, 1.0)

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(
                iteration, best_fit, params_dict,
                {'step_size': step_size, 'accepted': accept}
            )
            update_best(best_fit, params_dict, iteration)

            # Progress line (tracker throttles emission); Improved counts
            # cumulative accepted hops over hops so far.
            log_progress(
                self.name, iteration, best_fit,
                n_improved=sum(accept_history), pop_size=len(accept_history),
                unit='evals'
            )

        # Final statistics
        total_accepts = sum(accept_history)
        self.logger.info(
            f"Basin Hopping complete: {total_accepts}/{len(accept_history)} hops accepted "
            f"({100 * total_accepts / len(accept_history):.1f}%)"
        )

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'final_step_size': step_size,
            'accept_rate': total_accepts / len(accept_history),
            'n_hops': len(accept_history),
            'n_accepted': total_accepts,
        }

    def _local_minimize(
        self,
        x: np.ndarray,
        evaluate: Callable[[np.ndarray, int], float],
        n_steps: int,
        method: str,
        n_params: int
    ) -> Tuple[np.ndarray, float]:
        """
        Perform local minimization starting from x using SciPy.

        Args:
            x: Starting point (normalized [0,1])
            evaluate: Evaluation function
            n_steps: Maximum number of iterations
            method: Local optimization method ('nelder_mead' or 'gradient')
            n_params: Number of parameters

        Returns:
            Tuple of (optimized_position, fitness)
        """
        # Define objective function for minimization (negate fitness for maximization)
        # SciPy minimizes, we want to maximize fitness
        def objective(p):
            # Clip parameters to bounds before evaluation
            p_clipped = np.clip(p, 0, 1)
            # Evaluate (0 is passed as iteration number, though not strictly tracked inside local step)
            score = evaluate(p_clipped, 0)
            return -score

        # Map method names to SciPy methods
        scipy_method = 'Nelder-Mead'
        if method == 'gradient':
            scipy_method = 'L-BFGS-B'
        elif method.lower() == 'lbfgs':
            scipy_method = 'L-BFGS-B'
        elif method.lower() == 'powell':
            scipy_method = 'Powell'

        # Set bounds for methods that support them
        bounds = [(0, 1) for _ in range(n_params)]

        # Optimize
        try:
            # Note: Nelder-Mead in SciPy doesn't strictly support bounds, but we clip inside objective.
            # L-BFGS-B and others support bounds directly.
            res = minimize(
                objective,
                x,
                method=scipy_method,
                bounds=bounds if scipy_method in ['L-BFGS-B', 'TNC', 'SLSQP', 'Powell'] else None,
                options={'maxiter': n_steps, 'disp': False}
            )

            # Return optimized position (clipped to be safe) and actual fitness (negate back)
            best_pos = np.clip(res.x, 0, 1)
            best_fit = -res.fun
            return best_pos, best_fit

        except (ValueError, RuntimeError) as e:
            self.logger.warning(f"Local minimization failed: {e}. Returning original point.")
            current_fit = evaluate(x, 0)
            return x, current_fit
