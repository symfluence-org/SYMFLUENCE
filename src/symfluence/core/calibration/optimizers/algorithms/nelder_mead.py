#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Nelder-Mead Simplex Algorithm

A derivative-free optimization method that uses a simplex (polytope) of n+1
vertices to search for the optimum. The simplex adapts its shape through
reflection, expansion, contraction, and shrinkage operations.

Key Features:
    - No gradient information required
    - Simple and robust for low-dimensional problems
    - Adapts to local landscape geometry
    - Good for polishing solutions from global optimizers

Reference:
    Nelder, J.A. and Mead, R. (1965). A Simplex Method for Function Minimization.
    The Computer Journal, 7(4), 308-313.

    Gao, F. and Han, L. (2012). Implementing the Nelder-Mead simplex algorithm
    with adaptive parameters. Computational Optimization and Applications,
    51(1), 259-277.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import NelderMeadDefaults


class NelderMeadAlgorithm(OptimizationAlgorithm):
    """Nelder-Mead Simplex optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "NELDER-MEAD"

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
        Run Nelder-Mead simplex optimization.

        The algorithm maintains a simplex of n+1 vertices and iteratively
        transforms it using four operations:
        1. Reflection: Reflect worst point through centroid
        2. Expansion: If reflection is best, expand further
        3. Contraction: If reflection is poor, contract toward centroid
        4. Shrinkage: If contraction fails, shrink toward best point

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
        self.logger.info(f"Starting Nelder-Mead optimization with {n_params} parameters")

        # Nelder-Mead parameters (adaptive based on dimension)
        # Standard parameters
        alpha = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.alpha,
            default=NelderMeadDefaults.ALPHA,
            dict_key='NM_ALPHA'
        )
        gamma = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.gamma,
            default=NelderMeadDefaults.GAMMA,
            dict_key='NM_GAMMA'
        )
        rho = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.rho,
            default=NelderMeadDefaults.RHO,
            dict_key='NM_RHO'
        )
        sigma = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.sigma,
            default=NelderMeadDefaults.SIGMA,
            dict_key='NM_SIGMA'
        )

        # Initial simplex size
        simplex_size = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.simplex_size,
            default=NelderMeadDefaults.SIMPLEX_SIZE,
            dict_key='NM_SIMPLEX_SIZE'
        )

        # Convergence tolerances
        x_tol = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.x_tol,
            default=NelderMeadDefaults.X_TOL,
            dict_key='NM_X_TOL'
        )
        f_tol = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.f_tol,
            default=NelderMeadDefaults.F_TOL,
            dict_key='NM_F_TOL'
        )

        # Use adaptive parameters for high dimensions (Gao & Han, 2012)
        use_adaptive = self._get_config_value(
            lambda: self.config.optimization.nelder_mead.adaptive,
            default=NelderMeadDefaults.ADAPTIVE,
            dict_key='NM_ADAPTIVE'
        )

        # Validate Nelder-Mead parameters
        valid, warning = NelderMeadDefaults.validate_parameters(alpha, gamma, rho, sigma)
        if not valid:
            self.logger.warning(f"Nelder-Mead parameters may cause issues: {warning}")
        if use_adaptive and n_params > 2:
            alpha = 1.0
            gamma = 1.0 + 2.0 / n_params
            rho = 0.75 - 0.5 / n_params
            sigma = 1.0 - 1.0 / n_params
            self.logger.info(f"Using adaptive parameters: α={alpha:.3f}, γ={gamma:.3f}, ρ={rho:.3f}, σ={sigma:.3f}")

        # Initialize simplex
        # Start from center or random point
        x0 = self._start_point(n_params, kwargs, np.full(n_params, 0.5))  # default: centre

        simplex = np.zeros((n_params + 1, n_params))
        simplex[0] = x0.copy()

        # Create initial simplex vertices
        for i in range(n_params):
            simplex[i + 1] = x0.copy()
            # Step inward when the start point sits on the upper bound, so a
            # clipped vertex does not collapse onto x0.
            step = simplex_size if x0[i] + simplex_size <= 1.0 else -simplex_size
            simplex[i + 1, i] += step
            # Ensure bounds
            simplex[i + 1] = np.clip(simplex[i + 1], 0, 1)

        # Evaluate all simplex vertices
        fitness = np.array([evaluate_solution(simplex[i], 0) for i in range(n_params + 1)])
        eval_count = n_params + 1

        # Track best solution
        best_idx = np.argmax(fitness)
        best_pos = simplex[best_idx].copy()
        best_fit = fitness[best_idx]

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_params + 1, best_fit)

        # Track operations for logging
        operations = {'reflect': 0, 'expand': 0, 'contract_out': 0, 'contract_in': 0, 'shrink': 0}
        n_improvements = 0

        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            # Sort simplex by fitness (descending for maximization)
            order = np.argsort(-fitness)
            simplex = simplex[order]
            fitness = fitness[order]

            # Best, second worst, worst
            f_best = fitness[0]
            f_second_worst = fitness[-2]
            f_worst = fitness[-1]

            # Update global best
            if f_best > best_fit:
                best_fit = f_best
                best_pos = simplex[0].copy()

            # Compute centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_reflected = centroid + alpha * (centroid - simplex[-1])
            x_reflected = np.clip(x_reflected, 0, 1)
            f_reflected = evaluate_solution(x_reflected, iteration)
            eval_count += 1

            if f_best >= f_reflected > f_second_worst:
                # Accept reflection
                simplex[-1] = x_reflected
                fitness[-1] = f_reflected
                operations['reflect'] += 1

            elif f_reflected > f_best:
                # Try expansion
                x_expanded = centroid + gamma * (x_reflected - centroid)
                x_expanded = np.clip(x_expanded, 0, 1)
                f_expanded = evaluate_solution(x_expanded, iteration)
                eval_count += 1

                if f_expanded > f_reflected:
                    simplex[-1] = x_expanded
                    fitness[-1] = f_expanded
                    operations['expand'] += 1
                else:
                    simplex[-1] = x_reflected
                    fitness[-1] = f_reflected
                    operations['reflect'] += 1

            else:
                # Contraction
                if f_reflected > f_worst:
                    # Outside contraction
                    x_contracted = centroid + rho * (x_reflected - centroid)
                    x_contracted = np.clip(x_contracted, 0, 1)
                    f_contracted = evaluate_solution(x_contracted, iteration)
                    eval_count += 1

                    if f_contracted >= f_reflected:
                        simplex[-1] = x_contracted
                        fitness[-1] = f_contracted
                        operations['contract_out'] += 1
                    else:
                        # Shrink
                        self._shrink_simplex(simplex, fitness, sigma, evaluate_solution, iteration)
                        eval_count += n_params
                        operations['shrink'] += 1
                else:
                    # Inside contraction
                    x_contracted = centroid - rho * (centroid - simplex[-1])
                    x_contracted = np.clip(x_contracted, 0, 1)
                    f_contracted = evaluate_solution(x_contracted, iteration)
                    eval_count += 1

                    if f_contracted > f_worst:
                        simplex[-1] = x_contracted
                        fitness[-1] = f_contracted
                        operations['contract_in'] += 1
                    else:
                        # Shrink
                        self._shrink_simplex(simplex, fitness, sigma, evaluate_solution, iteration)
                        eval_count += n_params
                        operations['shrink'] += 1

            # Update best after operations
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fit:
                best_fit = fitness[current_best_idx]
                best_pos = simplex[current_best_idx].copy()
                n_improvements += 1

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict, {'eval_count': eval_count})
            update_best(best_fit, params_dict, iteration)

            # Progress line (tracker throttles emission); Improved counts
            # cumulative simplex-best improvements over iterations so far.
            log_progress(
                self.name, iteration, best_fit,
                n_improved=n_improvements, pop_size=iteration,
                unit='evals'
            )

            # Check convergence
            # Simplex size (spread of vertices)
            simplex_spread = np.max(np.std(simplex, axis=0))
            fitness_spread = np.std(fitness)

            if simplex_spread < x_tol and fitness_spread < f_tol:
                self.logger.info(
                    f"Nelder-Mead converged at iteration {iteration} "
                    f"(simplex spread: {simplex_spread:.2e}, fitness spread: {fitness_spread:.2e})"
                )
                break

        # Final statistics
        self.logger.info(
            f"Nelder-Mead complete: {eval_count} evaluations, "
            f"operations: reflect={operations['reflect']}, expand={operations['expand']}, "
            f"contract_out={operations['contract_out']}, contract_in={operations['contract_in']}, "
            f"shrink={operations['shrink']}"
        )

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'evaluations': eval_count,
            'operations': operations,
            'final_simplex': simplex,
            'final_fitness': fitness,
        }

    def _shrink_simplex(
        self,
        simplex: np.ndarray,
        fitness: np.ndarray,
        sigma: float,
        evaluate: Callable,
        iteration: int
    ) -> None:
        """
        Shrink the simplex toward the best vertex.

        All vertices except the best are moved toward the best vertex.

        Args:
            simplex: Simplex vertices (n+1, n)
            fitness: Fitness values (n+1,)
            sigma: Shrinkage coefficient
            evaluate: Evaluation function
            iteration: Current iteration
        """
        n_vertices = len(simplex)
        best_vertex = simplex[0].copy()

        for i in range(1, n_vertices):
            simplex[i] = best_vertex + sigma * (simplex[i] - best_vertex)
            simplex[i] = np.clip(simplex[i], 0, 1)
            fitness[i] = evaluate(simplex[i], iteration)
