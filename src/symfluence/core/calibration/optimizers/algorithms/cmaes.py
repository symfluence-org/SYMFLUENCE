#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

A state-of-the-art evolutionary algorithm for derivative-free optimization.
CMA-ES adapts the covariance matrix of a multivariate normal distribution
to efficiently search the parameter space, making it highly effective for
ill-conditioned and non-separable optimization problems.

Key Features:
    - Self-adaptive step size and search direction
    - Handles parameter correlations through covariance matrix adaptation
    - Invariant to rotation and scaling of the search space
    - No user-defined parameters except population size (robust defaults)

Reference:
    Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review.
    In Towards a New Evolutionary Computation, pp. 75-102.

    Hansen, N. and Ostermeier, A. (2001). Completely Derandomized Self-Adaptation
    in Evolution Strategies. Evolutionary Computation, 9(2), 159-195.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import CMAESDefaults


class CMAESAlgorithm(OptimizationAlgorithm):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "CMA-ES"

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
        Run CMA-ES optimization.

        CMA-ES maintains a multivariate normal distribution N(m, sigma^2 C) where:
        - m is the mean (current best estimate)
        - sigma is the global step size
        - C is the covariance matrix (adapted to capture parameter correlations)

        The algorithm iteratively:
        1. Samples lambda candidate solutions from N(m, sigma^2 C)
        2. Evaluates and ranks candidates by fitness
        3. Updates m toward weighted mean of best mu solutions
        4. Adapts sigma using cumulative step-size adaptation (CSA)
        5. Adapts C using rank-mu and rank-one updates

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
        self.logger.info(f"Starting CMA-ES optimization with {n_params} parameters")

        # Population size (lambda) - default heuristic from Hansen (2006)
        # lambda = 4 + floor(3 * ln(n)) provides good exploration-exploitation balance
        # See CMAESDefaults.compute_population_size for formula derivation
        lambda_ = self.population_size
        if lambda_ < CMAESDefaults.MIN_POPULATION:
            lambda_ = CMAESDefaults.compute_population_size(n_params)
            self.logger.info(f"Adjusted population size to {lambda_} for {n_params} parameters")

        # Number of parents (mu) - typically lambda/2
        mu = lambda_ // 2

        # Recombination weights (log-linear weighting)
        # Weights are proportional to log-rank, giving more influence to better solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()  # Variance-effective selection mass

        # Strategy parameter defaults from Hansen (2006), Table 1
        # These formulas are derived from theoretical analysis and empirical tuning
        strategy_params = CMAESDefaults.compute_strategy_parameters(n_params, mu, mu_eff)
        c_sigma = strategy_params['c_sigma']  # Step-size adaptation learning rate
        d_sigma = strategy_params['d_sigma']  # Step-size damping factor
        c_c = strategy_params['c_c']          # Covariance path learning rate
        c_1 = strategy_params['c_1']          # Rank-one update learning rate
        c_mu = strategy_params['c_mu']        # Rank-mu update learning rate
        chi_n = strategy_params['chi_n']      # Expected ||N(0,I)||

        # Initialize state
        # Mean at center of normalized space [0, 1]
        mean = self._start_point(n_params, kwargs, np.full(n_params, 0.5))

        sigma = self._get_config_value(
            lambda: self.config.optimization.cmaes.initial_sigma,
            default=CMAESDefaults.INITIAL_SIGMA,
            dict_key='CMAES_INITIAL_SIGMA'
        )

        # Covariance matrix (identity initially)
        C = np.eye(n_params)

        # Evolution paths
        p_sigma = np.zeros(n_params)  # Step-size evolution path
        p_c = np.zeros(n_params)  # Covariance matrix evolution path

        # Eigendecomposition (for sampling)
        B = np.eye(n_params)  # Eigenvectors
        D = np.ones(n_params)  # Sqrt of eigenvalues
        invsqrt_C = np.eye(n_params)
        eigeneval = 0  # Last eigendecomposition count

        # Track best solution
        best_pos = mean.copy()
        best_fit = float('-inf')
        prev_best_fit = float('-inf')  # Track previous generation's best for n_improved
        eval_count = 0

        # Main optimization loop
        for generation in range(1, self.max_iterations + 1):
            # Generate lambda offspring by sampling from N(mean, sigma^2 C)
            # x_k = mean + sigma * B * D * z_k, where z_k ~ N(0, I)
            population = np.zeros((lambda_, n_params))
            z_samples = np.zeros((lambda_, n_params))

            for k in range(lambda_):
                z = self._rng.standard_normal(n_params)
                z_samples[k] = z
                y = B @ (D * z)  # Transform by sqrt(C)
                x = mean + sigma * y
                # Clip to bounds [0, 1]
                population[k] = np.clip(x, 0, 1)

            # Evaluate population
            fitness = evaluate_population(population, generation)
            eval_count += lambda_

            # Handle NaN/Inf fitness values
            invalid_mask = ~np.isfinite(fitness)
            if invalid_mask.any():
                self.logger.warning(
                    f"Generation {generation}: {invalid_mask.sum()} solutions "
                    f"returned invalid fitness, assigning penalty"
                )
                fitness[invalid_mask] = float('-inf')

            # Sort by fitness (descending - we maximize)
            sorted_indices = np.argsort(-fitness)

            # Update best
            if fitness[sorted_indices[0]] > best_fit:
                best_fit = fitness[sorted_indices[0]]
                best_pos = population[sorted_indices[0]].copy()

            # Select mu best individuals
            selected_indices = sorted_indices[:mu]

            # Compute weighted mean of selected points
            old_mean = mean.copy()
            mean = np.zeros(n_params)
            for i, idx in enumerate(selected_indices):
                mean += weights[i] * population[idx]
            mean = np.clip(mean, 0, 1)

            # Update evolution paths with error handling
            try:
                # p_sigma: cumulative step-size adaptation path
                # This path accumulates normalized steps to detect if the algorithm
                # is making progress in a consistent direction
                mean_shift = (mean - old_mean) / sigma
                p_sigma = ((1 - c_sigma) * p_sigma +
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrt_C @ mean_shift))

                # Check for numerical issues in evolution path
                if not np.all(np.isfinite(p_sigma)):
                    self.logger.warning("Evolution path p_sigma contains invalid values, resetting")
                    p_sigma = np.zeros(n_params)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
                self.logger.warning(f"Error updating p_sigma: {e}, resetting")
                p_sigma = np.zeros(n_params)

            # Heaviside function for stalling detection
            # h_sigma = 1 unless ||p_sigma|| is unexpectedly large (indicates stalling)
            p_sigma_norm = np.linalg.norm(p_sigma)
            expectation_factor = np.sqrt(1 - (1 - c_sigma) ** (2 * generation))
            threshold = (1.4 + 2 / (n_params + 1)) * chi_n
            h_sigma = 1 if p_sigma_norm / expectation_factor < threshold else 0

            # p_c: covariance matrix adaptation path
            try:
                p_c = ((1 - c_c) * p_c +
                      h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_shift)

                if not np.all(np.isfinite(p_c)):
                    self.logger.warning("Evolution path p_c contains invalid values, resetting")
                    p_c = np.zeros(n_params)
            except (ValueError, FloatingPointError) as e:
                self.logger.warning(f"Error updating p_c: {e}, resetting")
                p_c = np.zeros(n_params)

            # Adapt covariance matrix C with error handling
            try:
                # Rank-one update: captures correlation from evolution path
                rank_one = np.outer(p_c, p_c)

                # Rank-mu update: uses information from all mu best samples
                rank_mu = np.zeros((n_params, n_params))
                for i, idx in enumerate(selected_indices):
                    y_i = (population[idx] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                # Combined update with learning rates c_1 and c_mu
                # The (1 - h_sigma) term adds extra variance when stalling is detected
                C = ((1 - c_1 - c_mu) * C +
                     c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) +
                     c_mu * rank_mu)

                # Enforce symmetry (numerical stability)
                C = (C + C.T) / 2

                # Check for invalid covariance matrix
                if not np.all(np.isfinite(C)):
                    raise ValueError("Covariance matrix contains invalid values")

            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
                self.logger.warning(f"Error updating covariance matrix: {e}, resetting to identity")
                C = np.eye(n_params)
                p_c = np.zeros(n_params)

            # Adapt step size sigma using cumulative step-size adaptation (CSA)
            # sigma increases if ||p_sigma|| > chi_n (making good progress)
            # sigma decreases if ||p_sigma|| < chi_n (overshooting or stalling)
            sigma = sigma * np.exp((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1))

            # Bound sigma to prevent explosion/collapse
            # sigma_min = 1e-10 signals convergence, sigma_max = 1.0 prevents overshooting
            # See CMAESDefaults.SIGMA_MIN and SIGMA_MAX for rationale
            sigma = max(CMAESDefaults.SIGMA_MIN, min(sigma, CMAESDefaults.SIGMA_MAX))

            # Update eigendecomposition periodically for sampling efficiency
            # Frequency based on adaptation rates to balance accuracy and cost
            if eval_count - eigeneval > lambda_ / (c_1 + c_mu) / n_params / 10:
                eigeneval = eval_count
                try:
                    # Ensure C is positive definite
                    C = (C + C.T) / 2
                    eigenvalues, B = np.linalg.eigh(C)
                    # Floor eigenvalues to prevent negative values from numerical errors
                    eigenvalues = np.maximum(eigenvalues, CMAESDefaults.EIGENVALUE_FLOOR)
                    D = np.sqrt(eigenvalues)
                    invsqrt_C = B @ np.diag(1 / D) @ B.T
                except np.linalg.LinAlgError as e:
                    self.logger.warning(f"Eigendecomposition failed: {e}, resetting covariance matrix")
                    C = np.eye(n_params)
                    B = np.eye(n_params)
                    D = np.ones(n_params)
                    invsqrt_C = np.eye(n_params)

            # Record iteration
            params_dict = denormalize_params(best_pos)
            # Count individuals that beat previous generation's best (meaningful improvement metric)
            n_improved = int(np.sum(fitness > prev_best_fit))
            record_iteration(generation, best_fit, params_dict, {'sigma': sigma, 'n_improved': n_improved})
            update_best(best_fit, params_dict, generation)

            # Progress line (tracker throttles emission)
            log_progress(self.name, generation, best_fit, n_improved, lambda_, unit='gens')

            # Update previous best for next iteration
            prev_best_fit = best_fit

            # Early stopping: sigma has collapsed onto its floor, i.e. the search
            # distribution has shrunk to a point (steps of 1e-10 in a unit box)
            # and further generations re-evaluate the same parameter set.
            if sigma <= CMAESDefaults.CONVERGENCE_THRESHOLD:
                self.logger.info(
                    f"CMA-ES converged at generation {generation} "
                    f"(sigma <= {CMAESDefaults.CONVERGENCE_THRESHOLD:g})"
                )
                break

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'final_sigma': sigma,
            'evaluations': eval_count
        }
