#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Bayesian Optimization

A surrogate-based global optimization method that uses a probabilistic model
(Gaussian Process) to guide the search. Particularly effective for expensive
objective functions where each evaluation is costly.

Key Features:
    - Builds a surrogate model of the objective function
    - Uses acquisition functions to balance exploration vs exploitation
    - Efficient for expensive-to-evaluate objectives
    - Provides uncertainty estimates

Reference:
    Snoek, J., Larochelle, H., and Adams, R.P. (2012). Practical Bayesian
    Optimization of Machine Learning Algorithms. NIPS.

    Jones, D.R., Schonlau, M., and Welch, W.J. (1998). Efficient Global
    Optimization of Expensive Black-Box Functions. Journal of Global
    Optimization, 13(4), 455-492.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from symfluence.core.metrics.penalty import is_penalty

from .base_algorithm import OptimizationAlgorithm
from .config_schema import BODefaults


class BayesianOptimizationAlgorithm(OptimizationAlgorithm):
    """Bayesian Optimization using Gaussian Process surrogate."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "BAYESIAN-OPT"

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
        Run Bayesian Optimization.

        The algorithm:
        1. Starts with initial random samples to build surrogate
        2. Fits a Gaussian Process (GP) to observed data
        3. Uses acquisition function to select next point
        4. Evaluates objective and updates GP
        5. Repeats until budget exhausted

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        # Per-run RNG (OptimizationAlgorithm._new_rng): never the process-global stream.
        self._rng = self._new_rng()
        self.logger.info(f"Starting Bayesian Optimization with {n_params} parameters")

        # BO parameters from config
        n_initial = max(5, self._get_config_value(
            lambda: self.config.optimization.bayesian_opt.initial_samples_factor,
            default=BODefaults.INITIAL_SAMPLES_FACTOR,
            dict_key='BO_INITIAL_SAMPLES_FACTOR'
        ) * n_params)

        acquisition = self._get_config_value(
            lambda: self.config.optimization.bayesian_opt.acquisition,
            default=BODefaults.ACQUISITION,
            dict_key='BO_ACQUISITION'
        )
        xi = self._get_config_value(
            lambda: self.config.optimization.bayesian_opt.xi,
            default=BODefaults.XI,
            dict_key='BO_XI'
        )
        kappa = self._get_config_value(
            lambda: self.config.optimization.bayesian_opt.kappa,
            default=BODefaults.KAPPA,
            dict_key='BO_KAPPA'
        )
        n_restarts = self._get_config_value(
            lambda: self.config.optimization.bayesian_opt.restarts,
            default=BODefaults.RESTARTS,
            dict_key='BO_RESTARTS'
        )

        # Validate BO configuration
        valid, warning = BODefaults.validate_acquisition(acquisition)
        if not valid:
            self.logger.warning(f"BO acquisition validation: {warning}")

        self.logger.info(
            f"BO settings: {n_initial} initial samples, acquisition={acquisition}, "
            f"xi={xi}, kappa={kappa}"
        )

        # Storage for observations
        X_observed: List[np.ndarray] = []
        y_observed: List[float] = []

        # Initial sampling (Latin Hypercube for better coverage)
        initial_samples = self._latin_hypercube_sample(n_initial, n_params)
        initial_fitness = evaluate_population(initial_samples, 0)

        for i in range(n_initial):
            X_observed.append(initial_samples[i])
            y_observed.append(initial_fitness[i])

        # Track best
        best_idx = np.argmax(y_observed)
        best_pos = X_observed[best_idx].copy()
        best_fit = y_observed[best_idx]

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_initial, best_fit)

        # Main BO loop
        n_improvements = 0
        for iteration in range(1, self.max_iterations + 1):
            # Convert to arrays; failed runs are re-valued for the surrogate
            # (see _surrogate_observations).
            X, y = self._surrogate_observations(
                np.array(X_observed), np.array(y_observed, dtype=float)
            )

            gp_params: Dict[str, Any] = {}
            if len(y) == 0:
                # Nothing but failed runs so far: there is no response surface
                # to model, so keep sampling at random until a run succeeds.
                next_point = self._rng.uniform(0, 1, n_params)
            else:
                # Fit GP surrogate
                gp_mean, gp_std, gp_params = self._fit_gp(X, y)

                # Optimize acquisition function to find next point
                next_point = self._optimize_acquisition(
                    gp_mean, gp_std, y, n_params, acquisition, xi, kappa, n_restarts
                )

            # Evaluate new point
            next_fitness = evaluate_solution(next_point, iteration)

            # Update observations
            X_observed.append(next_point)
            y_observed.append(next_fitness)

            # Update best
            if next_fitness > best_fit:
                best_fit = next_fitness
                best_pos = next_point.copy()
                n_improvements += 1

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(
                iteration, best_fit, params_dict,
                {'gp_lengthscale': gp_params.get('lengthscale', 0)}
            )
            update_best(best_fit, params_dict, iteration)

            # Progress line (tracker throttles emission); Improved counts
            # cumulative accepted improvements over evaluations so far.
            log_progress(
                self.name, iteration, best_fit,
                n_improved=n_improvements, pop_size=iteration,
                unit='evals'
            )

        self.logger.info(f"Bayesian Optimization complete: {len(y_observed)} evaluations")

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'X_observed': np.array(X_observed),
            'y_observed': np.array(y_observed),
            'n_evaluations': len(y_observed),
        }

    def _surrogate_observations(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Observations to fit the GP on, with failed runs re-valued.

        A failed run scores exactly the penalty constant (-9999). Fed to the GP
        as it is, it dominates the y normalisation (every valid score collapses
        to the same normalised value) and the surrogate can no longer tell good
        from mediocre. Failed points are kept -- the surrogate should still
        learn to avoid that region -- but valued just below the worst valid
        observation: worst minus the spread of the valid scores. With no valid
        observation yet they are dropped (empty arrays are returned).
        """
        failed = np.asarray(is_penalty(y, self.penalty_score), dtype=bool)
        if not failed.any():
            return X, y
        valid = y[~failed]
        if valid.size == 0:
            return X[:0], y[:0]
        worst = float(valid.min())
        spread = float(valid.max()) - worst
        margin = spread if spread > 0 else 0.1 * max(abs(worst), 1.0)
        y_fit = y.copy()
        y_fit[failed] = worst - margin
        return X, y_fit

    def _latin_hypercube_sample(self, n_samples: int, n_params: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n_samples, n_params))
        for j in range(n_params):
            intervals = np.arange(n_samples)
            self._rng.shuffle(intervals)
            for i in range(n_samples):
                low = intervals[i] / n_samples
                high = (intervals[i] + 1) / n_samples
                samples[i, j] = self._rng.uniform(low, high)
        return samples

    def _fit_gp(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Callable, Callable, Dict]:
        """
        Fit a simple Gaussian Process surrogate.

        Uses RBF kernel with automatic lengthscale estimation.

        Args:
            X: Observed inputs (n_obs, n_params)
            y: Observed outputs (n_obs,)

        Returns:
            Tuple of (mean_function, std_function, hyperparameters)
        """
        n_obs, n_params = X.shape

        # Normalize y for numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_normalized = (y - y_mean) / y_std

        # Estimate lengthscale from data spread
        lengthscale = np.std(X, axis=0) + 1e-8

        # Noise variance (nugget for numerical stability)
        noise_var = 1e-6

        def rbf_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
            """RBF (squared exponential) kernel."""
            # Scaled squared distance
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            scaled_diff = diff / lengthscale
            sq_dist = np.sum(scaled_diff ** 2, axis=2)
            return np.exp(-0.5 * sq_dist)

        # Compute kernel matrix
        K = rbf_kernel(X, X) + noise_var * np.eye(n_obs)

        # Cholesky decomposition for stable inversion
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_normalized))
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if Cholesky fails
            alpha = np.linalg.lstsq(K, y_normalized, rcond=None)[0]
            L = None

        def predict_mean(X_new: np.ndarray) -> np.ndarray:
            """Predict mean at new points."""
            K_star = rbf_kernel(X_new.reshape(-1, n_params), X)
            mean_normalized = K_star @ alpha
            return mean_normalized * y_std + y_mean

        def predict_std(X_new: np.ndarray) -> np.ndarray:
            """Predict standard deviation at new points."""
            X_new = X_new.reshape(-1, n_params)
            K_star = rbf_kernel(X_new, X)
            K_star_star = rbf_kernel(X_new, X_new)

            if L is not None:
                v = np.linalg.solve(L, K_star.T)
                var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
            else:
                var = np.diag(K_star_star) - np.diag(K_star @ np.linalg.lstsq(K, K_star.T, rcond=None)[0])

            var = np.maximum(var, 1e-10)  # Ensure positive
            return np.sqrt(var) * y_std

        params = {'lengthscale': np.mean(lengthscale), 'noise_var': noise_var}
        return predict_mean, predict_std, params

    def _optimize_acquisition(
        self,
        gp_mean: Callable,
        gp_std: Callable,
        y_observed: np.ndarray,
        n_params: int,
        acquisition: str,
        xi: float,
        kappa: float,
        n_restarts: int
    ) -> np.ndarray:
        """
        Optimize acquisition function to find next evaluation point.

        Args:
            gp_mean: GP mean prediction function
            gp_std: GP std prediction function
            y_observed: Observed fitness values
            n_params: Number of parameters
            acquisition: Acquisition function type
            xi: Exploration parameter
            kappa: UCB parameter
            n_restarts: Number of random restarts

        Returns:
            Next point to evaluate
        """
        y_best = np.max(y_observed)

        def acquisition_function(x: np.ndarray) -> float:
            """Compute acquisition value (to maximize)."""
            x = np.clip(x, 0, 1)
            mu = gp_mean(x)[0]
            sigma = gp_std(x)[0]

            if sigma < 1e-10:
                return mu

            if acquisition == 'ei':
                # Expected Improvement
                z = (mu - y_best - xi) / sigma
                ei = sigma * (z * self._norm_cdf(z) + self._norm_pdf(z))
                return ei
            elif acquisition == 'ucb':
                # Upper Confidence Bound
                return mu + kappa * sigma
            elif acquisition == 'pi':
                # Probability of Improvement
                z = (mu - y_best - xi) / sigma
                return self._norm_cdf(z)
            else:
                return mu  # Default to mean

        # Multi-start optimization of acquisition function
        best_acq = float('-inf')
        best_x = self._rng.uniform(0, 1, n_params)

        for _ in range(n_restarts):
            # Random starting point
            x0 = self._rng.uniform(0, 1, n_params)

            # Simple gradient-free optimization (coordinate descent)
            x = x0.copy()
            for _ in range(50):  # Local optimization steps
                for d in range(n_params):
                    # Line search along dimension d
                    best_val = acquisition_function(x)
                    for delta in [-0.1, -0.05, 0.05, 0.1]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x[d] + delta, 0, 1)
                        val = acquisition_function(x_new)
                        if val > best_val:
                            best_val = val
                            x = x_new

            acq_val = acquisition_function(x)
            if acq_val > best_acq:
                best_acq = acq_val
                best_x = x.copy()

        return best_x

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
