#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Approximate Bayesian Computation - Sequential Monte Carlo (ABC-SMC)

A likelihood-free inference method for Bayesian parameter estimation when the
likelihood function is intractable or too expensive to compute. Uses simulation
and summary statistics to approximate the posterior distribution.

Key Features:
    - Likelihood-free inference via simulation
    - Provides posterior distribution, not just point estimates
    - Handles complex, non-linear models
    - Sequential Monte Carlo for efficiency with adaptive tolerance
    - Proper importance sampling weights
    - Effective sample size monitoring with adaptive resampling

Reference:
    Sisson, S.A., Fan, Y., and Beaumont, M.A. (2018). Handbook of Approximate
    Bayesian Computation. Chapman and Hall/CRC.

    Beaumont, M.A., Cornuet, J.M., Marin, J.M., and Robert, C.P. (2009).
    Adaptive approximate Bayesian computation. Biometrika, 96(4), 983-990.

    Toni, T., Welch, D., Strelkowa, N., Ipsen, A., and Stumpf, M.P. (2009).
    Approximate Bayesian computation scheme for parameter inference and model
    selection in dynamical systems. Journal of the Royal Society Interface,
    6(31), 187-202.

    Del Moral, P., Doucet, A., and Jasra, A. (2012). An adaptive sequential
    Monte Carlo method for approximate Bayesian computation. Statistics and
    Computing, 22(5), 1009-1020.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .base_algorithm import OptimizationAlgorithm


class ABCAlgorithm(OptimizationAlgorithm):
    """
    Approximate Bayesian Computation with Sequential Monte Carlo.

    This implementation uses an adaptive ABC-SMC algorithm with:
    - Adaptive tolerance scheduling (geometric decay with quantile-based adaptation)
    - Proper importance sampling weights
    - Effective sample size (ESS) monitoring
    - Optimal Local Covariance Matrix (OLCM) perturbation kernel
    - Convergence monitoring based on tolerance improvement
    """

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "ABC"

    def _get_abc_config(self) -> Dict[str, Any]:
        """
        Get ABC configuration from typed config or dict.

        Returns:
            Dictionary of ABC configuration values
        """
        # Try typed config first
        try:
            abc_cfg = self.config.optimization.abc
            return {
                'n_particles': abc_cfg.n_particles,
                'n_generations': abc_cfg.n_generations,
                'initial_tolerance': abc_cfg.initial_tolerance,
                'final_tolerance': abc_cfg.final_tolerance,
                'tolerance_quantile': abc_cfg.tolerance_quantile,
                'tolerance_decay': abc_cfg.tolerance_decay,
                'perturbation_scale': abc_cfg.perturbation_scale,
                'kernel_type': abc_cfg.kernel_type,
                'use_olcm': abc_cfg.use_olcm,
                'min_acceptance_rate': abc_cfg.min_acceptance_rate,
                'min_ess_ratio': abc_cfg.min_ess_ratio,
                'convergence_threshold': abc_cfg.convergence_threshold,
                'min_generations': abc_cfg.min_generations,
            }
        except (AttributeError, TypeError):
            # Fall back to defaults (ABCConfig Pydantic defaults)
            return {
                'n_particles': self._get_config_value(
                    lambda: self.config.optimization.abc.n_particles,
                    dict_key='ABC_PARTICLES',
                ),
                'n_generations': self._get_config_value(
                    lambda: self.config.optimization.abc.n_generations,
                    dict_key='ABC_GENERATIONS',
                ),
                'initial_tolerance': self._get_config_value(
                    lambda: self.config.optimization.abc.initial_tolerance,
                    dict_key='ABC_INITIAL_TOLERANCE',
                ),
                'final_tolerance': self._get_config_value(
                    lambda: self.config.optimization.abc.final_tolerance,
                    dict_key='ABC_FINAL_TOLERANCE',
                ),
                'tolerance_quantile': self._get_config_value(
                    lambda: self.config.optimization.abc.tolerance_quantile,
                    dict_key='ABC_TOLERANCE_QUANTILE',
                ),
                'tolerance_decay': self._get_config_value(
                    lambda: self.config.optimization.abc.tolerance_decay,
                    dict_key='ABC_TOLERANCE_DECAY',
                ),
                'perturbation_scale': self._get_config_value(
                    lambda: self.config.optimization.abc.perturbation_scale,
                    dict_key='ABC_PERTURBATION_SCALE',
                ),
                'kernel_type': self._get_config_value(
                    lambda: self.config.optimization.abc.kernel_type,
                    dict_key='ABC_KERNEL_TYPE',
                ),
                'use_olcm': self._get_config_value(
                    lambda: self.config.optimization.abc.use_olcm,
                    dict_key='ABC_USE_OLCM',
                ),
                'min_acceptance_rate': self._get_config_value(
                    lambda: self.config.optimization.abc.min_acceptance_rate,
                    dict_key='ABC_MIN_ACCEPTANCE_RATE',
                ),
                'min_ess_ratio': self._get_config_value(
                    lambda: self.config.optimization.abc.min_ess_ratio,
                    dict_key='ABC_MIN_ESS_RATIO',
                ),
                'convergence_threshold': self._get_config_value(
                    lambda: self.config.optimization.abc.convergence_threshold,
                    dict_key='ABC_CONVERGENCE_THRESHOLD',
                ),
                'min_generations': self._get_config_value(
                    lambda: self.config.optimization.abc.min_generations,
                    dict_key='ABC_MIN_GENERATIONS',
                ),
            }

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
        Run ABC-SMC optimization.

        The algorithm:
        1. Sample initial population from prior (uniform [0,1])
        2. Evaluate all particles and compute distances
        3. Set initial tolerance based on distance distribution
        4. For each generation:
           a. Accept particles below current tolerance
           b. Compute importance weights
           c. Resample if ESS too low
           d. Perturb particles using adaptive kernel
           e. Reduce tolerance for next generation
        5. Return posterior samples and best solution

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
            Optimization results dictionary with posterior samples
        """
        # Per-run RNG (OptimizationAlgorithm._new_rng): never the process-global stream.
        self._rng = self._new_rng()
        self.logger.info(f"Starting ABC-SMC with {n_params} parameters")

        # Get ABC configuration
        cfg = self._get_abc_config()
        n_particles = cfg['n_particles']
        n_generations = cfg['n_generations']
        initial_tolerance = cfg['initial_tolerance']
        final_tolerance = cfg['final_tolerance']
        tolerance_quantile = cfg['tolerance_quantile']
        tolerance_decay = cfg['tolerance_decay']
        perturbation_scale = cfg['perturbation_scale']
        kernel_type = cfg['kernel_type']
        use_olcm = cfg['use_olcm']
        min_acceptance_rate = cfg['min_acceptance_rate']
        min_ess_ratio = cfg['min_ess_ratio']
        convergence_threshold = cfg['convergence_threshold']
        min_generations = cfg['min_generations']

        self.logger.info(
            f"ABC-SMC settings: {n_particles} particles, {n_generations} max generations, "
            f"tolerance {initial_tolerance:.3f} -> {final_tolerance:.3f}"
        )

        # Initialize particles from prior (uniform on [0,1]^d)
        particles = self._rng.uniform(0, 1, (n_particles, n_params))
        weights = np.ones(n_particles) / n_particles

        # Evaluate initial population
        fitness = evaluate_population(particles, 0)
        distances = self._fitness_to_distance(fitness)

        # Track best solution
        best_idx = np.argmin(distances)
        best_pos = particles[best_idx].copy()
        best_fit = fitness[best_idx]

        # Initialize tolerance based on distance distribution
        finite_distances = distances[np.isfinite(distances)]
        if len(finite_distances) > 0:
            # Start with tolerance that accepts most particles
            # Use high quantile (e.g., 90th percentile) of distances
            tolerance = np.percentile(finite_distances, 95)
            # Cap at initial_tolerance if specified
            tolerance = min(tolerance, initial_tolerance * np.median(finite_distances) / 0.5)
            self.logger.info(
                f"Initial distance distribution: min={np.min(finite_distances):.4f}, "
                f"median={np.median(finite_distances):.4f}, max={np.max(finite_distances):.4f}"
            )
            self.logger.info(f"Starting tolerance: {tolerance:.4f}")
        else:
            tolerance = initial_tolerance
            self.logger.warning("All initial distances are infinite, using default tolerance")

        # Storage for tracking
        tolerance_history = [tolerance]
        best_score_history = [best_fit]
        acceptance_history = []
        ess_history = []

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict, {
            'tolerance': tolerance,
            'acceptance_rate': 1.0,
            'n_accepted': n_particles,
            'ess': n_particles
        })
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_particles, best_fit)

        # ABC-SMC main loop
        iteration = 0
        prev_tolerance = tolerance

        for gen in range(n_generations):
            iteration += 1

            # Compute new tolerance using quantile-based adaptive schedule
            if gen > 0:
                tolerance = self._compute_next_tolerance(
                    distances, weights, tolerance,
                    tolerance_quantile, tolerance_decay, final_tolerance
                )
                tolerance_history.append(tolerance)

            # Accept particles below tolerance
            accepted_mask = distances <= tolerance
            n_accepted = np.sum(accepted_mask)

            if n_accepted == 0:
                # Fallback: increase tolerance to accept at least some particles
                self.logger.warning(
                    f"Gen {gen+1}: No particles below tolerance {tolerance:.4f}, "
                    "increasing tolerance"
                )
                finite_dists = distances[np.isfinite(distances)]
                if len(finite_dists) > 0:
                    tolerance = np.percentile(finite_dists, 90)
                    accepted_mask = distances <= tolerance
                    n_accepted = np.sum(accepted_mask)
                else:
                    # Accept all if no finite distances
                    accepted_mask = np.ones(n_particles, dtype=bool)
                    n_accepted = n_particles

            acceptance_rate = n_accepted / n_particles
            acceptance_history.append(acceptance_rate)

            # Update importance weights for accepted particles
            if gen > 0 and hasattr(self, '_prev_particles') and hasattr(self, '_kernel_cov'):
                weights = self._update_weights(
                    particles, self._prev_particles, self._prev_weights,  # type: ignore[has-type]
                    self._kernel_cov, accepted_mask  # type: ignore[has-type]
                )
            else:
                # First generation: uniform weights for accepted
                weights = np.zeros(n_particles)
                weights[accepted_mask] = 1.0 / n_accepted

            # Normalize weights
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(n_particles) / n_particles

            # Compute effective sample size
            ess = self._compute_ess(weights)
            ess_history.append(ess)

            # Update best from this generation
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fit:
                best_fit = fitness[gen_best_idx]
                best_pos = particles[gen_best_idx].copy()

            # Log progress
            self.logger.info(
                f"Gen {gen+1}/{n_generations}: tolerance={tolerance:.4f}, "
                f"accepted={n_accepted}/{n_particles} ({acceptance_rate:.1%}), "
                f"ESS={ess:.1f}, best_score={best_fit:.4f}"
            )

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(
                iteration, best_fit, params_dict,
                {
                    'tolerance': tolerance,
                    'acceptance_rate': acceptance_rate,
                    'n_accepted': n_accepted,
                    'ess': ess,
                    'generation': gen + 1
                }
            )
            update_best(best_fit, params_dict, iteration)
            log_progress(
                self.name, iteration, best_fit, n_accepted, n_particles,
                unit='gens', total=n_generations
            )
            best_score_history.append(best_fit)

            # Check stopping conditions
            if gen >= min_generations:
                # Check if tolerance has converged
                tol_improvement = (prev_tolerance - tolerance) / prev_tolerance if prev_tolerance > 0 else 0
                if tolerance <= final_tolerance:
                    self.logger.info(
                        f"Reached final tolerance {tolerance:.4f} <= {final_tolerance:.4f}"
                    )
                    break
                if tol_improvement < convergence_threshold:
                    self.logger.info(
                        f"Tolerance converged (improvement {tol_improvement:.4f} < {convergence_threshold})"
                    )
                    break
                if acceptance_rate < min_acceptance_rate:
                    self.logger.info(
                        f"Acceptance rate {acceptance_rate:.3f} below minimum {min_acceptance_rate}"
                    )
                    break

            prev_tolerance = tolerance

            # Skip resampling/perturbation on last iteration
            if gen >= n_generations - 1:
                break

            # Resample if ESS is too low
            min_ess = min_ess_ratio * n_particles
            if ess < min_ess:
                self.logger.debug(f"Resampling: ESS {ess:.1f} < {min_ess:.1f}")
                particles, weights = self._resample(particles, weights)

            # Store state for weight computation in next generation
            self._prev_particles = particles.copy()
            self._prev_weights = weights.copy()

            # Compute perturbation kernel covariance
            if use_olcm and n_accepted > n_params:
                # Use Optimal Local Covariance Matrix
                self._kernel_cov = self._compute_olcm(
                    particles, weights, perturbation_scale
                )
            else:
                # Use scaled identity or component-wise variance
                particle_std = np.std(particles, axis=0)
                particle_std[particle_std < 1e-10] = 0.1
                self._kernel_cov = np.diag((perturbation_scale * particle_std) ** 2)

            # Perturb particles
            particles = self._perturb_particles(
                particles, self._kernel_cov, kernel_type
            )

            # Evaluate new population
            fitness = evaluate_population(particles, iteration)
            distances = self._fitness_to_distance(fitness)

            # Update global best
            new_best_idx = np.argmax(fitness)
            if fitness[new_best_idx] > best_fit:
                best_fit = fitness[new_best_idx]
                best_pos = particles[new_best_idx].copy()

        # Compile posterior from final accepted particles
        accepted_mask = distances <= tolerance
        if np.sum(accepted_mask) > 0:
            posterior_particles = particles[accepted_mask]
            posterior_weights = weights[accepted_mask]
            posterior_weights = posterior_weights / np.sum(posterior_weights)
        else:
            # Use all particles weighted by distance
            posterior_particles = particles
            posterior_weights = 1.0 / (distances + 1e-10)
            posterior_weights = posterior_weights / np.sum(posterior_weights)

        # Compute posterior statistics
        posterior_mean = np.average(posterior_particles, weights=posterior_weights, axis=0)
        posterior_std = np.sqrt(np.average(
            (posterior_particles - posterior_mean) ** 2,
            weights=posterior_weights,
            axis=0
        ))

        # Compute credible intervals (2.5% and 97.5% quantiles)
        credible_intervals = self._compute_credible_intervals(
            posterior_particles, posterior_weights
        )

        self.logger.info(
            f"ABC-SMC complete. Generations: {iteration}, Final tolerance: {tolerance:.4f}, "
            f"Posterior samples: {len(posterior_particles)}, Best score: {best_fit:.4f}"
        )

        # Clean up temporary state
        if hasattr(self, '_prev_particles'):
            del self._prev_particles
        if hasattr(self, '_prev_weights'):
            del self._prev_weights
        if hasattr(self, '_kernel_cov'):
            del self._kernel_cov

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'posterior_samples': posterior_particles,
            'posterior_weights': posterior_weights,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_intervals': credible_intervals,
            'final_tolerance': tolerance,
            'n_generations': iteration,
            'tolerance_history': tolerance_history,
            'best_score_history': best_score_history,
            'acceptance_history': acceptance_history,
            'ess_history': ess_history,
        }

    def _fitness_to_distance(self, fitness: np.ndarray) -> np.ndarray:
        """
        Convert fitness scores to distances for ABC.

        For KGE and similar metrics where 1 is optimal:
        distance = 1 - fitness

        Args:
            fitness: Array of fitness scores

        Returns:
            Array of distances (lower = better)
        """
        distances = 1.0 - fitness
        # Handle invalid values
        distances[~np.isfinite(fitness)] = np.inf
        # Ensure non-negative (KGE can be very negative)
        return np.maximum(distances, 0)

    def _compute_next_tolerance(
        self,
        distances: np.ndarray,
        weights: np.ndarray,
        current_tolerance: float,
        quantile: float,
        decay: float,
        final_tolerance: float
    ) -> float:
        """
        Compute next tolerance using adaptive schedule.

        Uses weighted quantile of distances, bounded by geometric decay.

        Args:
            distances: Current distances
            weights: Particle weights
            current_tolerance: Current tolerance
            quantile: Quantile for new tolerance (e.g., 0.75)
            decay: Maximum decay factor (e.g., 0.9)
            final_tolerance: Minimum allowed tolerance

        Returns:
            New tolerance value
        """
        # Get finite distances below current tolerance
        valid = np.isfinite(distances) & (distances <= current_tolerance)
        if np.sum(valid) < 2:
            # Not enough valid particles, use geometric decay
            return max(current_tolerance * decay, final_tolerance)

        valid_distances = distances[valid]
        valid_weights = weights[valid]
        valid_weights = valid_weights / np.sum(valid_weights)

        # Compute weighted quantile
        sorted_idx = np.argsort(valid_distances)
        sorted_distances = valid_distances[sorted_idx]
        sorted_weights = valid_weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)

        # Find index where cumulative weight exceeds quantile
        quantile_idx = np.searchsorted(cumsum, quantile)
        quantile_idx = min(int(quantile_idx), len(sorted_distances) - 1)
        new_tolerance = sorted_distances[quantile_idx]

        # Bound by geometric decay and final tolerance
        max_tolerance = current_tolerance * decay
        new_tolerance = max(min(new_tolerance, max_tolerance), final_tolerance)

        return new_tolerance

    def _compute_ess(self, weights: np.ndarray) -> float:
        """
        Compute effective sample size from weights.

        ESS = 1 / sum(w_i^2) where weights are normalized.

        Args:
            weights: Normalized particle weights

        Returns:
            Effective sample size
        """
        weights = weights / np.sum(weights)
        return 1.0 / np.sum(weights ** 2)

    def _resample(
        self,
        particles: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample particles using systematic resampling.

        Systematic resampling is more efficient than multinomial
        and produces lower variance.

        Args:
            particles: Current particles
            weights: Current weights

        Returns:
            Resampled particles and uniform weights
        """
        n = len(particles)
        weights = weights / np.sum(weights)

        # Systematic resampling
        positions = (np.arange(n) + self._rng.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        return particles[indices].copy(), np.ones(n) / n

    def _update_weights(
        self,
        particles: np.ndarray,
        prev_particles: np.ndarray,
        prev_weights: np.ndarray,
        kernel_cov: np.ndarray,
        accepted_mask: np.ndarray
    ) -> np.ndarray:
        """
        Update importance weights after perturbation.

        For uniform prior, new weights are:
        w_i ∝ π(θ_i) / Σ_j w_j K(θ_i | θ_j)

        where π is prior (uniform) and K is perturbation kernel.

        Args:
            particles: New particles
            prev_particles: Previous particles
            prev_weights: Previous weights
            kernel_cov: Kernel covariance matrix
            accepted_mask: Which particles were accepted

        Returns:
            Updated weights
        """
        n = len(particles)
        weights = np.zeros(n)

        # Prior is uniform on [0,1]^d, so π(θ) = 1 if in bounds
        in_bounds = np.all((particles >= 0) & (particles <= 1), axis=1)

        # Compute kernel density for each new particle
        try:
            kernel_cov_reg = kernel_cov + 1e-6 * np.eye(kernel_cov.shape[0])
            kernel_inv = np.linalg.inv(kernel_cov_reg)
            kernel_det = np.linalg.det(kernel_cov_reg)
            d = particles.shape[1]

            for i in range(n):
                if not in_bounds[i]:
                    weights[i] = 0
                    continue

                # Compute sum of kernel evaluations from previous particles
                diffs = particles[i] - prev_particles
                # Mahalanobis distances
                mahal = np.sum(diffs @ kernel_inv * diffs, axis=1)
                # Gaussian kernel values
                kernel_vals = np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** d * kernel_det)
                # Weighted sum
                denominator = np.sum(prev_weights * kernel_vals)

                if denominator > 1e-300:
                    # Prior / weighted kernel sum (prior = 1 for uniform)
                    weights[i] = 1.0 / denominator
                else:
                    weights[i] = 0

        except np.linalg.LinAlgError:
            # Fallback to uniform weights if covariance is singular
            weights = np.ones(n)

        # Zero out rejected particles
        weights[~accepted_mask] = 0

        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback: uniform over accepted
            weights = np.zeros(n)
            weights[accepted_mask] = 1.0 / np.sum(accepted_mask)

        return weights

    def _compute_olcm(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        Compute Optimal Local Covariance Matrix for perturbation kernel.

        Uses weighted covariance of particles with optimal scaling.

        Reference:
            Filippi et al. (2013). On optimality of kernels for approximate
            Bayesian computation using sequential Monte Carlo.

        Args:
            particles: Current particles
            weights: Particle weights
            scale: Additional scaling factor

        Returns:
            Covariance matrix for perturbation kernel
        """
        n, d = particles.shape
        weights = weights / np.sum(weights)

        # Weighted mean
        mean = np.average(particles, weights=weights, axis=0)

        # Weighted covariance
        diffs = particles - mean
        cov = np.zeros((d, d))
        for i in range(n):
            cov += weights[i] * np.outer(diffs[i], diffs[i])

        # Optimal scaling for Gaussian kernel: (2 * d + 1)^(-1) ≈ 2
        # Combined with user scale
        optimal_scale = scale * 2.0

        # Regularize if needed
        min_var = 1e-6
        diag = np.diag(cov).copy()  # Copy to avoid read-only view issues
        diag[diag < min_var] = min_var
        np.fill_diagonal(cov, diag)

        return optimal_scale * cov

    def _perturb_particles(
        self,
        particles: np.ndarray,
        kernel_cov: np.ndarray,
        kernel_type: str
    ) -> np.ndarray:
        """
        Perturb particles using specified kernel type.

        Args:
            particles: Particles to perturb
            kernel_cov: Kernel covariance matrix
            kernel_type: Type of kernel ('gaussian', 'uniform', 'component_wise')

        Returns:
            Perturbed particles clipped to [0, 1]
        """
        n, d = particles.shape

        if kernel_type == 'gaussian':
            # Full multivariate Gaussian
            try:
                noise = self._rng.multivariate_normal(np.zeros(d), kernel_cov, size=n)
            except np.linalg.LinAlgError:
                # Fallback to component-wise
                std = np.sqrt(np.diag(kernel_cov))
                noise = self._rng.normal(0, 1, (n, d)) * std
        elif kernel_type == 'uniform':
            # Uniform box kernel
            std = np.sqrt(np.diag(kernel_cov))
            half_width = std * np.sqrt(3)  # Match variance
            noise = self._rng.uniform(-half_width, half_width, (n, d))
        else:  # component_wise (default)
            # Independent Gaussian for each component
            std = np.sqrt(np.diag(kernel_cov))
            noise = self._rng.normal(0, 1, (n, d)) * std

        perturbed = particles + noise
        return np.clip(perturbed, 0, 1)

    def _compute_credible_intervals(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute credible intervals for each parameter.

        Args:
            particles: Posterior samples
            weights: Sample weights
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Dict with 'lower', 'upper', and 'median' arrays
        """
        n_params = particles.shape[1]
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)
        median = np.zeros(n_params)

        weights = weights / np.sum(weights)

        for i in range(n_params):
            values = particles[:, i]
            sorted_idx = np.argsort(values)
            sorted_values = values[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)

            lower[i] = sorted_values[np.searchsorted(cumsum, alpha / 2)]
            upper[i] = sorted_values[np.searchsorted(cumsum, 1 - alpha / 2)]
            median[i] = sorted_values[np.searchsorted(cumsum, 0.5)]

        return {
            'lower': lower,
            'upper': upper,
            'median': median,
            'alpha': alpha
        }
