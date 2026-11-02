#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
DREAM (DiffeRential Evolution Adaptive Metropolis)

A Markov Chain Monte Carlo (MCMC) algorithm that uses differential evolution
for efficient proposal generation. DREAM is particularly popular in hydrology
for its ability to provide parameter uncertainty estimates alongside optimization.

Key Features:
    - Multiple parallel chains for better exploration
    - Differential evolution-based proposals for efficient jumping
    - Adaptive proposal scaling via self-adaptive randomized subspace sampling
    - Outlier chain detection and correction
    - Provides posterior samples for uncertainty quantification

Reference:
    Vrugt, J.A., ter Braak, C.J.F., Diks, C.G.H., Robinson, B.A., Hyman, J.M.,
    and Higdon, D. (2009). Accelerating Markov chain Monte Carlo simulation by
    differential evolution with self-adaptive randomized subspace sampling.
    International Journal of Nonlinear Sciences and Numerical Simulation, 10(3), 273-290.

    Vrugt, J.A. (2016). Markov chain Monte Carlo simulation using the DREAM
    software package: Theory, concepts, and MATLAB implementation.
    Environmental Modelling & Software, 75, 273-316.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import DREAMDefaults


class DREAMAlgorithm(OptimizationAlgorithm):
    """DREAM (DiffeRential Evolution Adaptive Metropolis) optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "DREAM"

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
        Run DREAM optimization.

        DREAM maintains N parallel Markov chains that explore the parameter space.
        Proposals are generated using differential evolution:
            x_proposal = x_current + gamma * (x_r1 - x_r2) + epsilon

        where gamma is adapted based on acceptance rate and epsilon is small random noise.

        The algorithm uses Metropolis-Hastings acceptance:
            alpha = min(1, p(x_proposal) / p(x_current))

        For optimization, we use likelihood proportional to exp(fitness / T) where T is temperature.

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
            Optimization results dictionary including posterior samples
        """
        self.logger.info(f"Starting DREAM optimization with {n_params} parameters")

        # DREAM parameters from config
        min_chains = DREAMDefaults.compute_min_chains(n_params)
        n_chains = max(min_chains, self.population_size)
        self.logger.info(f"Using {n_chains} chains for DREAM ({n_params} parameters)")

        n_pairs = self._get_config_value(
            lambda: self.config.optimization.dream.de_pairs,
            default=DREAMDefaults.DE_PAIRS,
            dict_key='DREAM_PAIRS'
        )
        n_pairs = min(n_pairs, (n_chains - 1) // 2)
        if n_pairs < 1:
            n_pairs = 1

        CR = self._get_config_value(
            lambda: self.config.optimization.dream.crossover_probability,
            default=DREAMDefaults.CROSSOVER_PROBABILITY,
            dict_key='DREAM_CR'
        )

        d_star = max(1, int(CR * n_params))
        gamma_base = DREAMDefaults.compute_optimal_gamma(n_pairs, d_star)

        eps_std = self._get_config_value(
            lambda: self.config.optimization.dream.epsilon_std,
            default=DREAMDefaults.EPSILON_STD,
            dict_key='DREAM_EPS'
        )

        temperature = self._get_config_value(
            lambda: self.config.optimization.dream.temperature,
            default=DREAMDefaults.TEMPERATURE,
            dict_key='DREAM_TEMPERATURE'
        )

        outlier_threshold = self._get_config_value(
            lambda: self.config.optimization.dream.outlier_threshold,
            default=DREAMDefaults.OUTLIER_THRESHOLD,
            dict_key='DREAM_OUTLIER_THRESHOLD'
        )

        # Initialize chains uniformly in [0, 1]
        chains = np.random.uniform(0, 1, (n_chains, n_params))

        # Evaluate initial chain positions
        fitness = evaluate_population(chains, 0)

        # Convert fitness to log-likelihood (for MCMC)
        # Higher fitness = higher likelihood
        log_likelihood = fitness / temperature

        # Track best solution
        best_idx = np.argmax(fitness)
        best_pos = chains[best_idx].copy()
        best_fit = fitness[best_idx]

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_chains, best_fit)

        # Storage for posterior samples (after burn-in)
        # First 20% of iterations are discarded as burn-in for posterior estimation
        # (Vrugt 2016, Section 3.1 - "Burn-in Period")
        burn_in = int(self.max_iterations * DREAMDefaults.BURN_IN_FRACTION)
        posterior_samples: List[np.ndarray] = []

        # Acceptance tracking for adaptive scaling
        acceptance_history = np.zeros((n_chains, 100))  # Rolling window
        acceptance_idx = 0

        # Main DREAM loop
        for iteration in range(1, self.max_iterations + 1):
            n_accepted = 0

            # Generate proposals for all chains with error handling
            proposals = np.zeros_like(chains)

            for i in range(n_chains):
                try:
                    # Select random chains for DE (excluding current chain)
                    available = [j for j in range(n_chains) if j != i]

                    # Select 2*n_pairs chains for differential evolution
                    selected = np.random.choice(available, size=2 * n_pairs, replace=False)
                    r1 = selected[:n_pairs]
                    r2 = selected[n_pairs:]

                    # Compute differential evolution jump
                    # diff = sum_{k=1}^{delta} (x_{r1[k]} - x_{r2[k]})
                    diff = np.zeros(n_params)
                    for k in range(n_pairs):
                        diff += chains[r1[k]] - chains[r2[k]]

                    # Subspace sampling: randomly select dimensions to update
                    # This improves mixing in high dimensions
                    # (Vrugt 2009, Section 2.3)
                    crossover_mask = np.random.random(n_params) < CR
                    if not crossover_mask.any():
                        # Ensure at least one dimension is updated
                        crossover_mask[np.random.randint(n_params)] = True

                    # Adaptive jump rate
                    # MODE_JUMP_PROBABILITY = 0.1: occasionally use gamma = 1 for mode jumping
                    # This helps escape local modes in multimodal distributions
                    # (Vrugt 2009, Section 2.2 - "Jump Rate")
                    if np.random.random() < DREAMDefaults.MODE_JUMP_PROBABILITY:
                        gamma = 1.0
                    else:
                        gamma = gamma_base

                    # Generate proposal: x_proposal = x_current + gamma * diff + noise
                    proposal = chains[i].copy()
                    jump = gamma * diff + eps_std * np.random.randn(n_params)
                    proposal[crossover_mask] += jump[crossover_mask]

                    # Reflect at bounds
                    proposal = self._reflect_at_bounds(proposal)
                    proposals[i] = proposal

                except (ValueError, FloatingPointError, IndexError) as e:
                    self.logger.warning(f"Error generating proposal for chain {i}: {e}")
                    proposals[i] = chains[i].copy()  # Keep current position

            # Evaluate all proposals
            try:
                proposal_fitness = evaluate_population(proposals, iteration)
                proposal_log_lik = proposal_fitness / temperature

                # Handle NaN/Inf fitness values
                invalid_mask = ~np.isfinite(proposal_fitness)
                if invalid_mask.any():
                    self.logger.warning(
                        f"Iteration {iteration}: {invalid_mask.sum()} proposals "
                        f"returned invalid fitness, rejecting"
                    )
                    proposal_log_lik[invalid_mask] = float('-inf')

            except (ValueError, FloatingPointError) as e:
                self.logger.warning(f"Error evaluating proposals: {e}")
                proposal_fitness = fitness.copy()
                proposal_log_lik = log_likelihood.copy()

            # Metropolis-Hastings acceptance for each chain
            for i in range(n_chains):
                # Log acceptance probability
                log_alpha = proposal_log_lik[i] - log_likelihood[i]

                # Accept or reject
                if np.log(np.random.random()) < log_alpha:
                    chains[i] = proposals[i]
                    fitness[i] = proposal_fitness[i]
                    log_likelihood[i] = proposal_log_lik[i]
                    n_accepted += 1
                    acceptance_history[i, acceptance_idx % 100] = 1

                    # Update best if improved
                    if fitness[i] > best_fit:
                        best_fit = fitness[i]
                        best_pos = chains[i].copy()
                else:
                    acceptance_history[i, acceptance_idx % 100] = 0

            acceptance_idx += 1

            # Outlier chain detection and correction
            # Detect and correct outlier chains that may be stuck in low-probability regions
            # (Vrugt 2009, Section 2.4)
            if iteration > 10 and iteration % 10 == 0:
                try:
                    self._correct_outlier_chains(
                        chains, fitness, log_likelihood,
                        outlier_threshold, temperature
                    )
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error in outlier correction: {e}")

            # Store posterior samples after burn-in
            if iteration > burn_in:
                for i in range(n_chains):
                    posterior_samples.append(chains[i].copy())

            # Record iteration
            params_dict = denormalize_params(best_pos)
            acceptance_rate = n_accepted / n_chains

            record_iteration(
                iteration, best_fit, params_dict,
                {'acceptance_rate': acceptance_rate, 'n_chains': n_chains}
            )
            update_best(best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, best_fit, n_accepted, n_chains, unit='gens')

            # Convergence check via Gelman-Rubin diagnostic (simplified)
            # R-hat < 1.1 indicates approximate convergence of chains
            # (Gelman & Rubin, 1992; Vrugt 2016, Section 3.2)
            if iteration > burn_in and iteration % 50 == 0:
                try:
                    r_hat = self._compute_gelman_rubin(chains, fitness)
                    if r_hat < DREAMDefaults.RHAT_THRESHOLD:
                        self.logger.info(
                            f"DREAM converged at iteration {iteration} (R-hat = {r_hat:.3f})"
                        )
                        # Continue to collect more samples rather than stopping
                        pass
                except (ValueError, FloatingPointError) as e:
                    self.logger.debug(f"Error computing R-hat: {e}")

        # Compute posterior statistics
        posterior_array = np.array(posterior_samples) if posterior_samples else chains
        posterior_mean = np.mean(posterior_array, axis=0)
        posterior_std = np.std(posterior_array, axis=0)

        # Compute credible intervals (95%)
        credible_lower = np.percentile(posterior_array, 2.5, axis=0)
        credible_upper = np.percentile(posterior_array, 97.5, axis=0)

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval_95': (credible_lower, credible_upper),
            'n_posterior_samples': len(posterior_samples),
            'final_chains': chains,
            'final_fitness': fitness
        }

    def _correct_outlier_chains(
        self,
        chains: np.ndarray,
        fitness: np.ndarray,
        log_likelihood: np.ndarray,
        threshold: float,
        temperature: float
    ) -> None:
        """
        Detect and correct outlier chains that may be stuck in low-probability regions.

        Uses interquartile range (IQR) of log-likelihood to identify outliers.
        Outlier chains are restarted from a randomly selected good chain.

        Args:
            chains: Current chain positions (n_chains, n_params)
            fitness: Current fitness values (n_chains,)
            log_likelihood: Current log-likelihood values (n_chains,)
            threshold: IQR multiplier for outlier detection
            temperature: Temperature for likelihood computation
        """
        n_chains = len(chains)
        if n_chains < 4:
            return

        # Compute IQR of log-likelihood
        q1, q3 = np.percentile(log_likelihood, [25, 75])
        iqr = q3 - q1

        # Identify outliers (below Q1 - threshold * IQR)
        outlier_threshold = q1 - threshold * iqr
        outliers = log_likelihood < outlier_threshold

        if outliers.any():
            # Get indices of good chains
            good_chains = np.where(~outliers)[0]
            if len(good_chains) == 0:
                return

            # Restart outlier chains from random good chains
            for i in np.where(outliers)[0]:
                source = np.random.choice(good_chains)
                chains[i] = chains[source].copy()
                fitness[i] = fitness[source]
                log_likelihood[i] = log_likelihood[source]

            self.logger.debug(f"Corrected {outliers.sum()} outlier chains")

    def _compute_gelman_rubin(
        self,
        chains: np.ndarray,
        fitness: np.ndarray
    ) -> float:
        """
        Compute simplified Gelman-Rubin convergence diagnostic.

        The R-hat statistic compares within-chain and between-chain variance.
        Values close to 1.0 indicate convergence.

        Reference:
            Gelman, A. and Rubin, D.B. (1992). Inference from iterative simulation
            using multiple sequences. Statistical Science, 7(4), 457-472.

        Args:
            chains: Current chain positions (n_chains, n_params)
            fitness: Current fitness values (for weighting)

        Returns:
            R-hat statistic (target: < 1.1 for convergence)
        """
        n_chains, n_params = chains.shape

        if n_chains < 2:
            return float('inf')

        # Compute between-chain variance
        _chain_means = np.mean(chains, axis=1)  # (n_chains,) for each param
        _overall_mean = np.mean(chains, axis=0)  # (n_params,)

        B = np.var(np.mean(chains, axis=0))  # Simplified: variance of means

        # Compute within-chain variance
        W = np.mean([np.var(chains[i]) for i in range(n_chains)])

        if W < 1e-10:
            return 1.0

        # Simplified R-hat
        # Full formula would use chain length, but we use current snapshot
        var_estimate = W + B
        r_hat = np.sqrt(var_estimate / W)

        return r_hat
