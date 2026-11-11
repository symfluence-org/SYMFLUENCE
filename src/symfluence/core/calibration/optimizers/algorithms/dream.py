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
        # Per-run RNG (OptimizationAlgorithm._new_rng): never the process-global stream.
        self._rng = self._new_rng()
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
        chains = self._rng.uniform(0, 1, (n_chains, n_params))

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
        # Chain positions at every iteration, for the R-hat diagnostic
        chain_history: List[np.ndarray] = [chains.copy()]
        r_hat = float('nan')

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
                    selected = self._rng.choice(available, size=2 * n_pairs, replace=False)
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
                    crossover_mask = self._rng.random(n_params) < CR
                    if not crossover_mask.any():
                        # Ensure at least one dimension is updated
                        crossover_mask[self._rng.integers(n_params)] = True

                    # Adaptive jump rate
                    # MODE_JUMP_PROBABILITY = 0.1: occasionally use gamma = 1 for mode jumping
                    # This helps escape local modes in multimodal distributions
                    # (Vrugt 2009, Section 2.2 - "Jump Rate")
                    if self._rng.random() < DREAMDefaults.MODE_JUMP_PROBABILITY:
                        gamma = 1.0
                    else:
                        gamma = gamma_base

                    # Generate proposal: x_proposal = x_current + gamma * diff + noise
                    proposal = chains[i].copy()
                    jump = gamma * diff + eps_std * self._rng.standard_normal(n_params)
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
                if np.log(self._rng.random()) < log_alpha:
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

            # Convergence diagnostic: Gelman-Rubin R-hat over the second half
            # of the chain histories. R-hat < 1.1 indicates approximate
            # convergence (Gelman & Rubin, 1992; Vrugt 2016, Section 3.2).
            # Diagnostic only: the run always continues to NUMBER_OF_ITERATIONS
            # so the posterior sample count does not depend on it.
            chain_history.append(chains.copy())
            if iteration > burn_in and iteration % 50 == 0:
                try:
                    r_hat = self._compute_gelman_rubin(np.array(chain_history))
                    verdict = (
                        "chains have converged"
                        if r_hat < DREAMDefaults.RHAT_THRESHOLD
                        else f"not converged (threshold {DREAMDefaults.RHAT_THRESHOLD})"
                    )
                    self.logger.info(
                        f"DREAM iteration {iteration}: max R-hat = {r_hat:.3f} ({verdict})"
                    )
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
            'final_fitness': fitness,
            'r_hat': r_hat,
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
                source = self._rng.choice(good_chains)
                chains[i] = chains[source].copy()
                fitness[i] = fitness[source]
                log_likelihood[i] = log_likelihood[source]

            self.logger.debug(f"Corrected {outliers.sum()} outlier chains")

    def _compute_gelman_rubin(self, chain_history: np.ndarray) -> float:
        """
        Gelman-Rubin potential scale reduction factor (R-hat).

        Computed per parameter from the second half of each chain's history
        (the first half is discarded as warm-up) and maximised over
        parameters. For ``m`` chains of ``n`` retained draws each::

            W      = mean over chains of the within-chain variance
            B / n  = variance of the chain means
            var+   = (n - 1) / n * W + B / n
            R-hat  = sqrt(var+ / W)

        Values close to 1.0 indicate the chains have mixed.

        Reference:
            Gelman, A. and Rubin, D.B. (1992). Inference from iterative simulation
            using multiple sequences. Statistical Science, 7(4), 457-472.

        Args:
            chain_history: Chain positions over iterations,
                shape (n_iterations, n_chains, n_params)

        Returns:
            Maximum R-hat over parameters (target: < 1.1 for convergence);
            ``inf`` when it cannot be computed (fewer than 2 chains or fewer
            than 2 retained draws per chain).
        """
        history = np.asarray(chain_history, dtype=float)
        if history.ndim != 3:
            raise ValueError(
                "chain_history must have shape (n_iterations, n_chains, n_params), "
                f"got {history.shape}"
            )
        draws = history[history.shape[0] // 2:]
        n, m, _ = draws.shape
        if m < 2 or n < 2:
            return float('inf')

        chain_means = draws.mean(axis=0)                 # (m, d)
        within = draws.var(axis=0, ddof=1).mean(axis=0)  # W, (d,)
        between_over_n = chain_means.var(axis=0, ddof=1)  # B/n, (d,)
        var_plus = (n - 1) / n * within + between_over_n

        r_hat = np.ones_like(within)
        moving = within > 0
        r_hat[moving] = np.sqrt(var_plus[moving] / within[moving])
        # No within-chain movement: converged only if the chains also agree.
        r_hat[~moving & (between_over_n > 0)] = np.inf
        return float(np.max(r_hat))
