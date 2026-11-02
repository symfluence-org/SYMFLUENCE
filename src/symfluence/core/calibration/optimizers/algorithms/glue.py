#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
GLUE (Generalized Likelihood Uncertainty Estimation)

A Monte Carlo-based method for parameter uncertainty estimation that has become
a standard approach in hydrological modeling. GLUE rejects the idea of a single
"optimal" parameter set, instead identifying all "behavioral" parameter sets
that produce acceptable model performance.

Key Features:
    - Monte Carlo sampling from parameter space (uniform or Latin Hypercube)
    - Likelihood-based weighting of model runs
    - Threshold-based behavioral/non-behavioral classification
    - Weighted prediction bounds for uncertainty quantification
    - Dotty plots for parameter sensitivity visualization

Reference:
    Beven, K. and Binley, A. (1992). The future of distributed models: Model
    calibration and uncertainty prediction. Hydrological Processes, 6(3), 279-298.

    Beven, K. (2006). A manifesto for the equifinality thesis. Journal of
    Hydrology, 320(1-2), 18-36.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import GLUEDefaults


class GLUEAlgorithm(OptimizationAlgorithm):
    """GLUE (Generalized Likelihood Uncertainty Estimation) algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "GLUE"

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
        Run GLUE uncertainty analysis.

        GLUE performs Monte Carlo sampling of the parameter space, evaluates
        each sample, and identifies "behavioral" parameter sets that exceed
        a performance threshold. The behavioral sets are then used to:
        1. Estimate parameter uncertainty (dotty plots)
        2. Generate weighted prediction bounds
        3. Identify the best-performing parameter set

        The likelihood L(θ|Y) for each parameter set θ is typically computed as:
            L = max(0, metric - threshold) ^ N

        where N is a shaping factor (default 1).

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
            Optimization results dictionary including behavioral samples
        """
        self.logger.info(f"Starting GLUE analysis with {n_params} parameters")

        # GLUE parameters from config
        n_samples = self.max_iterations * self.population_size
        batch_size = self.population_size

        threshold = self._get_config_value(
            lambda: self.config.optimization.glue.threshold,
            default=GLUEDefaults.THRESHOLD,
            dict_key='GLUE_THRESHOLD'
        )
        shaping_factor = self._get_config_value(
            lambda: self.config.optimization.glue.shaping_factor,
            default=GLUEDefaults.SHAPING_FACTOR,
            dict_key='GLUE_SHAPING_FACTOR'
        )
        sampling_method = self._get_config_value(
            lambda: self.config.optimization.glue.sampling,
            default=GLUEDefaults.SAMPLING,
            dict_key='GLUE_SAMPLING'
        )

        # Validate GLUE configuration
        valid, warning = GLUEDefaults.validate_threshold(threshold)
        if not valid:
            self.logger.warning(f"GLUE threshold validation: {warning}")

        self.logger.info(
            f"GLUE settings: {n_samples} samples, threshold={threshold}, "
            f"shaping_factor={shaping_factor}, sampling={sampling_method}"
        )

        # Storage for all samples
        all_samples: List[np.ndarray] = []
        all_fitness: List[float] = []

        # Track best solution
        best_pos = np.full(n_params, 0.5)
        best_fit = float('-inf')

        # Generate all samples upfront if using LHS
        if sampling_method == 'lhs':
            all_param_samples = self._latin_hypercube_sample(n_samples, n_params)
        else:
            all_param_samples = None

        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        _sample_idx = 0

        for batch in range(n_batches):
            # Determine batch boundaries
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            actual_batch_size = batch_end - batch_start

            # Generate or retrieve samples for this batch
            if all_param_samples is not None:
                # LHS: use pre-generated samples
                population = all_param_samples[batch_start:batch_end]
            else:
                # Uniform random sampling
                population = np.random.uniform(0, 1, (actual_batch_size, n_params))

            # Evaluate batch
            fitness = evaluate_population(population, batch)

            # Store results
            for i in range(actual_batch_size):
                all_samples.append(population[i].copy())
                all_fitness.append(fitness[i])

                # Update best
                if fitness[i] > best_fit:
                    best_fit = fitness[i]
                    best_pos = population[i].copy()

            # Record iteration (use batch number as iteration)
            params_dict = denormalize_params(best_pos)
            n_behavioral = sum(1 for f in all_fitness if f >= threshold)
            record_iteration(
                batch + 1, best_fit, params_dict,
                {'n_behavioral': n_behavioral, 'n_total': len(all_fitness)}
            )
            update_best(best_fit, params_dict, batch + 1)

            # Progress line every 10th batch and on the final batch. The
            # cumulative sample count doesn't align with the tracker's
            # iteration throttle (cadence would depend on batch_size), so
            # GLUE throttles on batch index and forces emission. GLUE has no
            # notion of "improvement", so the Improved field carries
            # behavioral/total — the closest comparable acceptance metric.
            if (batch + 1) % 10 == 0 or batch + 1 == n_batches:
                log_progress(
                    self.name, len(all_fitness), best_fit,
                    n_improved=n_behavioral, pop_size=len(all_fitness),
                    unit='evals', total=n_samples, force=True
                )
            self.logger.debug(
                f"GLUE batch {batch + 1}/{n_batches}: "
                f"{n_behavioral}/{len(all_fitness)} behavioral samples"
            )

        # Convert to arrays
        samples_array = np.array(all_samples)
        fitness_array = np.array(all_fitness)

        # Identify behavioral samples
        behavioral_mask = fitness_array >= threshold
        n_behavioral = behavioral_mask.sum()

        self.logger.info(
            f"GLUE complete: {n_behavioral}/{len(fitness_array)} behavioral samples "
            f"({100 * n_behavioral / len(fitness_array):.1f}%)"
        )

        if n_behavioral == 0:
            self.logger.warning(
                f"No behavioral samples found with threshold={threshold}. "
                f"Consider lowering GLUE_THRESHOLD. Best score was {best_fit:.4f}"
            )
            # Return best found even if non-behavioral
            return {
                'best_solution': best_pos,
                'best_score': best_fit,
                'best_params': denormalize_params(best_pos),
                'n_behavioral': 0,
                'n_total': len(fitness_array),
                'behavioral_samples': np.array([]),
                'behavioral_fitness': np.array([]),
                'behavioral_likelihoods': np.array([]),
            }

        # Extract behavioral samples
        behavioral_samples = samples_array[behavioral_mask]
        behavioral_fitness = fitness_array[behavioral_mask]

        # Compute likelihoods (rescaled for numerical stability)
        # L = (fitness - threshold) ^ shaping_factor
        likelihoods = (behavioral_fitness - threshold) ** shaping_factor
        likelihoods = likelihoods / likelihoods.sum()  # Normalize

        # Compute weighted statistics
        weighted_mean = np.average(behavioral_samples, axis=0, weights=likelihoods)
        weighted_std = np.sqrt(
            np.average((behavioral_samples - weighted_mean) ** 2, axis=0, weights=likelihoods)
        )

        # Compute parameter ranges (5th and 95th percentiles of behavioral)
        param_5th = np.percentile(behavioral_samples, 5, axis=0)
        param_95th = np.percentile(behavioral_samples, 95, axis=0)

        # Log parameter uncertainty
        self.logger.info("GLUE Parameter Uncertainty (behavioral samples):")
        params_dict_mean = denormalize_params(weighted_mean)
        for i, (name, value) in enumerate(params_dict_mean.items()):
            self.logger.info(
                f"  {name}: {value:.4f} (5-95%: [{param_5th[i]:.3f}, {param_95th[i]:.3f}])"
            )

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'n_behavioral': n_behavioral,
            'n_total': len(fitness_array),
            'behavioral_fraction': n_behavioral / len(fitness_array),
            'behavioral_samples': behavioral_samples,
            'behavioral_fitness': behavioral_fitness,
            'behavioral_likelihoods': likelihoods,
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'param_bounds_90': (param_5th, param_95th),
            'all_samples': samples_array,
            'all_fitness': fitness_array,
        }

    def _latin_hypercube_sample(self, n_samples: int, n_params: int) -> np.ndarray:
        """
        Generate Latin Hypercube samples for better space coverage.

        LHS ensures that each parameter's marginal distribution is sampled
        uniformly, providing better coverage than pure random sampling.

        Args:
            n_samples: Number of samples to generate
            n_params: Number of parameters

        Returns:
            Array of shape (n_samples, n_params) with values in [0, 1]
        """
        samples = np.zeros((n_samples, n_params))

        for j in range(n_params):
            # Create n_samples intervals and sample one point from each
            intervals = np.arange(n_samples)
            np.random.shuffle(intervals)

            for i in range(n_samples):
                # Random point within interval [intervals[i]/n, (intervals[i]+1)/n]
                low = intervals[i] / n_samples
                high = (intervals[i] + 1) / n_samples
                samples[i, j] = np.random.uniform(low, high)

        return samples
