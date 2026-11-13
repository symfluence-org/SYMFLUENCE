#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
DDS (Dynamically Dimensioned Search) Algorithm

A simple and effective algorithm for calibrating computationally expensive
hydrological models. DDS progressively focuses the search from global to local
as iterations progress.

Reference:
    Tolson, B.A. and Shoemaker, C.A. (2007). Dynamically dimensioned search
    algorithm for computationally efficient watershed model calibration.
    Water Resources Research, 43(1).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from symfluence.core.metrics.penalty import is_penalty

from .base_algorithm import OptimizationAlgorithm


class DDSAlgorithm(OptimizationAlgorithm):
    """Dynamically Dimensioned Search optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "DDS"

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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run DDS optimization.

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
        # DDS perturbation range (default 0.2, higher values explore more)
        r = self._get_config_value(lambda: self.config.optimization.dds.r, dict_key='DDS_R')

        # Minimum perturbation probability to ensure exploration even late in optimization
        p_min = self._get_config_value(
            lambda: None, default=0.05, dict_key='DDS_P_MIN'
        )

        # Initialize with random starting point or provided initial guess
        initial_guess = kwargs.get('initial_guess')
        if initial_guess is not None and len(initial_guess) == n_params:
            x_best = np.array(initial_guess, dtype=float)
            self.logger.info(f"Starting DDS optimization with {n_params} parameters (warm start)")
            self.logger.debug(
                f"DDS initial guess: mean={x_best.mean():.4f}, "
                f"range=[{x_best.min():.4f}, {x_best.max():.4f}]"
            )
        else:
            x_best = self._rng.uniform(0, 1, n_params)
            self.logger.info(f"Starting DDS optimization with {n_params} parameters (random start)")

        f_best = evaluate_solution(x_best, 0)

        # Record initial state
        params_dict = denormalize_params(x_best)
        record_iteration(0, f_best, params_dict)
        update_best(f_best, params_dict, 0)

        # Stagnation detection settings
        stagnation_threshold = self._get_config_value(
            lambda: self.config.optimization.dds.stagnation_threshold,
            dict_key='DDS_STAGNATION_THRESHOLD',
        )
        iterations_since_improvement = 0
        n_improvements = 0

        # Initialize current perturbation range
        current_r = r

        # DDS main loop
        for iteration in range(1, self.max_iterations + 1):
            # Stagnation detection: if no improvement for many iterations,
            # increase perturbation to explore more
            if iterations_since_improvement >= stagnation_threshold:
                current_r = min(r * 2.0, 0.5)  # Double r but cap at 0.5
                if iterations_since_improvement == stagnation_threshold:
                    self.logger.debug(
                        f"DDS stagnation detected at iteration {iteration} "
                        f"(no improvement for {iterations_since_improvement} iterations). "
                        f"Increasing perturbation to {current_r:.3f}"
                    )
            else:
                current_r = r

            # Calculate probability of perturbation (decreases with iterations)
            # Use p_min to maintain minimum exploration capability throughout optimization
            p = 1.0 - np.log(iteration) / np.log(self.max_iterations)
            p = max(p_min, max(1.0 / n_params, p))  # Ensure minimum perturbation probability

            # Select parameters to perturb
            perturb_mask = self._rng.random(n_params) < p

            # Ensure at least one parameter is perturbed
            if not perturb_mask.any():
                perturb_mask[self._rng.integers(n_params)] = True

            # Generate candidate solution
            x_new = x_best.copy()

            for i in range(n_params):
                if perturb_mask[i]:
                    perturbation = current_r * self._rng.standard_normal()
                    x_new[i] = x_best[i] + perturbation

                    # Reflect at boundaries (use while loop for large perturbations)
                    while x_new[i] < 0 or x_new[i] > 1:
                        if x_new[i] < 0:
                            x_new[i] = -x_new[i]
                        if x_new[i] > 1:
                            x_new[i] = 2.0 - x_new[i]

            # Evaluate candidate
            f_new = evaluate_solution(x_new, 0)

            # Update if better (DDS is greedy)
            # Don't count crashes toward stagnation — penalty scores indicate
            # model failure (e.g. CLASS energy balance crash), not search
            # exhaustion.  Counting them triggers perturbation doubling which
            # generates more extreme parameter combos, causing more crashes.
            is_crash = is_penalty(f_new, self.penalty_score)
            if f_new > f_best:
                x_best = x_new
                f_best = f_new
                iterations_since_improvement = 0
                n_improvements += 1
            elif not is_crash:
                iterations_since_improvement += 1

            # Record results
            params_dict = denormalize_params(x_best)
            record_iteration(iteration, f_best, params_dict)
            update_best(f_best, params_dict, iteration)

            # Progress line (tracker throttles emission; Improved counts
            # cumulative accepted improvements over evaluations so far)
            log_progress(
                self.name, iteration, f_best,
                n_improved=n_improvements, pop_size=iteration,
                unit='evals'
            )

        return {
            'best_solution': x_best,
            'best_score': f_best,
            'best_params': denormalize_params(x_best)
        }
