#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
DE (Differential Evolution) Algorithm

A population-based evolutionary algorithm that uses vector differences
for mutation. Effective for continuous optimization problems.

Reference:
    Storn, R. and Price, K. (1997). Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces.
    Journal of Global Optimization, 11(4), 341-359.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import DEDefaults


class DEAlgorithm(OptimizationAlgorithm):
    """Differential Evolution optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "DE"

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
        Run DE optimization.

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
        self.logger.info(f"Starting DE optimization with {n_params} parameters")

        # DE requires at least 4 individuals to sample 3 distinct ones
        pop_size = max(4, self.population_size)
        if pop_size != self.population_size:
            self.logger.warning(
                f"Population size {self.population_size} is too small for DE. Increasing to {pop_size}."
            )

        # DE parameters from config
        F = self._get_config_value(
            lambda: self.config.optimization.de.scaling_factor,
            default=DEDefaults.F,
            dict_key='DE_SCALING_FACTOR'
        )
        CR = self._get_config_value(
            lambda: self.config.optimization.de.crossover_rate,
            default=DEDefaults.CR,
            dict_key='DE_CROSSOVER_RATE'
        )

        # Validate DE parameters
        valid, warning = DEDefaults.validate_parameters(F, CR)
        if not valid:
            self.logger.warning(f"DE parameters may cause issues: {warning}")

        # Initialize population
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
        population = self._seed_population(np.random.uniform(0, 1, (pop_size, n_params)), kwargs)
        fitness = evaluate_population(population, 0)

        # Record initial best with enhanced tracking
        best_idx = np.argmax(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]

        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict, {
            'mean_score': float(np.mean(fitness)),
            'std_score': float(np.std(fitness)),
            'n_improved': 0,
        })
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, pop_size, best_fit)

        # DE main loop
        for iteration in range(1, self.max_iterations + 1):
            # Generate all trial solutions for this iteration
            trials = []
            for i in range(pop_size):
                # Select three random individuals (not i)
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation: DE/rand/1
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, 0, 1)

                # Crossover
                cross_points = np.random.random(n_params) < CR
                if not cross_points.any():
                    cross_points[np.random.randint(n_params)] = True

                trial = np.where(cross_points, mutant, population[i])
                trials.append(trial)

            # Evaluate all trials
            trial_population = np.array(trials)
            trial_fitness = evaluate_population(trial_population, iteration)

            # Selection - update population based on trial results
            n_improved = 0
            for i in range(pop_size):
                if trial_fitness[i] > fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]
                    n_improved += 1

                    if trial_fitness[i] > best_fit:
                        best_pos = trials[i].copy()
                        best_fit = trial_fitness[i]

            # Record results with enhanced tracking for response surface analysis
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict, {
                'mean_score': float(np.mean(fitness)),
                'std_score': float(np.std(fitness)),
                'n_improved': int(n_improved),
            })
            update_best(best_fit, params_dict, iteration)

            # Progress line (tracker throttles emission)
            log_progress(self.name, iteration, best_fit, n_improved, pop_size, unit='evals')

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos)
        }
