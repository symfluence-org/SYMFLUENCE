#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
GA (Genetic Algorithm)

A population-based evolutionary algorithm that uses selection, crossover,
and mutation to evolve solutions. Effective for continuous optimization problems.

Reference:
    Holland, J.H. (1975). Adaptation in Natural and Artificial Systems.
    University of Michigan Press.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import GADefaults


class GAAlgorithm(OptimizationAlgorithm):
    """Genetic Algorithm optimization."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "GA"

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
        Run GA optimization.

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
        self.logger.info(f"Starting GA optimization with {n_params} parameters")

        # GA requires at least 4 individuals
        pop_size = max(4, self.population_size)
        if pop_size != self.population_size:
            self.logger.warning(
                f"Population size {self.population_size} is too small for GA. Increasing to {pop_size}."
            )

        # Ensure even population size for crossover
        if pop_size % 2 != 0:
            pop_size += 1
            self.logger.info(f"Adjusted population size to {pop_size} for even pairing")

        # GA parameters from config
        crossover_rate = self._get_config_value(
            lambda: self.config.optimization.ga.crossover_rate,
            default=GADefaults.CROSSOVER_RATE,
            dict_key='GA_CROSSOVER_RATE'
        )
        mutation_rate = self._get_config_value(
            lambda: self.config.optimization.ga.mutation_rate,
            default=GADefaults.MUTATION_RATE,
            dict_key='GA_MUTATION_RATE'
        )
        mutation_scale = self._get_config_value(
            lambda: self.config.optimization.ga.mutation_scale,
            default=GADefaults.MUTATION_SCALE,
            dict_key='GA_MUTATION_SCALE'
        )
        tournament_size = self._get_config_value(
            lambda: self.config.optimization.ga.tournament_size,
            default=GADefaults.TOURNAMENT_SIZE,
            dict_key='GA_TOURNAMENT_SIZE'
        )
        elitism_count = self._get_config_value(
            lambda: self.config.optimization.ga.elitism_count,
            default=GADefaults.ELITISM_COUNT,
            dict_key='GA_ELITISM_COUNT'
        )

        # Validate GA parameters
        valid, warning = GADefaults.validate_rates(crossover_rate, mutation_rate)
        if not valid:
            self.logger.warning(f"GA parameters may cause issues: {warning}")

        # Ensure elitism doesn't exceed population
        elitism_count = min(elitism_count, pop_size // 2)

        self.logger.debug(
            f"GA parameters: crossover_rate={crossover_rate}, mutation_rate={mutation_rate}, "
            f"mutation_scale={mutation_scale}, tournament_size={tournament_size}, elitism={elitism_count}"
        )

        # Initialize population
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
        self._rng = self._new_rng()
        population = self._rng.uniform(0, 1, (pop_size, n_params))
        fitness = evaluate_population(population, 0)

        # Record initial best
        best_idx = np.argmax(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]

        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, pop_size, best_fit)

        # Track previous generation's best for n_improved metric
        prev_best_fit = best_fit

        # GA main loop
        for iteration in range(1, self.max_iterations + 1):
            # Sort population by fitness (descending)
            sorted_indices = np.argsort(-fitness)
            sorted_population = population[sorted_indices]
            _sorted_fitness = fitness[sorted_indices]  # kept for debugging

            # Elitism: keep top individuals
            new_population = list(sorted_population[:elitism_count])

            # Generate offspring until population is filled
            while len(new_population) < pop_size:
                # Tournament selection for two parents
                parent1 = self._tournament_select(
                    population, fitness, tournament_size
                )
                parent2 = self._tournament_select(
                    population, fitness, tournament_size
                )

                # Crossover
                if self._rng.random() < crossover_rate:
                    child1, child2 = self._sbx_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self._polynomial_mutation(
                    child1, mutation_rate, mutation_scale
                )
                child2 = self._polynomial_mutation(
                    child2, mutation_rate, mutation_scale
                )

                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)

            # Convert to array and evaluate
            population = np.array(new_population)
            fitness = evaluate_population(population, iteration)

            # Track improvements - count individuals that beat previous generation's best
            current_best_idx = np.argmax(fitness)
            n_improved = int(np.sum(fitness > prev_best_fit))

            if fitness[current_best_idx] > best_fit:
                best_pos = population[current_best_idx].copy()
                best_fit = fitness[current_best_idx]

            # Record results
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict)
            update_best(best_fit, params_dict, iteration)

            # Progress line (tracker throttles emission)
            log_progress(self.name, iteration, best_fit, n_improved, pop_size, unit='gens')

            # Update previous best for next iteration
            prev_best_fit = best_fit

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos)
        }

    def _tournament_select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        tournament_size: int
    ) -> np.ndarray:
        """
        Select an individual using tournament selection.

        Args:
            population: Current population
            fitness: Fitness values
            tournament_size: Number of individuals in tournament

        Returns:
            Selected individual
        """
        pop_size = len(population)
        tournament_size = min(tournament_size, pop_size)
        candidates = self._rng.choice(pop_size, tournament_size, replace=False)
        winner = candidates[np.argmax(fitness[candidates])]
        return population[winner].copy()

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        eta: float = 20.0
    ) -> tuple:
        """
        Simulated Binary Crossover (SBX).

        Args:
            parent1: First parent
            parent2: Second parent
            eta: Distribution index (higher = children closer to parents)

        Returns:
            Tuple of two children
        """
        n_params = len(parent1)
        child1 = np.empty(n_params)
        child2 = np.empty(n_params)

        for i in range(n_params):
            if self._rng.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    # Calculate beta
                    rand = self._rng.random()
                    beta = 1.0 + (2.0 * y1) / (y2 - y1 + 1e-14)
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[i] = np.clip(c1, 0, 1)
                    child2[i] = np.clip(c2, 0, 1)
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]

        return child1, child2

    def _polynomial_mutation(
        self,
        individual: np.ndarray,
        mutation_rate: float,
        mutation_scale: float,
        eta: float = 20.0
    ) -> np.ndarray:
        """
        Polynomial mutation.

        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each gene
            mutation_scale: Not used in polynomial mutation (kept for interface)
            eta: Distribution index (higher = smaller mutations)

        Returns:
            Mutated individual
        """
        mutant = individual.copy()

        for i in range(len(mutant)):
            if self._rng.random() < mutation_rate:
                y = mutant[i]
                delta1 = y  # y - 0 (lower bound)
                delta2 = 1 - y  # 1 - y (upper bound)

                rand = self._rng.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutant[i] = np.clip(y + deltaq, 0, 1)

        return mutant
