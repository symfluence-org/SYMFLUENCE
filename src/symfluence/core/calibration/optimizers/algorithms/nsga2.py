#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II)

A multi-objective evolutionary algorithm that uses non-dominated sorting
and crowding distance for maintaining diversity in the Pareto front.

Reference:
    Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and
    elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions
    on Evolutionary Computation, 6(2), 182-197.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from symfluence.core.metrics.penalty import is_penalty

from .base_algorithm import OptimizationAlgorithm
from .config_schema import NSGA2Defaults


class NSGA2Algorithm(OptimizationAlgorithm):
    """NSGA-II multi-objective optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "NSGA-II"

    def _clean_objectives(self, objectives: np.ndarray, label: str) -> np.ndarray:
        """Replace rows with any NaN/inf objective by the (finite) penalty.

        Applied to the initial population as well as to offspring: a NaN
        there poisons non-dominated sorting, crowding distance and the
        ``argmax`` that picks the reported best.
        """
        objectives = np.array(objectives, dtype=float)
        invalid_mask = ~np.isfinite(objectives).all(axis=1)
        if invalid_mask.any():
            self.logger.warning(
                f"{label}: {invalid_mask.sum()} individual(s) returned invalid "
                f"objectives, assigning penalty"
            )
            objectives[invalid_mask] = self.penalty_score
        return objectives

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
        objective_names: Optional[List[str]] = None,
        multiobjective: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run NSGA-II optimization.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            evaluate_population_objectives: Callback for multi-objective evaluation
            objective_names: List of objective metric names
            multiobjective: Whether to use multi-objective evaluation
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        self.logger.debug(f"Starting NSGA-II optimization with {n_params} parameters")

        pop_size = self.population_size
        objective_names = objective_names or ['KGE', 'NSE']
        num_objectives = len(objective_names)

        # NSGA-II parameters from config
        crossover_rate = self._get_config_value(
            lambda: self.config.optimization.nsga2.crossover_rate,
            default=NSGA2Defaults.CROSSOVER_RATE,
            dict_key='NSGA2_CROSSOVER_RATE'
        )
        mutation_rate = self._get_config_value(
            lambda: self.config.optimization.nsga2.mutation_rate,
            default=NSGA2Defaults.MUTATION_RATE,
            dict_key='NSGA2_MUTATION_RATE'
        )
        eta_c = self._get_config_value(
            lambda: self.config.optimization.nsga2.eta_c,
            default=NSGA2Defaults.ETA_C,
            dict_key='NSGA2_ETA_C'
        )
        eta_m = self._get_config_value(
            lambda: self.config.optimization.nsga2.eta_m,
            default=NSGA2Defaults.ETA_M,
            dict_key='NSGA2_ETA_M'
        )

        # Dedicated per-run RNG (see OptimizationAlgorithm._new_rng) so the
        # genetic operators are reproducible and isolated from the global state.
        self._rng = self._new_rng()

        # Initialize population
        self.logger.debug(f"Initializing population ({pop_size} individuals)...")
        population = self._seed_population(self._rng.uniform(0, 1, (pop_size, n_params)), kwargs)
        objectives = np.full((pop_size, num_objectives), np.nan)

        # Evaluate initial population
        if multiobjective and evaluate_population_objectives:
            objectives = evaluate_population_objectives(population, objective_names, 0)
        else:
            fitness = evaluate_population(population, 0)
            objectives[:, 0] = fitness
            if num_objectives > 1:
                objectives[:, 1] = fitness

        # Check for all-penalty objectives (indicates the worker could not evaluate
        # the multi-objective set). Rather than aborting the whole run, degrade to
        # single-objective evaluation on the primary metric and duplicate it across
        # objective columns -- this yields a valid calibration result and matches the
        # behaviour of workers that only expose single-objective evaluation, so the
        # algorithm x model matrix stays consistent instead of leaving holes.
        objectives = self._clean_objectives(objectives, "Initial population")
        if np.all(is_penalty(objectives, self.penalty_score)) and multiobjective and evaluate_population is not None:
            self.logger.warning(
                "Multi-objective evaluation returned all-penalty objectives; falling back "
                "to single-objective (%s) evaluation for NSGA-II.", objective_names[0]
            )
            fitness = np.asarray(evaluate_population(population, 0)).reshape(-1)
            objectives = np.repeat(fitness.reshape(-1, 1), num_objectives, axis=1)
            objectives = self._clean_objectives(objectives, "Initial population")

        if np.all(is_penalty(objectives, self.penalty_score)):
            raise ValueError(
                f"All {pop_size} individuals in the initial population returned penalty "
                f"objectives (all penalty or non-finite), even after falling back to "
                f"single-objective evaluation. This indicates a broken model or forcing "
                f"setup for this run rather than a multi-objective limitation."
            )

        # Perform NSGA-II selection
        ranks = self._fast_non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, ranks)

        # Find representative solution (best on first objective)
        best_idx = np.argmax(objectives[:, 0])
        best_solution = population[best_idx].copy()
        best_fitness = objectives[best_idx, 0]
        best_secondary = objectives[best_idx, 1] if num_objectives > 1 else None

        # Record initial best
        extra_metrics = {}
        if best_secondary is not None:
            extra_metrics[f"obj2_{objective_names[1]}"] = float(best_secondary)
        params_dict = denormalize_params(best_solution)
        record_iteration(0, best_fitness, params_dict, additional_metrics=extra_metrics or None)
        update_best(best_fitness, params_dict, 0)

        if best_secondary is not None:
            self.logger.info(
                "Initial population complete | Best obj1 (%s): %.4f | Best obj2 (%s): %.4f",
                objective_names[0], best_fitness, objective_names[1], best_secondary
            )
        else:
            self.logger.info(f"Initial population complete | Best obj1: {best_fitness:.4f}")

        # Main NSGA-II loop with error handling
        prev_best_fitness = best_fitness
        for generation in range(1, self.max_iterations + 1):
            n_improved = 0
            try:
                # Generate offspring through selection, crossover, and mutation
                offspring = np.zeros_like(population)
                for i in range(0, pop_size, 2):
                    # Tournament selection
                    p1_idx = self._tournament_selection(ranks, crowding_distances, pop_size)
                    p2_idx = self._tournament_selection(ranks, crowding_distances, pop_size)
                    p1, p2 = population[p1_idx], population[p2_idx]

                    # Crossover
                    if self._rng.random() < crossover_rate:
                        c1, c2 = self._sbx_crossover(p1, p2, eta_c)
                    else:
                        c1, c2 = p1.copy(), p2.copy()

                    # Mutation
                    offspring[i] = self._polynomial_mutation(c1, eta_m, mutation_rate)
                    if i + 1 < pop_size:
                        offspring[i + 1] = self._polynomial_mutation(c2, eta_m, mutation_rate)

                # Evaluate offspring
                if multiobjective and evaluate_population_objectives:
                    offspring_objectives = evaluate_population_objectives(
                        offspring, objective_names, generation
                    )
                else:
                    offspring_fitness = evaluate_population(offspring, generation)
                    offspring_objectives = np.full((pop_size, num_objectives), np.nan)
                    offspring_objectives[:, 0] = offspring_fitness
                    if num_objectives > 1:
                        offspring_objectives[:, 1] = offspring_fitness

                # Handle NaN/Inf objective values
                offspring_objectives = self._clean_objectives(
                    offspring_objectives, f"Generation {generation} offspring"
                )

                # Combine parent and offspring populations
                combined_pop = np.vstack([population, offspring])
                combined_obj = np.vstack([objectives, offspring_objectives])

                # Environmental selection
                selected_indices = self._environmental_selection(combined_obj, pop_size)
                population = combined_pop[selected_indices]
                objectives = combined_obj[selected_indices]

                # Update ranks and crowding distances
                ranks = self._fast_non_dominated_sort(objectives)
                crowding_distances = self._calculate_crowding_distance(objectives, ranks)

                # Offspring beating the previous generation's best (progress metric)
                n_improved = int(np.sum(offspring_objectives[:, 0] > prev_best_fitness))

                # Update best solution
                current_best_idx = np.argmax(objectives[:, 0])
                if objectives[current_best_idx, 0] > best_fitness:
                    best_solution = population[current_best_idx].copy()
                    best_fitness = objectives[current_best_idx, 0]
                    best_secondary = objectives[current_best_idx, 1] if num_objectives > 1 else None

            except (ValueError, FloatingPointError) as e:
                self.logger.warning(
                    f"Error in generation {generation}: {e}. "
                    f"Continuing with current population."
                )
                # Continue with current state rather than crashing

            # Record results
            params_dict = denormalize_params(best_solution)
            extra_metrics = {}
            if best_secondary is not None:
                extra_metrics[f"obj2_{objective_names[1]}"] = float(best_secondary)
            record_iteration(
                generation, best_fitness, params_dict,
                additional_metrics=extra_metrics or None
            )
            update_best(best_fitness, params_dict, generation)

            # Progress line (tracker throttles emission)
            log_progress(
                self.name, generation, best_fitness,
                n_improved=n_improved, pop_size=pop_size,
                secondary_score=best_secondary,
                secondary_label=objective_names[1] if best_secondary is not None else None,
                unit='gens'
            )
            prev_best_fitness = best_fitness

        return {
            'best_solution': best_solution,
            'best_score': best_fitness,
            'best_params': denormalize_params(best_solution),
            'pareto_front': population[ranks == 0] if ranks is not None else None,
            'pareto_objectives': objectives[ranks == 0] if ranks is not None else None
        }

    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> np.ndarray:
        """Fast non-dominated sorting for NSGA-II."""
        pop_size = len(objectives)
        ranks = np.zeros(pop_size, dtype=int)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions: List[List[int]] = [[] for _ in range(pop_size)]

        # Find domination relationships
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Assign ranks
        current_front = np.where(domination_count == 0)[0]
        rank = 0
        while len(current_front) > 0:
            ranks[current_front] = rank
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = np.array(next_front)
            rank += 1

        return ranks

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (maximization)."""
        return bool(np.all(obj1 >= obj2)) and bool(np.any(obj1 > obj2))

    def _calculate_crowding_distance(
        self, objectives: np.ndarray, ranks: np.ndarray
    ) -> np.ndarray:
        """Calculate crowding distance for each solution."""
        pop_size = len(objectives)
        num_objectives = objectives.shape[1]
        crowding_distance = np.zeros(pop_size)

        for rank in np.unique(ranks):
            rank_indices = np.where(ranks == rank)[0]
            if len(rank_indices) <= 2:
                crowding_distance[rank_indices] = np.inf
                continue

            for obj_idx in range(num_objectives):
                obj_values = objectives[rank_indices, obj_idx]
                sorted_indices = np.argsort(obj_values)
                sorted_rank_indices = rank_indices[sorted_indices]

                # Boundary solutions get infinite distance
                crowding_distance[sorted_rank_indices[0]] = np.inf
                crowding_distance[sorted_rank_indices[-1]] = np.inf

                # Calculate crowding distance for middle solutions
                obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        crowding_distance[sorted_rank_indices[i]] += (
                            (obj_values[sorted_indices[i + 1]] - obj_values[sorted_indices[i - 1]]) / obj_range
                        )

        return crowding_distance

    def _tournament_selection(
        self,
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        pop_size: int
    ) -> int:
        """Tournament selection for NSGA-II."""
        candidates = self._rng.choice(pop_size, 2, replace=False)
        best_idx = candidates[0]

        for candidate in candidates[1:]:
            if (ranks[candidate] < ranks[best_idx] or
                (ranks[candidate] == ranks[best_idx] and
                 crowding_distances[candidate] > crowding_distances[best_idx])):
                best_idx = candidate

        return best_idx

    def _environmental_selection(
        self, objectives: np.ndarray, target_size: int
    ) -> np.ndarray:
        """Select best individuals for next generation."""
        ranks = self._fast_non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, ranks)

        selected_indices: List[int] = []
        for rank in np.unique(ranks):
            rank_indices = np.where(ranks == rank)[0]
            if len(selected_indices) + len(rank_indices) <= target_size:
                selected_indices.extend(rank_indices)
            else:
                # Sort by crowding distance and select best
                remaining = target_size - len(selected_indices)
                rank_crowding = crowding_distances[rank_indices]
                sorted_indices = np.argsort(rank_crowding)[::-1]
                selected_indices.extend(rank_indices[sorted_indices[:remaining]])
                break

        return np.array(selected_indices)

    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        eta_c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX).

        SBX simulates the behavior of single-point crossover for binary strings
        in the continuous domain. Children are generated symmetrically around
        the parents with a spread factor beta.

        Reference:
            Deb, K. and Agrawal, R.B. (1995). Simulated binary crossover for
            continuous search space. Complex Systems, 9(2), 115-148.

        Args:
            p1: First parent (normalized [0,1])
            p2: Second parent (normalized [0,1])
            eta_c: Distribution index controlling spread (higher = closer to parents)

        Returns:
            Tuple of two children
        """
        c1, c2 = p1.copy(), p2.copy()
        n_params = len(p1)

        for i in range(n_params):
            # SBX_SWAP_PROBABILITY = 0.5: equal chance of swapping each gene
            # SBX_EPSILON = 1e-9: minimum parent distance to avoid numerical issues
            if (self._rng.random() < NSGA2Defaults.SBX_SWAP_PROBABILITY and
                    abs(p1[i] - p2[i]) > NSGA2Defaults.SBX_EPSILON):
                # Order parents so y1 <= y2
                if p1[i] < p2[i]:
                    y1, y2 = p1[i], p2[i]
                else:
                    y1, y2 = p2[i], p1[i]

                # Generate spread factor using polynomial distribution
                rand = self._rng.random()

                # Beta calculation for bounded SBX (Deb & Agrawal 1995, Eq. 9-11)
                # beta = (2 * y1 / (y2 - y1))^(eta+1) at lower bound
                # The formula ensures children stay within [0, 1] bounds
                beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta_c + 1.0)

                # Compute betaq from polynomial distribution
                # This creates a distribution biased toward children near parents
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

                # Generate children symmetrically around parent midpoint
                c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                # Ensure children stay within bounds
                c1[i] = np.clip(c1[i], 0, 1)
                c2[i] = np.clip(c2[i], 0, 1)

        return c1, c2

    def _polynomial_mutation(
        self,
        solution: np.ndarray,
        eta_m: float,
        mutation_rate: float
    ) -> np.ndarray:
        """
        Polynomial mutation operator.

        Creates a perturbation using polynomial probability distribution.
        The distribution is symmetric around zero with higher probability
        for small perturbations (controlled by eta_m).

        Reference:
            Deb, K. and Goyal, M. (1996). A combined genetic adaptive search
            (GeneAS) for engineering design. Computer Science and Informatics, 26, 30-45.

        Args:
            solution: Solution to mutate (normalized [0,1])
            eta_m: Distribution index (higher = smaller mutations)
            mutation_rate: Probability of mutating each gene

        Returns:
            Mutated solution
        """
        mutated = solution.copy()
        n_params = len(solution)

        for i in range(n_params):
            if self._rng.random() < mutation_rate:
                y = mutated[i]
                # Distance to bounds (used to bias mutation toward feasible region)
                delta1 = y - 0.0  # Distance to lower bound
                delta2 = 1.0 - y  # Distance to upper bound

                rand = self._rng.random()
                # Mutation power: 1/(eta_m + 1) controls perturbation magnitude
                mut_pow = 1.0 / (eta_m + 1.0)

                # Polynomial distribution biased by distance to bounds
                # (Deb & Goyal 1996, Equations 4-5)
                if rand < 0.5:
                    # Perturbation toward lower bound
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    # Perturbation toward upper bound
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutated[i] = y + deltaq
                mutated[i] = np.clip(mutated[i], 0, 1)

        return mutated
