#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Simulated Annealing (SA)

A probabilistic optimization algorithm inspired by the annealing process in
metallurgy. Uses a temperature-based acceptance criterion to escape local
minima by occasionally accepting worse solutions.

Key Features:
    - Escapes local minima via probabilistic acceptance
    - Temperature schedule controls exploration vs exploitation
    - Simple single-solution algorithm (low memory)
    - Effective for rugged fitness landscapes

Reference:
    Kirkpatrick, S., Gelatt, C.D., and Vecchi, M.P. (1983). Optimization by
    Simulated Annealing. Science, 220(4598), 671-680.

    Cerny, V. (1985). Thermodynamical approach to the traveling salesman
    problem: An efficient simulation algorithm. Journal of Optimization
    Theory and Applications, 45(1), 41-51.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import SADefaults


class SimulatedAnnealingAlgorithm(OptimizationAlgorithm):
    """Simulated Annealing global optimization."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "SA"

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
        Run Simulated Annealing optimization.

        The algorithm:
        1. Start with random solution and high temperature
        2. Generate neighbor by perturbing current solution
        3. Accept if better, or with probability exp(-delta/T) if worse
        4. Reduce temperature according to cooling schedule
        5. Repeat until temperature is cold or max iterations reached

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
        self.logger.info(f"Starting Simulated Annealing with {n_params} parameters")

        # SA parameters from config
        initial_temp = self._get_config_value(
            lambda: self.config.optimization.sa.initial_temp,
            default=SADefaults.INITIAL_TEMP,
            dict_key='SA_INITIAL_TEMP'
        )
        final_temp = self._get_config_value(
            lambda: self.config.optimization.sa.final_temp,
            default=SADefaults.FINAL_TEMP,
            dict_key='SA_FINAL_TEMP'
        )
        cooling_schedule = self._get_config_value(
            lambda: self.config.optimization.sa.cooling_schedule,
            default=SADefaults.COOLING_SCHEDULE,
            dict_key='SA_COOLING_SCHEDULE'
        )
        cooling_rate = self._get_config_value(
            lambda: self.config.optimization.sa.cooling_rate,
            default=SADefaults.COOLING_RATE,
            dict_key='SA_COOLING_RATE'
        )
        step_size = self._get_config_value(
            lambda: self.config.optimization.sa.step_size,
            default=SADefaults.STEP_SIZE,
            dict_key='SA_STEP_SIZE'
        )
        steps_per_temp = self._get_config_value(
            lambda: self.config.optimization.sa.steps_per_temp,
            default=SADefaults.STEPS_PER_TEMP,
            dict_key='SA_STEPS_PER_TEMP'
        )
        adaptive_step = self._get_config_value(
            lambda: self.config.optimization.sa.adaptive_step,
            default=SADefaults.ADAPTIVE_STEP,
            dict_key='SA_ADAPTIVE_STEP'
        )

        # Validate temperature parameters
        valid, msg = SADefaults.validate_temperatures(initial_temp, final_temp)
        if not valid:
            self.logger.warning(f"SA validation: {msg}")

        self.logger.info(
            f"SA settings: T0={initial_temp}, Tf={final_temp}, "
            f"cooling={cooling_schedule}, rate={cooling_rate}"
        )

        # Initialize current solution randomly
        current = self._start_point(n_params, kwargs, self._rng.uniform(0, 1, n_params))
        current_fit = evaluate_solution(current, 0)

        # Track best solution (global best)
        best_pos = current.copy()
        best_fit = current_fit

        # Temperature
        temperature = initial_temp

        # Tracking for adaptive step size
        n_accepted = 0
        n_total = 0
        target_acceptance = 0.4  # Target 40% acceptance rate

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, 1, best_fit)

        # Main SA loop
        for iteration in range(1, self.max_iterations + 1):
            n_accepted_iter = 0

            # Multiple steps at each temperature
            for _ in range(steps_per_temp):
                n_total += 1

                # Generate neighbor by perturbing current solution
                neighbor = self._generate_neighbor(current, step_size, n_params)

                # Evaluate neighbor
                neighbor_fit = evaluate_solution(neighbor, iteration)

                # Calculate delta (we're maximizing, so better = higher)
                delta = neighbor_fit - current_fit

                # Accept or reject
                if delta > 0:
                    # Better solution - always accept
                    current = neighbor
                    current_fit = neighbor_fit
                    n_accepted += 1
                    n_accepted_iter += 1
                else:
                    # Worse solution - accept with probability
                    # For maximization: P = exp(delta / T) where delta < 0
                    acceptance_prob = np.exp(delta / temperature)
                    if self._rng.random() < acceptance_prob:
                        current = neighbor
                        current_fit = neighbor_fit
                        n_accepted += 1
                        n_accepted_iter += 1

                # Update global best
                if current_fit > best_fit:
                    best_fit = current_fit
                    best_pos = current.copy()

            # Adaptive step size based on acceptance rate
            if adaptive_step and n_total > 0 and iteration % 10 == 0:
                acceptance_rate = n_accepted / n_total
                if acceptance_rate > target_acceptance + 0.1:
                    # Too many acceptances - increase step size
                    step_size = min(0.5, step_size * 1.1)
                elif acceptance_rate < target_acceptance - 0.1:
                    # Too few acceptances - decrease step size
                    step_size = max(0.01, step_size * 0.9)
                # Reset counters
                n_accepted = 0
                n_total = 0

            # Cool down temperature
            temperature = self._cool_temperature(
                temperature, initial_temp, iteration,
                cooling_schedule, cooling_rate, final_temp
            )

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(
                iteration, best_fit, params_dict,
                {'temperature': temperature, 'accepted': n_accepted_iter, 'step_size': step_size}
            )
            update_best(best_fit, params_dict, iteration)

            # Log progress
            # Progress line (tracker throttles emission); Improved counts
            # accepted moves at this temperature level.
            log_progress(
                self.name, iteration, best_fit, n_accepted_iter, steps_per_temp,
                unit='loops'
            )

            # Check if frozen
            if temperature < final_temp:
                self.logger.info(f"Temperature reached minimum at iteration {iteration}")
                break

        self.logger.info(f"Simulated Annealing complete. Final temp: {temperature:.2e}")

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'final_solution': current,
            'final_score': current_fit,
            'final_temperature': temperature,
        }

    def _generate_neighbor(
        self,
        current: np.ndarray,
        step_size: float,
        n_params: int
    ) -> np.ndarray:
        """
        Generate a neighbor solution by perturbing current solution.

        Uses Gaussian perturbation with adaptive step size.

        Args:
            current: Current solution
            step_size: Standard deviation for perturbation
            n_params: Number of parameters

        Returns:
            Neighbor solution (clipped to [0, 1])
        """
        # Gaussian perturbation
        perturbation = self._rng.normal(0, step_size, n_params)
        neighbor = current + perturbation

        # Clip to bounds
        neighbor = np.clip(neighbor, 0, 1)

        return neighbor

    def _cool_temperature(
        self,
        current_temp: float,
        initial_temp: float,
        iteration: int,
        schedule: str,
        cooling_rate: float,
        min_temp: float
    ) -> float:
        """
        Apply cooling schedule to reduce temperature.

        Supported schedules:
            - 'exponential': T = T * alpha (geometric cooling)
            - 'linear': T = T0 * (1 - k/max_k)
            - 'logarithmic': T = T0 / log(1 + k)
            - 'adaptive': Slower cooling when improvements found

        Args:
            current_temp: Current temperature
            initial_temp: Initial temperature
            iteration: Current iteration
            schedule: Cooling schedule name
            cooling_rate: Cooling rate parameter (alpha)
            min_temp: Minimum temperature

        Returns:
            New temperature
        """
        if schedule == 'exponential':
            # Classic geometric cooling: T_new = T * alpha
            new_temp = current_temp * cooling_rate

        elif schedule == 'linear':
            # Linear cooling: T = T0 * (1 - k/max_k)
            progress = iteration / self.max_iterations
            new_temp = initial_temp * (1 - progress)

        elif schedule == 'logarithmic':
            # Logarithmic cooling: T = T0 / log(1 + k)
            new_temp = initial_temp / np.log(1 + iteration)

        elif schedule == 'quadratic':
            # Quadratic cooling: slower at start, faster at end
            progress = iteration / self.max_iterations
            new_temp = initial_temp * (1 - progress) ** 2

        elif schedule == 'boltzmann':
            # Boltzmann schedule: T = T0 / (1 + k)
            new_temp = initial_temp / (1 + iteration)

        else:
            # Default to exponential
            new_temp = current_temp * cooling_rate

        return max(new_temp, min_temp)
