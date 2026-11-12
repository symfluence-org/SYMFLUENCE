#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Base Algorithm Interface

Abstract base class for optimization algorithms using the Strategy pattern.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


# Type alias for native gradient callback
# Signature: (x_normalized: np.ndarray) -> Tuple[loss: float, gradient: np.ndarray]
NativeGradientCallback = Callable[[np.ndarray], Tuple[float, np.ndarray]]


class OptimizationAlgorithm(ConfigMixin, ABC):
    """
    Abstract base class for optimization algorithms.

    Algorithms receive evaluation callbacks from the optimizer and return
    optimization results. This allows algorithms to be easily swapped
    while maintaining a consistent interface.
    """

    def __init__(self, config: Union['SymfluenceConfig', Dict[str, Any]], logger):
        """
        Initialize the algorithm.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
        """
        # Use centralized config coercion (handles dict -> SymfluenceConfig with fallback)
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        # Common algorithm parameters - use _get_config_value for typed access
        self.max_iterations = self._get_config_value(
            lambda: self.config.optimization.iterations,
            dict_key='NUMBER_OF_ITERATIONS',
        )
        self.population_size = self._get_config_value(
            lambda: self.config.optimization.population_size,
            dict_key='POPULATION_SIZE',
        )
        self.target_metric = self._get_config_value(
            lambda: self.config.optimization.metric,
            dict_key='OPTIMIZATION_METRIC',
        )
        self.penalty_score = ModelDefaults.PENALTY_SCORE
        self.random_seed = self._get_config_value(
            lambda: self.config.system.random_seed,
            default=None,
            dict_key='RANDOM_SEED'
        )

    @property
    def _rng(self) -> np.random.Generator:
        """Lazily-created per-run RNG.

        ``optimize()`` assigns a fresh one at the start of every run; accessing
        it before then (e.g. an operator exercised directly in a unit test)
        creates one on demand rather than failing.
        """
        rng = self.__dict__.get("_rng_instance")
        if rng is None:
            rng = self._new_rng()
            self.__dict__["_rng_instance"] = rng
        return rng

    @_rng.setter
    def _rng(self, value: np.random.Generator) -> None:
        self.__dict__["_rng_instance"] = value

    def _new_rng(self) -> np.random.Generator:
        """A fresh seeded RNG for one optimization run.

        Population algorithms (NSGA-II, MOEA/D, GA) draw selection, crossover
        and mutation randomness from this instead of the process-global
        ``np.random``. The global state is seeded once and then shared with
        everything else in the process, so any intervening consumer — or a
        run-dependent number of draws, e.g. when a model crashes on some
        parameter sets and not others — desynchronises it and the run diverges.
        Two runs of the same config on the same machine were producing KGE
        0.843 vs 0.728 for exactly this reason. A dedicated Generator seeded
        per run isolates the algorithm's randomness and makes it reproducible.
        """
        seed = self.random_seed
        if seed in (None, "None", ""):
            return np.random.default_rng()
        return np.random.default_rng(int(seed))

    def _initial_guess(self, n_params: int, kwargs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Normalized ``INITIAL_GUESS`` passed by the optimizer, or None.

        ``BaseModelOptimizer`` normalizes the config's INITIAL_GUESS and
        passes it as ``kwargs['initial_guess']``. Returns it clipped to
        [0, 1], or None when absent or of the wrong length.
        """
        guess = kwargs.get('initial_guess')
        if guess is None:
            return None
        guess = np.asarray(guess, dtype=float).reshape(-1)
        if guess.size != n_params:
            self.logger.warning(
                f"Ignoring initial_guess with {guess.size} values for {n_params} parameters"
            )
            return None
        return np.clip(guess, 0.0, 1.0)

    def _start_point(self, n_params: int, kwargs: Dict[str, Any], default: np.ndarray) -> np.ndarray:
        """Start point for single-point methods: the initial guess, else ``default``."""
        guess = self._initial_guess(n_params, kwargs)
        if guess is None:
            return default
        self.logger.info(f"{self.name}: starting from INITIAL_GUESS")
        return guess

    def _seed_population(self, population: np.ndarray, kwargs: Dict[str, Any]) -> np.ndarray:
        """Put the initial guess (if any) into the population as member 0."""
        guess = self._initial_guess(population.shape[1], kwargs)
        if guess is not None and len(population) > 0:
            population[0] = guess
            self.logger.info(f"{self.name}: seeded INITIAL_GUESS as population member 0")
        return population

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name (e.g., 'DDS', 'PSO', 'NSGA-II')."""
        pass

    @abstractmethod
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
        compute_gradient: Optional[NativeGradientCallback] = None,
        gradient_mode: str = 'auto',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the optimization algorithm.

        Args:
            n_params: Number of parameters to optimize
            evaluate_solution: Callback to evaluate a single normalized solution,
                ``evaluate_solution(x, step_id)``. The optional second argument
                is the algorithm's own step/iteration counter; it does not
                select a worker process (single evaluations always run in
                process 0).
            evaluate_population: Callback to evaluate a population of normalized solutions
            denormalize_params: Callback to convert normalized params to dict
            record_iteration: Callback to record iteration results
            update_best: Callback to update best solution
            log_progress: Callback to log optimization progress
            evaluate_population_objectives: Optional callback for multi-objective evaluation
            compute_gradient: Optional callback for native gradient computation.
                Signature: (x_normalized: np.ndarray) -> Tuple[loss, gradient_array]
                When provided, gradient-based algorithms can use this instead of
                finite differences for ~N times faster gradient computation.
            gradient_mode: How to compute gradients for gradient-based algorithms:
                - 'auto': Use native gradients if compute_gradient provided, else FD
                - 'native': Require native gradients (error if compute_gradient is None)
                - 'finite_difference': Always use FD even if native available
                Ignored by non-gradient algorithms (DDS, PSO, etc.)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dictionary containing:
                - best_solution: Best normalized solution found
                - best_score: Best fitness score
                - best_params: Best parameters as dictionary
                - history: Optimization history
        """
        pass

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to [0, 1] bounds."""
        return np.clip(x, 0, 1)

    def _reflect_at_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        Reflect solutions at bounds instead of clipping.

        This often produces better exploration than simple clipping.
        """
        result = x.copy()
        for i in range(len(result)):
            while result[i] < 0 or result[i] > 1:
                if result[i] < 0:
                    result[i] = -result[i]
                if result[i] > 1:
                    result[i] = 2.0 - result[i]
        return result

    # =========================================================================
    # Gradient computation utilities (for gradient-based algorithms)
    # =========================================================================

    def _should_use_native_gradients(
        self,
        compute_gradient: Optional[NativeGradientCallback],
        gradient_mode: str
    ) -> bool:
        """
        Determine whether to use native gradients based on availability and mode.

        Args:
            compute_gradient: Native gradient callback (None if not available)
            gradient_mode: User-specified mode ('auto', 'native', 'finite_difference')

        Returns:
            True if native gradients should be used

        Raises:
            ValueError: If gradient_mode='native' but compute_gradient is None
        """
        if gradient_mode == 'finite_difference':
            return False

        if gradient_mode == 'native':
            if compute_gradient is None:
                raise ValueError(
                    "gradient_mode='native' requires a compute_gradient callback, "
                    "but none was provided. The model may not support native gradients. "
                    "Use gradient_mode='auto' or 'finite_difference' instead."
                )
            return True

        # 'auto' mode: use native if available
        return compute_gradient is not None

    def _create_gradient_function(
        self,
        compute_gradient: Optional[NativeGradientCallback],
        evaluate_solution: Callable[[np.ndarray, int], float],
        gradient_mode: str,
        epsilon: float = 1e-4
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """
        Create unified gradient function that uses either native or FD gradients.

        This allows gradient-based algorithms to use a single interface regardless
        of whether native gradients are available.

        Args:
            compute_gradient: Native gradient callback (may be None)
            evaluate_solution: Fallback evaluation function for FD
            gradient_mode: Gradient computation mode
            epsilon: Perturbation size for finite differences

        Returns:
            Function with signature (x: np.ndarray) -> Tuple[fitness, gradient]
            where fitness is the objective value (higher is better) and gradient
            points in the direction of increasing fitness (for gradient ascent).

        Note:
            The returned function handles the sign conventions:
            - Native gradients compute d(loss)/d(x) for minimization
            - Returned gradients are for MAXIMIZATION (gradient ascent)
            - Both fitness and gradient are negated appropriately
        """
        use_native = self._should_use_native_gradients(compute_gradient, gradient_mode)

        if use_native and compute_gradient is not None:
            # Use native gradient callback
            def native_gradient_func(x: np.ndarray) -> Tuple[float, np.ndarray]:
                # Native callback returns (loss, grad) for minimization
                # We need (fitness, grad) for maximization
                loss, grad = compute_gradient(x)
                fitness = -loss  # Convert loss to fitness
                ascent_grad = -grad  # Convert descent to ascent direction
                return fitness, ascent_grad

            return native_gradient_func

        else:
            # Use finite difference gradients
            def fd_gradient_func(x: np.ndarray) -> Tuple[float, np.ndarray]:
                return self._compute_fd_gradients(x, evaluate_solution, epsilon)

            return fd_gradient_func

    def _compute_fd_gradients(
        self,
        x: np.ndarray,
        evaluate_func: Callable[[np.ndarray, int], float],
        epsilon: float
    ) -> Tuple[float, np.ndarray]:
        """
        Compute gradients using central finite differences.

        Central difference formula (O(ε²) accuracy):
            ∂f/∂x_i ≈ (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)

        Args:
            x: Current parameter values (normalized [0,1])
            evaluate_func: Function to evaluate fitness: f = evaluate_func(x, step_id)
            epsilon: Perturbation size (typically 1e-4)

        Returns:
            Tuple of (f_center, gradient_array):
            - f_center: Fitness at current point
            - gradient_array: Approximate gradient (shape: n_params)

        Note:
            Cost: 2*n_params + 1 function evaluations per call
        """
        n_params = len(x)
        gradient = np.zeros(n_params)

        # Evaluate at current point
        f_center = evaluate_func(x, 0)

        # Compute central differences
        for i in range(n_params):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] = min(1.0, x[i] + epsilon)
            x_minus[i] = max(0.0, x[i] - epsilon)

            f_plus = evaluate_func(x_plus, 0)
            f_minus = evaluate_func(x_minus, 0)

            # Central difference (for maximization, gradient points uphill).
            # Divide by the step actually taken: at a bound one side is
            # clipped, and dividing by 2*eps would halve the slope there.
            step = x_plus[i] - x_minus[i]
            gradient[i] = (f_plus - f_minus) / step if step > 0 else 0.0

        return f_center, gradient

    def _clip_gradient(self, gradient: np.ndarray, clip_value: float) -> np.ndarray:
        """
        Clip gradient norm to prevent exploding gradients.

        Rescales gradient to have maximum L2 norm of clip_value,
        preserving direction but reducing magnitude if necessary.

        Args:
            gradient: Gradient vector (shape: n_params)
            clip_value: Maximum L2 norm allowed. Set to 0 or negative to disable.

        Returns:
            np.ndarray: Clipped gradient with ||g|| ≤ clip_value (or unclipped if disabled)
        """
        # Replace NaN gradients with zero
        if np.any(np.isnan(gradient)):
            nan_count = np.sum(np.isnan(gradient))
            self.logger.warning(
                f"NaN detected in {nan_count}/{len(gradient)} gradient elements, replacing with 0"
            )
            gradient = np.where(np.isnan(gradient), 0.0, gradient)

        # clip_value <= 0 disables clipping
        if clip_value <= 0:
            return gradient

        norm = np.linalg.norm(gradient)
        if norm > clip_value:
            gradient = gradient * (clip_value / norm)
        return gradient
