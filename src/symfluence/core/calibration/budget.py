# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration evaluation budget.

``NUMBER_OF_ITERATIONS`` is one number for every model, so a 4-parameter GR
and a 30-parameter SUMMA calibration get the same budget -- a much smaller
budget *per dimension* for the larger model, which biases a multi-model
comparison towards parsimonious models for reasons unrelated to their
structure.

``CALIBRATION_BUDGET_MODE: per_parameter`` makes the budget proportional to
the number of calibrated parameters::

    evaluations = clip(EVALUATIONS_PER_PARAMETER * n_params,
                       CALIBRATION_BUDGET_MIN_EVALUATIONS,
                       CALIBRATION_BUDGET_MAX_EVALUATIONS)

and converts it to the algorithm's iteration unit: one evaluation per
iteration for single-point searches (DDS, SA, Nelder-Mead, ...), about
``POPULATION_SIZE`` per iteration (generation) for population algorithms.
The default ``fixed`` mode keeps ``NUMBER_OF_ITERATIONS`` unchanged.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

#: Algorithms (by their ``OptimizationAlgorithm.name``) whose iteration
#: evaluates a whole population of about ``POPULATION_SIZE`` candidates.
#: SCE-UA and the rest evaluate a varying number per iteration; for them the
#: conversion is an approximation, recorded as such.
POPULATION_ALGORITHMS = frozenset({
    'PSO', 'DE', 'GA', 'NSGA-II', 'MOEA/D', 'ABC', 'DREAM', 'GLUE', 'CMA-ES', 'SCE-UA',
})

BUDGET_MODES = ('fixed', 'per_parameter')


@dataclass(frozen=True)
class CalibrationBudget:
    """The resolved budget of one calibration."""

    mode: str
    n_params: int
    iterations: int
    evaluations_per_iteration: int
    target_evaluations: int
    evaluations_per_parameter: Optional[int] = None
    min_evaluations: Optional[int] = None
    max_evaluations: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready form for calibration outputs and provenance."""
        return asdict(self)


def evaluations_per_iteration(algorithm_name: str, population_size: int) -> int:
    """Approximate model evaluations per iteration of *algorithm_name*."""
    if str(algorithm_name).upper() in POPULATION_ALGORITHMS:
        return max(1, int(population_size))
    return 1


def resolve_budget(
    mode: Optional[str],
    n_params: int,
    iterations: int,
    algorithm_name: str,
    population_size: int,
    evaluations_per_parameter: int = 100,
    min_evaluations: int = 100,
    max_evaluations: Optional[int] = None,
) -> CalibrationBudget:
    """Resolve the iteration count a calibration runs with.

    Args:
        mode: ``'fixed'`` (or None) or ``'per_parameter'``.
        n_params: Number of calibrated parameters.
        iterations: Configured ``NUMBER_OF_ITERATIONS`` (used in fixed mode).
        algorithm_name: ``OptimizationAlgorithm.name`` (e.g. ``'DDS'``).
        population_size: Configured ``POPULATION_SIZE``.
        evaluations_per_parameter: Evaluations granted per parameter.
        min_evaluations / max_evaluations: Caps on the per-parameter budget.
    """
    mode = (mode or 'fixed').lower()
    if mode not in BUDGET_MODES:
        raise ValueError(f"Unknown CALIBRATION_BUDGET_MODE {mode!r}; expected one of {BUDGET_MODES}")
    per_iter = evaluations_per_iteration(algorithm_name, population_size)
    if mode == 'fixed':
        iterations = int(iterations)
        return CalibrationBudget(
            mode='fixed', n_params=int(n_params), iterations=iterations,
            evaluations_per_iteration=per_iter, target_evaluations=iterations * per_iter,
        )
    target = int(evaluations_per_parameter) * max(1, int(n_params))
    target = max(target, int(min_evaluations))
    if max_evaluations is not None:
        target = min(target, int(max_evaluations))
    return CalibrationBudget(
        mode='per_parameter', n_params=int(n_params),
        iterations=max(1, math.ceil(target / per_iter)),
        evaluations_per_iteration=per_iter, target_evaluations=target,
        evaluations_per_parameter=int(evaluations_per_parameter),
        min_evaluations=int(min_evaluations),
        max_evaluations=None if max_evaluations is None else int(max_evaluations),
    )
