#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Optimization Algorithms Package

This package contains optimization algorithms implemented using the Strategy pattern.
Each algorithm can be used interchangeably through the common OptimizationAlgorithm interface.

Usage:
    from symfluence.core.calibration.optimizers.algorithms import get_algorithm

    algorithm = get_algorithm('dds', config, logger)
    result = algorithm.optimize(...)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from .abc import ABCAlgorithm
from .adam import AdamAlgorithm
from .async_dds import AsyncDDSAlgorithm
from .base_algorithm import OptimizationAlgorithm
from .basin_hopping import BasinHoppingAlgorithm
from .bayesian_optimization import BayesianOptimizationAlgorithm
from .cmaes import CMAESAlgorithm
from .config_schema import (
    CMAESDefaults,
    DREAMDefaults,
    NSGA2Defaults,
    PSODefaults,
    get_algorithm_defaults,
    validate_hyperparameters,
)
from .dds import DDSAlgorithm
from .de import DEAlgorithm
from .dream import DREAMAlgorithm
from .ga import GAAlgorithm
from .glue import GLUEAlgorithm
from .lbfgs import LBFGSAlgorithm
from .moead import MOEADAlgorithm
from .nelder_mead import NelderMeadAlgorithm
from .nsga2 import NSGA2Algorithm
from .pso import PSOAlgorithm
from .sce_ua import SCEUAAlgorithm
from .simulated_annealing import SimulatedAnnealingAlgorithm

# Algorithm registry mapping names to classes
ALGORITHM_REGISTRY: Dict[str, Type[OptimizationAlgorithm]] = {
    'dds': DDSAlgorithm,
    'pso': PSOAlgorithm,
    'de': DEAlgorithm,
    'sce-ua': SCEUAAlgorithm,
    'sce_ua': SCEUAAlgorithm,  # Alternative name
    'sceua': SCEUAAlgorithm,   # Alternative name
    'async_dds': AsyncDDSAlgorithm,
    'asyncdds': AsyncDDSAlgorithm,  # Alternative name
    'nsga2': NSGA2Algorithm,
    'nsga-ii': NSGA2Algorithm,  # Alternative name
    'adam': AdamAlgorithm,
    'lbfgs': LBFGSAlgorithm,
    'l-bfgs': LBFGSAlgorithm,  # Alternative name
    'cmaes': CMAESAlgorithm,
    'cma-es': CMAESAlgorithm,  # Alternative name
    'dream': DREAMAlgorithm,
    'glue': GLUEAlgorithm,
    'basin_hopping': BasinHoppingAlgorithm,
    'basinhopping': BasinHoppingAlgorithm,  # Alternative name
    'bh': BasinHoppingAlgorithm,  # Short alias
    'nelder_mead': NelderMeadAlgorithm,
    'neldermead': NelderMeadAlgorithm,  # Alternative name
    'nelder-mead': NelderMeadAlgorithm,  # Alternative name
    'nm': NelderMeadAlgorithm,  # Short alias
    'simplex': NelderMeadAlgorithm,  # Alternative name
    'ga': GAAlgorithm,
    'bayesian_opt': BayesianOptimizationAlgorithm,
    'bayesian': BayesianOptimizationAlgorithm,  # Alternative name
    'bo': BayesianOptimizationAlgorithm,  # Short alias
    'moead': MOEADAlgorithm,
    'moea_d': MOEADAlgorithm,  # Alternative name
    'moea-d': MOEADAlgorithm,  # Alternative name
    'simulated_annealing': SimulatedAnnealingAlgorithm,
    'sa': SimulatedAnnealingAlgorithm,  # Short alias
    'annealing': SimulatedAnnealingAlgorithm,  # Alternative name
    'abc': ABCAlgorithm,
    'abc_smc': ABCAlgorithm,  # Alternative name
    'approximate_bayesian': ABCAlgorithm,  # Alternative name
}


# Primary (non-alias) names of the built-in algorithms; these are the names
# the BaseModelOptimizer.run_<algo>() wrappers pass to run_optimization().
_BUILTIN_PRIMARY_NAMES: List[str] = [
    'dds', 'pso', 'de', 'sce-ua', 'async_dds', 'nsga2', 'adam', 'lbfgs', 'cmaes',
    'dream', 'glue', 'basin_hopping', 'nelder_mead', 'ga', 'bayesian_opt', 'moead',
    'simulated_annealing', 'abc',
]

# Primary names contributed via register_algorithm() (external packages),
# kept separate so list_algorithms() can report them alongside the built-ins.
_EXTERNAL_PRIMARY_NAMES: List[str] = []


def _normalize_name(name: str) -> str:
    """The same normalization get_algorithm() applies to lookups."""
    return name.lower().replace('-', '_').replace(' ', '_')


def register_algorithm(
    name: str,
    algorithm_class: Optional[Type[OptimizationAlgorithm]] = None,
    *,
    aliases: Tuple[str, ...] = (),
):
    """Register an optimization algorithm with the calibration engine.

    The extensibility seam for external packages (calibration contract
    family, ADR-0009): call from a plugin's ``register()``, or use as a
    decorator. Names are normalized the same way :func:`get_algorithm`
    normalizes lookups, so every registered spelling is reachable.

    Args:
        name: Primary algorithm name.
        algorithm_class: The :class:`OptimizationAlgorithm` subclass; omit to
            use as a decorator.
        aliases: Additional accepted spellings.

    Raises:
        ValueError: On collision with an existing (built-in or registered)
            algorithm name.
    """
    def _register(cls: Type[OptimizationAlgorithm]) -> Type[OptimizationAlgorithm]:
        keys = [_normalize_name(name)] + [_normalize_name(a) for a in aliases]
        existing = {_normalize_name(k) for k in ALGORITHM_REGISTRY}
        taken = [k for k in keys if k in existing]
        if taken:
            raise ValueError(
                f"Algorithm name(s) already registered: {taken}. "
                f"Pick a different name/alias."
            )
        for key in keys:
            ALGORITHM_REGISTRY[key] = cls
        _EXTERNAL_PRIMARY_NAMES.append(_normalize_name(name))
        return cls

    if algorithm_class is None:
        return _register
    return _register(algorithm_class)


def get_algorithm(
    name: str,
    config: Dict[str, Any],
    logger: logging.Logger
) -> OptimizationAlgorithm:
    """
    Get an optimization algorithm instance by name.

    Args:
        name: Algorithm name (case-insensitive). Supported values:
              'dds', 'pso', 'de', 'sce-ua', 'async_dds', 'nsga2'
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Instantiated algorithm

    Raises:
        ValueError: If algorithm name is not recognized
    """
    return _lookup_class(name)(config, logger)


def _lookup_class(name: str) -> Type[OptimizationAlgorithm]:
    """Registry lookup with the same normalization on both sides.

    Registry keys are normalized too, so hyphenated aliases ('nsga-ii',
    'l-bfgs', 'cma-es') are reachable; normalizing only the query made them
    dead entries.
    """
    lookup = {_normalize_name(k): cls for k, cls in ALGORITHM_REGISTRY.items()}
    cls = lookup.get(_normalize_name(name))
    if cls is None:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available algorithms: {list_algorithms()}"
        )
    return cls


def resolve_algorithm_name(name: str) -> str:
    """Map any accepted spelling/alias to the algorithm's primary name.

    E.g. 'NSGA-II' -> 'nsga2', 'L-BFGS' -> 'lbfgs', 'SCE_UA' -> 'sce-ua'.
    The primary name is what ``BaseModelOptimizer.run_<algo>()`` passes to
    ``run_optimization`` (it drives result file names and per-algorithm
    branches), so callers dispatching by config string should use this.

    Raises:
        ValueError: If the name is not a registered algorithm.
    """
    cls = _lookup_class(name)
    for primary in [*_BUILTIN_PRIMARY_NAMES, *_EXTERNAL_PRIMARY_NAMES]:
        if _lookup_class(primary) is cls:
            return primary
    return _normalize_name(name)  # registered directly into ALGORITHM_REGISTRY


def algorithm_slug(name: str) -> str:
    """Canonical, filesystem-safe slug for an algorithm's result files.

    One name per algorithm regardless of how it was spelled: the primary
    registry name (``resolve_algorithm_name``), e.g. 'NSGA-II' / 'nsga2' ->
    'nsga2', 'CMA-ES' -> 'cmaes', 'MOEA/D' / 'moea_d' -> 'moead'. Result files
    used to be named after the algorithm class's display name
    (``algorithm.name.lower()``: 'nsga-ii', 'cma-es', 'asyncdds', 'moea_d',
    'sa', ...) while readers looked them up under the configured name, so the
    two only met for algorithms whose names happened to coincide. Unregistered
    names fall back to plain normalization.
    """
    try:
        slug = resolve_algorithm_name(name)
    except ValueError:
        slug = _normalize_name(str(name))
    return slug.replace('/', '_')


def list_algorithms() -> list:
    """
    List all available algorithm names.

    Returns:
        Sorted list of primary algorithm names
    """
    # Return only primary names (not aliases), built-in + externally registered
    return sorted(_BUILTIN_PRIMARY_NAMES + _EXTERNAL_PRIMARY_NAMES)


__all__ = [
    'register_algorithm',
    'resolve_algorithm_name',
    'algorithm_slug',
    # Base class
    'OptimizationAlgorithm',
    # Algorithm classes
    'DDSAlgorithm',
    'PSOAlgorithm',
    'DEAlgorithm',
    'SCEUAAlgorithm',
    'AsyncDDSAlgorithm',
    'NSGA2Algorithm',
    'AdamAlgorithm',
    'LBFGSAlgorithm',
    'CMAESAlgorithm',
    'DREAMAlgorithm',
    'GLUEAlgorithm',
    'BasinHoppingAlgorithm',
    'NelderMeadAlgorithm',
    'GAAlgorithm',
    'BayesianOptimizationAlgorithm',
    'MOEADAlgorithm',
    'SimulatedAnnealingAlgorithm',
    'ABCAlgorithm',
    # Config schema classes
    'CMAESDefaults',
    'NSGA2Defaults',
    'DREAMDefaults',
    'PSODefaults',
    'get_algorithm_defaults',
    'validate_hyperparameters',
    # Factory functions
    'get_algorithm',
    'list_algorithms',
    # Registry
    'ALGORITHM_REGISTRY',
]
