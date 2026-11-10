# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Multi-objective optimizers degrade to single-objective on all-penalty.

Some model workers advertise a multi-objective evaluation callback but return
penalty objectives for every candidate (multi-objective evaluation is not wired
up for that model). NSGA-II and MOEA/D must then fall back to single-objective
evaluation and still produce a valid calibration result, rather than aborting
the whole run -- otherwise those model x algorithm cells go blank.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from symfluence.core.calibration.optimizers.algorithms.moead import MOEADAlgorithm
from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm
from symfluence.core.constants import ModelDefaults

# Failure is signalled by exactly the penalty score (the population evaluator
# maps every failed/NaN objective to it); magnitude thresholds are gone.
PENALTY = ModelDefaults.PENALTY_SCORE
GOOD = 0.75


def _config():
    # Flat-dict config; the optimizers resolve values via _get_config_value.
    return {
        "NUMBER_OF_ITERATIONS": 1,
        "POPULATION_SIZE": 6,
        "OPTIMIZATION_METRIC": "KGE",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "NSGA-II",
    }


def _callbacks():
    """Minimal callbacks; track the best single-objective fitness seen."""
    seen = {"best": None}

    def evaluate_solution(x, iteration):
        return GOOD

    def evaluate_population(pop, iteration):
        return np.full(len(pop), GOOD)

    def evaluate_population_objectives(pop, names, iteration):
        # All-penalty multi-objective evaluation (worker can't do MO).
        return np.full((len(pop), len(names)), PENALTY)

    def denormalize_params(x):
        return {"p": float(np.ravel(x)[0])}

    def record_iteration(*a, **k):
        return None

    def update_best(score, params, iteration):
        if seen["best"] is None or score > seen["best"]:
            seen["best"] = score

    def log_progress(*a, **k):
        return None

    return seen, dict(
        evaluate_solution=evaluate_solution,
        evaluate_population=evaluate_population,
        evaluate_population_objectives=evaluate_population_objectives,
        denormalize_params=denormalize_params,
        record_iteration=record_iteration,
        update_best=update_best,
        log_progress=log_progress,
    )


@pytest.mark.parametrize("algo_cls", [NSGA2Algorithm, MOEADAlgorithm])
def test_all_penalty_multiobjective_falls_back(algo_cls):
    np.random.seed(0)
    algo = algo_cls(_config(), logging.getLogger("test.mo.fallback"))
    seen, cb = _callbacks()

    # Must NOT raise despite the multi-objective evaluation returning all-penalty.
    result = algo.optimize(
        n_params=3,
        objective_names=["KGE", "NSE"],
        multiobjective=True,
        **cb,
    )

    assert result is not None
    # The best recorded score comes from the single-objective fallback, not penalty.
    assert seen["best"] is not None
    assert seen["best"] > -900.0
    assert seen["best"] == pytest.approx(GOOD)
