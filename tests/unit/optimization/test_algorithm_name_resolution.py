"""Algorithm names from config resolve through the registry, not a side table."""
from __future__ import annotations

import logging

import pytest

from symfluence.core.calibration.optimizers.algorithms import (
    ALGORITHM_REGISTRY,
    get_algorithm,
    list_algorithms,
    resolve_algorithm_name,
)

pytestmark = pytest.mark.unit

# Every spelling the old OptimizationManager name->method table accepted,
# with the name its run_<algo>() wrapper passed to run_optimization().
LEGACY_TABLE = {
    'DDS': 'dds', 'ASYNC-DDS': 'async_dds', 'ASYNCDDS': 'async_dds', 'ASYNC_DDS': 'async_dds',
    'PSO': 'pso', 'SCE-UA': 'sce-ua', 'DE': 'de', 'NSGA-II': 'nsga2',
    'ADAM': 'adam', 'LBFGS': 'lbfgs', 'CMA-ES': 'cmaes', 'CMAES': 'cmaes',
    'DREAM': 'dream', 'GLUE': 'glue',
    'BASIN-HOPPING': 'basin_hopping', 'BASINHOPPING': 'basin_hopping', 'BH': 'basin_hopping',
    'NELDER-MEAD': 'nelder_mead', 'NELDERMEAD': 'nelder_mead', 'NM': 'nelder_mead',
    'SIMPLEX': 'nelder_mead', 'GA': 'ga',
    'BAYESIAN-OPT': 'bayesian_opt', 'BAYESIAN_OPT': 'bayesian_opt', 'BAYESIAN': 'bayesian_opt',
    'BO': 'bayesian_opt', 'MOEAD': 'moead', 'MOEA-D': 'moead', 'MOEA_D': 'moead',
    'SIMULATED-ANNEALING': 'simulated_annealing', 'SIMULATED_ANNEALING': 'simulated_annealing',
    'SA': 'simulated_annealing', 'ANNEALING': 'simulated_annealing',
    'ABC': 'abc', 'ABC-SMC': 'abc', 'ABC_SMC': 'abc', 'APPROXIMATE-BAYESIAN': 'abc',
}


@pytest.mark.parametrize("name,expected", sorted(LEGACY_TABLE.items()))
def test_legacy_spellings_resolve_to_same_primary(name, expected):
    assert resolve_algorithm_name(name) == expected


@pytest.mark.parametrize("name,expected", [
    # Rejected by the old side table although the registry has them.
    ('L-BFGS', 'lbfgs'), ('SCE_UA', 'sce-ua'), ('SCEUA', 'sce-ua'), ('NSGA2', 'nsga2'),
    # Lower-case config values.
    ('dds', 'dds'), ('pso', 'pso'),
])
def test_registry_names_previously_rejected(name, expected):
    assert resolve_algorithm_name(name) == expected


def test_every_registry_key_is_reachable():
    # Hyphenated keys ('nsga-ii', 'l-bfgs', 'cma-es') used to be dead because
    # only the query was normalized.
    logger = logging.getLogger("test")
    for key, cls in ALGORITHM_REGISTRY.items():
        assert type(get_algorithm(key, {}, logger)) is cls, key


def test_every_primary_resolves_to_itself():
    for primary in list_algorithms():
        assert resolve_algorithm_name(primary) == primary


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        resolve_algorithm_name("not-an-algorithm")
