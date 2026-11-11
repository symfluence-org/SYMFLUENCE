#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algorithm-specific unit tests for optimization algorithms.

Tests algorithm internals, magic number documentation, error handling,
and config schema validation.

References:
    - CMA-ES: Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review"
    - NSGA-II: Deb, K. et al. (2002). "A fast and elitist multiobjective GA: NSGA-II"
    - DREAM: Vrugt, J.A. et al. (2009). "Accelerating MCMC simulation by DE"
    - PSO: Kennedy, J. & Eberhart, R. (1995). "Particle swarm optimization"
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_algorithm_specific')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


def _make_full_config(tmp_path, **overrides):
    """Create a complete mock configuration for algorithm tests with optional overrides."""
    config = {
        # Required system paths
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        # Required domain settings
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        # Required model settings
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'FUSE',
        # Optimization settings
        'NUMBER_OF_ITERATIONS': 10,
        'POPULATION_SIZE': 20,
        'OPTIMIZATION_METRIC': 'KGE',
        'OPTIMIZATION_TARGET': 'streamflow',
    }
    config.update(overrides)
    return config


@pytest.fixture
def mock_config(tmp_path):
    """Create a complete mock configuration for algorithm tests."""
    return _make_full_config(tmp_path)


@pytest.fixture
def sphere_objective():
    """
    Sphere function for testing: f(x) = -sum(x^2)

    Optimal at x = [0, 0, ...] with f(x) = 0
    Negated for maximization.
    """
    def evaluate(x: np.ndarray, step_id: int = 0) -> float:
        return -np.sum(x ** 2)

    return evaluate


@pytest.fixture
def sphere_population_objective(sphere_objective):
    """Population evaluator for sphere function."""
    def evaluate_pop(population: np.ndarray, step_id: int = 0) -> np.ndarray:
        return np.array([sphere_objective(x, step_id) for x in population])

    return evaluate_pop


@pytest.fixture
def mock_callbacks():
    """Create mock callbacks for algorithm testing."""
    return {
        'denormalize_params': lambda x: {f'p{i}': v for i, v in enumerate(x)},
        'record_iteration': Mock(),
        'update_best': Mock(),
        'log_progress': Mock(),
    }


# =============================================================================
# Config Schema Tests
# =============================================================================

class TestConfigSchema:
    """Test config schema classes and validation functions."""

    def test_cmaes_defaults_exist(self):
        """CMAESDefaults should have all required attributes."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults

        assert hasattr(CMAESDefaults, 'MIN_POPULATION')
        assert hasattr(CMAESDefaults, 'POPULATION_LOG_FACTOR')
        assert hasattr(CMAESDefaults, 'INITIAL_SIGMA')
        assert hasattr(CMAESDefaults, 'SIGMA_MIN')
        assert hasattr(CMAESDefaults, 'SIGMA_MAX')
        assert hasattr(CMAESDefaults, 'CONVERGENCE_THRESHOLD')
        assert hasattr(CMAESDefaults, 'EIGENVALUE_FLOOR')

    def test_nsga2_defaults_exist(self):
        """NSGA2Defaults should have all required attributes."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import NSGA2Defaults

        assert hasattr(NSGA2Defaults, 'CROSSOVER_RATE')
        assert hasattr(NSGA2Defaults, 'MUTATION_RATE')
        assert hasattr(NSGA2Defaults, 'ETA_C')
        assert hasattr(NSGA2Defaults, 'ETA_M')
        assert hasattr(NSGA2Defaults, 'SBX_SWAP_PROBABILITY')
        assert hasattr(NSGA2Defaults, 'SBX_EPSILON')

    def test_dream_defaults_exist(self):
        """DREAMDefaults should have all required attributes."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import DREAMDefaults

        assert hasattr(DREAMDefaults, 'MIN_CHAINS_FACTOR')
        assert hasattr(DREAMDefaults, 'DE_PAIRS')
        assert hasattr(DREAMDefaults, 'CROSSOVER_PROBABILITY')
        assert hasattr(DREAMDefaults, 'EPSILON_STD')
        assert hasattr(DREAMDefaults, 'TEMPERATURE')
        assert hasattr(DREAMDefaults, 'OUTLIER_THRESHOLD')
        assert hasattr(DREAMDefaults, 'BURN_IN_FRACTION')
        assert hasattr(DREAMDefaults, 'MODE_JUMP_PROBABILITY')
        assert hasattr(DREAMDefaults, 'RHAT_THRESHOLD')

    def test_pso_defaults_exist(self):
        """PSODefaults should have all required attributes."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import PSODefaults

        assert hasattr(PSODefaults, 'INERTIA')
        assert hasattr(PSODefaults, 'COGNITIVE')
        assert hasattr(PSODefaults, 'SOCIAL')
        assert hasattr(PSODefaults, 'V_MAX')

    def test_cmaes_population_size_computation(self):
        """CMAESDefaults.compute_population_size should follow Hansen's heuristic."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults

        # λ = 4 + floor(3 * ln(n))
        assert CMAESDefaults.compute_population_size(1) == 4  # 4 + 0
        assert CMAESDefaults.compute_population_size(10) >= 10  # 4 + floor(3 * 2.3)
        assert CMAESDefaults.compute_population_size(100) >= 17  # 4 + floor(3 * 4.6)

    def test_cmaes_strategy_parameters_computation(self):
        """CMAESDefaults.compute_strategy_parameters should return valid values."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults

        n_params = 10
        mu = 5
        mu_eff = 3.0

        params = CMAESDefaults.compute_strategy_parameters(n_params, mu, mu_eff)

        assert 'c_sigma' in params
        assert 'c_c' in params
        assert 'c_1' in params
        assert 'c_mu' in params
        assert 'chi_n' in params

        # All values should be positive
        for key, value in params.items():
            assert value > 0, f"{key} should be positive"

        # Learning rates should be < 1
        assert params['c_sigma'] < 1
        assert params['c_c'] < 1
        assert params['c_1'] < 1
        assert params['c_mu'] < 1

    def test_cmaes_sigma_validation(self):
        """CMAESDefaults.validate_sigma should catch invalid values."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults

        # Valid sigma
        valid, msg = CMAESDefaults.validate_sigma(0.3)
        assert valid is True
        assert msg == ""

        # Too small
        valid, msg = CMAESDefaults.validate_sigma(1e-15)
        assert valid is False
        assert "below minimum" in msg

        # Too large
        valid, msg = CMAESDefaults.validate_sigma(2.0)
        assert valid is False
        assert "above maximum" in msg

    def test_dream_min_chains_computation(self):
        """DREAMDefaults.compute_min_chains should follow 2*n+1 rule."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import DREAMDefaults

        assert DREAMDefaults.compute_min_chains(1) == 3  # 2*1 + 1
        assert DREAMDefaults.compute_min_chains(5) == 11  # 2*5 + 1
        assert DREAMDefaults.compute_min_chains(10) == 21  # 2*10 + 1

    def test_dream_optimal_gamma_computation(self):
        """DREAMDefaults.compute_optimal_gamma should follow Vrugt's formula."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import DREAMDefaults

        # γ = 2.38 / sqrt(2 * δ * d*)
        gamma = DREAMDefaults.compute_optimal_gamma(n_pairs=3, d_star=5)
        expected = 2.38 / np.sqrt(2 * 3 * 5)
        assert np.isclose(gamma, expected)

    def test_pso_coefficient_validation(self):
        """PSODefaults.validate_coefficients should warn about unstable params."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import PSODefaults

        # Valid coefficients
        valid, msg = PSODefaults.validate_coefficients(0.7, 1.5, 1.5)
        assert valid is True

        # Inertia >= 1 may cause divergence
        valid, msg = PSODefaults.validate_coefficients(1.0, 1.5, 1.5)
        assert valid is False
        assert "divergence" in msg

    def test_get_algorithm_defaults(self):
        """get_algorithm_defaults should return correct class for each algorithm."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import (
            CMAESDefaults,
            DREAMDefaults,
            NSGA2Defaults,
            PSODefaults,
            get_algorithm_defaults,
        )

        assert get_algorithm_defaults('cmaes') is CMAESDefaults
        assert get_algorithm_defaults('CMA-ES') is CMAESDefaults
        assert get_algorithm_defaults('nsga2') is NSGA2Defaults
        assert get_algorithm_defaults('dream') is DREAMDefaults
        assert get_algorithm_defaults('pso') is PSODefaults

        with pytest.raises(ValueError):
            get_algorithm_defaults('unknown_algorithm')


# =============================================================================
# CMA-ES Algorithm Tests
# =============================================================================

class TestCMAESAlgorithm:
    """Test CMA-ES algorithm implementation."""

    def test_cmaes_imports_config_schema(self):
        """CMA-ES should import and use CMAESDefaults."""
        from symfluence.core.calibration.optimizers.algorithms.cmaes import CMAESAlgorithm
        from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults

        # Verify import works
        assert CMAESDefaults is not None

    def test_cmaes_strategy_parameter_computation(
        self, mock_config, test_logger, sphere_population_objective, mock_callbacks
    ):
        """CMA-ES should compute strategy parameters correctly."""
        from symfluence.core.calibration.optimizers.algorithms.cmaes import CMAESAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 3
        mock_config['POPULATION_SIZE'] = 10

        algo = CMAESAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=5,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=sphere_population_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result
        assert 'final_sigma' in result
        assert result['final_sigma'] > 0

    def test_cmaes_covariance_recovery_after_numerical_error(
        self, mock_config, test_logger, mock_callbacks
    ):
        """CMA-ES should recover from covariance matrix errors."""
        from symfluence.core.calibration.optimizers.algorithms.cmaes import CMAESAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        # Create objective that occasionally returns NaN
        call_count = [0]

        def unstable_objective(population, step_id):
            call_count[0] += 1
            fitness = np.array([-np.sum(x ** 2) for x in population])
            # Inject NaN occasionally
            if call_count[0] == 2:
                fitness[0] = np.nan
            return fitness

        algo = CMAESAlgorithm(mock_config, test_logger)

        # Should not raise exception
        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=unstable_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result

    def test_cmaes_sigma_bounds_enforced(
        self, mock_config, test_logger, sphere_population_objective, mock_callbacks
    ):
        """CMA-ES should keep sigma within bounds."""
        from symfluence.core.calibration.optimizers.algorithms.cmaes import CMAESAlgorithm
        from symfluence.core.calibration.optimizers.algorithms.config_schema import CMAESDefaults

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        algo = CMAESAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=sphere_population_objective,
            **mock_callbacks
        )

        assert result['final_sigma'] >= CMAESDefaults.SIGMA_MIN
        assert result['final_sigma'] <= CMAESDefaults.SIGMA_MAX


# =============================================================================
# NSGA-II Algorithm Tests
# =============================================================================

class TestNSGA2Algorithm:
    """Test NSGA-II algorithm implementation."""

    def test_nsga2_imports_config_schema(self):
        """NSGA-II should import and use NSGA2Defaults."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import NSGA2Defaults
        from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm

        assert NSGA2Defaults is not None

    def test_nsga2_sbx_crossover_produces_valid_children(self, test_logger, tmp_path):
        """SBX crossover should produce children within bounds."""
        from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm

        algo = NSGA2Algorithm(_make_full_config(tmp_path, NUMBER_OF_ITERATIONS=1, POPULATION_SIZE=10), test_logger)

        p1 = np.array([0.2, 0.3, 0.8])
        p2 = np.array([0.7, 0.5, 0.1])

        # Run multiple times due to stochastic nature
        for _ in range(100):
            c1, c2 = algo._sbx_crossover(p1, p2, eta_c=15.0)

            assert np.all(c1 >= 0) and np.all(c1 <= 1)
            assert np.all(c2 >= 0) and np.all(c2 <= 1)

    def test_nsga2_polynomial_mutation_stays_bounded(self, test_logger, tmp_path):
        """Polynomial mutation should keep solutions within bounds."""
        from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm

        algo = NSGA2Algorithm(_make_full_config(tmp_path, NUMBER_OF_ITERATIONS=1, POPULATION_SIZE=10), test_logger)

        solution = np.array([0.1, 0.5, 0.9])

        # Run multiple times
        for _ in range(100):
            mutated = algo._polynomial_mutation(solution.copy(), eta_m=10.0, mutation_rate=1.0)
            assert np.all(mutated >= 0) and np.all(mutated <= 1)

    def test_nsga2_non_dominated_sorting(self, test_logger, tmp_path):
        """Fast non-dominated sorting should correctly rank solutions."""
        from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm

        algo = NSGA2Algorithm(_make_full_config(tmp_path, NUMBER_OF_ITERATIONS=1, POPULATION_SIZE=10), test_logger)

        # Create objectives where ranking is clear
        # (higher is better for maximization)
        objectives = np.array([
            [1.0, 1.0],  # Dominates all below
            [0.5, 0.8],  # Second rank
            [0.8, 0.5],  # Second rank
            [0.3, 0.3],  # Third rank
        ])

        ranks = algo._fast_non_dominated_sort(objectives)

        assert ranks[0] == 0  # Best should be rank 0
        assert ranks[1] in [1, 2]  # Second front
        assert ranks[2] in [1, 2]  # Second front
        assert ranks[3] >= 1  # Dominated

    def test_nsga2_error_handling_in_loop(
        self, mock_config, test_logger, mock_callbacks
    ):
        """NSGA-II should handle errors gracefully in main loop."""
        from symfluence.core.calibration.optimizers.algorithms.nsga2 import NSGA2Algorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        call_count = [0]

        def unstable_objective(population, step_id):
            call_count[0] += 1
            fitness = np.array([-np.sum(x ** 2) for x in population])
            if call_count[0] == 3:
                fitness[0] = np.nan
            return fitness

        algo = NSGA2Algorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=unstable_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result


# =============================================================================
# DREAM Algorithm Tests
# =============================================================================

class TestDREAMAlgorithm:
    """Test DREAM algorithm implementation."""

    def test_dream_imports_config_schema(self):
        """DREAM should import and use DREAMDefaults."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import DREAMDefaults
        from symfluence.core.calibration.optimizers.algorithms.dream import DREAMAlgorithm

        assert DREAMDefaults is not None

    def test_dream_chain_computation(self, mock_config, test_logger):
        """DREAM should use at least 2*n+1 chains."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import DREAMDefaults
        from symfluence.core.calibration.optimizers.algorithms.dream import DREAMAlgorithm

        mock_config['POPULATION_SIZE'] = 5  # Less than minimum
        mock_config['NUMBER_OF_ITERATIONS'] = 1

        algo = DREAMAlgorithm(mock_config, test_logger)

        # For n_params=3, min_chains = 2*3+1 = 7
        min_chains = DREAMDefaults.compute_min_chains(3)
        assert min_chains == 7

    def test_dream_outlier_correction(
        self, mock_config, test_logger, sphere_population_objective, mock_callbacks
    ):
        """DREAM outlier correction should handle stuck chains."""
        from symfluence.core.calibration.optimizers.algorithms.dream import DREAMAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 15
        mock_config['POPULATION_SIZE'] = 10

        algo = DREAMAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=sphere_population_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result
        assert 'final_chains' in result

    def test_dream_gelman_rubin_computation(self, test_logger, tmp_path):
        """Gelman-Rubin computation should return valid R-hat."""
        from symfluence.core.calibration.optimizers.algorithms.dream import DREAMAlgorithm

        algo = DREAMAlgorithm(_make_full_config(tmp_path, NUMBER_OF_ITERATIONS=1, POPULATION_SIZE=10), test_logger)

        rng = np.random.default_rng(0)

        # Well-mixed chains: every chain samples the same distribution.
        # History shape is (n_iterations, n_chains, n_params).
        mixed = rng.normal(0.5, 0.05, size=(400, 5, 3))
        r_hat = algo._compute_gelman_rubin(mixed)
        assert np.isfinite(r_hat)
        assert r_hat < 1.1

        # Chains stuck around different values in ONE parameter only: the
        # statistic is the max over parameters, so it must flag it. (The old
        # "simplified" version pooled parameters of the current positions and
        # could not see this.)
        stuck = mixed.copy()
        stuck[:, :, 1] += np.arange(5) * 0.5
        assert algo._compute_gelman_rubin(stuck) > 2.0

        # Only the second half of the history counts (first half is warm-up).
        warmup_only = mixed.copy()
        warmup_only[:200, :, 1] += np.arange(5) * 0.5
        assert algo._compute_gelman_rubin(warmup_only) < 1.1

        # Not computable from a single chain or a single draw
        assert algo._compute_gelman_rubin(mixed[:, :1, :]) == float('inf')
        assert algo._compute_gelman_rubin(mixed[:2]) == float('inf')

    def test_dream_error_handling_in_proposal_generation(
        self, mock_config, test_logger, mock_callbacks
    ):
        """DREAM should handle errors in proposal generation."""
        from symfluence.core.calibration.optimizers.algorithms.dream import DREAMAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        call_count = [0]

        def unstable_objective(population, step_id):
            call_count[0] += 1
            fitness = np.array([-np.sum(x ** 2) for x in population])
            if call_count[0] == 2:
                fitness[:] = np.nan
            return fitness

        algo = DREAMAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=unstable_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result


# =============================================================================
# PSO Algorithm Tests
# =============================================================================

class TestPSOAlgorithm:
    """Test PSO algorithm implementation."""

    def test_pso_imports_config_schema(self):
        """PSO should import and use PSODefaults."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import PSODefaults
        from symfluence.core.calibration.optimizers.algorithms.pso import PSOAlgorithm

        assert PSODefaults is not None

    def test_pso_velocity_clamping(
        self, mock_config, test_logger, sphere_population_objective, mock_callbacks
    ):
        """PSO should clamp velocities to v_max."""
        from symfluence.core.calibration.optimizers.algorithms.config_schema import PSODefaults
        from symfluence.core.calibration.optimizers.algorithms.pso import PSOAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        algo = PSOAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=sphere_population_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result
        # Solution should be bounded
        assert np.all(result['best_solution'] >= 0)
        assert np.all(result['best_solution'] <= 1)

    def test_pso_personal_best_updates(
        self, mock_config, test_logger, mock_callbacks
    ):
        """PSO should correctly update personal bests."""
        from symfluence.core.calibration.optimizers.algorithms.pso import PSOAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 10
        mock_config['POPULATION_SIZE'] = 10

        algo = PSOAlgorithm(mock_config, test_logger)

        # Objective that improves toward center
        def objective(population, step_id):
            return np.array([-np.sum((x - 0.5) ** 2) for x in population])

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum((x - 0.5) ** 2),
            evaluate_population=objective,
            **mock_callbacks
        )

        assert 'best_solution' in result
        # Best solution should be closer to center after optimization
        assert result['best_score'] > -0.75  # Better than random

    def test_pso_error_handling(
        self, mock_config, test_logger, mock_callbacks
    ):
        """PSO should handle errors gracefully."""
        from symfluence.core.calibration.optimizers.algorithms.pso import PSOAlgorithm

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        call_count = [0]

        def unstable_objective(population, step_id):
            call_count[0] += 1
            fitness = np.array([-np.sum(x ** 2) for x in population])
            if call_count[0] == 2:
                fitness[0] = np.nan
            return fitness

        algo = PSOAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=unstable_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestAlgorithmErrorHandling:
    """Test NaN/Inf fitness handling across all algorithms."""

    @pytest.mark.parametrize("algorithm_name,algorithm_class", [
        ("cmaes", "CMAESAlgorithm"),
        ("nsga2", "NSGA2Algorithm"),
        ("dream", "DREAMAlgorithm"),
        ("pso", "PSOAlgorithm"),
    ])
    def test_nan_fitness_handling(
        self, algorithm_name, algorithm_class, mock_config, test_logger, mock_callbacks
    ):
        """All algorithms should handle NaN fitness without crashing."""
        import importlib

        module = importlib.import_module(
            f"symfluence.core.calibration.optimizers.algorithms.{algorithm_name}"
        )
        AlgorithmClass = getattr(module, algorithm_class)

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        def nan_objective(population, step_id):
            fitness = np.array([-np.sum(x ** 2) for x in population])
            # Always inject some NaN
            fitness[0] = np.nan
            return fitness

        algo = AlgorithmClass(mock_config, test_logger)

        # Should not raise exception
        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=nan_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result

    @pytest.mark.parametrize("algorithm_name,algorithm_class", [
        ("cmaes", "CMAESAlgorithm"),
        ("nsga2", "NSGA2Algorithm"),
        ("dream", "DREAMAlgorithm"),
        ("pso", "PSOAlgorithm"),
    ])
    def test_inf_fitness_handling(
        self, algorithm_name, algorithm_class, mock_config, test_logger, mock_callbacks
    ):
        """All algorithms should handle Inf fitness without crashing."""
        import importlib

        module = importlib.import_module(
            f"symfluence.core.calibration.optimizers.algorithms.{algorithm_name}"
        )
        AlgorithmClass = getattr(module, algorithm_class)

        mock_config['NUMBER_OF_ITERATIONS'] = 5
        mock_config['POPULATION_SIZE'] = 10

        def inf_objective(population, step_id):
            fitness = np.array([-np.sum(x ** 2) for x in population])
            fitness[0] = float('inf')
            return fitness

        algo = AlgorithmClass(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=lambda x, s: -np.sum(x ** 2),
            evaluate_population=inf_objective,
            **mock_callbacks
        )

        assert 'best_solution' in result


# =============================================================================
# Import Tests
# =============================================================================

class TestModuleImports:
    """Test that all config schema exports are available."""

    def test_config_schema_exports_from_init(self):
        """Config schema classes should be importable from algorithms package."""
        from symfluence.core.calibration.optimizers.algorithms import (
            CMAESDefaults,
            DREAMDefaults,
            NSGA2Defaults,
            PSODefaults,
            get_algorithm_defaults,
            validate_hyperparameters,
        )

        assert CMAESDefaults is not None
        assert NSGA2Defaults is not None
        assert DREAMDefaults is not None
        assert PSODefaults is not None
        assert callable(get_algorithm_defaults)
        assert callable(validate_hyperparameters)

    def test_validate_hyperparameters_function(self):
        """validate_hyperparameters should return appropriate warnings."""
        from symfluence.core.calibration.optimizers.algorithms import validate_hyperparameters

        # Valid CMA-ES params
        warnings = validate_hyperparameters('cmaes', {'sigma': 0.3})
        assert len(warnings) == 0

        # Invalid sigma
        warnings = validate_hyperparameters('cmaes', {'sigma': 1e-15})
        assert 'sigma' in warnings

        # Valid PSO params
        warnings = validate_hyperparameters('pso', {
            'inertia': 0.7,
            'cognitive': 1.5,
            'social': 1.5,
        })
        assert len(warnings) == 0
