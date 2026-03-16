"""
Integration tests comparing finite-difference vs native gradient optimization.

These tests verify that:
1. Native gradients produce equivalent or better results than FD
2. Native gradients are significantly faster
3. Both methods converge to similar optima
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

# Skip all tests if JAX not available
jax = pytest.importorskip("jax", reason="JAX required for native gradient tests")
import jax.numpy as jnp

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_gradient_comparison')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


@pytest.fixture
def rosenbrock_objective():
    """
    Rosenbrock function - a challenging optimization test case.

    f(x, y) = (a - x)^2 + b*(y - x^2)^2

    Optimal at (a, a^2) = (1, 1) for a=1, b=100
    We normalize to [0, 1] space: x_norm * 4 - 2 maps [0,1] -> [-2, 2]
    """
    a, b = 1.0, 100.0

    def evaluate(x_norm: np.ndarray, step_id: int = 0) -> float:
        # Denormalize from [0,1] to [-2, 2]
        x = x_norm * 4 - 2

        # Rosenbrock (negated for maximization)
        val = 0.0
        for i in range(len(x) - 1):
            val += (a - x[i])**2 + b * (x[i+1] - x[i]**2)**2
        return -val  # Negate for maximization

    return evaluate


@pytest.fixture
def rosenbrock_gradient_jax():
    """
    Native JAX gradient for Rosenbrock function.

    Returns (loss, gradient) for minimization.
    """
    a, b = 1.0, 100.0

    def loss_fn(x_norm):
        # Denormalize
        x = x_norm * 4 - 2

        # Rosenbrock
        val = 0.0
        for i in range(len(x) - 1):
            val = val + (a - x[i])**2 + b * (x[i+1] - x[i]**2)**2
        return val

    # Create gradient function
    grad_fn = jax.grad(loss_fn)

    def compute_gradient(x_norm: np.ndarray) -> Tuple[float, np.ndarray]:
        x_jax = jnp.array(x_norm)
        loss = float(loss_fn(x_jax))
        grad = np.array(grad_fn(x_jax))
        return loss, grad

    return compute_gradient


@pytest.fixture
def base_config():
    """Base configuration for gradient optimization tests."""
    return {
        'GRADIENT_MODE': 'auto',
        'GRADIENT_EPSILON': 1e-5,
        'GRADIENT_CLIP_VALUE': 10.0,
        'ADAM_STEPS': 100,
        'ADAM_LR': 0.05,
        'ADAM_BETA1': 0.9,
        'ADAM_BETA2': 0.999,
        'LBFGS_STEPS': 50,
        'LBFGS_LR': 0.5,
        'NUMBER_OF_ITERATIONS': 100,
    }


# ============================================================================
# Comparison tests
# ============================================================================

class TestGradientMethodComparison:
    """Compare FD and native gradient methods."""

    def test_adam_fd_vs_native_convergence(
        self, base_config, test_logger, rosenbrock_objective, rosenbrock_gradient_jax
    ):
        """Adam should converge similarly with FD and native gradients."""
        from symfluence.optimization.optimizers.algorithms.adam import AdamAlgorithm

        n_params = 4
        base_config['ADAM_STEPS'] = 200
        base_config['ADAM_LR'] = 0.02

        # Run with FD
        algo_fd = AdamAlgorithm(base_config.copy(), test_logger)
        result_fd = algo_fd.optimize(
            n_params=n_params,
            evaluate_solution=rosenbrock_objective,
            evaluate_population=lambda p, i: np.array([rosenbrock_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference'
        )

        # Run with native
        algo_native = AdamAlgorithm(base_config.copy(), test_logger)
        result_native = algo_native.optimize(
            n_params=n_params,
            evaluate_solution=rosenbrock_objective,
            evaluate_population=lambda p, i: np.array([rosenbrock_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=rosenbrock_gradient_jax,
            gradient_mode='native'
        )

        # Both should find reasonable solutions
        assert result_fd['best_score'] > -100  # Not too bad
        assert result_native['best_score'] > -100

        # Native should be at least as good (often better due to exact gradients)
        # Allow some tolerance due to stochastic nature
        assert result_native['best_score'] >= result_fd['best_score'] - 10

        test_logger.info(f"FD best score: {result_fd['best_score']:.4f}")
        test_logger.info(f"Native best score: {result_native['best_score']:.4f}")

    def test_lbfgs_fd_vs_native_convergence(
        self, base_config, test_logger, rosenbrock_objective, rosenbrock_gradient_jax
    ):
        """L-BFGS should converge similarly with FD and native gradients."""
        from symfluence.optimization.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        n_params = 4
        base_config['LBFGS_STEPS'] = 100
        base_config['LBFGS_LR'] = 1.0

        # Run with FD
        algo_fd = LBFGSAlgorithm(base_config.copy(), test_logger)
        result_fd = algo_fd.optimize(
            n_params=n_params,
            evaluate_solution=rosenbrock_objective,
            evaluate_population=lambda p, i: np.array([rosenbrock_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference'
        )

        # Run with native
        algo_native = LBFGSAlgorithm(base_config.copy(), test_logger)
        result_native = algo_native.optimize(
            n_params=n_params,
            evaluate_solution=rosenbrock_objective,
            evaluate_population=lambda p, i: np.array([rosenbrock_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=rosenbrock_gradient_jax,
            gradient_mode='native'
        )

        # Both should find reasonable solutions
        assert result_fd['best_score'] > -100
        assert result_native['best_score'] > -100

        test_logger.info(f"L-BFGS FD best score: {result_fd['best_score']:.4f}")
        test_logger.info(f"L-BFGS Native best score: {result_native['best_score']:.4f}")


class TestGradientSpeedup:
    """Test that native gradients provide speedup."""

    def test_native_gradients_faster_than_fd(
        self, base_config, test_logger, rosenbrock_objective, rosenbrock_gradient_jax
    ):
        """Native gradients should be faster than FD for many parameters."""
        from symfluence.optimization.optimizers.algorithms.adam import AdamAlgorithm

        n_params = 10  # More params = more FD evaluations
        n_steps = 20

        base_config['ADAM_STEPS'] = n_steps
        base_config['ADAM_LR'] = 0.01

        # Count function evaluations
        fd_eval_count = 0
        native_eval_count = 0

        def counting_objective_fd(x, step_id=0):
            nonlocal fd_eval_count
            fd_eval_count += 1
            return rosenbrock_objective(x, step_id)

        def counting_gradient(x):
            nonlocal native_eval_count
            native_eval_count += 1
            return rosenbrock_gradient_jax(x)

        # Run with FD
        algo_fd = AdamAlgorithm(base_config.copy(), test_logger)
        start_fd = time.time()
        algo_fd.optimize(
            n_params=n_params,
            evaluate_solution=counting_objective_fd,
            evaluate_population=lambda p, i: np.array([counting_objective_fd(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference'
        )
        time_fd = time.time() - start_fd

        # Run with native
        algo_native = AdamAlgorithm(base_config.copy(), test_logger)
        start_native = time.time()
        algo_native.optimize(
            n_params=n_params,
            evaluate_solution=rosenbrock_objective,  # Not counted (not used)
            evaluate_population=lambda p, i: np.array([rosenbrock_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=counting_gradient,
            gradient_mode='native'
        )
        time_native = time.time() - start_native

        # FD should use ~(2*n_params + 1) * n_steps evaluations
        expected_fd_evals = (2 * n_params + 1) * n_steps
        # Native should use ~n_steps evaluations
        expected_native_evals = n_steps

        test_logger.info(f"FD evaluations: {fd_eval_count} (expected ~{expected_fd_evals})")
        test_logger.info(f"Native evaluations: {native_eval_count} (expected ~{expected_native_evals})")
        test_logger.info(f"FD time: {time_fd:.3f}s, Native time: {time_native:.3f}s")

        # Native should use significantly fewer gradient computations
        assert native_eval_count < fd_eval_count / n_params


class TestGradientAccuracy:
    """Test that native and FD gradients are consistent."""

    def test_fd_approximates_native_gradient(self, base_config, test_logger):
        """FD gradient should approximate native gradient accurately."""
        from symfluence.optimization.optimizers.algorithms.base_algorithm import OptimizationAlgorithm

        # Simple quadratic for testing
        def loss_fn(x):
            return float(jnp.sum((x - 0.5) ** 2))

        grad_fn = jax.grad(lambda x: jnp.sum((x - 0.5) ** 2))

        def native_gradient(x):
            return loss_fn(jnp.array(x)), np.array(grad_fn(jnp.array(x)))

        def evaluate(x, step_id=0):
            return -loss_fn(x)  # Negate for maximization

        # Create minimal algorithm for testing
        from symfluence.optimization.optimizers.algorithms.adam import AdamAlgorithm
        algo = AdamAlgorithm(base_config, test_logger)

        # Test at several points
        test_points = [
            np.array([0.3, 0.3, 0.3]),
            np.array([0.7, 0.2, 0.9]),
            np.array([0.5, 0.5, 0.5]),
        ]

        for x in test_points:
            # Get native gradient
            _, native_grad = native_gradient(x)

            # Get FD gradient (returns fitness and ascent gradient)
            _, fd_grad = algo._compute_fd_gradients(x, evaluate, epsilon=1e-5)

            # FD gradient is for maximization (ascent), native is for minimization
            # So they should have opposite signs
            # Use relaxed tolerances since FD has inherent numerical approximation error
            np.testing.assert_allclose(-fd_grad, native_grad, atol=0.005, rtol=0.005)


# ============================================================================
# HBV-specific tests (if HBV worker available)
# ============================================================================

@pytest.mark.skipif(
    not Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE/src/symfluence/models/hbv').exists(),
    reason="HBV model not available"
)
class TestHBVGradientSupport:
    """Test HBV worker gradient support."""

    def test_hbv_worker_supports_native_gradients(self):
        """HBVWorker should report gradient support when JAX available."""
        try:
            from jhbv.calibration.worker import HAS_JAX, HBVWorker

            worker = HBVWorker({}, logging.getLogger('test'))
            assert worker.supports_native_gradients() == HAS_JAX
        except ImportError:
            pytest.skip("HBV worker not available")

    def test_hbv_gradient_callback_created_when_jax_available(self):
        """BaseModelOptimizer should create gradient callback for HBV."""
        # This would require more setup - mark as integration test
        pytest.skip("Requires full HBV model setup")
