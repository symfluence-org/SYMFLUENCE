"""
Unit tests for native gradient support in optimization algorithms.

Tests the gradient callback infrastructure, including:
- BaseWorker.supports_native_gradients()
- Algorithm gradient mode selection (_should_use_native_gradients)
- Unified gradient function creation (_create_gradient_function)
- Gradient chain rule transformation (physical -> normalized space)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_native_gradients')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def mock_config(tmp_path):
    """Create mock optimization config with all required fields."""
    return {
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
        'OPTIMIZATION_TARGET': 'streamflow',
        'OPTIMIZATION_METRIC': 'KGE',
        # Gradient-specific settings
        'GRADIENT_MODE': 'auto',
        'GRADIENT_EPSILON': 1e-4,
        'GRADIENT_CLIP_VALUE': 1.0,
        'ADAM_STEPS': 10,
        'ADAM_LR': 0.01,
        'LBFGS_STEPS': 10,
        'LBFGS_LR': 0.1,
        'NUMBER_OF_ITERATIONS': 10,
    }


@pytest.fixture
def quadratic_objective():
    """
    Create a simple quadratic objective for testing.

    f(x) = sum((x - 0.7)^2)

    Optimal at x = [0.7, 0.7, ...]
    """
    def evaluate(x: np.ndarray, step_id: int = 0) -> float:
        # Maximize negative quadratic (optimal at 0.7)
        return -np.sum((x - 0.7) ** 2)

    return evaluate


@pytest.fixture
def quadratic_gradient():
    """
    Analytical gradient for quadratic objective.

    For f(x) = sum((x - 0.7)^2), gradient = 2*(x - 0.7)
    For maximization (negative of f): gradient = -2*(x - 0.7)

    Native callback returns (loss, grad) for minimization.
    """
    def compute_gradient(x: np.ndarray) -> Tuple[float, np.ndarray]:
        loss = np.sum((x - 0.7) ** 2)  # Loss for minimization
        grad = 2 * (x - 0.7)  # Gradient for minimization
        return loss, grad

    return compute_gradient


# Ill-conditioned anisotropic quadratic. Steepest descent / gradient ascent
# zig-zags badly here, so it only converges quickly if the L-BFGS curvature
# machinery actually engages. The optimum is at 0.6 in every dimension.
ILL_COND_CURVATURE = np.array([1.0, 30.0, 1000.0])
ILL_COND_OPTIMUM = np.array([0.6, 0.6, 0.6])


@pytest.fixture
def ill_conditioned_objective():
    """Maximization objective f(x) = -Σ c_i (x_i - 0.6)^2 (condition number 1000)."""
    def evaluate(x: np.ndarray, step_id: int = 0) -> float:
        return -np.sum(ILL_COND_CURVATURE * (x - ILL_COND_OPTIMUM) ** 2)

    return evaluate


@pytest.fixture
def ill_conditioned_gradient():
    """Native (minimization) gradient for the ill-conditioned quadratic."""
    def compute_gradient(x: np.ndarray) -> Tuple[float, np.ndarray]:
        loss = np.sum(ILL_COND_CURVATURE * (x - ILL_COND_OPTIMUM) ** 2)
        grad = 2 * ILL_COND_CURVATURE * (x - ILL_COND_OPTIMUM)
        return loss, grad

    return compute_gradient


# ============================================================================
# BaseWorker gradient support tests
# ============================================================================

class TestBaseWorkerGradientSupport:
    """Test BaseWorker native gradient interface."""

    def test_default_supports_native_gradients_returns_false(self):
        """Default worker should not support native gradients."""
        from symfluence.core.calibration.workers.base_worker import BaseWorker

        # Create a minimal concrete implementation
        class MinimalWorker(BaseWorker):
            def apply_parameters(self, params, settings_dir, **kwargs):
                return True
            def run_model(self, config, settings_dir, output_dir, **kwargs):
                return True
            def calculate_metrics(self, output_dir, config, **kwargs):
                return {'KGE': 0.8}

        worker = MinimalWorker()
        assert worker.supports_native_gradients() is False

    def test_compute_gradient_returns_none_by_default(self):
        """Default compute_gradient should return None."""
        from symfluence.core.calibration.workers.base_worker import BaseWorker

        class MinimalWorker(BaseWorker):
            def apply_parameters(self, params, settings_dir, **kwargs):
                return True
            def run_model(self, config, settings_dir, output_dir, **kwargs):
                return True
            def calculate_metrics(self, output_dir, config, **kwargs):
                return {'KGE': 0.8}

        worker = MinimalWorker()
        result = worker.compute_gradient({'param1': 0.5}, 'kge')
        assert result is None

    def test_evaluate_with_gradient_raises_not_implemented(self):
        """Default evaluate_with_gradient should raise NotImplementedError."""
        from symfluence.core.calibration.workers.base_worker import BaseWorker

        class MinimalWorker(BaseWorker):
            def apply_parameters(self, params, settings_dir, **kwargs):
                return True
            def run_model(self, config, settings_dir, output_dir, **kwargs):
                return True
            def calculate_metrics(self, output_dir, config, **kwargs):
                return {'KGE': 0.8}

        worker = MinimalWorker()
        with pytest.raises(NotImplementedError):
            worker.evaluate_with_gradient({'param1': 0.5}, 'kge')

    def test_custom_worker_can_override_gradient_support(self):
        """Workers can override to support native gradients."""
        from symfluence.core.calibration.workers.base_worker import BaseWorker

        class GradientCapableWorker(BaseWorker):
            def apply_parameters(self, params, settings_dir, **kwargs):
                return True
            def run_model(self, config, settings_dir, output_dir, **kwargs):
                return True
            def calculate_metrics(self, output_dir, config, **kwargs):
                return {'KGE': 0.8}

            def supports_native_gradients(self) -> bool:
                return True

            def evaluate_with_gradient(self, params, metric='kge'):
                # Simple mock implementation
                loss = 0.5
                grad = {k: 0.1 for k in params}
                return loss, grad

        worker = GradientCapableWorker()
        assert worker.supports_native_gradients() is True
        loss, grad = worker.evaluate_with_gradient({'p1': 0.5, 'p2': 0.3})
        assert loss == 0.5
        assert grad == {'p1': 0.1, 'p2': 0.1}


# ============================================================================
# Algorithm gradient mode selection tests
# ============================================================================

class TestGradientModeSelection:
    """Test _should_use_native_gradients logic."""

    def test_finite_difference_mode_always_returns_false(
        self, mock_config, test_logger, quadratic_gradient
    ):
        """gradient_mode='finite_difference' should always use FD."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        # Even with callback provided, should return False
        result = algo._should_use_native_gradients(quadratic_gradient, 'finite_difference')
        assert result is False

    def test_native_mode_requires_callback(self, mock_config, test_logger):
        """gradient_mode='native' should error without callback."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        with pytest.raises(ValueError, match="requires a compute_gradient callback"):
            algo._should_use_native_gradients(None, 'native')

    def test_native_mode_with_callback_returns_true(
        self, mock_config, test_logger, quadratic_gradient
    ):
        """gradient_mode='native' with callback should return True."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        result = algo._should_use_native_gradients(quadratic_gradient, 'native')
        assert result is True

    def test_auto_mode_uses_native_when_available(
        self, mock_config, test_logger, quadratic_gradient
    ):
        """gradient_mode='auto' should use native when callback available."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        result = algo._should_use_native_gradients(quadratic_gradient, 'auto')
        assert result is True

    def test_auto_mode_falls_back_to_fd_when_no_callback(
        self, mock_config, test_logger
    ):
        """gradient_mode='auto' should use FD when no callback."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        result = algo._should_use_native_gradients(None, 'auto')
        assert result is False


# ============================================================================
# Unified gradient function tests
# ============================================================================

class TestUnifiedGradientFunction:
    """Test _create_gradient_function factory."""

    def test_fd_gradient_function_approximates_correctly(
        self, mock_config, test_logger, quadratic_objective
    ):
        """Finite difference gradients should be approximately correct."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        grad_func = algo._create_gradient_function(
            compute_gradient=None,
            evaluate_solution=quadratic_objective,
            gradient_mode='finite_difference',
            epsilon=1e-4
        )

        # Test at x = [0.5, 0.5, 0.5]
        x = np.array([0.5, 0.5, 0.5])
        fitness, gradient = grad_func(x)

        # Expected gradient for maximization: 2*(0.7 - 0.5) = 0.4 for each dim
        expected_grad = np.array([0.4, 0.4, 0.4])

        np.testing.assert_allclose(gradient, expected_grad, atol=1e-3)

    def test_native_gradient_function_uses_callback(
        self, mock_config, test_logger, quadratic_gradient
    ):
        """Native gradients should use the provided callback."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        # Mock evaluate_solution (shouldn't be called)
        mock_eval = Mock()

        grad_func = algo._create_gradient_function(
            compute_gradient=quadratic_gradient,
            evaluate_solution=mock_eval,
            gradient_mode='native',
            epsilon=1e-4
        )

        x = np.array([0.5, 0.5, 0.5])
        fitness, gradient = grad_func(x)

        # Native callback returns (loss, grad) for minimization
        # Factory should negate for maximization
        expected_fitness = -np.sum((x - 0.7) ** 2)  # -loss
        expected_grad = -2 * (x - 0.7)  # -grad (ascent direction)

        np.testing.assert_allclose(fitness, expected_fitness, atol=1e-10)
        np.testing.assert_allclose(gradient, expected_grad, atol=1e-10)

        # Verify evaluate_solution was NOT called
        mock_eval.assert_not_called()

    def test_gradient_clipping_works(self, mock_config, test_logger):
        """Gradient clipping should limit gradient norm."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        # Large gradient
        large_grad = np.array([10.0, 10.0, 10.0])
        clip_value = 1.0

        clipped = algo._clip_gradient(large_grad.copy(), clip_value)

        # Check norm is at most clip_value
        assert np.linalg.norm(clipped) <= clip_value + 1e-10

        # Check direction is preserved
        np.testing.assert_allclose(
            clipped / np.linalg.norm(clipped),
            large_grad / np.linalg.norm(large_grad),
            atol=1e-10
        )

    def test_small_gradient_not_clipped(self, mock_config, test_logger):
        """Small gradients should not be modified by clipping."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        algo = AdamAlgorithm(mock_config, test_logger)

        small_grad = np.array([0.1, 0.1, 0.1])
        clip_value = 1.0

        clipped = algo._clip_gradient(small_grad.copy(), clip_value)

        np.testing.assert_allclose(clipped, small_grad, atol=1e-10)


# ============================================================================
# Adam algorithm gradient tests
# ============================================================================

class TestAdamWithGradients:
    """Test Adam algorithm with native vs FD gradients."""

    def test_adam_converges_with_fd_gradients(
        self, mock_config, test_logger, quadratic_objective
    ):
        """Adam should converge using finite-difference gradients."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        mock_config['ADAM_STEPS'] = 50
        mock_config['ADAM_LR'] = 0.1
        algo = AdamAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=quadratic_objective,
            evaluate_population=lambda p, i: np.array([quadratic_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference'
        )

        # Should converge close to 0.7 (tolerance allows for variation in optimization)
        np.testing.assert_allclose(result['best_solution'], 0.7, atol=0.15)
        assert result['gradient_method'] == 'finite_difference'

    def test_adam_converges_with_native_gradients(
        self, mock_config, test_logger, quadratic_objective, quadratic_gradient
    ):
        """Adam should converge using native gradients."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        mock_config['ADAM_STEPS'] = 50
        mock_config['ADAM_LR'] = 0.1
        algo = AdamAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=quadratic_objective,
            evaluate_population=lambda p, i: np.array([quadratic_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=quadratic_gradient,
            gradient_mode='native'
        )

        # Should converge close to 0.7 (tolerance allows for variation in optimization)
        np.testing.assert_allclose(result['best_solution'], 0.7, atol=0.15)
        assert result['gradient_method'] == 'native'

    def test_adam_returns_gradient_method_in_result(
        self, mock_config, test_logger, quadratic_objective
    ):
        """Adam result should include which gradient method was used."""
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm

        mock_config['ADAM_STEPS'] = 5
        algo = AdamAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=2,
            evaluate_solution=quadratic_objective,
            evaluate_population=lambda p, i: np.array([quadratic_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            gradient_mode='finite_difference'
        )

        assert 'gradient_method' in result
        assert result['gradient_method'] == 'finite_difference'


# ============================================================================
# L-BFGS algorithm gradient tests
# ============================================================================

class TestLBFGSWithGradients:
    """Test L-BFGS algorithm with native vs FD gradients."""

    def test_lbfgs_converges_with_fd_gradients(
        self, mock_config, test_logger, quadratic_objective
    ):
        """L-BFGS should converge using finite-difference gradients."""
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        mock_config['LBFGS_STEPS'] = 30
        mock_config['LBFGS_LR'] = 0.5
        algo = LBFGSAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=quadratic_objective,
            evaluate_population=lambda p, i: np.array([quadratic_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference'
        )

        # Should converge close to 0.7 (tolerance allows for variation in optimization)
        np.testing.assert_allclose(result['best_solution'], 0.7, atol=0.15)
        assert result['gradient_method'] == 'finite_difference'

    def test_lbfgs_converges_with_native_gradients(
        self, mock_config, test_logger, quadratic_objective, quadratic_gradient
    ):
        """L-BFGS should converge using native gradients."""
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        mock_config['LBFGS_STEPS'] = 30
        mock_config['LBFGS_LR'] = 0.5
        algo = LBFGSAlgorithm(mock_config, test_logger)

        result = algo.optimize(
            n_params=3,
            evaluate_solution=quadratic_objective,
            evaluate_population=lambda p, i: np.array([quadratic_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=quadratic_gradient,
            gradient_mode='native'
        )

        # Should converge close to 0.7 (tolerance allows for variation in optimization)
        np.testing.assert_allclose(result['best_solution'], 0.7, atol=0.15)
        assert result['gradient_method'] == 'native'


class TestLBFGSQuasiNewton:
    """Guard that L-BFGS actually behaves as a quasi-Newton method.

    Regression coverage for the maximization sign-convention bug: the algorithm
    minimizes g = -fitness internally, so the curvature condition y·s > 0 must be
    satisfiable near a maximum of fitness. Before the fix, the secant history
    stayed permanently empty on any locally-concave objective and L-BFGS silently
    degraded to plain gradient ascent — these tests would have caught that.
    """

    @staticmethod
    def _run(algo, n_params, objective, compute_gradient, gradient_mode):
        """Run the algorithm while recording the (s, y) history length seen."""
        history_lengths = []
        original_direction = algo._lbfgs_direction

        def spy(gradient, s_history, y_history):
            history_lengths.append(len(s_history))
            return original_direction(gradient, s_history, y_history)

        algo._lbfgs_direction = spy

        result = algo.optimize(
            n_params=n_params,
            evaluate_solution=objective,
            evaluate_population=lambda p, i: np.array([objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=compute_gradient,
            gradient_mode=gradient_mode,
        )
        return result, history_lengths

    def test_secant_history_populates_on_concave_objective(
        self, mock_config, test_logger, ill_conditioned_objective
    ):
        """The (s, y) history must accumulate — the core of the sign-bug fix.

        On a locally-concave (for maximization) objective the buggy code kept the
        history empty forever. Here it must grow above zero.
        """
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        # Set both keys so step count is 40 regardless of resolution precedence.
        mock_config['LBFGS_STEPS'] = 40
        mock_config['NUMBER_OF_ITERATIONS'] = 40
        mock_config['LBFGS_LR'] = 1.0
        mock_config['GRADIENT_EPSILON'] = 1e-6
        mock_config['GRADIENT_CLIP_VALUE'] = 0.0  # disable clipping for clean curvature
        algo = LBFGSAlgorithm(mock_config, test_logger)

        _, history_lengths = self._run(
            algo, 3, ill_conditioned_objective, None, 'finite_difference'
        )

        assert max(history_lengths) > 0, (
            "L-BFGS secant history never populated — algorithm degraded to "
            "gradient ascent (maximization sign-convention regression)."
        )

    def test_lbfgs_converges_on_ill_conditioned_quadratic(
        self, mock_config, test_logger, ill_conditioned_objective
    ):
        """Quasi-Newton steps must solve an ill-conditioned quadratic tightly."""
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        mock_config['LBFGS_STEPS'] = 60
        mock_config['NUMBER_OF_ITERATIONS'] = 60
        mock_config['LBFGS_LR'] = 1.0
        mock_config['GRADIENT_EPSILON'] = 1e-6
        mock_config['GRADIENT_CLIP_VALUE'] = 0.0
        algo = LBFGSAlgorithm(mock_config, test_logger)

        result, _ = self._run(
            algo, 3, ill_conditioned_objective, None, 'finite_difference'
        )

        np.testing.assert_allclose(result['best_solution'], ILL_COND_OPTIMUM, atol=1e-2)

    def test_lbfgs_native_populates_history_and_converges(
        self, mock_config, test_logger, ill_conditioned_objective, ill_conditioned_gradient
    ):
        """Native-gradient path must also engage curvature and converge."""
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        mock_config['LBFGS_STEPS'] = 60
        mock_config['NUMBER_OF_ITERATIONS'] = 60
        mock_config['LBFGS_LR'] = 1.0
        mock_config['GRADIENT_CLIP_VALUE'] = 0.0
        algo = LBFGSAlgorithm(mock_config, test_logger)

        result, history_lengths = self._run(
            algo, 3, ill_conditioned_objective, ill_conditioned_gradient, 'native'
        )

        assert max(history_lengths) > 0
        assert result['gradient_method'] == 'native'
        np.testing.assert_allclose(result['best_solution'], ILL_COND_OPTIMUM, atol=1e-2)

    def test_lbfgs_outperforms_gradient_ascent_under_equal_budget(
        self, mock_config, test_logger, ill_conditioned_objective
    ):
        """With curvature working, L-BFGS must beat first-order Adam on this problem.

        On an ill-conditioned quadratic a genuine quasi-Newton method converges
        far faster than gradient ascent. If L-BFGS had degraded to gradient ascent
        (the bug), the two would be comparable and this margin would vanish.
        """
        from symfluence.core.calibration.optimizers.algorithms.adam import AdamAlgorithm
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        steps = 25
        common = dict(
            n_params=3,
            evaluate_solution=ill_conditioned_objective,
            evaluate_population=lambda p, i: np.array([ill_conditioned_objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference',
        )

        lbfgs_cfg = dict(mock_config)
        lbfgs_cfg.update({'LBFGS_STEPS': steps, 'NUMBER_OF_ITERATIONS': steps, 'LBFGS_LR': 1.0,
                          'GRADIENT_EPSILON': 1e-6, 'GRADIENT_CLIP_VALUE': 0.0})
        lbfgs_result = LBFGSAlgorithm(lbfgs_cfg, test_logger).optimize(**common)

        adam_cfg = dict(mock_config)
        adam_cfg.update({'ADAM_STEPS': steps, 'NUMBER_OF_ITERATIONS': steps, 'ADAM_LR': 0.05,
                         'GRADIENT_EPSILON': 1e-6, 'GRADIENT_CLIP_VALUE': 0.0})
        adam_result = AdamAlgorithm(adam_cfg, test_logger).optimize(**common)

        # L-BFGS should be at least an order of magnitude closer to the optimum
        # (both scores are <= 0; closer to 0 is better).
        assert lbfgs_result['best_score'] > 10 * adam_result['best_score']


# ============================================================================
# Gradient chain rule transformation tests
# ============================================================================

class TestGradientChainRule:
    """Test gradient transformation between physical and normalized space."""

    def test_gradient_scaling_with_different_bounds(self):
        """Gradient should scale correctly based on parameter bounds."""
        # For x_phys = x_norm * (max - min) + min
        # dx_phys/dx_norm = (max - min)
        # d(loss)/dx_norm = d(loss)/dx_phys * (max - min)

        bounds = {
            'p1': {'min': 0.0, 'max': 1.0},    # scale = 1
            'p2': {'min': 0.0, 'max': 100.0},  # scale = 100
            'p3': {'min': -10.0, 'max': 10.0}, # scale = 20
        }

        param_names = ['p1', 'p2', 'p3']

        # Physical gradients
        grad_physical = np.array([0.1, 0.1, 0.1])

        # Expected normalized gradients
        scale_factors = np.array([1.0, 100.0, 20.0])
        expected_normalized = grad_physical * scale_factors

        np.testing.assert_allclose(
            expected_normalized,
            np.array([0.1, 10.0, 2.0])
        )

    def test_gradient_direction_preserved_after_scaling(self):
        """Gradient direction should be preserved after scaling."""
        grad_physical = np.array([1.0, -2.0, 0.5])
        scale_factors = np.array([10.0, 5.0, 20.0])

        grad_normalized = grad_physical * scale_factors

        # Directions should match (signs preserved)
        assert np.sign(grad_normalized[0]) == np.sign(grad_physical[0])
        assert np.sign(grad_normalized[1]) == np.sign(grad_physical[1])
        assert np.sign(grad_normalized[2]) == np.sign(grad_physical[2])

    def test_gradient_callback_handles_param_without_bounds(self, test_logger):
        """A parameter in all_param_names but absent from the bounds dict (no
        calibration range, e.g. SAC-SMA's PXADJ) must not KeyError. It is held
        fixed: scale factor 0.0 and gradient 0.0, so the optimiser never moves
        it and the arrays stay aligned with all_param_names."""
        from unittest.mock import Mock

        from symfluence.core.calibration.optimizers.base_model_optimizer import (
            BaseModelOptimizer,
        )

        class _ConcreteOpt(BaseModelOptimizer):
            def _get_model_name(self):
                return 'SACSMA'

            def _get_final_file_manager_path(self, *a, **k):
                return None

            def _run_model_for_final_evaluation(self, *a, **k):
                return None

        opt = _ConcreteOpt.__new__(_ConcreteOpt)
        opt.logger = test_logger
        opt.target_metric = 'KGE'

        # 'PXADJ' has no bounds; the worker/denormalize therefore never sees it.
        opt.param_manager = Mock()
        opt.param_manager.all_param_names = ['p1', 'p2', 'PXADJ']
        opt.param_manager.get_parameter_bounds.return_value = {
            'p1': {'min': 0.0, 'max': 1.0},
            'p2': {'min': 0.0, 'max': 10.0},
        }
        opt.param_manager.denormalize_parameters = lambda x: {'p1': x[0], 'p2': x[1]}

        opt.worker = Mock()
        opt.worker.supports_native_gradients.return_value = True
        # Worker returns gradients only for the parameters it actually ran.
        opt.worker.evaluate_with_gradient.return_value = (0.4, {'p1': 0.5, 'p2': -0.3})

        callback = opt._create_gradient_callback()
        assert callback is not None

        loss, grad = callback(np.array([0.5, 0.5, 0.5]))
        assert loss == 0.4
        # p1: 0.5*(1-0)=0.5 ; p2: -0.3*(10-0)=-3.0 ; PXADJ: held fixed at 0.0
        np.testing.assert_allclose(grad, [0.5, -3.0, 0.0])
        assert np.all(np.isfinite(grad))


# ============================================================================
# Config option tests
# ============================================================================

class TestGradientConfigOptions:
    """Test gradient-related configuration options."""

    def test_gradient_mode_config_option_exists(self):
        """OptimizationConfig should have gradient_mode field."""
        from symfluence.core.config.models.optimization import OptimizationConfig

        config = OptimizationConfig()
        assert hasattr(config, 'gradient_mode')
        assert config.gradient_mode == 'auto'

    def test_gradient_epsilon_config_option_exists(self):
        """OptimizationConfig should have gradient_epsilon field."""
        from symfluence.core.config.models.optimization import OptimizationConfig

        config = OptimizationConfig()
        assert hasattr(config, 'gradient_epsilon')
        assert config.gradient_epsilon == 1e-4

    def test_gradient_clip_value_config_option_exists(self):
        """OptimizationConfig should have gradient_clip_value field."""
        from symfluence.core.config.models.optimization import OptimizationConfig

        config = OptimizationConfig()
        assert hasattr(config, 'gradient_clip_value')
        assert config.gradient_clip_value == 1.0

    def test_gradient_mode_accepts_valid_values(self):
        """gradient_mode should accept 'auto', 'native', 'finite_difference'."""
        from symfluence.core.config.models.optimization import OptimizationConfig

        for mode in ['auto', 'native', 'finite_difference']:
            config = OptimizationConfig(gradient_mode=mode)
            assert config.gradient_mode == mode


class TestLBFGSDegenerateWarning:
    """The live L-BFGS reports once when it is not running as quasi-Newton."""

    @staticmethod
    def _run(algo, objective):
        return algo.optimize(
            n_params=3,
            evaluate_solution=objective,
            evaluate_population=lambda p, i: np.array([objective(x, i) for x in p]),
            denormalize_params=lambda x: {f'p{i}': v for i, v in enumerate(x)},
            record_iteration=lambda *args, **kwargs: None,
            update_best=lambda *args, **kwargs: None,
            log_progress=lambda *args, **kwargs: None,
            compute_gradient=None,
            gradient_mode='finite_difference',
        )

    def test_reports_once_when_every_step_is_steepest_descent(
        self, mock_config, test_logger, caplog
    ):
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        # A linear objective has a constant gradient, so no secant pair ever
        # passes the curvature condition and every step is steepest descent.
        mock_config['LBFGS_STEPS'] = 30
        mock_config['NUMBER_OF_ITERATIONS'] = 30
        mock_config['LBFGS_LR'] = 0.01
        mock_config['GRADIENT_EPSILON'] = 1e-4
        mock_config['GRADIENT_CLIP_VALUE'] = 0.0
        algo = LBFGSAlgorithm(mock_config, test_logger)

        with caplog.at_level(logging.ERROR, logger=test_logger.name):
            self._run(algo, lambda x, step_id=0: float(np.sum(x)))

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "steepest descent" in errors[0].getMessage()
        assert "gradient_epsilon" in errors[0].getMessage()

    def test_silent_on_a_healthy_run(self, mock_config, test_logger, caplog):
        from symfluence.core.calibration.optimizers.algorithms.lbfgs import LBFGSAlgorithm

        mock_config['LBFGS_STEPS'] = 40
        mock_config['NUMBER_OF_ITERATIONS'] = 40
        mock_config['LBFGS_LR'] = 1.0
        mock_config['GRADIENT_EPSILON'] = 1e-6
        mock_config['GRADIENT_CLIP_VALUE'] = 0.0
        algo = LBFGSAlgorithm(mock_config, test_logger)
        scales = np.array([1.0, 10.0, 100.0])

        with caplog.at_level(logging.ERROR, logger=test_logger.name):
            self._run(algo, lambda x, step_id=0: -float(np.sum(scales * (x - 0.7) ** 2)))

        assert not [r for r in caplog.records if r.levelno == logging.ERROR]
