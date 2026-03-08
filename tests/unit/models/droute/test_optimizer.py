"""Tests for dRoute optimizer."""

import pytest


class TestDRouteOptimizerRegistration:
    """Tests for dRoute optimizer registration."""

    def test_optimizer_can_be_imported(self):
        from droute.calibration.optimizer import DRouteModelOptimizer
        assert DRouteModelOptimizer is not None

    def test_optimizer_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'DROUTE' in OptimizerRegistry._optimizers

    def test_optimizer_is_correct_class(self):
        from droute.calibration.optimizer import DRouteModelOptimizer

        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry._optimizers.get('DROUTE') == DRouteModelOptimizer


class TestDRouteWorkerRegistration:
    """Tests for dRoute worker registration."""

    def test_worker_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'DROUTE' in OptimizerRegistry._workers

    def test_worker_is_correct_class(self):
        from droute.calibration.worker import DRouteWorker

        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry._workers.get('DROUTE') == DRouteWorker


class TestDRouteGradientSupport:
    """Tests for dRoute gradient support delegation."""

    def test_gradient_support_returns_bool(self):
        from droute.calibration.worker import DRouteWorker
        worker = DRouteWorker()
        result = worker.supports_native_gradients()
        assert isinstance(result, bool)

    def test_gradient_support_without_droute(self):
        """Without droute installed, gradients should not be available."""
        from droute.calibration.worker import HAS_DROUTE, DRouteWorker
        worker = DRouteWorker()
        if not HAS_DROUTE:
            assert worker.supports_native_gradients() is False
