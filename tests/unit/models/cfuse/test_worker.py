"""Tests for cFUSE calibration worker."""

import pytest


class TestCFUSEWorkerRegistration:
    """Tests for cFUSE worker registration."""

    def test_worker_can_be_imported(self):
        from cfuse.calibration.worker import CFUSEWorker
        assert CFUSEWorker is not None

    def test_worker_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'CFUSE' in OptimizerRegistry._workers

    def test_worker_is_correct_class(self):
        from cfuse.calibration.worker import CFUSEWorker

        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry._workers.get('CFUSE') == CFUSEWorker


class TestCFUSEWorkerProperties:
    """Tests for cFUSE worker properties."""

    def test_gradient_support_returns_bool(self):
        from cfuse.calibration.worker import CFUSEWorker
        worker = CFUSEWorker()
        result = worker.supports_native_gradients()
        assert isinstance(result, bool)

    def test_worker_has_penalty_score(self):
        from cfuse.calibration.worker import CFUSEWorker
        worker = CFUSEWorker()
        assert hasattr(worker, 'penalty_score')

    def test_worker_has_evaluate_worker_function(self):
        """Worker should have a static function for process pool."""
        from cfuse.calibration.worker import CFUSEWorker
        assert hasattr(CFUSEWorker, 'evaluate_worker_function')
        assert callable(CFUSEWorker.evaluate_worker_function)
