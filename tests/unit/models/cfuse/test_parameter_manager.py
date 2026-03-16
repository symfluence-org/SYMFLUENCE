"""Tests for cFUSE parameter manager."""

import logging
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def logger():
    return logging.getLogger('test_cfuse_pm')


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cfuse_config(temp_dir):
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'CFUSE_PARAMS_TO_CALIBRATE': 'S1_max,S2_max,ku,ki,ks',
    }


class TestCFUSEParameterManagerRegistration:
    """Tests for cFUSE parameter manager registration."""

    def test_parameter_manager_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'CFUSE' in OptimizerRegistry._parameter_managers

    def test_parameter_manager_is_correct_class(self):
        from cfuse.calibration.parameter_manager import CFUSEParameterManager

        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry._parameter_managers.get('CFUSE') == CFUSEParameterManager


class TestCFUSEParameterManagerInstance:
    """Tests for cFUSE parameter manager instances."""

    def test_can_instantiate(self, cfuse_config, logger, temp_dir):
        from cfuse.calibration.parameter_manager import CFUSEParameterManager
        manager = CFUSEParameterManager(cfuse_config, logger, temp_dir)
        assert manager is not None

    def test_parameter_names(self, cfuse_config, logger, temp_dir):
        from cfuse.calibration.parameter_manager import CFUSEParameterManager
        manager = CFUSEParameterManager(cfuse_config, logger, temp_dir)
        names = manager._get_parameter_names()
        assert 'S1_max' in names
        assert 'S2_max' in names
        assert len(names) == 5

    def test_load_bounds(self, cfuse_config, logger, temp_dir):
        from cfuse.calibration.parameter_manager import CFUSEParameterManager
        manager = CFUSEParameterManager(cfuse_config, logger, temp_dir)
        bounds = manager._load_parameter_bounds()
        assert len(bounds) > 0
        for param in ['S1_max', 'S2_max', 'ku', 'ki', 'ks']:
            assert param in bounds

    def test_normalize_denormalize_roundtrip(self, cfuse_config, logger, temp_dir):
        from cfuse.calibration.parameter_manager import CFUSEParameterManager
        manager = CFUSEParameterManager(cfuse_config, logger, temp_dir)

        bounds = manager._load_parameter_bounds()
        params = {}
        for name in manager._get_parameter_names():
            b = bounds[name]
            params[name] = (b['min'] + b['max']) / 2.0

        normalized = manager.normalize_parameters(params)
        denormalized = manager.denormalize_parameters(normalized)

        for key in params:
            assert abs(denormalized[key] - params[key]) < 0.1, \
                f"Roundtrip failed for {key}: {params[key]} -> {denormalized[key]}"


class TestCFUSECalibrationBoundsFunction:
    """Tests for the convenience bounds function."""

    def test_get_cfuse_calibration_bounds(self):
        from cfuse.calibration.parameter_manager import get_cfuse_calibration_bounds
        bounds = get_cfuse_calibration_bounds()
        assert isinstance(bounds, dict)
        assert len(bounds) > 0
