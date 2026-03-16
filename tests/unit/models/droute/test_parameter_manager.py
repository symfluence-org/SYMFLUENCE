"""Tests for dRoute parameter manager."""

import logging
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def logger():
    return logging.getLogger('test_droute_pm')


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def droute_config(temp_dir):
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'DROUTE_PARAMS_TO_CALIBRATE': 'velocity,diffusivity,manning_n',
    }


class TestDRouteParameterManagerRegistration:
    """Tests for parameter manager registration."""

    def test_parameter_manager_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'DROUTE' in OptimizerRegistry._parameter_managers

    def test_parameter_manager_is_correct_class(self):
        from droute.calibration.parameter_manager import DRouteParameterManager

        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry._parameter_managers.get('DROUTE') == DRouteParameterManager


class TestDRouteParameterBounds:
    """Tests for dRoute parameter bounds."""

    def test_droute_bounds_available(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_droute_bounds
        bounds = get_droute_bounds()
        assert len(bounds) == 5

    def test_velocity_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_droute_bounds
        bounds = get_droute_bounds()
        assert 'velocity' in bounds
        assert bounds['velocity']['min'] == 0.1
        assert bounds['velocity']['max'] == 5.0

    def test_diffusivity_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_droute_bounds
        bounds = get_droute_bounds()
        assert 'diffusivity' in bounds
        assert bounds['diffusivity']['min'] == 100.0
        assert bounds['diffusivity']['max'] == 5000.0

    def test_muskingum_x_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_droute_bounds
        bounds = get_droute_bounds()
        assert 'muskingum_x' in bounds
        assert bounds['muskingum_x']['min'] == 0.0
        assert bounds['muskingum_x']['max'] == 0.5

    def test_manning_n_bounds(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_droute_bounds
        bounds = get_droute_bounds()
        assert 'manning_n' in bounds
        assert bounds['manning_n']['min'] == 0.01
        assert bounds['manning_n']['max'] == 0.1


class TestDRouteParameterManagerInstance:
    """Tests for dRoute parameter manager instances."""

    def test_can_instantiate(self, droute_config, logger, temp_dir):
        from droute.calibration.parameter_manager import DRouteParameterManager
        manager = DRouteParameterManager(droute_config, logger, temp_dir)
        assert manager is not None

    def test_parameter_names(self, droute_config, logger, temp_dir):
        from droute.calibration.parameter_manager import DRouteParameterManager
        manager = DRouteParameterManager(droute_config, logger, temp_dir)
        names = manager._get_parameter_names()
        assert names == ['velocity', 'diffusivity', 'manning_n']

    def test_load_bounds(self, droute_config, logger, temp_dir):
        from droute.calibration.parameter_manager import DRouteParameterManager
        manager = DRouteParameterManager(droute_config, logger, temp_dir)
        bounds = manager._load_parameter_bounds()
        assert 'velocity' in bounds
        assert 'diffusivity' in bounds
        assert 'manning_n' in bounds

    def test_normalize_denormalize_roundtrip(self, droute_config, logger, temp_dir):
        from droute.calibration.parameter_manager import DRouteParameterManager
        manager = DRouteParameterManager(droute_config, logger, temp_dir)

        params = {'velocity': 2.5, 'diffusivity': 2500.0, 'manning_n': 0.05}
        normalized = manager.normalize_parameters(params)
        denormalized = manager.denormalize_parameters(normalized)

        for key in params:
            assert abs(denormalized[key] - params[key]) < 0.01, \
                f"Roundtrip failed for {key}: {params[key]} -> {denormalized[key]}"

    def test_get_initial_parameters(self, droute_config, logger, temp_dir):
        from droute.calibration.parameter_manager import DRouteParameterManager
        manager = DRouteParameterManager(droute_config, logger, temp_dir)
        initial = manager.get_initial_parameters()
        assert initial is not None
        assert 'velocity' in initial
        assert 'diffusivity' in initial

    def test_update_model_files_no_config(self, droute_config, logger, temp_dir):
        """Without a config file, update should succeed silently."""
        from droute.calibration.parameter_manager import DRouteParameterManager
        manager = DRouteParameterManager(droute_config, logger, temp_dir)
        result = manager.update_model_files({'velocity': 1.0, 'diffusivity': 500.0})
        assert result is True

    def test_update_model_files_with_yaml(self, droute_config, logger, temp_dir):
        """With a YAML config file, parameters should be written."""
        import yaml
        from droute.calibration.parameter_manager import DRouteParameterManager

        config_path = temp_dir / 'droute_config.yaml'
        config_path.write_text(yaml.dump({'routing': {'method': 'muskingum_cunge'}}))

        manager = DRouteParameterManager(droute_config, logger, temp_dir)
        result = manager.update_model_files({'velocity': 2.0, 'manning_n': 0.03})
        assert result is True

        # Verify written values
        with open(config_path) as f:
            updated = yaml.safe_load(f)
        assert updated['routing']['velocity'] == 2.0
        assert updated['routing']['manning_n'] == 0.03
