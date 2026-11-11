"""
Unit tests for HYPE Parameter Manager

Tests that the HYPE parameter manager can:
1. Load parameter bounds from registry
2. Normalize and denormalize parameters correctly
3. Handle parameter initialization
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from symfluence.models.hype.calibration.parameter_manager import HYPEParameterManager


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_logger')


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hype_config(temp_dir):
    """Create a basic HYPE configuration."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'HYPE_PARAMS_TO_CALIBRATE': 'ttmp,cmlt,cevp,lp,rrcs1',
    }


class TestHYPEParameterManager:
    """Tests for HYPE Parameter Manager."""

    def test_can_instantiate(self, hype_config, logger, temp_dir):
        """Test that HYPEParameterManager can be instantiated."""
        manager = HYPEParameterManager(hype_config, logger, temp_dir)
        assert manager is not None
        assert manager.domain_name == 'test_domain'
        assert manager.experiment_id == 'test_exp'

    def test_parameter_names(self, hype_config, logger, temp_dir):
        """Test that parameter names are parsed correctly."""
        manager = HYPEParameterManager(hype_config, logger, temp_dir)
        param_names = manager._get_parameter_names()
        assert param_names == ['ttmp', 'cmlt', 'cevp', 'lp', 'rrcs1']

    def test_load_bounds(self, hype_config, logger, temp_dir):
        """Test that parameter bounds can be loaded from registry."""
        manager = HYPEParameterManager(hype_config, logger, temp_dir)
        bounds = manager._load_parameter_bounds()

        assert 'ttmp' in bounds
        assert 'cmlt' in bounds
        assert 'cevp' in bounds

        # Check specific bounds (from central parameter registry)
        assert bounds['ttmp']['min'] == -5.0
        assert bounds['ttmp']['max'] == 5.0
        assert bounds['cmlt']['min'] == 0.5
        assert bounds['cmlt']['max'] == 20.0

    def test_normalize_denormalize(self, hype_config, logger, temp_dir):
        """Test parameter normalization and denormalization roundtrip."""
        manager = HYPEParameterManager(hype_config, logger, temp_dir)

        # Test parameters
        params = {'ttmp': 0.0, 'cmlt': 5.0, 'cevp': 0.5, 'lp': 0.6, 'rrcs1': 0.2}

        # Normalize
        normalized = manager.normalize_parameters(params)
        assert len(normalized) == 5

        # All normalized values should be between 0 and 1
        for val in normalized:
            assert 0 <= val <= 1

        # Denormalize
        denorm = manager.denormalize_parameters(normalized)

        # Check roundtrip accuracy
        for param_name in params:
            assert abs(denorm[param_name] - params[param_name]) < 1e-6

    def test_get_default_initial_values(self, hype_config, logger, temp_dir):
        """Test that default initial values are midpoint of bounds."""
        manager = HYPEParameterManager(hype_config, logger, temp_dir)
        defaults = manager._get_default_initial_values()

        # ttmp bounds: -5.0 to 5.0, midpoint should be 0.0
        assert abs(defaults['ttmp'] - 0.0) < 1e-6

        # cmlt bounds: 0.5 to 20.0, midpoint should be 10.25
        assert abs(defaults['cmlt'] - 10.25) < 1e-6

        # cevp bounds: 0.1 to 2.0, midpoint should be 1.05
        assert abs(defaults['cevp'] - 1.05) < 1e-6

    def test_update_par_file_creates_warning_if_missing(self, hype_config, logger, temp_dir, caplog):
        """Test that update_par_file handles missing file gracefully."""
        manager = HYPEParameterManager(hype_config, logger, temp_dir)

        params = {'ttmp': 1.0}
        result = manager.update_par_file(params)

        # Should fail gracefully
        assert result is False


def test_update_par_file_keeps_small_values(hype_config, logger, temp_dir):
    """Regression: '.6f' wrote 1.4e-5 as 0.000014 and 4e-7 as 0.000000."""
    manager = HYPEParameterManager(hype_config, logger, temp_dir)
    manager.par_file_path.parent.mkdir(parents=True, exist_ok=True)
    manager.par_file_path.write_text("!! par\nrcgrw\t0.1\nrrcs1\t0.2\t0.3\t!! per soil\nlp\t0.5\n")
    assert manager.update_par_file({'rcgrw': 1.4e-5, 'rrcs1': 4e-7, 'lp': [0.123456789, 2.5e-8]})
    rows = {ln.split()[0]: ln.split('!!')[0].split()[1:] for ln in manager.par_file_path.read_text().splitlines()[1:]}
    assert [float(v) for v in rows['rcgrw']] == [1.4e-5]
    assert [float(v) for v in rows['rrcs1']] == [4e-7, 4e-7]
    assert [float(v) for v in rows['lp']] == [0.123456789, 2.5e-8]
