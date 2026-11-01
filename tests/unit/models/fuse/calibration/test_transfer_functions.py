"""
Unit Tests for FUSE Transfer Functions.

Tests the MPR-style transfer function framework:
- Individual transfer function classes (Linear, Power, Exponential, Constant, FlexiblePower)
- ParameterTransferManager
- Coefficient-to-parameter conversion
- Spatial variation summary
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mark all tests in this module
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_transfer_functions')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_attributes_csv(tmp_path):
    """Create a sample subcatchment attributes CSV."""
    df = pd.DataFrame({
        'id': range(1, 6),
        'elev_m': [500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        'precip_mm_yr': [600.0, 800.0, 1000.0, 1200.0, 1400.0],
        'temp_C': [8.0, 5.0, 2.0, -1.0, -4.0],
        'aridity': [1.0, 0.8, 0.6, 0.4, 0.2],
        'snow_frac': [0.1, 0.3, 0.5, 0.7, 0.9],
    })
    path = tmp_path / "subcatchment_attributes.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def param_bounds():
    """Standard parameter bounds."""
    return {
        'MAXWATR_1': (25.0, 500.0),
        'BASERTE': (0.001, 0.1),
        'MBASE': (-5.0, 5.0),
    }


# =============================================================================
# LinearTF Tests
# =============================================================================

class TestLinearTF:
    """Tests for LinearTF transfer function."""

    def test_apply_linear(self):
        """Should compute a + b * attr."""
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF

        tf = LinearTF(a_bounds=(0, 100), b_bounds=(-10, 10))
        attr = np.array([0.0, 0.5, 1.0])
        coeffs = np.array([10.0, 5.0])  # a=10, b=5

        result = tf.apply(attr, coeffs)

        np.testing.assert_array_almost_equal(result, [10.0, 12.5, 15.0])

    def test_coefficient_bounds(self):
        """Should return 2 sets of bounds."""
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF

        tf = LinearTF(a_bounds=(0, 100), b_bounds=(-10, 10))
        bounds = tf.get_coefficient_bounds()

        assert len(bounds) == 2
        assert bounds[0] == (0, 100)
        assert bounds[1] == (-10, 10)

    def test_name_is_linear(self):
        """Name should be 'linear'."""
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF

        tf = LinearTF(a_bounds=(0, 1), b_bounds=(0, 1))
        assert tf.name == 'linear'

    def test_n_coefficients(self):
        """Should have 2 coefficients."""
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF

        tf = LinearTF(a_bounds=(0, 1), b_bounds=(0, 1))
        assert tf.n_coefficients == 2


# =============================================================================
# PowerTF Tests
# =============================================================================

class TestPowerTF:
    """Tests for PowerTF transfer function."""

    def test_apply_power(self):
        """Should compute a * (attr + 0.01)^b."""
        from symfluence.core.calibration.regionalization.transfer_functions import PowerTF

        tf = PowerTF(a_bounds=(0, 100), b_bounds=(-2, 2))
        attr = np.array([1.0, 2.0, 4.0])
        coeffs = np.array([1.0, 2.0])  # a=1, b=2

        result = tf.apply(attr, coeffs)

        # 1.0 * (attr + 0.01)^2
        expected = 1.0 * np.power(attr + 0.01, 2.0)
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_handles_zero_attribute(self):
        """Should handle zero attributes (via +0.01 offset)."""
        from symfluence.core.calibration.regionalization.transfer_functions import PowerTF

        tf = PowerTF(a_bounds=(0, 100), b_bounds=(-2, 2))
        attr = np.array([0.0])
        coeffs = np.array([1.0, 0.5])

        result = tf.apply(attr, coeffs)

        assert np.isfinite(result[0])

    def test_name_is_power(self):
        """Name should be 'power'."""
        from symfluence.core.calibration.regionalization.transfer_functions import PowerTF

        tf = PowerTF(a_bounds=(0, 1), b_bounds=(0, 1))
        assert tf.name == 'power'


# =============================================================================
# ExponentialTF Tests
# =============================================================================

class TestExponentialTF:
    """Tests for ExponentialTF transfer function."""

    def test_apply_exponential(self):
        """Should compute a * exp(b * attr)."""
        from symfluence.core.calibration.regionalization.transfer_functions import ExponentialTF

        tf = ExponentialTF(a_bounds=(0, 100), b_bounds=(-2, 2))
        attr = np.array([0.0, 1.0])
        coeffs = np.array([2.0, 0.5])  # a=2, b=0.5

        result = tf.apply(attr, coeffs)

        expected = 2.0 * np.exp(0.5 * attr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_name_is_exponential(self):
        """Name should be 'exponential'."""
        from symfluence.core.calibration.regionalization.transfer_functions import ExponentialTF

        tf = ExponentialTF(a_bounds=(0, 1), b_bounds=(0, 1))
        assert tf.name == 'exponential'


# =============================================================================
# ConstantTF Tests
# =============================================================================

class TestConstantTF:
    """Tests for ConstantTF transfer function."""

    def test_apply_constant(self):
        """Should return uniform value regardless of attribute."""
        from symfluence.core.calibration.regionalization.transfer_functions import ConstantTF

        tf = ConstantTF(a_bounds=(0, 100))
        attr = np.array([0.0, 0.5, 1.0, 100.0])
        coeffs = np.array([42.0])

        result = tf.apply(attr, coeffs)

        np.testing.assert_array_equal(result, [42.0, 42.0, 42.0, 42.0])

    def test_single_coefficient(self):
        """Should have exactly 1 coefficient."""
        from symfluence.core.calibration.regionalization.transfer_functions import ConstantTF

        tf = ConstantTF(a_bounds=(0, 100))
        assert tf.n_coefficients == 1
        assert len(tf.get_coefficient_bounds()) == 1

    def test_name_is_constant(self):
        """Name should be 'constant'."""
        from symfluence.core.calibration.regionalization.transfer_functions import ConstantTF

        tf = ConstantTF(a_bounds=(0, 1))
        assert tf.name == 'constant'


# =============================================================================
# FlexiblePowerTF Tests
# =============================================================================

class TestFlexiblePowerTF:
    """Tests for FlexiblePowerTF transfer function."""

    def test_with_calibrated_exponent(self):
        """When calibrate_exponent=True, should use both a and b."""
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF

        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        assert tf.n_coefficients == 2

    def test_without_calibrated_exponent(self):
        """When calibrate_exponent=False, should use fixed exponent."""
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF

        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=False, fixed_exponent=0.0)
        assert tf.n_coefficients == 1

    def test_fixed_exponent_zero_gives_constant(self):
        """b=0 should give constant output equal to a."""
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF

        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=False, fixed_exponent=0.0)
        attr = np.array([0.5, 1.0, 2.0])
        coeffs = np.array([42.0])

        result = tf.apply(attr, coeffs)

        np.testing.assert_array_almost_equal(result, [42.0, 42.0, 42.0])

    def test_calibrated_b_zero_gives_constant(self):
        """Even with calibrate_exponent=True, b~0 gives constant."""
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF

        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        attr = np.array([0.5, 1.0, 2.0])
        coeffs = np.array([42.0, 0.0])  # b = 0

        result = tf.apply(attr, coeffs)

        np.testing.assert_array_almost_equal(result, [42.0, 42.0, 42.0])

    def test_positive_exponent_increases_with_attribute(self):
        """b > 0 should give increasing output with increasing attr."""
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF

        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        attr = np.array([0.1, 0.5, 1.0])
        coeffs = np.array([10.0, 1.0])  # a=10, b=1

        result = tf.apply(attr, coeffs)

        # Values should increase
        assert result[0] < result[1] < result[2]

    def test_handles_zero_attribute_safely(self):
        """Should handle zero attributes without error."""
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF

        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        attr = np.array([0.0, 0.0])
        coeffs = np.array([10.0, 1.5])

        result = tf.apply(attr, coeffs)

        assert np.all(np.isfinite(result))


# =============================================================================
# ParameterTransferManager Tests
# =============================================================================

class TestParameterTransferManager:
    """Tests for ParameterTransferManager."""

    def test_initialization(self, sample_attributes_csv, param_bounds, test_logger):
        """Should initialize and load attributes."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=param_bounds,
            logger=test_logger,
        )

        assert manager.n_subcatchments == 5

    def test_normalizes_attributes(self, sample_attributes_csv, param_bounds, test_logger):
        """Should create normalized attribute columns."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=param_bounds,
            logger=test_logger,
        )

        assert 'elev_m_norm' in manager.attributes.columns
        vals = manager.attributes['elev_m_norm'].values
        assert vals.min() == pytest.approx(0.0)
        assert vals.max() == pytest.approx(1.0)

    def test_get_calibration_parameters_returns_dict(
        self, sample_attributes_csv, param_bounds, test_logger
    ):
        """Should return coefficient bounds as dictionary."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=param_bounds,
            logger=test_logger,
        )

        cal_params = manager.get_calibration_parameters()

        assert isinstance(cal_params, dict)
        # Each parameter should have at least an 'a' coefficient
        for param_name in param_bounds:
            assert f'{param_name}_a' in cal_params

    def test_coefficients_to_parameters_shape(
        self, sample_attributes_csv, param_bounds, test_logger
    ):
        """Should return array with correct shape."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=param_bounds,
            logger=test_logger,
        )

        # Build coefficients from calibration parameters
        cal_params = manager.get_calibration_parameters()
        coefficients = {name: (bounds[0] + bounds[1]) / 2 for name, bounds in cal_params.items()}

        param_array, param_names = manager.coefficients_to_parameters(coefficients)

        assert param_array.shape[0] == 5  # n_subcatchments
        assert len(param_names) > 0

    def test_parameters_within_bounds(
        self, sample_attributes_csv, param_bounds, test_logger
    ):
        """Output parameters should be clipped to original bounds."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=param_bounds,
            logger=test_logger,
        )

        cal_params = manager.get_calibration_parameters()
        # Use extreme coefficient values
        coefficients = {}
        for name, bounds in cal_params.items():
            if name.endswith('_a'):
                coefficients[name] = bounds[1] * 2  # Way above max
            else:
                coefficients[name] = bounds[1]  # Max slope

        param_array, param_names = manager.coefficients_to_parameters(coefficients)

        for i, pname in enumerate(param_names):
            if pname in param_bounds:
                p_min, p_max = param_bounds[pname]
                assert param_array[:, i].min() >= p_min - 1e-10
                assert param_array[:, i].max() <= p_max + 1e-10

    def test_summarize_spatial_variation(
        self, sample_attributes_csv, param_bounds, test_logger
    ):
        """Should return a summary DataFrame."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=param_bounds,
            logger=test_logger,
        )

        cal_params = manager.get_calibration_parameters()
        coefficients = {name: (bounds[0] + bounds[1]) / 2 for name, bounds in cal_params.items()}

        summary = manager.summarize_spatial_variation(coefficients)

        assert isinstance(summary, pd.DataFrame)
        assert 'parameter' in summary.columns
        assert 'min' in summary.columns
        assert 'max' in summary.columns
        assert 'mean' in summary.columns
        assert 'std' in summary.columns

    def test_ignores_params_not_in_bounds(
        self, sample_attributes_csv, test_logger
    ):
        """Should skip parameters from DEFAULT_PARAM_CONFIG not in param_bounds."""
        from symfluence.models.fuse.calibration.transfer_functions import (
            ParameterTransferManager,
        )

        # Only MBASE in bounds, even though DEFAULT_PARAM_CONFIG has many more
        narrow_bounds = {'MBASE': (-5.0, 5.0)}

        manager = ParameterTransferManager(
            attributes_path=Path(sample_attributes_csv),
            param_bounds=narrow_bounds,
            logger=test_logger,
        )

        assert len(manager.transfer_functions) == 1
        assert 'MBASE' in manager.transfer_functions
