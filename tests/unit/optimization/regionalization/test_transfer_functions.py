"""Unit Tests for shared transfer function classes."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.optimization]


class TestLinearTF:
    def test_apply(self):
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF
        tf = LinearTF(a_bounds=(0, 100), b_bounds=(-10, 10))
        result = tf.apply(np.array([0.0, 0.5, 1.0]), np.array([10.0, 5.0]))
        np.testing.assert_array_almost_equal(result, [10.0, 12.5, 15.0])

    def test_bounds(self):
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF
        tf = LinearTF(a_bounds=(0, 100), b_bounds=(-10, 10))
        assert len(tf.get_coefficient_bounds()) == 2

    def test_name(self):
        from symfluence.core.calibration.regionalization.transfer_functions import LinearTF
        assert LinearTF(a_bounds=(0, 1), b_bounds=(0, 1)).name == 'linear'


class TestPowerTF:
    def test_apply(self):
        from symfluence.core.calibration.regionalization.transfer_functions import PowerTF
        tf = PowerTF(a_bounds=(0, 100), b_bounds=(-2, 2))
        attr = np.array([1.0, 2.0, 4.0])
        result = tf.apply(attr, np.array([1.0, 2.0]))
        expected = 1.0 * np.power(attr + 0.01, 2.0)
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_handles_zero(self):
        from symfluence.core.calibration.regionalization.transfer_functions import PowerTF
        tf = PowerTF(a_bounds=(0, 100), b_bounds=(-2, 2))
        result = tf.apply(np.array([0.0]), np.array([1.0, 0.5]))
        assert np.isfinite(result[0])


class TestExponentialTF:
    def test_apply(self):
        from symfluence.core.calibration.regionalization.transfer_functions import ExponentialTF
        tf = ExponentialTF(a_bounds=(0, 100), b_bounds=(-2, 2))
        attr = np.array([0.0, 1.0])
        result = tf.apply(attr, np.array([2.0, 0.5]))
        np.testing.assert_array_almost_equal(result, 2.0 * np.exp(0.5 * attr))


class TestConstantTF:
    def test_apply(self):
        from symfluence.core.calibration.regionalization.transfer_functions import ConstantTF
        tf = ConstantTF(a_bounds=(0, 100))
        result = tf.apply(np.array([0.0, 0.5, 1.0, 100.0]), np.array([42.0]))
        np.testing.assert_array_equal(result, [42.0, 42.0, 42.0, 42.0])

    def test_single_coefficient(self):
        from symfluence.core.calibration.regionalization.transfer_functions import ConstantTF
        tf = ConstantTF(a_bounds=(0, 100))
        assert tf.n_coefficients == 1


class TestFlexiblePowerTF:
    def test_calibrated_exponent(self):
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF
        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        assert tf.n_coefficients == 2

    def test_fixed_exponent_zero_constant(self):
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF
        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=False, fixed_exponent=0.0)
        result = tf.apply(np.array([0.5, 1.0, 2.0]), np.array([42.0]))
        np.testing.assert_array_almost_equal(result, [42.0, 42.0, 42.0])

    def test_b_zero_gives_constant(self):
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF
        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        result = tf.apply(np.array([0.5, 1.0, 2.0]), np.array([42.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [42.0, 42.0, 42.0])

    def test_positive_exponent_increases(self):
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF
        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        result = tf.apply(np.array([0.1, 0.5, 1.0]), np.array([10.0, 1.0]))
        assert result[0] < result[1] < result[2]

    def test_handles_zero_safely(self):
        from symfluence.core.calibration.regionalization.transfer_functions import FlexiblePowerTF
        tf = FlexiblePowerTF(a_bounds=(0, 100), calibrate_exponent=True)
        result = tf.apply(np.array([0.0, 0.0]), np.array([10.0, 1.5]))
        assert np.all(np.isfinite(result))
