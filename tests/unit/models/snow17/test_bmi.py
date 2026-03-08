"""Tests for Snow-17 BMI interface."""

import numpy as np
import pytest
from jsnow17.bmi import Snow17BMI
from jsnow17.parameters import SNOW17_DEFAULTS


class TestSnow17BMI:
    """Test BMI-like interface lifecycle."""

    def test_initialize_finalize(self):
        """BMI init/finalize lifecycle should work."""
        bmi = Snow17BMI()
        bmi.initialize()
        assert bmi._initialized
        bmi.finalize()
        assert not bmi._initialized

    def test_update_produces_output(self):
        """Single update should produce valid output."""
        bmi = Snow17BMI(latitude=51.0)
        bmi.initialize()

        # Cold day with snow
        rpm = bmi.update(10.0, -10.0, 15)
        assert isinstance(rpm, float)
        assert rpm >= 0.0
        assert np.isfinite(rpm)

        bmi.finalize()

    def test_update_auto_initializes(self):
        """Update without explicit initialize should auto-init."""
        bmi = Snow17BMI()
        rpm = bmi.update(5.0, -5.0, 30)
        assert rpm >= 0.0

    def test_batch_update(self):
        """Batch update should produce array of correct length."""
        bmi = Snow17BMI(latitude=45.0)
        bmi.initialize()

        n = 365
        t = np.arange(n)
        temp = 5.0 + 15.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
        precip = np.ones(n) * 3.0
        doy = t + 1

        rpm = bmi.update_batch(precip, temp, doy)
        assert len(rpm) == n
        assert np.all(rpm >= 0.0)
        assert np.all(np.isfinite(rpm))

    def test_batch_matches_sequential(self):
        """Batch update should match sequential updates."""
        n = 100
        t = np.arange(n)
        temp = -5.0 + 10.0 * np.sin(2 * np.pi * t / 365.0)
        precip = np.ones(n) * 2.5
        doy = (t % 365) + 1

        # Sequential
        bmi_seq = Snow17BMI(latitude=45.0)
        bmi_seq.initialize()
        rpm_seq = np.zeros(n)
        for i in range(n):
            rpm_seq[i] = bmi_seq.update(precip[i], temp[i], int(doy[i]))

        # Batch
        bmi_batch = Snow17BMI(latitude=45.0)
        bmi_batch.initialize()
        rpm_batch = bmi_batch.update_batch(precip, temp, doy)

        np.testing.assert_allclose(rpm_batch, rpm_seq, atol=1e-10)

    def test_get_state(self):
        """get_state should return valid Snow17State."""
        bmi = Snow17BMI()
        bmi.initialize()

        # Accumulate some snow
        for _ in range(30):
            bmi.update(5.0, -10.0, 15)

        state = bmi.get_state()
        assert float(state.w_i) > 0.0  # Should have ice

    def test_get_value(self):
        """get_value should return correct variable values."""
        bmi = Snow17BMI()
        bmi.initialize()

        # Accumulate snow
        bmi.update(10.0, -15.0, 15)

        swe = bmi.get_value('swe')
        assert swe >= 0.0

        w_i = bmi.get_value('w_i')
        assert w_i >= 0.0

        rpm = bmi.get_value('rain_plus_melt')
        assert rpm >= 0.0

    def test_get_output_var_names(self):
        """Should return list of output variable names."""
        bmi = Snow17BMI()
        names = bmi.get_output_var_names()
        assert 'rain_plus_melt' in names
        assert 'swe' in names

    def test_custom_params(self):
        """Custom parameters should be accepted."""
        params = SNOW17_DEFAULTS.copy()
        params['SCF'] = 1.3
        params['MFMAX'] = 1.8

        bmi = Snow17BMI(params=params, latitude=60.0)
        bmi.initialize()
        rpm = bmi.update(10.0, -5.0, 30)
        assert rpm >= 0.0
