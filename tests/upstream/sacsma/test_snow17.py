"""Tests for Snow-17 model (redirected to shared module).

These tests verify that the SAC-SMA package correctly imports Snow-17
from the shared ``symfluence.models.snow17`` module.
"""

import numpy as np
import pytest
from jsacsma.parameters import (
    Snow17Parameters,
    create_snow17_params,
)
from jsnow17.model import (
    areal_depletion,
    seasonal_melt_factor,
    snow17_simulate_numpy,
    snow17_step,
)
from jsnow17.parameters import (
    DEFAULT_ADC,
    SNOW17_DEFAULTS,
    Snow17Params,
    Snow17State,
    params_dict_to_namedtuple,
)


@pytest.fixture
def default_params():
    return create_snow17_params(SNOW17_DEFAULTS)


@pytest.fixture
def no_snow_state():
    return Snow17State(w_i=0.0, w_q=0.0, w_qx=0.0, deficit=0.0, ati=0.0, swe=0.0)


@pytest.fixture
def snowy_state():
    return Snow17State(w_i=50.0, w_q=2.0, w_qx=2.5, deficit=5.0, ati=-2.0, swe=55.0)


class TestSharedModuleImport:
    """Verify shared module imports work from SAC-SMA package."""

    def test_snow17_parameters_alias(self):
        """Snow17Parameters should alias Snow17Params from shared module."""
        assert Snow17Parameters is Snow17Params

    def test_create_snow17_params_returns_shared_type(self):
        """create_snow17_params should return shared Snow17Params type."""
        params = create_snow17_params(SNOW17_DEFAULTS)
        assert isinstance(params, Snow17Params)


class TestSeasonalMeltFactor:
    """Test seasonal melt factor sinusoid."""

    def test_peaks_near_summer_solstice(self):
        mfmax, mfmin = 1.5, 0.3
        mf_172 = float(seasonal_melt_factor(172, mfmax, mfmin, xp=np))
        assert mf_172 > (mfmax + mfmin) / 2

    def test_trough_near_winter_solstice(self):
        mfmax, mfmin = 1.5, 0.3
        mf_355 = float(seasonal_melt_factor(355, mfmax, mfmin, xp=np))
        assert mf_355 < (mfmax + mfmin) / 2

    def test_average_at_equinox(self):
        mfmax, mfmin = 1.5, 0.3
        mf_81 = float(seasonal_melt_factor(81, mfmax, mfmin, xp=np))
        expected_avg = (mfmax + mfmin) / 2
        assert abs(mf_81 - expected_avg) < 0.01

    def test_bounded_between_mfmin_mfmax(self):
        mfmax, mfmin = 1.5, 0.3
        for doy in range(1, 366):
            mf = float(seasonal_melt_factor(doy, mfmax, mfmin, xp=np))
            assert mfmin - 0.001 <= mf <= mfmax + 0.001, f"Day {doy}: mf={mf}"

    def test_southern_hemisphere(self):
        mfmax, mfmin = 1.5, 0.3
        mf_n_summer = float(seasonal_melt_factor(172, mfmax, mfmin, lat=45.0, xp=np))
        mf_s_summer = float(seasonal_melt_factor(172, mfmax, mfmin, lat=-45.0, xp=np))
        assert mf_n_summer > mf_s_summer


class TestArealDepletion:
    """Test areal depletion curve."""

    def test_zero_swe(self):
        cover = float(areal_depletion(np.float64(0.0), 100.0, DEFAULT_ADC, xp=np))
        assert cover == pytest.approx(0.0, abs=0.01)

    def test_full_coverage_at_si(self):
        cover = float(areal_depletion(np.float64(100.0), 100.0, DEFAULT_ADC, xp=np))
        assert cover == pytest.approx(1.0)

    def test_partial_coverage(self):
        cover = float(areal_depletion(np.float64(50.0), 100.0, DEFAULT_ADC, xp=np))
        assert 0.0 < cover < 1.0

    def test_monotonic_increasing(self):
        si = 100.0
        prev = 0.0
        for swe in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            cover = float(areal_depletion(np.float64(swe), si, DEFAULT_ADC, xp=np))
            assert cover >= prev - 0.001, f"Non-monotonic at SWE={swe}"
            prev = cover


class TestRainSnowPartition:
    """Test rain/snow partitioning in snow17_step."""

    def test_all_snow_below_threshold(self, default_params, no_snow_state):
        temp = float(default_params.PXTEMP) - 2.0
        state, outflow = snow17_step(
            np.float64(10.0), np.float64(temp), 1.0,
            no_snow_state, default_params,
            doy=np.float64(1), xp=np,
        )
        assert state.w_i > 0

    def test_all_rain_above_threshold(self, default_params, no_snow_state):
        temp = float(default_params.PXTEMP) + 2.0
        state, outflow = snow17_step(
            np.float64(10.0), np.float64(temp), 1.0,
            no_snow_state, default_params,
            doy=np.float64(172), xp=np,
        )
        assert state.w_i == 0.0
        assert outflow > 0


class TestSnow17Step:
    """Test individual Snow-17 timesteps."""

    def test_no_precip_no_change(self, default_params, no_snow_state):
        state, outflow = snow17_step(
            np.float64(0.0), np.float64(5.0), 1.0,
            no_snow_state, default_params,
            doy=np.float64(172), xp=np,
        )
        assert state.w_i == 0.0
        assert outflow == 0.0

    def test_melt_in_warm_conditions(self, default_params, snowy_state):
        state, outflow = snow17_step(
            np.float64(0.0), np.float64(10.0), 1.0,
            snowy_state, default_params,
            doy=np.float64(172), xp=np,
        )
        assert state.w_i < snowy_state.w_i or outflow > 0

    def test_snowfall_accumulation(self, default_params, no_snow_state):
        state, _ = snow17_step(
            np.float64(20.0), np.float64(-10.0), 1.0,
            no_snow_state, default_params,
            doy=np.float64(1), xp=np,
        )
        assert state.w_i > 0

    def test_non_negative_outputs(self, default_params, snowy_state):
        for temp in [-20, -5, 0, 5, 15, 30]:
            state, outflow = snow17_step(
                np.float64(5.0), np.float64(float(temp)), 1.0,
                snowy_state, default_params,
                doy=np.float64(172), xp=np,
            )
            assert state.w_i >= 0
            assert state.w_q >= 0
            assert state.deficit >= 0
            assert outflow >= 0


class TestSnow17Simulate:
    """Test full Snow-17 simulation."""

    def test_basic_simulation(self, default_params):
        n = 365
        precip = np.full(n, 3.0)
        temp = 10.0 * np.sin(np.arange(n) * 2 * np.pi / 365 - np.pi / 2)
        doy = np.arange(1, n + 1)

        rpm, final = snow17_simulate_numpy(precip, temp, doy, default_params)

        assert len(rpm) == n
        assert np.all(rpm >= 0)
        assert rpm.sum() > 0

    def test_all_rain_passthrough(self, default_params):
        n = 30
        precip = np.full(n, 5.0)
        temp = np.full(n, 20.0)
        doy = np.full(n, 172, dtype=int)

        rpm, final = snow17_simulate_numpy(precip, temp, doy, default_params)

        total_in = precip.sum()
        total_out = rpm.sum()
        assert total_out > 0.8 * total_in

    def test_returns_final_state(self, default_params):
        n = 30
        precip = np.full(n, 3.0)
        temp = np.full(n, -5.0)
        doy = np.arange(1, n + 1)

        _, final = snow17_simulate_numpy(precip, temp, doy, default_params)

        assert isinstance(final, Snow17State)
        assert final.w_i > 0
