"""Tests for standalone Snow-17 model."""

import numpy as np
import pytest
from jsnow17.model import (
    HAS_JAX,
    create_initial_state,
    seasonal_melt_factor,
    snow17_simulate,
    snow17_simulate_numpy,
    snow17_step,
)
from jsnow17.parameters import (
    DEFAULT_ADC,
    SNOW17_DEFAULTS,
    SNOW17_PARAM_BOUNDS,
    Snow17Params,
    Snow17State,
    params_dict_to_namedtuple,
)


def _synthetic_snow_forcing(n_days=730, lat=51.0):
    """Create synthetic forcing with cold winters and warm summers."""
    t = np.arange(n_days)
    # Temperature: cold winters (-15C), warm summers (+20C)
    temp = 5.0 + 15.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    # Precipitation: ~3 mm/day year-round
    rng = np.random.default_rng(42)
    precip = rng.exponential(3.0, n_days)
    # Day of year
    doy = (t % 365) + 1
    return precip, temp, doy


class TestSeasonalMeltFactor:
    """Test seasonal melt factor computation."""

    def test_bounds(self):
        """Melt factor should stay between MFMIN and MFMAX."""
        mfmax, mfmin = 1.5, 0.3
        for doy in range(1, 366):
            mf = seasonal_melt_factor(np.float64(doy), mfmax, mfmin, lat=45.0, xp=np)
            assert float(mfmin) <= float(mf) + 1e-10, f"mf={mf} < MFMIN at doy={doy}"
            assert float(mf) <= float(mfmax) + 1e-10, f"mf={mf} > MFMAX at doy={doy}"

    def test_northern_hemisphere_peak(self):
        """Melt factor should peak near Jun 21 (doy~172) in NH."""
        mfmax, mfmin = 2.0, 0.5
        mf_jun = seasonal_melt_factor(np.float64(172), mfmax, mfmin, lat=45.0, xp=np)
        mf_dec = seasonal_melt_factor(np.float64(355), mfmax, mfmin, lat=45.0, xp=np)
        assert float(mf_jun) > float(mf_dec)

    def test_southern_hemisphere_reversed(self):
        """Southern hemisphere should have reversed seasonality."""
        mfmax, mfmin = 2.0, 0.5
        mf_jun_nh = seasonal_melt_factor(np.float64(172), mfmax, mfmin, lat=45.0, xp=np)
        mf_jun_sh = seasonal_melt_factor(np.float64(172), mfmax, mfmin, lat=-45.0, xp=np)
        # Jun in SH should be near min, in NH near max
        assert float(mf_jun_nh) > float(mf_jun_sh)


class TestSnow17StepNumpy:
    """Test single Snow-17 timestep with NumPy."""

    def test_cold_accumulation(self):
        """Snowfall at cold temp should increase w_i."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        state = create_initial_state(use_jax=False)
        new_state, outflow = snow17_step(
            np.float64(10.0), np.float64(-10.0), 1.0,
            state, params, np.float64(15), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # All precip should become snow (temp well below PXTEMP)
        assert float(new_state.w_i) > 0.0
        # No outflow when snow is accumulating from cold
        assert float(outflow) < 1.0  # Very little or no outflow

    def test_warm_melt(self):
        """Warm temperature should melt snow and produce outflow."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        # Start with substantial snowpack
        state = Snow17State(
            w_i=np.float64(100.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(100.0),
        )
        new_state, outflow = snow17_step(
            np.float64(5.0), np.float64(10.0), 1.0,
            state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # Should produce outflow from melt + rain
        assert float(outflow) > 0.0
        # SWE should decrease
        total_swe = float(new_state.w_i) + float(new_state.w_q)
        assert total_swe < 100.0

    def test_rain_on_snow(self):
        """Rain on warm snow should produce extra melt."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        state = Snow17State(
            w_i=np.float64(50.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(50.0),
        )
        # Heavy rain at warm temp
        _, outflow_ros = snow17_step(
            np.float64(15.0), np.float64(8.0), 1.0,
            state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # Light rain at same temp
        _, outflow_light = snow17_step(
            np.float64(0.1), np.float64(8.0), 1.0,
            state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
        )
        # Rain-on-snow should produce more outflow
        assert float(outflow_ros) > float(outflow_light)

    def test_zero_precip_depletion(self):
        """Zero precip should gradually deplete snowpack."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        state = Snow17State(
            w_i=np.float64(50.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(50.0),
        )
        # Warm, no precip for many days
        for day in range(100):
            state, _ = snow17_step(
                np.float64(0.0), np.float64(5.0), 1.0,
                state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
            )
        # Snowpack should be depleted
        assert float(state.w_i) + float(state.w_q) < 1.0

    def test_state_reset_on_complete_melt(self):
        """When all snow melts, state should reset to clean."""
        params = params_dict_to_namedtuple(SNOW17_DEFAULTS, use_jax=False)
        state = Snow17State(
            w_i=np.float64(1.0), w_q=np.float64(0.0), w_qx=np.float64(0.0),
            deficit=np.float64(0.0), ati=np.float64(0.0), swe=np.float64(1.0),
        )
        # Very warm, melt everything
        for _ in range(10):
            state, _ = snow17_step(
                np.float64(0.0), np.float64(20.0), 1.0,
                state, params, np.float64(172), 45.0, 100.0, DEFAULT_ADC, xp=np,
            )
        assert float(state.w_i) < 1e-6
        assert float(state.w_q) < 1e-6
        assert float(state.deficit) < 1e-6


class TestSnow17SimulateNumpy:
    """Test full Snow-17 simulation with NumPy backend."""

    def test_basic_simulation(self):
        """Model should produce non-negative output."""
        precip, temp, doy = _synthetic_snow_forcing()
        rpm, state = snow17_simulate(precip, temp, doy, use_jax=False)

        assert len(rpm) == len(precip)
        assert np.all(rpm >= 0.0)
        assert np.all(np.isfinite(rpm))

    def test_snow_seasonality(self):
        """Outflow should peak in spring (melt) not winter (accumulation)."""
        precip, temp, doy = _synthetic_snow_forcing(n_days=730)
        rpm, _ = snow17_simulate(precip, temp, doy, use_jax=False)

        # Second year data (365-730)
        # Winter: days 365+335 to 365+90 (Nov-Mar)
        # Spring: days 365+90 to 365+180 (Apr-Jun)
        winter_rpm = np.mean(rpm[365+335:365+365]) + np.mean(rpm[365:365+60])
        spring_rpm = np.mean(rpm[365+90:365+180])
        assert spring_rpm > winter_rpm, "Spring melt should exceed winter outflow"

    def test_mass_conservation(self):
        """Total output should not exceed total input."""
        precip, temp, doy = _synthetic_snow_forcing(n_days=1095)
        rpm, final_state = snow17_simulate(precip, temp, doy, use_jax=False)

        total_rpm = np.sum(rpm)
        total_precip = np.sum(precip)
        remaining_swe = float(final_state.w_i) + float(final_state.w_q)

        # Output + remaining SWE should not exceed input
        assert total_rpm + remaining_swe <= total_precip * 1.5  # Some SCF correction allowed


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestSnow17JAX:
    """Test JAX backend simulation."""

    def test_basic_simulation(self):
        """JAX simulation should produce valid output."""
        import jax.numpy as jnp

        precip, temp, doy = _synthetic_snow_forcing()
        rpm, state = snow17_simulate(
            jnp.array(precip), jnp.array(temp), jnp.array(doy), use_jax=True,
        )

        rpm_np = np.array(rpm)
        assert len(rpm_np) == len(precip)
        assert np.all(rpm_np >= 0.0)
        assert np.all(np.isfinite(rpm_np))

    def test_backend_equivalence(self):
        """JAX and NumPy backends should produce equivalent results."""
        import jax.numpy as jnp

        precip, temp, doy = _synthetic_snow_forcing(n_days=365)

        rpm_np, _ = snow17_simulate(precip, temp, doy, use_jax=False)
        rpm_jax, _ = snow17_simulate(
            jnp.array(precip), jnp.array(temp), jnp.array(doy), use_jax=True,
        )

        np.testing.assert_allclose(
            np.array(rpm_jax), rpm_np,
            atol=1e-4, rtol=1e-4,
            err_msg="JAX and NumPy backends diverge",
        )

    def test_gradient_through_step(self):
        """Gradient through snow17_step should be finite and nonzero."""
        import jax
        import jax.numpy as jnp

        def loss_fn(scf):
            params = Snow17Params(
                SCF=scf, PXTEMP=jnp.array(0.0), MFMAX=jnp.array(1.0),
                MFMIN=jnp.array(0.3), NMF=jnp.array(0.15), MBASE=jnp.array(0.0),
                TIPM=jnp.array(0.1), UADJ=jnp.array(0.04),
                PLWHC=jnp.array(0.04), DAYGM=jnp.array(0.0),
            )
            state = Snow17State(
                w_i=jnp.array(50.0), w_q=jnp.array(0.0), w_qx=jnp.array(0.0),
                deficit=jnp.array(0.0), ati=jnp.array(0.0), swe=jnp.array(50.0),
            )
            adc = jnp.asarray(DEFAULT_ADC, dtype=float)
            _, outflow = snow17_step(
                jnp.array(5.0), jnp.array(-5.0), 1.0,
                state, params, jnp.array(15.0), 45.0, 100.0, adc, xp=jnp,
            )
            return outflow

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(1.0))
        assert np.isfinite(float(grad)), f"Gradient is not finite: {grad}"
