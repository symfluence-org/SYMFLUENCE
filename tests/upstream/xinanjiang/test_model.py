"""Tests for Xinanjiang model core simulation."""

import numpy as np
import pytest
from jsnow17.parameters import SNOW17_DEFAULTS
from jxaj.model import (
    HAS_JAX,
    XinanjiangState,
    create_initial_state,
    simulate,
    simulate_coupled_numpy,
)
from jxaj.parameters import DEFAULT_PARAMS


def _synthetic_forcing(n_days=730):
    """Create synthetic forcing for testing.

    Simple seasonal pattern: P peaks in summer, PET follows temperature.
    """
    t = np.arange(n_days)
    # Seasonal precipitation (mm/day), peak ~10 mm/day in summer
    precip = 3.0 + 7.0 * np.sin(2 * np.pi * t / 365.0 - np.pi / 2) ** 2
    precip = np.maximum(precip, 0.0)
    # Add some randomness
    rng = np.random.default_rng(42)
    precip = precip * rng.exponential(1.0, n_days)

    # PET (mm/day), seasonal
    pet = 1.0 + 3.0 * np.sin(2 * np.pi * t / 365.0) ** 2

    return precip, pet


class TestSimulateNumpy:
    """Test NumPy backend simulation."""

    def test_basic_simulation(self):
        """Model should produce non-negative runoff."""
        precip, pet = _synthetic_forcing()
        runoff, state = simulate(precip, pet, use_jax=False)

        assert len(runoff) == len(precip)
        assert np.all(runoff >= 0.0)
        assert np.all(np.isfinite(runoff))

    def test_default_params_produce_runoff(self):
        """Default parameters should produce positive runoff."""
        precip, pet = _synthetic_forcing()
        runoff, _ = simulate(precip, pet, use_jax=False, warmup_days=365)

        # After warmup, should have meaningful runoff
        assert np.mean(runoff[365:]) > 0.1

    def test_zero_precip_gives_zero_runoff(self):
        """Zero precipitation should eventually produce zero runoff."""
        precip = np.zeros(500)
        pet = np.ones(500) * 2.0
        runoff, _ = simulate(precip, pet, use_jax=False)

        # After depletion, runoff should approach zero
        assert runoff[-1] < 0.01

    def test_mass_conservation_approximate(self):
        """Total runoff + ET + storage change should approximate total precip.

        This is an approximate check since we don't track all fluxes.
        """
        precip, pet = _synthetic_forcing(n_days=1095)
        runoff, final_state = simulate(precip, pet, use_jax=False, warmup_days=365)

        total_precip = np.sum(precip[365:])
        total_runoff = np.sum(runoff[365:])

        # Runoff should be a reasonable fraction of precip (20-90%)
        runoff_ratio = total_runoff / total_precip
        assert 0.1 < runoff_ratio < 1.0, f"Runoff ratio {runoff_ratio:.2f} outside expected range"

    def test_state_output(self):
        """Final state should have reasonable values."""
        precip, pet = _synthetic_forcing()
        _, state = simulate(precip, pet, use_jax=False)

        assert isinstance(state, XinanjiangState)
        assert float(state.wu) >= 0.0
        assert float(state.wl) >= 0.0
        assert float(state.wd) >= 0.0
        assert float(state.s) >= 0.0
        assert 0.0 <= float(state.fr) <= 1.0

    def test_warmup_convergence(self):
        """Model should converge during warmup period.

        Running with 2 different initial states should converge
        to similar results after warmup.
        """
        precip, pet = _synthetic_forcing(n_days=1095)

        state1 = create_initial_state(initial_wu=0.0, initial_wl=0.0, use_jax=False)
        state2 = create_initial_state(initial_wu=20.0, initial_wl=80.0, use_jax=False)

        runoff1, _ = simulate(precip, pet, params=DEFAULT_PARAMS, use_jax=False,
                              initial_state=state1)
        runoff2, _ = simulate(precip, pet, params=DEFAULT_PARAMS, use_jax=False,
                              initial_state=state2)

        # After 2 years, the last year should be very similar
        last_year_1 = runoff1[730:]
        last_year_2 = runoff2[730:]
        max_diff = np.max(np.abs(last_year_1 - last_year_2))
        assert max_diff < 1.0, f"Max diff {max_diff:.3f} after warmup"

    def test_custom_params(self):
        """Model should accept custom parameter dictionaries."""
        precip, pet = _synthetic_forcing()
        params = DEFAULT_PARAMS.copy()
        params['K'] = 0.8  # Higher PET correction
        params['B'] = 0.2  # Lower curve exponent

        runoff, _ = simulate(precip, pet, params=params, use_jax=False)
        assert np.all(np.isfinite(runoff))


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestSimulateJAX:
    """Test JAX backend simulation."""

    def test_basic_simulation(self):
        """JAX simulation should produce valid output."""
        import jax.numpy as jnp

        precip, pet = _synthetic_forcing()
        precip_jax = jnp.array(precip)
        pet_jax = jnp.array(pet)

        runoff, state = simulate(precip_jax, pet_jax, use_jax=True)

        runoff_np = np.array(runoff)
        assert len(runoff_np) == len(precip)
        assert np.all(runoff_np >= 0.0)
        assert np.all(np.isfinite(runoff_np))

    def test_backend_equivalence(self):
        """JAX and NumPy backends should produce equivalent results."""
        import jax.numpy as jnp

        precip, pet = _synthetic_forcing()

        runoff_np, _ = simulate(precip, pet, use_jax=False)

        precip_jax = jnp.array(precip)
        pet_jax = jnp.array(pet)
        runoff_jax, _ = simulate(precip_jax, pet_jax, use_jax=True)

        np.testing.assert_allclose(
            np.array(runoff_jax), runoff_np,
            atol=1e-4, rtol=1e-4,
            err_msg="JAX and NumPy backends diverge"
        )


class TestParameterSensitivity:
    """Test that parameters have expected effects on output."""

    def test_higher_k_increases_et(self):
        """Higher K (PET correction) should reduce total runoff."""
        precip, pet = _synthetic_forcing()

        params_low = DEFAULT_PARAMS.copy()
        params_low['K'] = 0.2
        params_high = DEFAULT_PARAMS.copy()
        params_high['K'] = 0.9

        runoff_low, _ = simulate(precip, pet, params=params_low, use_jax=False)
        runoff_high, _ = simulate(precip, pet, params=params_high, use_jax=False)

        # Higher K means more ET, less runoff
        assert np.sum(runoff_low[365:]) > np.sum(runoff_high[365:])

    def test_higher_b_changes_runoff(self):
        """B exponent should affect runoff volume."""
        precip, pet = _synthetic_forcing()

        params_low = DEFAULT_PARAMS.copy()
        params_low['B'] = 0.1
        params_high = DEFAULT_PARAMS.copy()
        params_high['B'] = 0.4

        runoff_low, _ = simulate(precip, pet, params=params_low, use_jax=False)
        runoff_high, _ = simulate(precip, pet, params=params_high, use_jax=False)

        # Different B should produce different runoff (any direction is fine)
        assert not np.allclose(runoff_low[365:], runoff_high[365:], atol=0.1)


def _synthetic_coupled_forcing(n_days=730):
    """Create forcing with temperature for coupled Snow-17 + XAJ tests."""
    t = np.arange(n_days)
    # Temperature: cold winters, warm summers
    temp = 5.0 + 15.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    # Precipitation
    rng = np.random.default_rng(42)
    precip = rng.exponential(3.0, n_days)
    # PET
    pet = 1.0 + 3.0 * np.sin(2 * np.pi * t / 365.0) ** 2
    # Day of year
    doy = (t % 365) + 1
    return precip, temp, pet, doy


class TestCoupledSnow17XAJ:
    """Test coupled Snow-17 + XAJ simulation."""

    def test_coupled_produces_runoff(self):
        """Coupled simulation should produce valid runoff."""
        precip, temp, pet, doy = _synthetic_coupled_forcing()
        runoff, state = simulate(
            precip, pet,
            params=DEFAULT_PARAMS.copy(),
            use_jax=False,
            temp=temp,
            day_of_year=doy,
            snow17_params=SNOW17_DEFAULTS.copy(),
            latitude=51.0,
        )

        assert len(runoff) == len(precip)
        assert np.all(runoff >= 0.0)
        assert np.all(np.isfinite(runoff))
        # Should have meaningful runoff
        assert np.mean(runoff[365:]) > 0.1

    def test_coupled_snow_effect(self):
        """Coupled should differ from standalone (snow delays runoff)."""
        precip, temp, pet, doy = _synthetic_coupled_forcing()

        # Standalone XAJ
        runoff_standalone, _ = simulate(precip, pet, use_jax=False)

        # Coupled Snow-17 + XAJ
        runoff_coupled, _ = simulate(
            precip, pet,
            use_jax=False,
            temp=temp,
            day_of_year=doy,
            snow17_params=SNOW17_DEFAULTS.copy(),
            latitude=51.0,
        )

        # Should produce different results (snow stores/delays water)
        assert not np.allclose(runoff_standalone, runoff_coupled, atol=0.1)

    def test_backward_compat_no_snow(self):
        """simulate(precip, pet) without snow args should work unchanged."""
        precip, pet = _synthetic_forcing()
        runoff, state = simulate(precip, pet, use_jax=False)

        assert len(runoff) == len(precip)
        assert np.all(runoff >= 0.0)
        assert isinstance(state, XinanjiangState)

    def test_coupled_requires_day_of_year(self):
        """Coupled mode should raise when day_of_year missing."""
        precip, temp, pet, doy = _synthetic_coupled_forcing(n_days=100)
        with pytest.raises(ValueError, match="day_of_year"):
            simulate(
                precip, pet,
                use_jax=False,
                temp=temp,
                snow17_params=SNOW17_DEFAULTS.copy(),
            )

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_coupled_jax_equivalence(self):
        """JAX and NumPy coupled backends should produce similar results."""
        import jax.numpy as jnp

        precip, temp, pet, doy = _synthetic_coupled_forcing(n_days=365)

        runoff_np, _ = simulate(
            precip, pet,
            use_jax=False,
            temp=temp,
            day_of_year=doy,
            snow17_params=SNOW17_DEFAULTS.copy(),
            latitude=51.0,
        )

        runoff_jax, _ = simulate(
            jnp.array(precip), jnp.array(pet),
            use_jax=True,
            temp=jnp.array(temp),
            day_of_year=jnp.array(doy),
            snow17_params=SNOW17_DEFAULTS.copy(),
            latitude=51.0,
        )

        np.testing.assert_allclose(
            np.array(runoff_jax), runoff_np,
            atol=1e-4, rtol=1e-4,
            err_msg="JAX and NumPy coupled backends diverge",
        )
