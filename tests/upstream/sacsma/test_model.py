"""Tests for coupled Snow-17 + SAC-SMA model orchestrator (dual-backend)."""

import numpy as np
import pytest
from jsacsma.model import HAS_JAX, SacSmaSnow17State, simulate
from jsacsma.parameters import DEFAULT_PARAMS
from jsacsma.sacsma import SacSmaState
from jsnow17.parameters import Snow17State


class TestCoupledSimulation:
    """Test coupled Snow-17 + SAC-SMA execution."""

    def test_basic_simulation(self):
        """Run a basic year-long simulation."""
        n = 365
        precip = np.full(n, 3.0)
        temp = 10.0 * np.sin(np.arange(n) * 2 * np.pi / 365 - np.pi / 2)
        pet = np.maximum(0.0, 2.0 + 1.5 * np.sin(np.arange(n) * 2 * np.pi / 365 - np.pi / 2))

        flow, state = simulate(precip, temp, pet, start_date='2004-01-01')

        assert len(flow) == n
        assert np.all(flow >= 0)
        assert flow.sum() > 0

    def test_with_default_params(self):
        """Should work with None params (uses defaults)."""
        n = 100
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow, _ = simulate(precip, temp, pet, params=None, start_date='2004-01-01')
        assert len(flow) == n

    def test_with_custom_params(self):
        """Should accept partial parameter dicts."""
        n = 100
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        custom = {'SCF': 1.2, 'UZTWM': 80.0}
        flow, _ = simulate(precip, temp, pet, params=custom, start_date='2004-01-01')
        assert len(flow) == n

    def test_all_rain_passthrough(self):
        """Warm temps: snow module passes rain -> SAC-SMA generates flow."""
        n = 100
        precip = np.full(n, 5.0)
        temp = np.full(n, 25.0)  # No snow
        pet = np.full(n, 2.0)

        flow, state = simulate(precip, temp, pet, start_date='2004-06-01')

        assert flow.sum() > 0
        assert state.snow17.w_i == pytest.approx(0.0, abs=1e-10)

    def test_cold_accumulation_then_melt(self):
        """Cold period stores snow, warm period melts -> delayed flow."""
        n = 200
        precip = np.full(n, 3.0)
        temp = np.concatenate([np.full(100, -10.0), np.full(100, 15.0)])
        pet = np.concatenate([np.full(100, 0.5), np.full(100, 3.0)])

        flow, _ = simulate(precip, temp, pet, start_date='2004-01-01')

        cold_flow = flow[:100].mean()
        warm_flow = flow[100:].mean()
        assert warm_flow > cold_flow

    def test_returns_combined_state(self):
        """Final state should be SacSmaSnow17State."""
        n = 30
        precip = np.full(n, 3.0)
        temp = np.full(n, 5.0)
        pet = np.full(n, 2.0)

        _, state = simulate(precip, temp, pet, start_date='2004-01-01')

        assert isinstance(state, SacSmaSnow17State)
        assert isinstance(state.snow17, Snow17State)
        assert isinstance(state.sacsma, SacSmaState)

    def test_day_of_year_from_start_date(self):
        """Should generate correct day_of_year from start_date."""
        n = 100
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow, _ = simulate(precip, temp, pet, start_date='2004-07-01')
        assert len(flow) == n

    def test_day_of_year_explicit(self):
        """Should accept explicit day_of_year."""
        n = 100
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)
        doy = np.arange(1, n + 1)

        flow, _ = simulate(precip, temp, pet, day_of_year=doy)
        assert len(flow) == n

    def test_missing_doy_raises_in_coupled_mode(self):
        """Should raise ValueError when no day_of_year or start_date in coupled mode."""
        n = 30
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        with pytest.raises(ValueError, match="day_of_year or start_date required"):
            simulate(precip, temp, pet)

    def test_zero_precip_drains_storage(self):
        """No precip: flow should come from draining initial storage."""
        n = 500
        precip = np.zeros(n)
        temp = np.full(n, 10.0)
        pet = np.full(n, 1.0)

        flow, _ = simulate(precip, temp, pet, start_date='2004-01-01')

        assert flow[:30].sum() > 0
        assert flow[-10:].mean() < 0.01

    def test_positive_flow_reasonable_range(self):
        """Flow should be positive and in a reasonable range (mm/day)."""
        n = 730
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow, _ = simulate(precip, temp, pet, start_date='2004-01-01')

        assert flow.min() >= 0
        assert flow[365:].mean() < precip.mean()

    def test_custom_initial_state(self):
        """Should accept custom initial state."""
        n = 30
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        init_state = SacSmaSnow17State(
            snow17=Snow17State(w_i=0.0, w_q=0.0, w_qx=0.0, deficit=0.0, ati=0.0, swe=0.0),
            sacsma=SacSmaState(uztwc=10.0, uzfwc=5.0, lztwc=50.0, lzfpc=20.0, lzfsc=10.0, adimc=60.0),
        )

        flow, _ = simulate(precip, temp, pet, initial_state=init_state, start_date='2004-01-01')
        assert len(flow) == n

    def test_start_date_leap_year(self):
        """start_date on a leap year should produce DOY=366 for Dec 31."""
        n = 366  # Full leap year
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow, _ = simulate(precip, temp, pet, start_date='2004-01-01')
        assert len(flow) == n


class TestStandaloneMode:
    """Test standalone SAC-SMA (no snow module)."""

    def test_standalone_runs(self):
        """snow_module='none' should run without snow processing."""
        n = 100
        precip = np.full(n, 5.0)
        temp = np.full(n, 10.0)  # temp unused in 'none' mode
        pet = np.full(n, 2.0)

        flow, state = simulate(precip, temp, pet, snow_module='none')
        assert len(flow) == n
        assert np.all(flow >= 0)
        assert flow.sum() > 0

    def test_standalone_no_snow_state_change(self):
        """In standalone mode, snow17 state should remain unchanged."""
        n = 100
        precip = np.full(n, 5.0)
        temp = np.full(n, -10.0)  # Cold, but no snow module
        pet = np.full(n, 2.0)

        _, state = simulate(precip, temp, pet, snow_module='none')
        # Snow state should be initial (all zeros)
        assert state.snow17.w_i == 0.0
        assert state.snow17.swe == 0.0

    def test_standalone_no_doy_required(self):
        """Standalone mode should not require day_of_year or start_date."""
        n = 50
        precip = np.full(n, 5.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow, _ = simulate(precip, temp, pet, snow_module='none')
        assert len(flow) == n


class TestParameterSensitivity:
    """Verify parameters affect simulation as expected."""

    def _run(self, **kwargs):
        n = 365
        precip = np.full(n, 4.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)
        params = {**DEFAULT_PARAMS, **kwargs}
        flow, _ = simulate(precip, temp, pet, params=params, start_date='2004-01-01')
        return flow

    def test_higher_uzk_more_interflow(self):
        flow_low = self._run(UZK=0.15)
        flow_high = self._run(UZK=0.7)
        assert not np.allclose(flow_low, flow_high)

    def test_higher_lzpk_more_baseflow(self):
        flow_low = self._run(LZPK=0.002)
        flow_high = self._run(LZPK=0.04)
        assert not np.allclose(flow_low, flow_high)

    def test_larger_uztwm_more_storage(self):
        flow_small = self._run(UZTWM=5.0)
        flow_large = self._run(UZTWM=140.0)
        assert flow_large.max() <= flow_small.max() + 1.0


class TestCoupledJAX:
    """Test JAX backend for coupled simulation."""

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_coupled_jax_runs(self):
        """Coupled simulation with use_jax=True should produce results."""
        n = 100
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow, state = simulate(precip, temp, pet, use_jax=True, start_date='2004-01-01')
        assert len(flow) == n
        assert float(flow.sum()) > 0

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_coupled_jax_numpy_consistency(self):
        """JAX and NumPy coupled modes should produce similar results."""
        n = 100
        precip = np.full(n, 3.0)
        temp = np.full(n, 10.0)
        pet = np.full(n, 2.0)

        flow_np, _ = simulate(precip, temp, pet, use_jax=False, start_date='2004-01-01')
        flow_jax, _ = simulate(precip, temp, pet, use_jax=True, start_date='2004-01-01')

        np.testing.assert_allclose(np.array(flow_jax), flow_np, rtol=1e-4, atol=1e-5)
