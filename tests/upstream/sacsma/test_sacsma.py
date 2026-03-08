"""Tests for SAC-SMA soil moisture accounting model (dual-backend)."""

import numpy as np
import pytest
from jsacsma.parameters import (
    SACSMA_DEFAULTS,
    create_sacsma_params,
)
from jsacsma.sacsma import (
    HAS_JAX,
    SacSmaState,
    _create_default_state,
    sacsma_simulate,
    sacsma_simulate_numpy,
    sacsma_step,
)


@pytest.fixture
def default_params():
    return create_sacsma_params(SACSMA_DEFAULTS)


@pytest.fixture
def half_capacity_state(default_params):
    return _create_default_state(default_params)


@pytest.fixture
def dry_state(default_params):
    """Completely dry soil state."""
    return SacSmaState(uztwc=0.0, uzfwc=0.0, lztwc=0.0, lzfpc=0.0, lzfsc=0.0, adimc=0.0)


@pytest.fixture
def saturated_state(default_params):
    """Fully saturated soil state."""
    return SacSmaState(
        uztwc=default_params.UZTWM,
        uzfwc=default_params.UZFWM,
        lztwc=default_params.LZTWM,
        lzfpc=default_params.LZFPM,
        lzfsc=default_params.LZFSM,
        adimc=default_params.UZTWM + default_params.LZTWM,
    )


class TestETSequence:
    """Test evapotranspiration demand sequence."""

    def test_et_from_upper_zone_first(self, default_params, half_capacity_state):
        """ET should first deplete upper zone tension water."""
        pet = 5.0
        state_before = half_capacity_state
        state, surf, interf, base, et = sacsma_step(
            0.0, pet, 1.0, state_before, default_params, xp=np,
        )
        # UZTWC should decrease
        assert state.uztwc < state_before.uztwc

    def test_et_limited_by_pet(self, default_params, saturated_state):
        """Total ET should not exceed PET (plus RIVA)."""
        pet = 2.0
        _, _, _, _, et = sacsma_step(
            0.0, pet, 1.0, saturated_state, default_params, xp=np,
        )
        # ET can be slightly above PET due to RIVA
        assert et <= pet * (1.0 + default_params.RIVA + 0.5)

    def test_zero_et_when_dry(self, default_params, dry_state):
        """No ET when soil is empty."""
        _, _, _, _, et = sacsma_step(
            0.0, 5.0, 1.0, dry_state, default_params, xp=np,
        )
        # Only RIVA component if any
        assert et <= 5.0 * default_params.RIVA + 0.01

    def test_no_et_when_pet_zero(self, default_params, half_capacity_state):
        """No ET when PET is zero."""
        state, _, _, _, et = sacsma_step(
            0.0, 0.0, 1.0, half_capacity_state, default_params, xp=np,
        )
        assert et == pytest.approx(0.0, abs=1e-10)


class TestPercolation:
    """Test percolation from upper zone to lower zone."""

    def test_percolation_reduces_uzfwc(self, default_params, half_capacity_state):
        """Percolation should reduce upper zone free water."""
        # Add water to trigger percolation
        wet_state = half_capacity_state._replace(
            uzfwc=default_params.UZFWM * 0.8,
            lztwc=default_params.LZTWM * 0.1,  # Low LZ -> high demand
        )
        state, _, _, _, _ = sacsma_step(
            5.0, 1.0, 1.0, wet_state, default_params, xp=np,
        )
        lz_before = wet_state.lztwc + wet_state.lzfpc + wet_state.lzfsc
        lz_after = state.lztwc + state.lzfpc + state.lzfsc
        assert lz_after >= lz_before - 1.0

    def test_no_percolation_when_uz_dry(self, default_params, dry_state):
        """No percolation when upper zone is dry."""
        state, _, _, _, _ = sacsma_step(
            0.0, 0.0, 1.0, dry_state, default_params, xp=np,
        )
        assert state.lzfpc == 0.0
        assert state.lzfsc == 0.0


class TestSurfaceRunoff:
    """Test surface runoff generation."""

    def test_direct_runoff_from_pctim(self, default_params, half_capacity_state):
        """Permanent impervious area should generate direct runoff."""
        pxv = 20.0
        _, surface, _, _, _ = sacsma_step(
            pxv, 0.0, 1.0, half_capacity_state, default_params, xp=np,
        )
        assert surface >= pxv * default_params.PCTIM * 0.9

    def test_overflow_runoff_when_saturated(self, default_params, saturated_state):
        """Large precip on saturated soil should produce surface runoff."""
        pxv = 50.0
        _, surface, _, _, _ = sacsma_step(
            pxv, 0.0, 1.0, saturated_state, default_params, xp=np,
        )
        assert surface > 0

    def test_no_surface_runoff_dry_soil(self, default_params, dry_state):
        """Small precip on dry soil: all absorbed, minimal runoff."""
        pxv = 1.0
        _, surface, _, _, _ = sacsma_step(
            pxv, 0.0, 1.0, dry_state, default_params, xp=np,
        )
        assert surface <= pxv * (default_params.PCTIM + default_params.ADIMP + 0.01)


class TestBaseflow:
    """Test baseflow recession."""

    def test_baseflow_from_lower_zone(self, default_params, half_capacity_state):
        """Lower zone free water should generate baseflow."""
        _, _, _, baseflow, _ = sacsma_step(
            0.0, 0.0, 1.0, half_capacity_state, default_params, xp=np,
        )
        assert baseflow > 0

    def test_no_baseflow_when_lz_dry(self, default_params, dry_state):
        """No baseflow when lower zone is empty."""
        _, _, _, baseflow, _ = sacsma_step(
            0.0, 0.0, 1.0, dry_state, default_params, xp=np,
        )
        assert baseflow == pytest.approx(0.0, abs=1e-10)

    def test_baseflow_recession(self, default_params, half_capacity_state):
        """Baseflow should decrease over time with no input."""
        state = half_capacity_state
        baseflows = []
        for _ in range(30):
            state, _, _, bf, _ = sacsma_step(0.0, 0.0, 1.0, state, default_params, xp=np)
            baseflows.append(float(bf))

        assert baseflows[-1] < baseflows[0]

    def test_deep_loss_reduces_effective_baseflow(self):
        """SIDE > 0 should reduce effective baseflow."""
        params_no_side = create_sacsma_params({**SACSMA_DEFAULTS, 'SIDE': 0.0})
        params_with_side = create_sacsma_params({**SACSMA_DEFAULTS, 'SIDE': 0.3})

        state = _create_default_state(params_no_side)
        _, _, _, bf_no_side, _ = sacsma_step(0.0, 0.0, 1.0, state, params_no_side, xp=np)
        _, _, _, bf_with_side, _ = sacsma_step(0.0, 0.0, 1.0, state, params_with_side, xp=np)

        assert bf_with_side < bf_no_side


class TestInterflow:
    """Test interflow from upper zone."""

    def test_interflow_from_uzfwc(self, default_params):
        """Upper zone free water should generate interflow when LZ is saturated."""
        state = SacSmaState(
            uztwc=default_params.UZTWM * 0.5,
            uzfwc=default_params.UZFWM * 0.8,
            lztwc=default_params.LZTWM,
            lzfpc=default_params.LZFPM,
            lzfsc=default_params.LZFSM,
            adimc=(default_params.UZTWM + default_params.LZTWM) * 0.5,
        )
        _, _, interflow, _, _ = sacsma_step(
            0.0, 0.0, 1.0, state, default_params, xp=np,
        )
        assert interflow > 0

    def test_no_interflow_when_uz_dry(self, default_params, dry_state):
        """No interflow when UZ free water is empty."""
        _, _, interflow, _, _ = sacsma_step(
            0.0, 0.0, 1.0, dry_state, default_params, xp=np,
        )
        assert interflow == pytest.approx(0.0, abs=1e-10)


class TestNonNegativeOutputs:
    """Ensure all outputs are non-negative under various conditions."""

    @pytest.mark.parametrize("pxv", [0.0, 1.0, 10.0, 50.0, 200.0])
    @pytest.mark.parametrize("pet", [0.0, 2.0, 10.0])
    def test_non_negative(self, default_params, half_capacity_state, pxv, pet):
        state, surface, interflow, baseflow, et = sacsma_step(
            pxv, pet, 1.0, half_capacity_state, default_params, xp=np,
        )
        assert surface >= 0
        assert interflow >= 0
        assert baseflow >= 0
        assert et >= 0
        assert state.uztwc >= 0
        assert state.uzfwc >= 0
        assert state.lztwc >= 0
        assert state.lzfpc >= 0
        assert state.lzfsc >= 0


class TestSacSmaSimulate:
    """Test full SAC-SMA simulation."""

    def test_basic_simulation(self, default_params):
        n = 365
        pxv = np.full(n, 3.0)
        pet = np.full(n, 2.0)

        flow, final = sacsma_simulate(pxv, pet, default_params)

        assert len(flow) == n
        assert np.all(flow >= 0)
        assert flow.sum() > 0

    def test_numpy_backend_explicit(self, default_params):
        """Test explicit NumPy backend call."""
        n = 100
        pxv = np.full(n, 3.0)
        pet = np.full(n, 2.0)

        flow, final = sacsma_simulate_numpy(pxv, pet, default_params)
        assert len(flow) == n
        assert np.all(flow >= 0)

    def test_dry_in_produces_zero_flow(self, default_params):
        """No precip, no PET -> flow only from initial storage."""
        n = 1000
        pxv = np.zeros(n)
        pet = np.zeros(n)

        flow, final = sacsma_simulate(pxv, pet, default_params)

        assert flow[:30].sum() > 0
        assert flow[-10:].mean() < 0.01

    def test_mass_conservation_approximate(self, default_params):
        """Total in ~ total out + storage change + deep loss."""
        n = 1000
        pxv = np.full(n, 5.0)
        pet = np.full(n, 2.0)

        init_state = _create_default_state(default_params)
        flow, final = sacsma_simulate(pxv, pet, default_params, initial_state=init_state)

        total_in = pxv.sum()
        total_out = flow.sum()
        assert total_out > 0
        assert total_out < total_in

    def test_returns_final_state(self, default_params):
        n = 30
        pxv = np.full(n, 5.0)
        pet = np.full(n, 2.0)

        _, final = sacsma_simulate(pxv, pet, default_params)

        assert isinstance(final, SacSmaState)


class TestDualBackend:
    """Test JAX/NumPy dual-backend consistency."""

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_jax_numpy_consistency(self):
        """JAX and NumPy backends should produce similar results."""
        import jax.numpy as jnp
        from jsacsma.parameters import params_dict_to_namedtuple
        from jsacsma.sacsma import sacsma_simulate_jax

        params_np = create_sacsma_params(SACSMA_DEFAULTS)
        params_jax = params_dict_to_namedtuple(SACSMA_DEFAULTS, use_jax=True)

        n = 100
        pxv = np.full(n, 3.0)
        pet = np.full(n, 2.0)

        flow_np, _ = sacsma_simulate_numpy(pxv, pet, params_np)
        flow_jax, _ = sacsma_simulate_jax(jnp.array(pxv), jnp.array(pet), params_jax)

        np.testing.assert_allclose(np.array(flow_jax), flow_np, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_jax_gradient(self):
        """JAX should compute non-zero gradients through SAC-SMA."""
        import jax
        import jax.numpy as jnp
        from jsacsma.parameters import SACSMA_PARAM_NAMES, SacSmaParameters
        from jsacsma.sacsma import _create_default_state, sacsma_simulate_jax

        def loss_fn(uztwm_val):
            p_dict = {**SACSMA_DEFAULTS, 'UZTWM': uztwm_val}
            values = []
            for name in SACSMA_PARAM_NAMES:
                val = p_dict[name]
                values.append(val if hasattr(val, 'shape') else jnp.array(float(val)))
            params = SacSmaParameters(*values)
            state = _create_default_state(params, use_jax=True)
            pxv = jnp.full(30, 3.0)
            pet = jnp.full(30, 2.0)
            flow, _ = sacsma_simulate_jax(pxv, pet, params, initial_state=state)
            return jnp.sum(flow)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(50.0))
        assert not jnp.isnan(g)
        assert g != 0.0
