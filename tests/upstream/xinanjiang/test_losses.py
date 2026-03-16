"""Tests for Xinanjiang loss functions and gradient utilities."""

import numpy as np
import pytest
from jxaj.losses import kge_loss, nse_loss
from jxaj.model import HAS_JAX
from jxaj.parameters import DEFAULT_PARAMS


def _synthetic_data(n_days=730):
    """Create synthetic forcing and 'observed' data for testing."""
    rng = np.random.default_rng(42)
    t = np.arange(n_days)
    precip = 3.0 + 7.0 * np.sin(2 * np.pi * t / 365.0 - np.pi / 2) ** 2
    precip = np.maximum(precip * rng.exponential(1.0, n_days), 0.0)
    pet = 1.0 + 3.0 * np.sin(2 * np.pi * t / 365.0) ** 2

    # Generate "observations" by running with default params
    from jxaj.model import simulate
    obs, _ = simulate(precip, pet, params=DEFAULT_PARAMS, use_jax=False, warmup_days=0)

    return precip, pet, obs


class TestLossFunctions:
    """Test NSE and KGE loss computation."""

    def test_nse_perfect_match(self):
        """NSE loss should be -1 (best) when sim == obs."""
        precip, pet, obs = _synthetic_data()
        loss = nse_loss(DEFAULT_PARAMS, precip, pet, obs, warmup_days=365, use_jax=False)
        # With default params and obs generated from default params, NSE should be ~1
        assert loss < -0.99, f"NSE loss = {loss}, expected ~ -1.0"

    def test_kge_perfect_match(self):
        """KGE loss should be -1 (best) when sim == obs."""
        precip, pet, obs = _synthetic_data()
        loss = kge_loss(DEFAULT_PARAMS, precip, pet, obs, warmup_days=365, use_jax=False)
        assert loss < -0.99, f"KGE loss = {loss}, expected ~ -1.0"

    def test_nse_worse_with_bad_params(self):
        """NSE should worsen with wrong parameters."""
        precip, pet, obs = _synthetic_data()
        good_loss = nse_loss(DEFAULT_PARAMS, precip, pet, obs, warmup_days=365, use_jax=False)

        bad_params = DEFAULT_PARAMS.copy()
        bad_params['K'] = 0.1
        bad_params['SM'] = 100.0
        bad_loss = nse_loss(bad_params, precip, pet, obs, warmup_days=365, use_jax=False)

        # Bad params should give worse (more positive) loss
        assert bad_loss > good_loss

    def test_loss_finite(self):
        """Loss functions should return finite values."""
        precip, pet, obs = _synthetic_data()
        nse = nse_loss(DEFAULT_PARAMS, precip, pet, obs, warmup_days=365, use_jax=False)
        kge = kge_loss(DEFAULT_PARAMS, precip, pet, obs, warmup_days=365, use_jax=False)
        assert np.isfinite(nse)
        assert np.isfinite(kge)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestGradients:
    """Test gradient computation via JAX autodiff."""

    def test_kge_gradient_nonzero(self):
        """KGE gradient should be non-zero for non-optimal parameters."""
        import jax.numpy as jnp
        from jxaj.losses import get_kge_gradient_fn
        from jxaj.parameters import PARAM_NAMES

        precip, pet, obs = _synthetic_data()
        precip_j = jnp.array(precip)
        pet_j = jnp.array(pet)
        obs_j = jnp.array(obs)

        grad_fn = get_kge_gradient_fn(precip_j, pet_j, obs_j, warmup_days=365)
        assert grad_fn is not None

        # Use perturbed parameters (not optimal)
        params = DEFAULT_PARAMS.copy()
        params['K'] = 0.3
        params['SM'] = 50.0
        param_names = list(params.keys())
        params_array = jnp.array([params[n] for n in param_names])

        grads = grad_fn(params_array, param_names)
        grads_np = np.array(grads)

        assert np.all(np.isfinite(grads_np)), "Gradients contain NaN/Inf"
        assert np.any(np.abs(grads_np) > 1e-10), "All gradients are zero"

    def test_nse_gradient_nonzero(self):
        """NSE gradient should be non-zero for non-optimal parameters."""
        import jax.numpy as jnp
        from jxaj.losses import get_nse_gradient_fn

        precip, pet, obs = _synthetic_data()
        precip_j = jnp.array(precip)
        pet_j = jnp.array(pet)
        obs_j = jnp.array(obs)

        grad_fn = get_nse_gradient_fn(precip_j, pet_j, obs_j, warmup_days=365)
        assert grad_fn is not None

        params = DEFAULT_PARAMS.copy()
        params['K'] = 0.3
        param_names = list(params.keys())
        params_array = jnp.array([params[n] for n in param_names])

        grads = grad_fn(params_array, param_names)
        grads_np = np.array(grads)

        assert np.all(np.isfinite(grads_np)), "Gradients contain NaN/Inf"
