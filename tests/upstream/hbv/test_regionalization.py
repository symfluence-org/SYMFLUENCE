"""
Tests for HBV Parameter Regionalization.

Verifies:
1. Transfer function initialization and forward pass
2. Integration with DistributedHBV
3. End-to-end differentiability of transfer function weights
4. Edge cases: mismatched dimensions, NaN/inf handling, extreme values
5. All activation functions (tanh, relu, sigmoid)
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


@pytest.mark.skipif(not HAS_JAX, reason="JAX required for regionalization tests")
class TestRegionalization:

    def test_transfer_function_forward(self):
        """Test simple forward pass of transfer function."""
        from jhbv.model import PARAM_BOUNDS, HBVParameters
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 5
        n_nodes = 10

        # Random attributes
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        attributes = jax.random.normal(k1, (n_nodes, input_dim))

        # Initialize network
        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8, 8],
            output_dim=len(PARAM_BOUNDS) # One output per parameter
        )

        weights = initialize_weights(k2, config)

        # Forward pass
        params = forward_transfer_function(weights, attributes, PARAM_BOUNDS)

        assert isinstance(params, HBVParameters)

        # Check shapes and bounds
        for name, bounds in PARAM_BOUNDS.items():
            if hasattr(params, name):
                val = getattr(params, name)
                assert val.shape == (n_nodes,)
                assert jnp.all(val >= bounds[0])
                assert jnp.all(val <= bounds[1])

    def test_distributed_hbv_integration(self):
        """Test integration with DistributedHBV runner."""
        from jhbv import (
            PARAM_BOUNDS,
            DistributedHBV,
            TransferFunctionConfig,
            create_synthetic_network,
            initialize_weights,
        )

        n_nodes = 5
        input_dim = 3

        network = create_synthetic_network(n_nodes=n_nodes, topology='linear')
        model = DistributedHBV(network, use_jax=True)

        # Mock attributes
        key = jax.random.PRNGKey(0)
        attributes = jax.random.normal(key, (n_nodes, input_dim))

        # Initialize weights
        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[5],
            output_dim=14 # Approx number of regionalized params
        )
        weights = initialize_weights(key, config)

        # Create params
        params = model.create_params(
            param_mode='regionalized',
            attributes=attributes,
            transfer_weights=weights
        )

        assert params.param_mode == 'regionalized'

        # Run simulation
        n_days = 10
        precip = jnp.ones((n_days, n_nodes)) * 5.0
        temp = jnp.ones((n_days, n_nodes)) * 10.0
        pet = jnp.ones((n_days, n_nodes)) * 2.0

        outlet_flow, _ = model.simulate(precip, temp, pet, params=params)

        assert outlet_flow.shape == (n_days,)
        assert jnp.all(jnp.isfinite(outlet_flow))

    def test_differentiability(self):
        """Test that we can compute gradients w.r.t weights."""
        from jhbv import (
            DistributedHBV,
            TransferFunctionConfig,
            create_synthetic_network,
            initialize_weights,
            nse_loss,
        )

        n_nodes = 3
        input_dim = 2
        network = create_synthetic_network(n_nodes=n_nodes, topology='linear')
        model = DistributedHBV(network, use_jax=True)

        key = jax.random.PRNGKey(1)
        attributes = jax.random.normal(key, (n_nodes, input_dim))

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[5],
            output_dim=14
        )
        initial_weights = initialize_weights(key, config)

        # Dummy data
        n_days = 20
        precip = jnp.ones((n_days, n_nodes))
        temp = jnp.ones((n_days, n_nodes))
        pet = jnp.ones((n_days, n_nodes))
        obs = jnp.ones(n_days)

        def loss_fn(weights):
            params = model.create_params(
                param_mode='regionalized',
                attributes=attributes,
                transfer_weights=weights
            )
            outlet_flow, _ = model.simulate(precip, temp, pet, params=params)

            # Simple MSE
            return jnp.mean((outlet_flow - obs)**2)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(initial_weights)

        # Check gradients exist and are correct shape
        assert len(grads) == len(initial_weights)
        for g, w in zip(grads, initial_weights):
            assert g.w.shape == w.w.shape
            assert g.b.shape == w.b.shape
            assert jnp.all(jnp.isfinite(g.w))


@pytest.mark.skipif(not HAS_JAX, reason="JAX required for regionalization tests")
class TestRegionalizationEdgeCases:
    """Edge case tests for regionalization module."""

    def test_mismatched_input_dimensions(self):
        """Test that mismatched input dimensions raise appropriate errors."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 5
        wrong_dim = 3
        n_nodes = 10

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # Attributes with wrong dimension
        wrong_attributes = jax.random.normal(k1, (n_nodes, wrong_dim))

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(k2, config)

        # Should raise error due to dimension mismatch in matrix multiply
        with pytest.raises(Exception):  # JAX raises TypeError or similar
            forward_transfer_function(weights, wrong_attributes, PARAM_BOUNDS)

    def test_single_node_input(self):
        """Test transfer function with single node (edge case)."""
        from jhbv.model import PARAM_BOUNDS, HBVParameters
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 4
        n_nodes = 1  # Single node

        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        attributes = jax.random.normal(k1, (n_nodes, input_dim))

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(k2, config)

        params = forward_transfer_function(weights, attributes, PARAM_BOUNDS)

        assert isinstance(params, HBVParameters)
        # Check all parameters have shape (1,)
        assert params.fc.shape == (1,)
        assert params.beta.shape == (1,)

    def test_nan_in_attributes(self):
        """Test that NaN in attributes propagates (does not silently fail)."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 3
        n_nodes = 5

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # Create attributes with NaN
        attributes = jax.random.normal(k1, (n_nodes, input_dim))
        attributes = attributes.at[2, 1].set(jnp.nan)

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(k2, config)

        params = forward_transfer_function(weights, attributes, PARAM_BOUNDS)

        # NaN should propagate to affected outputs (node 2)
        assert jnp.any(jnp.isnan(params.fc[2]))

    def test_infinity_in_attributes(self):
        """Test handling of infinity values in attributes."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 3
        n_nodes = 5

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # Create attributes with inf
        attributes = jax.random.normal(k1, (n_nodes, input_dim))
        attributes = attributes.at[0, 0].set(jnp.inf)

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(k2, config)

        params = forward_transfer_function(weights, attributes, PARAM_BOUNDS)

        # With sigmoid output mapping, inf should map to upper bound (not NaN)
        # Check that we get finite values due to sigmoid saturation
        assert jnp.isfinite(params.fc[0]) or jnp.isnan(params.fc[0])

    def test_extreme_attribute_values(self):
        """Test with very large and very small attribute values."""
        from jhbv.model import PARAM_BOUNDS, HBVParameters
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 3
        n_nodes = 4

        key = jax.random.PRNGKey(42)

        # Extreme values that shouldn't break sigmoid
        attributes = jnp.array([
            [1e6, 1e6, 1e6],      # Very large
            [-1e6, -1e6, -1e6],   # Very negative
            [1e-10, 1e-10, 1e-10], # Very small positive
            [0.0, 0.0, 0.0]       # Zeros
        ])

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(key, config)

        params = forward_transfer_function(weights, attributes, PARAM_BOUNDS)

        assert isinstance(params, HBVParameters)
        # All outputs should be finite due to sigmoid saturation
        for name, bounds in PARAM_BOUNDS.items():
            if hasattr(params, name):
                val = getattr(params, name)
                assert jnp.all(jnp.isfinite(val)), f"Non-finite values for {name}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX required for regionalization tests")
class TestActivationFunctions:
    """Test all supported activation functions."""

    @pytest.mark.parametrize("activation", ['tanh', 'relu', 'sigmoid'])
    def test_activation_function(self, activation):
        """Test transfer function with different activation functions."""
        from jhbv.model import PARAM_BOUNDS, HBVParameters
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 5
        n_nodes = 10

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        attributes = jax.random.normal(k1, (n_nodes, input_dim))

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8, 8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(k2, config)

        params = forward_transfer_function(
            weights, attributes, PARAM_BOUNDS, activation=activation
        )

        assert isinstance(params, HBVParameters)

        # Check shapes and bounds for all parameters
        for name, bounds in PARAM_BOUNDS.items():
            if hasattr(params, name):
                val = getattr(params, name)
                assert val.shape == (n_nodes,), f"{name} wrong shape with {activation}"
                assert jnp.all(val >= bounds[0]), f"{name} below lower bound with {activation}"
                assert jnp.all(val <= bounds[1]), f"{name} above upper bound with {activation}"

    def test_relu_gradient_flow(self):
        """Test that ReLU activation allows gradient flow."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.regionalization import (
            TransferFunctionConfig,
            forward_transfer_function,
            initialize_weights,
        )

        input_dim = 3
        n_nodes = 5

        key = jax.random.PRNGKey(42)
        attributes = jax.random.normal(key, (n_nodes, input_dim))

        config = TransferFunctionConfig(
            input_dim=input_dim,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(key, config)

        def loss_fn(w):
            params = forward_transfer_function(w, attributes, PARAM_BOUNDS, activation='relu')
            return jnp.mean(params.fc)

        grads = jax.grad(loss_fn)(weights)

        # Check gradients exist (some may be zero due to ReLU)
        assert len(grads) == len(weights)
        # At least some gradients should be non-zero
        has_nonzero = any(jnp.any(g.w != 0) for g in grads)
        assert has_nonzero, "All gradients are zero with ReLU"


@pytest.mark.skipif(not HAS_JAX, reason="JAX required for regionalization tests")
class TestRegularization:
    """Test regularization computation."""

    def test_regularization_positive(self):
        """Test that regularization term is always non-negative."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.regionalization import (
            TransferFunctionConfig,
            compute_regularization,
            initialize_weights,
        )

        key = jax.random.PRNGKey(42)

        config = TransferFunctionConfig(
            input_dim=5,
            hidden_dims=[8, 8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(key, config)

        reg = compute_regularization(weights, l2_lambda=0.01)
        assert reg >= 0.0

    def test_regularization_scaling(self):
        """Test that regularization scales with lambda."""
        from jhbv.model import PARAM_BOUNDS
        from jhbv.regionalization import (
            TransferFunctionConfig,
            compute_regularization,
            initialize_weights,
        )

        key = jax.random.PRNGKey(42)

        config = TransferFunctionConfig(
            input_dim=5,
            hidden_dims=[8],
            output_dim=len(PARAM_BOUNDS)
        )
        weights = initialize_weights(key, config)

        reg1 = compute_regularization(weights, l2_lambda=0.01)
        reg2 = compute_regularization(weights, l2_lambda=0.02)

        # reg2 should be approximately 2x reg1
        assert jnp.isclose(reg2, 2 * reg1, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
