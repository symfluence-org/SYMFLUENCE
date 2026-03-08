"""
Tests for Distributed HBV with Graph-Based Muskingum-Cunge Routing.

Tests cover:
1. River network graph construction and topology
2. Muskingum-Cunge routing implementation
3. Distributed HBV model integration
4. Gradient computation for calibration
5. End-to-end simulation accuracy
"""

from unittest.mock import patch

import numpy as np
import pytest

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


class TestRiverNetwork:
    """Tests for river network graph construction."""

    def test_create_linear_network(self):
        """Test creation of simple linear network."""
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=5, topology='linear', use_jax=False)

        assert network.n_nodes == 5
        assert network.n_edges == 4  # Linear: n-1 edges
        assert len(network.topo_order) == 5
        assert len(network.outlet_idx) == 1

        # Check topological order: should go from upstream to downstream
        # For linear network, upstream nodes should come first
        topo = np.asarray(network.topo_order)
        assert topo[-1] == network.outlet_idx[0]  # Outlet should be last

    def test_create_binary_tree_network(self):
        """Test creation of binary tree network."""
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=7, topology='binary_tree', use_jax=False)

        assert network.n_nodes == 7
        assert network.n_edges == 6  # Tree: n-1 edges
        assert len(network.outlet_idx) == 1

        # Check upstream counts (binary tree: root has 2 children, leaves have 0)
        upstream_count = np.asarray(network.upstream_count)
        assert upstream_count[0] == 2  # Root has 2 children
        assert np.sum(upstream_count == 0) >= 3  # At least 3 leaves

    def test_create_fishbone_network(self):
        """Test creation of fishbone (main stem + tributaries) network."""
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=8, topology='fishbone', use_jax=False)

        assert network.n_nodes == 8
        assert network.n_edges == 7
        assert len(network.outlet_idx) == 1

    def test_topological_order_valid(self):
        """Test that topological order is valid (upstream before downstream)."""
        from jhbv.network import create_synthetic_network

        for topology in ['linear', 'binary_tree', 'fishbone']:
            network = create_synthetic_network(n_nodes=7, topology=topology, use_jax=False)

            topo_order = np.asarray(network.topo_order)
            downstream_idx = np.asarray(network.downstream_idx)

            # For each node, its downstream should appear later in topo order
            topo_positions = {node: i for i, node in enumerate(topo_order)}

            for node in topo_order:
                down = downstream_idx[node]
                if down >= 0:  # Not an outlet
                    assert topo_positions[node] < topo_positions[down], \
                        f"Topology {topology}: node {node} should come before downstream {down}"

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_network_jax_arrays(self):
        """Test that JAX arrays are created when requested."""
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=5, topology='linear', use_jax=True)

        # Check that arrays are JAX arrays
        assert hasattr(network.node_ids, 'device')  # JAX array attribute
        assert hasattr(network.topo_order, 'device')


class TestMuskingumCungeRouting:
    """Tests for Muskingum-Cunge routing implementation."""

    def test_routing_coefficients_sum_to_one(self):
        """Test that routing coefficients C0 + C1 + C2 = 1."""
        from jhbv.routing import muskingum_coefficients

        # Test various K and x values
        test_cases = [
            (3600, 0.2),   # 1 hour travel time
            (7200, 0.3),   # 2 hours
            (86400, 0.1),  # 1 day
        ]

        for K, x in test_cases:
            dt = 3600  # 1 hour timestep
            C0, C1, C2 = muskingum_coefficients(K, x, dt)
            total = C0 + C1 + C2
            np.testing.assert_almost_equal(total, 1.0, decimal=10,
                                          err_msg=f"K={K}, x={x}")

    def test_single_reach_mass_balance(self):
        """Test mass balance through single reach."""
        from jhbv.routing import route_reach_step

        K = 3600.0  # 1 hour
        x = 0.2
        dt = 3600.0

        # Constant inflow should eventually produce constant outflow
        Q_prev = 0.0
        I = 10.0  # 10 m³/s constant inflow

        outflows = []
        for _ in range(100):
            Q = route_reach_step(I, I, Q_prev, K, x, dt)
            outflows.append(float(Q))
            Q_prev = Q

        # After equilibration, outflow should equal inflow
        np.testing.assert_almost_equal(outflows[-1], I, decimal=2)

    def test_adaptive_substepping(self):
        """Test that adaptive sub-stepping maintains stability."""
        from jhbv.routing import route_reach_adaptive

        K = 600.0  # 10 minutes - short travel time
        x = 0.2
        dt = 3600.0  # 1 hour timestep - longer than K

        # This should require sub-stepping for stability
        I_prev = 5.0
        I_curr = 15.0  # Sharp increase
        Q_prev = 5.0

        # With sub-stepping, output should be reasonable
        Q = route_reach_adaptive(I_prev, I_curr, Q_prev, K, x, dt, n_substeps=6)

        assert Q >= 0, "Output should be non-negative"
        assert Q < I_curr * 2, "Output should be bounded"

    def test_runoff_unit_conversion(self):
        """Test runoff conversion from mm to m³/s."""
        from jhbv.routing import runoff_mm_to_cms

        # 1 mm/day over 1 km² = 1000 m³/day = 1000/86400 m³/s
        runoff_mm = np.array([1.0])
        area_m2 = np.array([1e6])  # 1 km²
        dt = 86400.0  # 1 day

        Q = runoff_mm_to_cms(runoff_mm, area_m2, dt)

        expected = 1.0 / 1000.0 * 1e6 / 86400.0  # ≈ 0.01157 m³/s
        np.testing.assert_almost_equal(Q[0], expected, decimal=5)


class TestDistributedHBV:
    """Tests for the distributed HBV model."""

    def test_model_initialization(self):
        """Test basic model initialization."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=5, topology='linear', use_jax=False)
        model = DistributedHBV(network, use_jax=False)

        assert model.network.n_nodes == 5
        assert model.param_mode == 'uniform'

    def test_create_initial_state(self):
        """Test initial state creation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, use_jax=False)

        state = model.create_initial_state(
            initial_snow=10.0,
            initial_sm=200.0,
            initial_suz=15.0,
            initial_slz=20.0,
            initial_Q=1.0
        )

        assert len(state.hbv_states) == 3
        assert state.hbv_states[0].snow == 10.0
        assert state.hbv_states[0].sm == 200.0

    def test_create_uniform_params(self):
        """Test uniform parameter creation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, param_mode='uniform', use_jax=False)

        params = model.create_params(hbv_params={'fc': 300.0, 'beta': 2.0})

        assert params.param_mode == 'uniform'
        assert params.hbv_params.fc == 300.0

    def test_simulate_basic(self):
        """Test basic simulation runs without error."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        # Create synthetic forcing (100 days, 3 GRUs)
        n_days = 100
        n_nodes = 3
        np.random.seed(42)

        precip = np.random.exponential(5, (n_days, n_nodes))  # mm/day
        temp = 10 + 5 * np.sin(np.arange(n_days) * 2 * np.pi / 365)  # Seasonal
        temp = np.broadcast_to(temp[:, np.newaxis], (n_days, n_nodes))
        pet = np.full((n_days, n_nodes), 3.0)  # mm/day

        outlet_flow, state = model.simulate(precip, temp, pet)

        assert len(outlet_flow) == n_days
        assert np.all(outlet_flow >= 0), "Flow should be non-negative"
        assert np.any(outlet_flow > 0), "Should have some positive flow"

    def test_simulate_with_gru_runoff(self):
        """Test simulation returns GRU runoff when requested."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 50
        n_nodes = 3
        precip = np.random.exponential(5, (n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)

        outlet_flow, gru_runoff, state = model.simulate(
            precip, temp, pet, return_gru_runoff=True
        )

        assert gru_runoff.shape == (n_days, n_nodes)
        assert np.all(gru_runoff >= 0)

    def test_mass_balance(self):
        """Test approximate mass balance through the system."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        # Long simulation with constant forcing
        n_days = 500
        n_nodes = 3
        precip = np.full((n_days, n_nodes), 5.0)  # 5 mm/day constant
        temp = np.full((n_days, n_nodes), 15.0)   # Warm, no snow
        pet = np.full((n_days, n_nodes), 2.0)     # 2 mm/day PET

        outlet_flow, gru_runoff, _ = model.simulate(
            precip, temp, pet, return_gru_runoff=True
        )

        # After equilibration (skip first 200 days), check balance
        equilibration_period = 200

        # Total runoff volume from GRUs (mm -> m³)
        total_areas = float(np.sum(np.asarray(network.node_areas)))
        gru_runoff_m3 = np.sum(gru_runoff[equilibration_period:]) / 1000.0 * total_areas / n_nodes

        # Total outlet volume (m³/s -> m³)
        dt_seconds = 24 * 3600
        outlet_volume = np.sum(outlet_flow[equilibration_period:]) * dt_seconds

        # Should be approximately equal (allow for timing differences)
        ratio = outlet_volume / (gru_runoff_m3 + 1e-10)
        assert 0.5 < ratio < 2.0, f"Mass balance ratio {ratio} outside acceptable range"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestDistributedHBVJAX:
    """Tests for JAX-specific functionality."""

    def test_simulate_jax(self):
        """Test simulation with JAX backend."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=True)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=True)

        n_days = 100
        n_nodes = 3

        precip = jnp.array(np.random.exponential(5, (n_days, n_nodes)))
        temp = jnp.full((n_days, n_nodes), 10.0)
        pet = jnp.full((n_days, n_nodes), 3.0)

        outlet_flow, state = model.simulate(precip, temp, pet)

        assert len(outlet_flow) == n_days
        assert jnp.all(outlet_flow >= 0)

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        from jhbv.distributed import DistributedHBV
        from jhbv.model import DEFAULT_PARAMS
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=True)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=True)

        n_days = 50
        n_nodes = 3

        precip = jnp.array(np.random.exponential(5, (n_days, n_nodes)))
        temp = jnp.full((n_days, n_nodes), 10.0)
        pet = jnp.full((n_days, n_nodes), 3.0)

        # Synthetic observations
        obs = jnp.array(np.random.exponential(1, n_days))

        param_names = ['fc', 'beta', 'k1', 'k2']
        grad_fn = model.get_gradient_function(
            precip, temp, pet, obs,
            metric='nse',
            param_names=param_names
        )

        assert grad_fn is not None

        # Compute gradients
        x0 = jnp.array([DEFAULT_PARAMS[p] for p in param_names])
        grads = grad_fn(x0)

        assert len(grads) == len(param_names)
        assert jnp.all(jnp.isfinite(grads)), "Gradients should be finite"

    def test_value_and_grad(self):
        """Test combined value and gradient computation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.model import DEFAULT_PARAMS
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=True)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=True)

        n_days = 30
        n_nodes = 3

        precip = jnp.array(np.random.exponential(5, (n_days, n_nodes)))
        temp = jnp.full((n_days, n_nodes), 10.0)
        pet = jnp.full((n_days, n_nodes), 3.0)
        obs = jnp.array(np.random.exponential(1, n_days))

        param_names = ['fc', 'k1']
        val_grad_fn = model.get_value_and_grad_function(
            precip, temp, pet, obs,
            metric='nse',
            param_names=param_names
        )

        x0 = jnp.array([DEFAULT_PARAMS[p] for p in param_names])
        loss, grads = val_grad_fn(x0)

        assert jnp.isfinite(loss)
        assert len(grads) == 2
        assert jnp.all(jnp.isfinite(grads))


class TestCalibration:
    """Tests for calibration functionality."""

    def test_compute_loss_nse(self):
        """Test NSE loss computation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 100
        n_nodes = 3

        precip = np.random.exponential(5, (n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)

        # Run with default params to get "obs"
        outlet_flow, _ = model.simulate(precip, temp, pet)
        obs = outlet_flow + np.random.normal(0, 0.01, n_days)  # Add small noise

        # Loss should be close to 0 (high NSE)
        loss = model.compute_loss(
            {}, precip, temp, pet, obs, metric='nse', warmup_timesteps=0
        )

        # Negative NSE, so loss should be negative (close to -1 for perfect fit)
        assert loss < 0, "Loss should be negative (NSE > 0)"
        assert loss > -2, "Loss should be reasonable"

    def test_compute_loss_kge(self):
        """Test KGE loss computation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 100
        n_nodes = 3

        precip = np.random.exponential(5, (n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)

        outlet_flow, _ = model.simulate(precip, temp, pet)
        obs = outlet_flow + np.random.normal(0, 0.01, n_days)

        loss = model.compute_loss(
            {}, precip, temp, pet, obs, metric='kge', warmup_timesteps=0
        )

        assert loss < 0, "Loss should be negative (KGE > 0)"

    def test_compute_loss_ignores_nan_observations(self):
        """Loss computation should be robust to NaNs in observed series."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 80
        n_nodes = 3
        precip = np.random.exponential(5, (n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)

        outlet_flow, _ = model.simulate(precip, temp, pet)
        obs = outlet_flow.copy()
        obs[10:15] = np.nan

        loss = model.compute_loss(
            {}, precip, temp, pet, obs, metric='nse', warmup_timesteps=0
        )

        assert np.isfinite(loss)

    def test_get_loss_function(self):
        """Test loss function creation for scipy.optimize."""
        from jhbv.distributed import DistributedHBV
        from jhbv.model import DEFAULT_PARAMS
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 50
        n_nodes = 3

        precip = np.random.exponential(5, (n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)
        obs = np.random.exponential(1, n_days)

        param_names = ['fc', 'beta']
        loss_fn = model.get_loss_function(
            precip, temp, pet, obs,
            param_names=param_names
        )

        x0 = np.array([DEFAULT_PARAMS[p] for p in param_names])
        loss = loss_fn(x0)

        assert np.isfinite(loss)


class TestNetworkTopologies:
    """Tests for different network topologies."""

    @pytest.mark.parametrize("topology,n_nodes", [
        ('linear', 3),
        ('linear', 10),
        ('binary_tree', 7),
        ('binary_tree', 15),
        ('fishbone', 6),
        ('fishbone', 12),
    ])
    def test_topology_simulation(self, topology, n_nodes):
        """Test simulation works for various topologies."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=n_nodes, topology=topology, use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 30
        precip = np.random.exponential(5, (n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)

        outlet_flow, _ = model.simulate(precip, temp, pet)

        assert len(outlet_flow) == n_days
        assert np.all(outlet_flow >= 0)
        assert np.all(np.isfinite(outlet_flow))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_node_network(self):
        """Test model with single node (no routing)."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        # Single node network
        network = create_synthetic_network(n_nodes=1, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 30
        precip = np.random.exponential(5, (n_days, 1))
        temp = np.full((n_days, 1), 10.0)
        pet = np.full((n_days, 1), 3.0)

        outlet_flow, _ = model.simulate(precip, temp, pet)

        assert len(outlet_flow) == n_days
        assert np.all(outlet_flow >= 0)

    def test_zero_precipitation(self):
        """Test model with no precipitation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 100
        n_nodes = 3
        precip = np.zeros((n_days, n_nodes))
        temp = np.full((n_days, n_nodes), 10.0)
        pet = np.full((n_days, n_nodes), 3.0)

        outlet_flow, _ = model.simulate(precip, temp, pet)

        # Flow should decrease over time (recession)
        assert outlet_flow[-1] < outlet_flow[0] or outlet_flow[0] == 0

    def test_cold_conditions(self):
        """Test model with snow accumulation."""
        from jhbv.distributed import DistributedHBV
        from jhbv.network import create_synthetic_network

        network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=False)
        model = DistributedHBV(network, timestep_hours=24, warmup_days=0, use_jax=False)

        n_days = 100
        n_nodes = 3
        precip = np.full((n_days, n_nodes), 5.0)
        temp = np.full((n_days, n_nodes), -5.0)  # Cold, snow accumulates
        pet = np.full((n_days, n_nodes), 0.0)     # No ET in winter

        outlet_flow, _ = model.simulate(precip, temp, pet)

        # Flow should be minimal (precip stored as snow)
        assert np.mean(outlet_flow) < 1.0  # Low flow


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
def test_calibrate_distributed_hbv_adam_passes_n_iterations_to_two_phase():
    """Two-phase Adam calibration should honor the n_iterations argument."""
    from jhbv import distributed as distributed_mod
    from jhbv.network import create_synthetic_network

    network = create_synthetic_network(n_nodes=3, topology='linear', use_jax=True)
    model = distributed_mod.DistributedHBV(
        network, timestep_hours=24, warmup_days=0, use_jax=True
    )

    n_days = 20
    precip = np.ones((n_days, 3), dtype=np.float32)
    temp = np.ones((n_days, 3), dtype=np.float32) * 10.0
    pet = np.ones((n_days, 3), dtype=np.float32) * 3.0
    obs = np.ones(n_days, dtype=np.float32)

    with patch.object(distributed_mod, "_calibrate_two_phase", return_value="ok") as mock_two_phase:
        result = distributed_mod.calibrate_distributed_hbv_adam(
            model,
            precip,
            temp,
            pet,
            obs,
            n_iterations=123,
            two_phase=True,
            verbose=False,
        )

    assert result == "ok"
    assert mock_two_phase.call_args[0][14] == 123


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
