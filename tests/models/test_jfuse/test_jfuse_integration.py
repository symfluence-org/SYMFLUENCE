# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Unit tests for jFUSE model integration.

Tests cover:
- Configuration validation
- Parameter manager bounds and transformations
- Worker initialization and gradient computation
- Lumped and distributed mode simulation
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from jfuse.runner import HAS_JAX as _HAS_JAX
    from jfuse.runner import HAS_JFUSE as _HAS_JFUSE
    HAS_JFUSE = _HAS_JFUSE and _HAS_JAX
except ImportError:
    HAS_JFUSE = False

from symfluence.models.spatial_modes import SpatialMode

if HAS_JFUSE:
    import jax
    import jax.numpy as jnp
    import jfuse
    from jfuse import PARAM_BOUNDS, CoupledModel, Parameters, create_fuse_model

    def kge_loss(sim, obs):
        """Compute KGE loss (1 - KGE) for use in optimization."""
        sim_mean = jnp.mean(sim)
        obs_mean = jnp.mean(obs)
        sim_std = jnp.std(sim)
        obs_std = jnp.std(obs)
        cov = jnp.mean((sim - sim_mean) * (obs - obs_mean))
        r = cov / (sim_std * obs_std + 1e-10)
        beta = sim_mean / (obs_mean + 1e-10)
        gamma = sim_std / (obs_std + 1e-10)
        kge = 1 - jnp.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
        return 1 - kge
else:
    jax = None
    jnp = None

# Import jfuse calibration components (these don't require JAX)
from jfuse.calibration.parameter_manager import (
    DEFAULT_PARAMS,
    FALLBACK_PARAM_BOUNDS,
    JFUSEParameterManager,
)
from jfuse.calibration.parameter_manager import (
    PARAM_BOUNDS as SYMFLUENCE_PARAM_BOUNDS,
)
from jfuse.calibration.worker import JFUSEWorker
from jfuse.sfconfig import JFUSEConfig, JFUSEConfigAdapter

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'ROOT_PATH': '/tmp/test',
        'JFUSE_MODEL_CONFIG_NAME': 'prms',
        'JFUSE_ENABLE_SNOW': True,
        'JFUSE_WARMUP_DAYS': 365,
        'JFUSE_SPATIAL_MODE': 'lumped',
        'JFUSE_PARAMS_TO_CALIBRATE': 'S1_max,S2_max,ku,ki,ks',
    }


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_jfuse')


@pytest.fixture
def sample_forcing():
    """Create sample forcing data."""
    n_days = 730  # 2 years
    np.random.seed(42)

    return {
        'precip': np.random.exponential(2.0, n_days).astype(np.float32),
        'temp': np.sin(np.linspace(0, 4*np.pi, n_days)) * 15 + 5,  # -10 to 20 C
        'pet': np.maximum(0, np.sin(np.linspace(0, 4*np.pi, n_days)) * 3 + 2),
    }


@pytest.fixture
def sample_observations(sample_forcing):
    """Create sample observations (synthetic streamflow)."""
    n_days = len(sample_forcing['precip'])
    np.random.seed(42)
    base = 10.0
    precip = sample_forcing['precip']
    response = np.convolve(precip, np.exp(-np.arange(10)/3), mode='same')
    return (base + response * 2 + np.random.normal(0, 1, n_days)).astype(np.float32)


# =============================================================================
# Test: Configuration
# =============================================================================

class TestConfig:
    """Test jFUSE configuration."""

    def test_config_defaults(self):
        """Test JFUSEConfig has sensible defaults."""
        config = JFUSEConfig()

        assert config.model_config_name == 'prms_gradient'
        assert config.enable_snow is True
        assert config.spatial_mode == 'auto'
        assert config.warmup_days == 365
        assert config.n_hrus == 1

    def test_config_distributed_mode(self):
        """Test distributed mode configuration."""
        config = JFUSEConfig(
            spatial_mode='distributed',
            n_hrus=10,
            network_file='network.nc'
        )

        assert config.spatial_mode == 'distributed'
        assert config.n_hrus == 10
        assert config.network_file == 'network.nc'

    def test_config_adapter(self, sample_config):
        """Test config adapter converts dict to JFUSEConfig."""
        adapter = JFUSEConfigAdapter(model_name='JFUSE')
        jfuse_config = adapter.from_dict(sample_config)

        assert jfuse_config.model_config_name == 'prms'
        assert jfuse_config.enable_snow is True
        assert jfuse_config.warmup_days == 365


# =============================================================================
# Test: Parameter Manager
# =============================================================================

class TestParameterManager:
    """Test jFUSE parameter manager."""

    def test_parameter_bounds_loaded(self):
        """Test that parameter bounds are loaded correctly."""
        assert len(SYMFLUENCE_PARAM_BOUNDS) > 0

        assert 'S1_max' in SYMFLUENCE_PARAM_BOUNDS
        assert 'S2_max' in SYMFLUENCE_PARAM_BOUNDS
        assert 'ku' in SYMFLUENCE_PARAM_BOUNDS

    def test_parameter_bounds_valid(self):
        """Test that parameter bounds are valid (min < max)."""
        for name, bounds in SYMFLUENCE_PARAM_BOUNDS.items():
            assert len(bounds) == 2, f"Bounds for {name} should be tuple of 2"
            assert bounds[0] < bounds[1], f"min >= max for {name}"

    def test_default_params_within_bounds(self):
        """Test that default parameters are within bounds."""
        for name, default in DEFAULT_PARAMS.items():
            if name in SYMFLUENCE_PARAM_BOUNDS:
                low, high = SYMFLUENCE_PARAM_BOUNDS[name]
                assert low <= default <= high, \
                    f"Default {name}={default} outside bounds [{low}, {high}]"

    def test_parameter_manager_init(self, sample_config, logger):
        """Test parameter manager initialization."""
        pm = JFUSEParameterManager(
            config=sample_config,
            logger=logger,
            jfuse_settings_dir=Path('/tmp/test')
        )

        assert pm.domain_name == 'test_domain'
        assert len(pm.calibration_params) == 5  # S1_max,S2_max,ku,ki,ks

    def test_get_bounds_array(self, sample_config, logger):
        """Test getting bounds as arrays."""
        pm = JFUSEParameterManager(
            config=sample_config,
            logger=logger,
            jfuse_settings_dir=Path('/tmp/test')
        )

        lower, upper = pm.get_bounds_array()

        assert len(lower) == len(pm.calibration_params)
        assert len(upper) == len(pm.calibration_params)
        assert all(lower < upper)

    def test_normalize_denormalize(self, sample_config, logger):
        """Test parameter normalization and denormalization."""
        pm = JFUSEParameterManager(
            config=sample_config,
            logger=logger,
            jfuse_settings_dir=Path('/tmp/test')
        )

        params = pm.get_initial_parameters()

        normalized = pm.normalize(params)
        assert all(0 <= n <= 1 for n in normalized), "Normalized values should be in [0, 1]"

        denormalized = pm.denormalize(normalized)

        for name in pm.calibration_params:
            assert abs(params[name] - denormalized[name]) < 1e-6, \
                f"Round-trip failed for {name}"

    def test_validate_params(self, sample_config, logger):
        """Test parameter validation."""
        pm = JFUSEParameterManager(
            config=sample_config,
            logger=logger,
            jfuse_settings_dir=Path('/tmp/test')
        )

        valid_params = pm.get_initial_parameters()
        is_valid, violations = pm.validate(valid_params)
        assert is_valid
        assert len(violations) == 0

        invalid_params = valid_params.copy()
        invalid_params['S1_max'] = -100
        is_valid, violations = pm.validate(invalid_params)
        assert not is_valid
        assert len(violations) > 0


# =============================================================================
# Test: Worker - Lumped Mode
# =============================================================================

@pytest.mark.skipif(not HAS_JFUSE, reason="jFUSE/JAX not installed")
class TestWorkerLumped:
    """Test jFUSE worker in lumped mode."""

    def test_worker_init(self, sample_config, logger):
        """Test worker initialization."""
        worker = JFUSEWorker(config=sample_config, logger=logger)

        assert worker.model_config_name == 'prms'
        assert worker.warmup_days == 365
        assert worker.spatial_mode == SpatialMode.LUMPED

    def test_supports_native_gradients(self, sample_config, logger):
        """Test native gradient support detection."""
        worker = JFUSEWorker(config=sample_config, logger=logger)

        assert worker.supports_native_gradients() is True

    def test_dict_to_params(self, sample_config, logger, sample_forcing):
        """Test parameter dict to Parameters object conversion."""
        worker = JFUSEWorker(config=sample_config, logger=logger)

        worker._model = create_fuse_model('prms', n_hrus=1)
        worker._default_params = Parameters.default(n_hrus=1)
        worker._forcing = {k: jnp.array(v) for k, v in sample_forcing.items()}
        worker._forcing_tuple = (
            worker._forcing['precip'],
            worker._forcing['pet'],
            worker._forcing['temp']
        )
        worker._initialized = True

        param_dict = {'S1_max': 150.0, 'S2_max': 600.0}
        params_obj = worker._dict_to_params(param_dict)

        assert float(params_obj.S1_max) == 150.0
        assert float(params_obj.S2_max) == 600.0

    def test_lumped_simulation(self, sample_config, logger, sample_forcing):
        """Test lumped mode simulation."""
        worker = JFUSEWorker(config=sample_config, logger=logger)

        worker._model = create_fuse_model('prms', n_hrus=1)
        worker._default_params = Parameters.default(n_hrus=1)
        worker._forcing = {k: jnp.array(v) for k, v in sample_forcing.items()}
        worker._forcing_tuple = (
            worker._forcing['precip'],
            worker._forcing['pet'],
            worker._forcing['temp']
        )
        worker._initialized = True
        worker.warmup_days = 30

        params_obj = worker._default_params
        runoff, _ = worker._model.simulate(worker._forcing_tuple, params_obj)

        assert runoff.shape[0] == len(sample_forcing['precip'])
        assert not np.any(np.isnan(runoff))
        assert np.all(runoff >= 0)

    def test_gradient_computation(self, sample_config, logger, sample_forcing, sample_observations):
        """Test gradient computation via JAX autodiff."""
        worker = JFUSEWorker(config=sample_config, logger=logger)

        worker._model = create_fuse_model('prms', n_hrus=1)
        worker._default_params = Parameters.default(n_hrus=1)
        worker._forcing = {k: jnp.array(v) for k, v in sample_forcing.items()}
        worker._forcing_tuple = (
            worker._forcing['precip'],
            worker._forcing['pet'],
            worker._forcing['temp']
        )
        worker._observations = jnp.array(sample_observations[30:])
        worker._initialized = True
        worker.warmup_days = 30

        params = {'S1_max': 200.0, 'S2_max': 800.0, 'ku': 0.1}
        gradients = worker.compute_gradient(params, metric='kge')

        assert gradients is not None
        assert len(gradients) == len(params)
        assert all(isinstance(g, float) for g in gradients.values())
        assert all(not np.isnan(g) for g in gradients.values())

    def test_evaluate_with_gradient(self, sample_config, logger, sample_forcing, sample_observations):
        """Test combined loss and gradient computation."""
        worker = JFUSEWorker(config=sample_config, logger=logger)

        worker._model = create_fuse_model('prms', n_hrus=1)
        worker._default_params = Parameters.default(n_hrus=1)
        worker._forcing = {k: jnp.array(v) for k, v in sample_forcing.items()}
        worker._forcing_tuple = (
            worker._forcing['precip'],
            worker._forcing['pet'],
            worker._forcing['temp']
        )
        worker._observations = jnp.array(sample_observations[30:])
        worker._initialized = True
        worker.warmup_days = 30

        params = {'S1_max': 200.0, 'S2_max': 800.0}
        loss, gradients = worker.evaluate_with_gradient(params, metric='kge')

        assert isinstance(loss, float)
        assert 0 <= loss <= 2
        assert gradients is not None
        assert len(gradients) == len(params)


# =============================================================================
# Test: Worker - Distributed Mode
# =============================================================================

@pytest.mark.skipif(not HAS_JFUSE, reason="jFUSE/JAX not installed")
class TestWorkerDistributed:
    """Test jFUSE worker in distributed mode."""

    def test_distributed_config(self):
        """Test distributed mode configuration."""
        config = {
            'JFUSE_SPATIAL_MODE': 'distributed',
            'JFUSE_N_HRUS': 3,
        }
        worker = JFUSEWorker(config=config, logger=logging.getLogger('test'))

        assert worker.spatial_mode == SpatialMode.DISTRIBUTED

    def test_multi_hru_parameters(self):
        """Test multi-HRU parameter handling."""
        n_hrus = 5
        params = Parameters.default(n_hrus=n_hrus)

        assert params.S1_max.shape == (n_hrus,)
        assert params.S2_max.shape == (n_hrus,)
        assert params.ku.shape == (n_hrus,)

    def test_multi_hru_simulation(self):
        """Test multi-HRU simulation without routing."""
        n_hrus = 3
        n_days = 100

        model = create_fuse_model('prms', n_hrus=n_hrus)
        params = Parameters.default(n_hrus=n_hrus)

        np.random.seed(42)
        precip = np.random.exponential(2.0, (n_days, n_hrus)).astype(np.float32)
        pet = np.ones((n_days, n_hrus), dtype=np.float32) * 2.0
        temp = np.ones((n_days, n_hrus), dtype=np.float32) * 10.0

        forcing_tuple = (jnp.array(precip), jnp.array(pet), jnp.array(temp))

        runoff, _ = model.simulate(forcing_tuple, params)

        assert runoff.shape == (n_days, n_hrus)
        assert not np.any(np.isnan(runoff))

    def test_coupled_model_creation(self):
        """Test CoupledModel can be created."""
        from jfuse import CoupledModel, create_network_from_topology

        reach_ids = [1, 2, 3]
        downstream_ids = [3, 3, -1]
        lengths = [1000.0, 1500.0, 2000.0]
        slopes = [0.01, 0.01, 0.005]
        areas = [1e6, 1.5e6, 0.5e6]

        network = create_network_from_topology(
            reach_ids=reach_ids,
            downstream_ids=downstream_ids,
            lengths=lengths,
            slopes=slopes,
            areas=areas
        )

        coupled = CoupledModel(
            network=network.to_arrays(),
            hru_areas=jnp.array(areas),
            n_hrus=3
        )

        assert coupled.fuse_model.n_hrus == 3

    def test_coupled_simulation(self):
        """Test coupled FUSE + routing simulation."""
        from jfuse import CoupledModel, create_network_from_topology

        n_hrus = 3
        n_days = 100

        reach_ids = [1, 2, 3]
        downstream_ids = [3, 3, -1]
        lengths = [1000.0, 1500.0, 2000.0]
        slopes = [0.01, 0.01, 0.005]
        areas = [1e6, 1.5e6, 0.5e6]

        network = create_network_from_topology(
            reach_ids=reach_ids,
            downstream_ids=downstream_ids,
            lengths=lengths,
            slopes=slopes,
            areas=areas
        )

        coupled = CoupledModel(
            network=network.to_arrays(),
            hru_areas=jnp.array(areas),
            n_hrus=n_hrus
        )

        np.random.seed(42)
        precip = np.random.exponential(2.0, (n_days, n_hrus)).astype(np.float32)
        pet = np.ones((n_days, n_hrus), dtype=np.float32) * 2.0
        temp = np.ones((n_days, n_hrus), dtype=np.float32) * 10.0

        forcing_tuple = (jnp.array(precip), jnp.array(pet), jnp.array(temp))

        params = coupled.default_params()
        outlet_Q, runoff = coupled.simulate(forcing_tuple, params)

        assert outlet_Q.shape == (n_days,)
        assert runoff.shape == (n_days, n_hrus)

        assert np.all(outlet_Q >= 0)
        assert np.all(runoff >= 0)


# =============================================================================
# Test: Gradient Correctness
# =============================================================================

@pytest.mark.skipif(not HAS_JFUSE, reason="jFUSE/JAX not installed")
class TestGradientCorrectness:
    """Test that gradients are computed correctly."""

    def test_gradient_finite_difference_check(self):
        """Verify JAX gradients against finite differences."""
        import jax

        n_days = 200
        warmup = 30

        model = create_fuse_model('prms', n_hrus=1)
        default_params = Parameters.default(n_hrus=1)

        np.random.seed(42)
        precip = jnp.array(np.random.exponential(2.0, n_days).astype(np.float32))
        pet = jnp.array(np.ones(n_days, dtype=np.float32) * 2.0)
        temp = jnp.array(np.ones(n_days, dtype=np.float32) * 10.0)
        forcing = (precip, pet, temp)

        obs = jnp.array(np.random.uniform(5, 20, n_days - warmup).astype(np.float32))

        param_names = ['S1_max', 'S2_max']
        param_values = jnp.array([200.0, 800.0])

        import equinox as eqx

        def loss_fn(param_array):
            params = default_params
            for i, name in enumerate(param_names):
                params = eqx.tree_at(lambda p, _name=name: getattr(p, _name), params, param_array[i])
            runoff, _ = model.simulate(forcing, params)
            sim = runoff[warmup:]
            return kge_loss(sim[:len(obs)], obs)

        jax_grad = jax.grad(loss_fn)(param_values)

        eps = 1e-4
        fd_grad = []
        base_loss = float(loss_fn(param_values))
        for i in range(len(param_values)):
            perturbed = param_values.at[i].set(param_values[i] + eps)
            perturbed_loss = float(loss_fn(perturbed))
            fd_grad.append((perturbed_loss - base_loss) / eps)
        fd_grad = np.array(fd_grad)

        for i, name in enumerate(param_names):
            jax_g = float(jax_grad[i])
            fd_g = fd_grad[i]
            if abs(fd_g) > 1e-6:
                rel_err = abs(jax_g - fd_g) / abs(fd_g)
                assert rel_err < 0.1, \
                    f"Gradient mismatch for {name}: JAX={jax_g:.6f}, FD={fd_g:.6f}, rel_err={rel_err:.2%}"


# =============================================================================
# Test: Integration
# =============================================================================

@pytest.mark.skipif(not HAS_JFUSE, reason="jFUSE/JAX not installed")
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_calibration_improves_kge(self, sample_forcing, sample_observations):
        """Test that gradient-based calibration improves KGE."""
        import equinox as eqx
        import jax

        n_days = len(sample_forcing['precip'])
        warmup = 30

        model = create_fuse_model('prms', n_hrus=1)
        default_params = Parameters.default(n_hrus=1)

        precip = jnp.array(sample_forcing['precip'])
        pet = jnp.array(sample_forcing['pet'])
        temp = jnp.array(sample_forcing['temp'])
        forcing = (precip, pet, temp)
        obs = jnp.array(sample_observations[warmup:])

        param_names = ['S1_max', 'S2_max', 'ku']

        param_array = jnp.array([500.0, 2000.0, 0.3])

        def loss_fn(arr):
            params = default_params
            for i, name in enumerate(param_names):
                params = eqx.tree_at(lambda p, _name=name: getattr(p, _name), params, arr[i])
            runoff, _ = model.simulate(forcing, params)
            return kge_loss(runoff[warmup:warmup+len(obs)], obs)

        initial_loss = float(loss_fn(param_array))

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        lr = 0.5

        for _ in range(20):
            loss, grad = loss_and_grad(param_array)
            param_array = param_array - lr * grad
            param_array = jnp.clip(param_array,
                                   jnp.array([50.0, 100.0, 0.001]),
                                   jnp.array([5000.0, 20000.0, 1.0]))

        final_loss = float(loss_fn(param_array))

        assert final_loss < initial_loss, \
            f"Calibration did not improve: initial={initial_loss:.4f}, final={final_loss:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
