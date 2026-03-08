"""
Unit Tests for jFUSE JAX-Differentiable Transfer Functions.

Tests the JAX transfer function framework:
- JaxTransferFunctionConfig initialization and attribute normalization
- Coefficient names and bounds generation
- Default coefficient computation
- The pure JAX apply_transfer_functions function
- Module-level constants (PARAM_ATTR_MAP, DEFAULT_B_BOUNDS)

Note: Tests that require JAX/jFUSE are skipped if those packages are not installed.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

# Check for jFUSE/JAX availability
try:
    from jfuse import HAS_JAX as _HAS_JAX
    from jfuse import HAS_JFUSE as _HAS_JFUSE
    HAS_JFUSE = _HAS_JFUSE and _HAS_JAX
except ImportError:
    HAS_JFUSE = False

if HAS_JFUSE:
    try:
        import jax.numpy as jnp
        from jfuse.fuse.state import NUM_PARAMETERS, PARAM_BOUNDS, PARAM_NAMES
    except ImportError:
        HAS_JFUSE = False

# Skip entire module if jFUSE is not available
pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.skipif(not HAS_JFUSE, reason="jFUSE/JAX not installed"),
]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_jfuse_transfer_functions')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_gru_attributes(tmp_path):
    """Create sample GRU attributes CSV (minimal for testing)."""
    n_grus = 20
    np.random.seed(42)

    df = pd.DataFrame({
        'gru_id': range(n_grus),
        'elev_m': np.linspace(200, 2500, n_grus),
        'precip_mm_yr': np.linspace(500, 1500, n_grus),
        'temp_C': np.linspace(10, -5, n_grus),
        'aridity': np.linspace(1.5, 0.3, n_grus),
        'snow_frac': np.linspace(0.0, 0.8, n_grus),
        'is_coastal': np.zeros(n_grus, dtype=int),
    })

    path = tmp_path / "subcatchment_attributes.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tf_config(sample_gru_attributes, test_logger):
    """Create a JaxTransferFunctionConfig instance."""
    from jfuse.calibration.transfer_functions import (
        JaxTransferFunctionConfig,
    )

    return JaxTransferFunctionConfig(
        attributes_path=sample_gru_attributes,
        logger=test_logger,
    )


# =============================================================================
# Module Constants Tests
# =============================================================================

class TestModuleConstants:
    """Test module-level constants."""

    def test_param_attr_map_covers_default_params(self):
        """PARAM_ATTR_MAP should cover all default calibrated params."""
        from jfuse.calibration.transfer_functions import (
            DEFAULT_CALIBRATED_PARAMS,
            PARAM_ATTR_MAP,
        )

        for param in DEFAULT_CALIBRATED_PARAMS:
            assert param in PARAM_ATTR_MAP, f"{param} missing from PARAM_ATTR_MAP"

    def test_default_b_bounds_are_symmetric(self):
        """DEFAULT_B_BOUNDS should be symmetric around zero."""
        from jfuse.calibration.transfer_functions import (
            DEFAULT_B_BOUNDS,
        )

        assert DEFAULT_B_BOUNDS[0] == -DEFAULT_B_BOUNDS[1]

    def test_default_calibrated_params_length(self):
        """Should have 14 default calibrated parameters."""
        from jfuse.calibration.transfer_functions import (
            DEFAULT_CALIBRATED_PARAMS,
        )

        assert len(DEFAULT_CALIBRATED_PARAMS) == 14

    def test_smooth_frac_is_constant(self):
        """smooth_frac should be mapped to 'constant'."""
        from jfuse.calibration.transfer_functions import (
            PARAM_ATTR_MAP,
        )

        assert PARAM_ATTR_MAP['smooth_frac'] == 'constant'


# =============================================================================
# JaxTransferFunctionConfig Initialization Tests
# =============================================================================

class TestJaxTransferFunctionConfigInit:
    """Tests for JaxTransferFunctionConfig initialization."""

    def test_loads_gru_count(self, tf_config):
        """Should detect correct number of non-coastal GRUs."""
        assert tf_config.n_grus == 20

    def test_filters_coastal_grus(self, tmp_path, test_logger):
        """Should auto-filter coastal GRUs."""
        from jfuse.calibration.transfer_functions import (
            JaxTransferFunctionConfig,
        )

        df = pd.DataFrame({
            'gru_id': range(10),
            'elev_m': np.ones(10) * 500,
            'precip_mm_yr': np.ones(10) * 800,
            'temp_C': np.ones(10) * 5,
            'aridity': np.ones(10) * 0.8,
            'snow_frac': np.ones(10) * 0.3,
            'is_coastal': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        })
        path = tmp_path / "attrs_with_coastal.csv"
        df.to_csv(path, index=False)

        config = JaxTransferFunctionConfig(
            attributes_path=str(path), logger=test_logger
        )

        assert config.n_grus == 7  # 10 - 3 coastal

    def test_uses_non_coastal_indices(self, tmp_path, test_logger):
        """Should use explicit non_coastal_indices if provided."""
        from jfuse.calibration.transfer_functions import (
            JaxTransferFunctionConfig,
        )

        df = pd.DataFrame({
            'gru_id': range(10),
            'elev_m': np.arange(10) * 100,
            'precip_mm_yr': np.ones(10) * 800,
            'temp_C': np.ones(10) * 5,
            'aridity': np.ones(10) * 0.8,
            'snow_frac': np.ones(10) * 0.3,
            'is_coastal': np.zeros(10, dtype=int),
        })
        path = tmp_path / "attrs.csv"
        df.to_csv(path, index=False)

        indices = np.array([0, 2, 4, 6, 8])
        config = JaxTransferFunctionConfig(
            attributes_path=str(path),
            non_coastal_indices=indices,
            logger=test_logger,
        )

        assert config.n_grus == 5

    def test_custom_calibrated_params(self, sample_gru_attributes, test_logger):
        """Should accept custom list of calibrated parameters."""
        from jfuse.calibration.transfer_functions import (
            JaxTransferFunctionConfig,
        )

        custom_params = ['S1_max', 'S2_max', 'ku']
        config = JaxTransferFunctionConfig(
            attributes_path=sample_gru_attributes,
            calibrated_params=custom_params,
            logger=test_logger,
        )

        assert config.n_calibrated_params == 3
        assert config.calibrated_params == custom_params


# =============================================================================
# Properties Tests
# =============================================================================

class TestJaxTransferFunctionConfigProperties:
    """Tests for JaxTransferFunctionConfig properties."""

    def test_n_calibrated_params(self, tf_config):
        """Should match the number of calibrated parameters."""
        assert tf_config.n_calibrated_params == 14

    def test_n_coefficients_is_double_params(self, tf_config):
        """Should have 2 coefficients (a, b) per parameter."""
        assert tf_config.n_coefficients == 28  # 14 * 2

    def test_coefficient_names_format(self, tf_config):
        """Coefficient names should follow param_a/param_b pattern."""
        names = tf_config.coefficient_names

        assert len(names) == 28
        for i in range(0, len(names), 2):
            assert names[i].endswith('_a')
            assert names[i + 1].endswith('_b')

    def test_coefficient_bounds_length(self, tf_config):
        """Should have bounds for each coefficient."""
        bounds = tf_config.coefficient_bounds

        assert len(bounds) == 28

    def test_coefficient_bounds_are_tuples(self, tf_config):
        """Each bound should be (min, max) tuple."""
        for bound in tf_config.coefficient_bounds:
            assert len(bound) == 2
            assert bound[0] < bound[1]

    def test_param_indices_shape(self, tf_config):
        """param_indices should have one entry per calibrated param."""
        assert tf_config.param_indices.shape == (14,)

    def test_param_indices_valid(self, tf_config):
        """Each index should point to a valid PARAM_NAMES position."""
        for idx in tf_config.param_indices:
            assert 0 <= idx < NUM_PARAMETERS


# =============================================================================
# Default Coefficients Tests
# =============================================================================

class TestDefaultCoefficients:
    """Tests for get_default_coefficients method."""

    def test_default_shape(self, tf_config):
        """Default coefficients should have correct shape."""
        defaults = tf_config.get_default_coefficients()

        assert defaults.shape == (28,)

    def test_default_b_values_are_zero(self, tf_config):
        """Default b coefficients should all be 0 (uniform params)."""
        defaults = tf_config.get_default_coefficients()

        # b coefficients are at odd indices
        b_values = defaults[1::2]
        np.testing.assert_array_equal(b_values, 0.0)

    def test_default_a_values_are_finite(self, tf_config):
        """Default a coefficients should be finite numbers."""
        defaults = tf_config.get_default_coefficients()

        # a coefficients are at even indices
        a_values = defaults[0::2]
        assert np.all(np.isfinite(a_values))


# =============================================================================
# JAX Array Accessor Tests
# =============================================================================

class TestJAXArrayAccessors:
    """Tests for JAX array conversion methods."""

    def test_attr_matrix_shape(self, tf_config):
        """Attribute matrix should be (n_grus, n_calibrated_params)."""
        attr_jax = tf_config.get_attr_matrix_jax()

        assert attr_jax.shape == (20, 14)

    def test_attr_matrix_normalized(self, tf_config):
        """Attribute matrix values should be in [0, 1] range."""
        attr_jax = tf_config.get_attr_matrix_jax()

        # Non-constant attributes should be in [0, 1]
        assert float(jnp.min(attr_jax)) >= -0.01
        assert float(jnp.max(attr_jax)) <= 1.01

    def test_constant_attribute_is_zero(self, tf_config):
        """Columns for 'constant' attributes (smooth_frac) should be zero."""
        from jfuse.calibration.transfer_functions import (
            PARAM_ATTR_MAP,
        )

        attr_jax = tf_config.get_attr_matrix_jax()

        for i, pname in enumerate(tf_config.calibrated_params):
            if PARAM_ATTR_MAP.get(pname) == 'constant':
                np.testing.assert_array_equal(
                    np.array(attr_jax[:, i]), 0.0,
                    err_msg=f"Column for {pname} should be all zeros",
                )

    def test_default_full_params_shape(self, tf_config):
        """Full default params should have shape (30,)."""
        defaults = tf_config.get_default_full_params_jax()

        assert defaults.shape == (NUM_PARAMETERS,)

    def test_bounds_shapes(self, tf_config):
        """Lower and upper bounds should have shape (30,)."""
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        assert lower.shape == (NUM_PARAMETERS,)
        assert upper.shape == (NUM_PARAMETERS,)

    def test_lower_less_than_upper(self, tf_config):
        """Lower bounds should be less than upper bounds."""
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        assert jnp.all(lower < upper)


# =============================================================================
# apply_transfer_functions Tests
# =============================================================================

class TestApplyTransferFunctions:
    """Tests for the pure JAX apply_transfer_functions function."""

    def test_output_shape(self, tf_config):
        """Should return (n_grus, NUM_PARAMETERS) array."""
        from jfuse.calibration.transfer_functions import (
            apply_transfer_functions,
        )

        coeffs = jnp.array(tf_config.get_default_coefficients())
        attr_matrix = tf_config.get_attr_matrix_jax()
        defaults = tf_config.get_default_full_params_jax()
        indices = tf_config.get_param_indices_jax()
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        result = apply_transfer_functions(
            coeffs, attr_matrix, defaults, indices, lower, upper,
            n_grus=tf_config.n_grus,
        )

        assert result.shape == (20, NUM_PARAMETERS)

    def test_default_coeffs_give_uniform_params(self, tf_config):
        """With b=0, all GRUs should get the same parameter values."""
        from jfuse.calibration.transfer_functions import (
            apply_transfer_functions,
        )

        coeffs = jnp.array(tf_config.get_default_coefficients())
        attr_matrix = tf_config.get_attr_matrix_jax()
        defaults = tf_config.get_default_full_params_jax()
        indices = tf_config.get_param_indices_jax()
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        result = apply_transfer_functions(
            coeffs, attr_matrix, defaults, indices, lower, upper,
            n_grus=tf_config.n_grus,
        )

        # All rows should be identical (b=0 means no spatial variation)
        for col in range(NUM_PARAMETERS):
            col_std = float(jnp.std(result[:, col]))
            assert col_std < 1e-5, f"Column {col} has std {col_std} > 0"

    def test_results_within_bounds(self, tf_config):
        """All output values should be within parameter bounds."""
        from jfuse.calibration.transfer_functions import (
            apply_transfer_functions,
        )

        # Use non-zero b values to create variation
        coeffs_np = tf_config.get_default_coefficients()
        coeffs_np[1::2] = 0.5  # Set all b = 0.5
        coeffs = jnp.array(coeffs_np)

        attr_matrix = tf_config.get_attr_matrix_jax()
        defaults = tf_config.get_default_full_params_jax()
        indices = tf_config.get_param_indices_jax()
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        result = apply_transfer_functions(
            coeffs, attr_matrix, defaults, indices, lower, upper,
            n_grus=tf_config.n_grus,
        )

        # Check bounds
        assert jnp.all(result >= lower[None, :] - 1e-6)
        assert jnp.all(result <= upper[None, :] + 1e-6)

    def test_nonzero_b_creates_spatial_variation(self, tf_config):
        """Non-zero b should create spatial variation in calibrated params."""
        from jfuse.calibration.transfer_functions import (
            PARAM_ATTR_MAP,
            apply_transfer_functions,
        )

        coeffs_np = tf_config.get_default_coefficients()
        # Set large b for non-constant params
        for i, pname in enumerate(tf_config.calibrated_params):
            if PARAM_ATTR_MAP.get(pname) != 'constant':
                coeffs_np[2 * i + 1] = 2.0  # b = 2.0
        coeffs = jnp.array(coeffs_np)

        attr_matrix = tf_config.get_attr_matrix_jax()
        defaults = tf_config.get_default_full_params_jax()
        indices = tf_config.get_param_indices_jax()
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        result = apply_transfer_functions(
            coeffs, attr_matrix, defaults, indices, lower, upper,
            n_grus=tf_config.n_grus,
        )

        # At least some calibrated columns should have variation
        has_variation = False
        for i, pname in enumerate(tf_config.calibrated_params):
            if PARAM_ATTR_MAP.get(pname) != 'constant':
                idx = int(indices[i])
                col_std = float(jnp.std(result[:, idx]))
                if col_std > 0.01:
                    has_variation = True
                    break

        assert has_variation, "Expected spatial variation for non-constant params"

    def test_non_calibrated_params_unchanged(self, tf_config):
        """Non-calibrated params should remain at default values."""
        from jfuse.calibration.transfer_functions import (
            apply_transfer_functions,
        )

        coeffs = jnp.array(tf_config.get_default_coefficients())
        attr_matrix = tf_config.get_attr_matrix_jax()
        defaults = tf_config.get_default_full_params_jax()
        indices = tf_config.get_param_indices_jax()
        lower = tf_config.get_lower_bounds_jax()
        upper = tf_config.get_upper_bounds_jax()

        result = apply_transfer_functions(
            coeffs, attr_matrix, defaults, indices, lower, upper,
            n_grus=tf_config.n_grus,
        )

        # Find non-calibrated parameter indices
        calibrated_set = set(int(idx) for idx in indices)
        for p_idx in range(NUM_PARAMETERS):
            if p_idx not in calibrated_set:
                expected = float(defaults[p_idx])
                actual = float(result[0, p_idx])
                # Should match after clipping
                clipped_expected = np.clip(expected, float(lower[p_idx]), float(upper[p_idx]))
                assert actual == pytest.approx(clipped_expected, abs=1e-5), \
                    f"Param index {p_idx}: expected {clipped_expected}, got {actual}"
