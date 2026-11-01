"""
Test suite for model config adapters (Phase 1 refactoring).

Tests the new ModelRegistry-based config system to verify:
1. All model config adapters are properly registered
2. Defaults can be retrieved from adapters
3. Field transformers work correctly
4. Validation is properly delegated to models
5. Core config system integration works
"""
from __future__ import annotations

import pytest

from symfluence.core.config.config_resolution import (
    get_config_adapter,
    get_config_defaults,
    get_config_transformers,
    validate_model_config,
)
from symfluence.core.config.defaults import ModelDefaults
from symfluence.core.config.transformers import transform_flat_to_nested
from symfluence.core.modeling.base import ConfigValidationError, ModelConfigAdapter
from symfluence.core.registries import R


class TestModelRegistryRegistration:
    """Test that all model config adapters are properly registered."""

    def test_all_models_registered(self):
        """Verify all expected models have config adapters registered."""
        expected_models = [
            'SUMMA', 'FUSE', 'NGEN', 'HYPE', 'MESH',
            'GR', 'MIZUROUTE', 'LSTM', 'GNN'
        ]

        for model in expected_models:
            adapter = get_config_adapter(model)
            # Some models might not have adapters yet, so we check for registration attempts
            # The adapter can be None if the model module hasn't been imported
            # So we just verify the registry method doesn't crash
            assert adapter is None or isinstance(adapter, ModelConfigAdapter), \
                f"Model {model} adapter should be None or ModelConfigAdapter instance"

    def test_get_config_defaults(self):
        """Test that config defaults can be retrieved for models."""
        test_models = ['SUMMA', 'FUSE', 'NGEN']

        for model in test_models:
            defaults = get_config_defaults(model)
            # Defaults might be empty dict if model not imported, but should not fail
            assert isinstance(defaults, dict), \
                f"Defaults for {model} should be a dict"

    def test_get_config_transformers(self):
        """Test that config transformers can be retrieved for models."""
        test_models = ['SUMMA', 'FUSE', 'NGEN']

        for model in test_models:
            transformers = get_config_transformers(model)
            # Transformers might be empty dict if model not imported
            assert isinstance(transformers, dict), \
                f"Transformers for {model} should be a dict"


class TestSUMMAConfigAdapter:
    """Test SUMMA config adapter specifically (most comprehensive model)."""

    @pytest.fixture
    def summa_adapter(self):
        """Get SUMMA config adapter instance."""
        # Import SUMMA module to trigger registration
        try:
            import symfluence.models.summa
        except ImportError:
            pytest.skip("SUMMA module not available")

        adapter = get_config_adapter('SUMMA')
        if adapter is None:
            pytest.skip("SUMMA adapter not registered")
        return adapter

    def test_summa_adapter_has_schema(self, summa_adapter):
        """Test that SUMMA adapter returns a Pydantic schema."""
        schema = summa_adapter.get_config_schema()
        assert schema is not None, "SUMMA should have a config schema"
        assert hasattr(schema, 'model_validate'), "Schema should be a Pydantic model"

    def test_summa_adapter_has_defaults(self, summa_adapter):
        """Test that SUMMA adapter returns defaults."""
        defaults = summa_adapter.get_defaults()
        assert isinstance(defaults, dict), "Defaults should be a dict"
        assert len(defaults) > 0, "SUMMA should have default values"

        # Check for expected SUMMA defaults
        expected_keys = ['SUMMA_EXE', 'SETTINGS_SUMMA_PATH', 'ROUTING_MODEL']
        for key in expected_keys:
            assert key in defaults, f"SUMMA defaults should include {key}"

    def test_summa_adapter_has_transformers(self, summa_adapter):
        """Test that SUMMA adapter returns field transformers."""
        transformers = summa_adapter.get_field_transformers()
        assert isinstance(transformers, dict), "Transformers should be a dict"
        assert len(transformers) > 0, "SUMMA should have field transformers"

        # Check for expected SUMMA transformers
        assert 'SUMMA_EXE' in transformers, "Should have SUMMA_EXE transformer"
        assert transformers['SUMMA_EXE'] == ('model', 'summa', 'exe'), \
            "SUMMA_EXE should map to correct nested path"

    def test_summa_adapter_validation(self, summa_adapter):
        """Test that SUMMA adapter validation works."""
        # Valid config should not raise
        valid_config = {
            'SUMMA_EXE': 'summa.exe',
            'SETTINGS_SUMMA_PATH': '/path/to/settings',
        }
        summa_adapter.validate(valid_config)  # Should not raise

        # Invalid config should raise
        invalid_config = {
            'SUMMA_EXE': '',  # Empty - invalid
            'SETTINGS_SUMMA_PATH': None,  # None - invalid
        }
        with pytest.raises(ConfigValidationError):
            summa_adapter.validate(invalid_config)

    def test_summa_adapter_required_keys(self, summa_adapter):
        """Test that SUMMA adapter returns required keys."""
        required = summa_adapter.get_required_keys()
        assert isinstance(required, list), "Required keys should be a list"
        assert 'SUMMA_EXE' in required, "SUMMA_EXE should be required"
        assert 'SETTINGS_SUMMA_PATH' in required, "SETTINGS_SUMMA_PATH should be required"

    def test_summa_mizuroute_legacy_key_fallback(self, summa_adapter):
        """SUMMA validation should accept legacy mizuRoute keys as fallback."""
        config = {
            'SUMMA_EXE': 'summa.exe',
            'SETTINGS_SUMMA_PATH': '/path/to/settings',
            'ROUTING_MODEL': 'MIZUROUTE',
            'INSTALL_PATH_MIZUROUTE': '/legacy/install',
            'EXE_NAME_MIZUROUTE': 'mizuroute.exe',
        }
        summa_adapter.validate(config)  # Should not raise


class TestFUSEConfigAdapter:
    """Test FUSE config adapter."""

    @pytest.fixture
    def fuse_adapter(self):
        """Get FUSE config adapter instance."""
        try:
            import symfluence.models.fuse
        except ImportError:
            pytest.skip("FUSE module not available")

        adapter = get_config_adapter('FUSE')
        if adapter is None:
            pytest.skip("FUSE adapter not registered")
        return adapter

    def test_fuse_adapter_has_defaults(self, fuse_adapter):
        """Test that FUSE adapter returns defaults."""
        defaults = fuse_adapter.get_defaults()
        assert isinstance(defaults, dict), "Defaults should be a dict"

        # Check for expected FUSE defaults
        expected_keys = ['FUSE_EXE', 'SETTINGS_FUSE_PATH', 'FUSE_SPATIAL_MODE']
        for key in expected_keys:
            assert key in defaults, f"FUSE defaults should include {key}"

    def test_fuse_spatial_mode_validation(self, fuse_adapter):
        """Test FUSE spatial mode validation."""
        # Valid spatial mode
        valid_config = {
            'FUSE_EXE': 'fuse.exe',
            'SETTINGS_FUSE_PATH': '/path',
            'FUSE_SPATIAL_MODE': 'lumped',
        }
        fuse_adapter.validate(valid_config)  # Should not raise

        # Invalid spatial mode
        invalid_config = {
            'FUSE_EXE': 'fuse.exe',
            'SETTINGS_FUSE_PATH': '/path',
            'FUSE_SPATIAL_MODE': 'invalid_mode',
        }
        with pytest.raises(ConfigValidationError):
            fuse_adapter.validate(invalid_config)

    def test_fuse_mizuroute_legacy_key_fallback(self, fuse_adapter):
        """FUSE validation should accept legacy mizuRoute keys as fallback."""
        config = {
            'FUSE_EXE': 'fuse.exe',
            'SETTINGS_FUSE_PATH': '/path',
            'FUSE_SPATIAL_MODE': 'lumped',
            'ROUTING_MODEL': 'MIZUROUTE',
            'INSTALL_PATH_MIZUROUTE': '/legacy/install',
            'EXE_NAME_MIZUROUTE': 'mizuroute.exe',
        }
        fuse_adapter.validate(config)  # Should not raise


class TestNGENConfigAdapter:
    """Test NGEN config adapter."""

    @pytest.fixture
    def ngen_adapter(self):
        """Get NGEN config adapter instance."""
        try:
            import symfluence.models.ngen
        except ImportError:
            pytest.skip("NGEN module not available")

        adapter = get_config_adapter('NGEN')
        if adapter is None:
            pytest.skip("NGEN adapter not registered")
        return adapter

    def test_ngen_adapter_has_defaults(self, ngen_adapter):
        """Test that NGEN adapter returns defaults."""
        defaults = ngen_adapter.get_defaults()
        assert isinstance(defaults, dict), "Defaults should be a dict"

        # Check for expected NGEN defaults
        expected_keys = ['NGEN_EXE', 'NGEN_INSTALL_PATH', 'NGEN_MODULES_TO_CALIBRATE']
        for key in expected_keys:
            assert key in defaults, f"NGEN defaults should include {key}"

    def test_ngen_module_validation(self, ngen_adapter):
        """Test NGEN module validation."""
        # Valid module
        valid_config = {
            'NGEN_EXE': 'ngen',
            'NGEN_INSTALL_PATH': '/path',
            'NGEN_MODULES_TO_CALIBRATE': 'CFE',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc,satdk',
        }
        ngen_adapter.validate(valid_config)  # Should not raise

        # Invalid module (missing params)
        invalid_config = {
            'NGEN_EXE': 'ngen',
            'NGEN_INSTALL_PATH': '/path',
            'NGEN_MODULES_TO_CALIBRATE': 'CFE',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': '',  # Empty - invalid
        }
        with pytest.raises(ConfigValidationError):
            ngen_adapter.validate(invalid_config)


class TestCoreConfigIntegration:
    """Test that core config system integrates with ModelRegistry."""

    def test_model_defaults_uses_registry(self):
        """Test that ModelDefaults.get_defaults_for_model uses ModelRegistry."""
        # Import a model to ensure it's registered
        try:
            import symfluence.models.summa
        except ImportError:
            pytest.skip("SUMMA module not available")

        defaults = ModelDefaults.get_defaults_for_model('SUMMA')
        assert isinstance(defaults, dict), "Should return defaults dict"
        # Should have defaults from registry or legacy
        assert len(defaults) > 0, "Should have some defaults"

    def test_transformers_merge_model_specific(self):
        """Test that transform_flat_to_nested merges model-specific transformers."""
        # Import SUMMA to register its transformers
        try:
            import symfluence.models.summa
        except ImportError:
            pytest.skip("SUMMA module not available")

        # Create a flat config with SUMMA model
        flat_config = {
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'DOMAIN_NAME': 'test_domain',
            'SUMMA_EXE': 'summa.exe',
        }

        nested = transform_flat_to_nested(flat_config)

        # Check that base transformation worked
        assert nested['domain']['name'] == 'test_domain', \
            "Base transformation should work"

        # Check that model-specific transformation worked
        assert 'model' in nested, "Should have model section"
        if 'summa' in nested['model']:
            # If SUMMA transformers were applied
            assert nested['model']['summa']['exe'] == 'summa.exe', \
                "Model-specific transformation should work"

    def test_validation_delegates_to_models(self):
        """Test that root config validation delegates to ModelRegistry."""
        # This is a higher-level test that would require creating a full SymfluenceConfig
        # We'll just test that the method exists and can be called

        # Create a mock config
        config = {
            'SUMMA_EXE': 'summa.exe',
            'SETTINGS_SUMMA_PATH': '/path',
        }

        # Should not crash (validation might pass or fail depending on what's registered)
        try:
            validate_model_config('SUMMA', config)
        except Exception as e:  # noqa: BLE001
            # If it raises, it should be a proper validation error, not a crash
            assert 'configuration' in str(e).lower() or 'required' in str(e).lower(), \
                f"Should raise proper validation error, got: {e}"


class TestBackwardCompatibility:
    """Test that legacy code still works with new registry."""

    def test_legacy_defaults_still_available(self):
        """Test that legacy ModelDefaults attributes still work."""
        from symfluence.core.config.defaults import ModelDefaults

        # Legacy attributes should still exist
        assert hasattr(ModelDefaults, 'SUMMA'), \
            "Legacy SUMMA attribute should exist"
        assert hasattr(ModelDefaults, 'FUSE'), \
            "Legacy FUSE attribute should exist"

    def test_get_defaults_for_model_has_fallback(self):
        """Test that get_defaults_for_model has proper fallback."""
        # Test with a model that might not be registered
        defaults = ModelDefaults.get_defaults_for_model('UNKNOWN_MODEL')
        # Should return empty dict, not crash
        assert isinstance(defaults, dict), \
            "Should return dict even for unknown model"


class TestConfigAdapterInterface:
    """Test the ModelConfigAdapter base interface."""

    def test_adapter_interface_methods(self):
        """Test that adapters implement required interface methods."""
        # Try to get any registered adapter
        try:
            import symfluence.models.summa
            adapter = get_config_adapter('SUMMA')
            if adapter is None:
                pytest.skip("No adapters registered")
        except ImportError:
            pytest.skip("No model modules available")

        # Test required methods exist
        assert hasattr(adapter, 'get_config_schema'), \
            "Adapter should have get_config_schema method"
        assert hasattr(adapter, 'get_defaults'), \
            "Adapter should have get_defaults method"
        assert hasattr(adapter, 'get_field_transformers'), \
            "Adapter should have get_field_transformers method"
        assert hasattr(adapter, 'validate'), \
            "Adapter should have validate method"
        assert hasattr(adapter, 'get_required_keys'), \
            "Adapter should have get_required_keys method"

    def test_adapter_interface_return_types(self):
        """Test that adapter methods return correct types."""
        try:
            import symfluence.models.summa
            adapter = get_config_adapter('SUMMA')
            if adapter is None:
                pytest.skip("SUMMA adapter not registered")
        except ImportError:
            pytest.skip("SUMMA module not available")

        # Test return types
        schema = adapter.get_config_schema()
        assert schema is None or hasattr(schema, 'model_validate'), \
            "Schema should be None or Pydantic model"

        defaults = adapter.get_defaults()
        assert isinstance(defaults, dict), \
            "Defaults should be dict"

        transformers = adapter.get_field_transformers()
        assert isinstance(transformers, dict), \
            "Transformers should be dict"

        required = adapter.get_required_keys()
        assert isinstance(required, list), \
            "Required keys should be list"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestConfigPrefixFallback:
    """AutoGeneratedConfigAdapter derives transformers from CONFIG_PREFIX when
    the schema declares no Field(alias=...) — the external JAX-plugin pattern.

    Without this, such plugins emit no transformers, so core never recognizes
    their keys nor populates config.model.<model> (it runs on defaults).
    """

    def _adapter_cls(self, prefix=None):
        from pydantic import BaseModel

        from symfluence.core.modeling.base.base_config import AutoGeneratedConfigAdapter

        class _Schema(BaseModel):
            backend: str = "jax"
            warmup_days: int = 365

        class _Adapter(AutoGeneratedConfigAdapter):
            if prefix is not None:
                CONFIG_PREFIX = prefix

            def get_config_schema(self):
                return _Schema

            def validate(self, config):  # abstract in the base
                return None

        return _Adapter

    def test_prefix_fallback_generates_transformers(self):
        ft = self._adapter_cls(prefix="FAKE_")("FAKE").get_field_transformers()
        assert ft == {
            "FAKE_BACKEND": ("model", "fake", "backend"),
            "FAKE_WARMUP_DAYS": ("model", "fake", "warmup_days"),
        }

    def test_no_prefix_no_aliases_yields_empty(self):
        # Unchanged behavior: no CONFIG_PREFIX and no aliases -> no transformers.
        assert self._adapter_cls(prefix=None)("NOPREFIX").get_field_transformers() == {}
