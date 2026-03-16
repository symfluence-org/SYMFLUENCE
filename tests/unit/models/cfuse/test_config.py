"""Tests for cFUSE configuration."""

import pytest


class TestCFUSEConfig:
    """Tests for cFUSE configuration schema."""

    def test_config_can_be_imported(self):
        from cfuse.sfconfig import CFUSEConfig
        assert CFUSEConfig is not None

    def test_config_defaults(self):
        from cfuse.sfconfig import CFUSEConfig
        config = CFUSEConfig()
        assert config.model_structure == 'prms'
        assert config.warmup_days == 365
        assert config.timestep_days == 1.0

    def test_config_spatial_mode(self):
        from cfuse.sfconfig import CFUSEConfig
        config = CFUSEConfig()
        assert config.spatial_mode in ('lumped', 'distributed', 'auto')

    def test_config_gradient_settings(self):
        from cfuse.sfconfig import CFUSEConfig
        config = CFUSEConfig()
        assert hasattr(config, 'use_native_gradients')
        assert hasattr(config, 'device')


class TestCFUSEConfigAdapter:
    """Tests for cFUSE config adapter."""

    def test_adapter_can_be_imported(self):
        from cfuse.sfconfig import CFUSEConfigAdapter
        assert CFUSEConfigAdapter is not None

    def test_adapter_has_from_dict(self):
        from cfuse.sfconfig import CFUSEConfigAdapter
        adapter = CFUSEConfigAdapter('CFUSE')
        assert hasattr(adapter, 'from_dict') or hasattr(adapter, 'get_config_schema')


class TestCFUSEOptimizerRegistration:
    """Tests for cFUSE optimizer registration."""

    def test_optimizer_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'CFUSE' in OptimizerRegistry._optimizers

    def test_optimizer_is_correct_class(self):
        from cfuse.calibration.optimizer import CFUSEModelOptimizer

        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry._optimizers.get('CFUSE') == CFUSEModelOptimizer


class TestCFUSEModelStructures:
    """Tests for cFUSE model structure availability."""

    def test_available_structures(self):
        from cfuse.config import PRMS_CONFIG, SACRAMENTO_CONFIG, TOPMODEL_CONFIG
        assert PRMS_CONFIG is not None
        assert TOPMODEL_CONFIG is not None
        assert SACRAMENTO_CONFIG is not None
