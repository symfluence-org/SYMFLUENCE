"""Tests for HYPE model configuration."""
from __future__ import annotations

import pytest

from symfluence.core.modeling.base import ConfigValidationError


class TestHYPEConfigAdapter:
    """Tests for HYPEConfigAdapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        assert HYPEConfigAdapter is not None

    def test_adapter_creation(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        assert adapter is not None

    def test_get_config_schema_returns_hype_config(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema is not None
        assert schema.__name__ == "HYPEConfig"

    def test_required_keys_includes_settings_path(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        keys = adapter.get_required_keys()
        assert "SETTINGS_HYPE_PATH" in keys

    def test_validate_passes_with_required_keys(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        config = {"SETTINGS_HYPE_PATH": "/path/to/hype"}
        adapter.validate(config)

    def test_validate_raises_on_missing_keys(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        config = {"SETTINGS_HYPE_PATH": None}
        with pytest.raises(ConfigValidationError, match="Missing required"):
            adapter.validate(config)


def test_timeshift_and_frac_threshold_are_schema_fields():
    """They were read from the typed config but had no field, so were ignored."""
    from symfluence.models.hype.config_schema import HYPEConfig

    default = HYPEConfig()
    assert default.timeshift == 0 and default.frac_threshold == 0.05
    cfg = HYPEConfig(HYPE_TIMESHIFT=-1, HYPE_FRAC_THRESHOLD=0.02)
    assert cfg.timeshift == -1 and cfg.frac_threshold == 0.02


def test_gnn_checkpoint_interval_is_a_schema_field():
    from symfluence.models.gnn.config_schema import GNNConfig

    assert GNNConfig().checkpoint_interval == 50
    assert GNNConfig(GNN_CHECKPOINT_INTERVAL=5).checkpoint_interval == 5
