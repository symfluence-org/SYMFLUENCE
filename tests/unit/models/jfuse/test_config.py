"""Tests for jFUSE model configuration."""

import pytest
from jfuse.sfconfig import JFUSEConfig, JFUSEConfigAdapter

# =============================================================================
# JFUSEConfig Pydantic model
# =============================================================================

class TestJFUSEConfig:
    """Tests for JFUSEConfig Pydantic schema."""

    def test_default_creation(self):
        config = JFUSEConfig()
        assert config.model_config_name == "prms_gradient"
        assert config.enable_snow is True

    def test_spatial_mode_literals(self):
        for mode in ("lumped", "distributed", "auto"):
            config = JFUSEConfig(spatial_mode=mode)
            assert config.spatial_mode == mode

    def test_invalid_spatial_mode_raises(self):
        with pytest.raises(Exception):
            JFUSEConfig(spatial_mode="invalid_mode")

    def test_model_config_name_literals(self):
        for name in ("prms", "prms_gradient", "max_gradient", "sacramento", "topmodel", "vic"):
            config = JFUSEConfig(model_config_name=name)
            assert config.model_config_name == name

    def test_invalid_model_config_name_raises(self):
        with pytest.raises(Exception):
            JFUSEConfig(model_config_name="nonexistent_model")

    def test_warmup_days_non_negative(self):
        config = JFUSEConfig(warmup_days=0)
        assert config.warmup_days == 0
        with pytest.raises(Exception):
            JFUSEConfig(warmup_days=-1)

    def test_timestep_days_bounds(self):
        config = JFUSEConfig(timestep_days=0.5)
        assert config.timestep_days == 0.5
        with pytest.raises(Exception):
            JFUSEConfig(timestep_days=0.001)  # below min
        with pytest.raises(Exception):
            JFUSEConfig(timestep_days=2.0)  # above max

    def test_n_hrus_positive(self):
        config = JFUSEConfig(n_hrus=5)
        assert config.n_hrus == 5
        with pytest.raises(Exception):
            JFUSEConfig(n_hrus=0)

    def test_calibration_metric_literals(self):
        for metric in ("KGE", "NSE"):
            config = JFUSEConfig(calibration_metric=metric)
            assert config.calibration_metric == metric

    def test_get_calibration_params(self):
        config = JFUSEConfig(params_to_calibrate="S1_max,S2_max,ku")
        params = config.get_calibration_params()
        assert params == ["S1_max", "S2_max", "ku"]

    def test_get_calibration_params_strips_whitespace(self):
        config = JFUSEConfig(params_to_calibrate=" S1_max , S2_max ")
        params = config.get_calibration_params()
        assert params == ["S1_max", "S2_max"]

    def test_get_routing_calibration_params_empty_default(self):
        config = JFUSEConfig()
        assert config.get_routing_calibration_params() == []

    def test_get_routing_calibration_params_with_values(self):
        config = JFUSEConfig(routing_params_to_calibrate="mannings_n,slope")
        params = config.get_routing_calibration_params()
        assert params == ["mannings_n", "slope"]

    def test_mannings_n_bounds(self):
        config = JFUSEConfig(default_mannings_n=0.05)
        assert config.default_mannings_n == 0.05
        with pytest.raises(Exception):
            JFUSEConfig(default_mannings_n=0.005)  # below 0.01
        with pytest.raises(Exception):
            JFUSEConfig(default_mannings_n=0.3)  # above 0.2

    def test_output_frequency_literals(self):
        for freq in ("daily", "timestep"):
            config = JFUSEConfig(output_frequency=freq)
            assert config.output_frequency == freq


# =============================================================================
# JFUSEConfigAdapter
# =============================================================================

class TestJFUSEConfigAdapter:
    """Tests for JFUSEConfigAdapter."""

    def test_model_name(self):
        assert JFUSEConfigAdapter.MODEL_NAME == "JFUSE"

    def test_config_prefix(self):
        assert JFUSEConfigAdapter.CONFIG_PREFIX == "JFUSE_"

    def test_get_config_schema(self):
        schema = JFUSEConfigAdapter.get_config_schema()
        assert schema is JFUSEConfig

    def test_get_defaults_returns_dict(self):
        defaults = JFUSEConfigAdapter.get_defaults()
        assert isinstance(defaults, dict)
        assert "JFUSE_MODEL_CONFIG_NAME" in defaults
        assert defaults["JFUSE_MODEL_CONFIG_NAME"] == "prms_gradient"

    def test_get_field_transformers(self):
        transformers = JFUSEConfigAdapter.get_field_transformers()
        assert "JFUSE_SPATIAL_MODE" in transformers
        field_name, field_type = transformers["JFUSE_SPATIAL_MODE"]
        assert field_name == "spatial_mode"
        assert field_type == str

    def test_from_dict(self):
        config_dict = {
            "JFUSE_MODEL_CONFIG_NAME": "sacramento",
            "JFUSE_SPATIAL_MODE": "distributed",
            "JFUSE_N_HRUS": "10",
        }
        config = JFUSEConfigAdapter.from_dict(config_dict)
        assert isinstance(config, JFUSEConfig)
        assert config.model_config_name == "sacramento"
        assert config.spatial_mode == "distributed"
        assert config.n_hrus == 10

    def test_to_dict(self):
        config = JFUSEConfig(model_config_name="vic", spatial_mode="lumped")
        result = JFUSEConfigAdapter.to_dict(config)
        assert result["JFUSE_MODEL_CONFIG_NAME"] == "vic"
        assert result["JFUSE_SPATIAL_MODE"] == "lumped"

    def test_from_dict_uses_defaults_for_missing_keys(self):
        config = JFUSEConfigAdapter.from_dict({})
        assert config.model_config_name == "prms_gradient"
        assert config.warmup_days == 365
