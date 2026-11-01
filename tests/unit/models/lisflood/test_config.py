# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD model configuration."""
from __future__ import annotations

import pytest


class TestLisfloodConfigSchema:
    """Tests for LisfloodConfig Pydantic schema."""

    def test_lisflood_config_has_install_path_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        fields = LisfloodConfig.model_fields
        assert "install_path" in fields

    def test_lisflood_config_has_exe_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "exe" in LisfloodConfig.model_fields

    def test_lisflood_config_has_settings_path_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "settings_path" in LisfloodConfig.model_fields

    def test_lisflood_config_has_settings_file_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "settings_file" in LisfloodConfig.model_fields

    def test_lisflood_config_has_spatial_mode_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "spatial_mode" in LisfloodConfig.model_fields

    def test_lisflood_config_has_experiment_output_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "experiment_output" in LisfloodConfig.model_fields

    def test_lisflood_config_has_pcraster_path_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "pcraster_path" in LisfloodConfig.model_fields

    def test_lisflood_config_has_num_threads_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "num_threads" in LisfloodConfig.model_fields

    def test_lisflood_config_has_pet_method_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "pet_method" in LisfloodConfig.model_fields

    def test_lisflood_config_has_params_to_calibrate_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "params_to_calibrate" in LisfloodConfig.model_fields

    def test_lisflood_config_has_timeout_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "timeout" in LisfloodConfig.model_fields

    def test_lisflood_config_has_forest_fraction_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "forest_fraction" in LisfloodConfig.model_fields

    def test_lisflood_config_has_forcing_timestep_field(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert "forcing_timestep" in LisfloodConfig.model_fields

    def test_lisflood_config_has_all_expected_fields(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        expected = {
            "install_path",
            "exe",
            "settings_path",
            "settings_file",
            "spatial_mode",
            "experiment_output",
            "pcraster_path",
            "num_threads",
            "pet_method",
            "params_to_calibrate",
            "timeout",
            "forest_fraction",
            "forcing_timestep",
        }
        actual = set(LisfloodConfig.model_fields.keys())
        assert expected.issubset(actual), f"Missing fields: {expected - actual}"


class TestLisfloodConfigDefaults:
    """Tests for LisfloodConfig default values."""

    def test_default_install_path(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.install_path == "default"

    def test_default_exe(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.exe == "lisflood"

    def test_default_settings_file(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.settings_file == "settings.xml"

    def test_default_spatial_mode(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.spatial_mode == "lumped"

    def test_default_num_threads(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.num_threads == 1

    def test_default_pet_method(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.pet_method == "oudin"

    def test_default_timeout(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.timeout == 7200

    def test_default_forest_fraction(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.forest_fraction == pytest.approx(0.3)

    def test_default_forcing_timestep(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig()
        assert cfg.forcing_timestep == 86400


class TestLisfloodConfigAliases:
    """Tests for LisfloodConfig alias mapping."""

    def test_install_path_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_INSTALL_PATH": "/custom/path"})
        assert cfg.install_path == "/custom/path"

    def test_exe_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_EXE": "my_lisflood"})
        assert cfg.exe == "my_lisflood"

    def test_settings_file_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_SETTINGS_FILE": "custom.xml"})
        assert cfg.settings_file == "custom.xml"

    def test_spatial_mode_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_SPATIAL_MODE": "distributed"})
        assert cfg.spatial_mode == "distributed"

    def test_num_threads_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_NUM_THREADS": 4})
        assert cfg.num_threads == 4

    def test_timeout_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_TIMEOUT": 3600})
        assert cfg.timeout == 3600

    def test_forest_fraction_alias(self):
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        cfg = LisfloodConfig(**{"LISFLOOD_FOREST_FRACTION": 0.5})
        assert cfg.forest_fraction == pytest.approx(0.5)


class TestLisfloodConfigAdapter:
    """Tests for LisfloodConfigAdapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        assert LisfloodConfigAdapter is not None

    def test_adapter_creation(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        adapter = LisfloodConfigAdapter()
        assert adapter is not None

    def test_get_config_schema_returns_lisflood_config(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        adapter = LisfloodConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema is not None
        assert schema.__name__ == "LisfloodConfig"

    def test_validate_accepts_xml_extension(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        adapter = LisfloodConfigAdapter()
        # Should not raise
        adapter.validate({"settings_file": "settings.xml"})

    def test_validate_rejects_non_xml(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        adapter = LisfloodConfigAdapter()
        with pytest.raises(ValueError, match=r"\.xml"):
            adapter.validate({"settings_file": "settings.txt"})

    def test_validate_rejects_json_extension(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        adapter = LisfloodConfigAdapter()
        with pytest.raises(ValueError, match=r"\.xml"):
            adapter.validate({"settings_file": "settings.json"})

    def test_validate_default_settings_file_passes(self):
        from symfluence.models.lisflood.config import LisfloodConfigAdapter

        adapter = LisfloodConfigAdapter()
        # Default is 'settings.xml', should pass
        adapter.validate({})


class TestLisfloodInRegistry:
    """Tests for LISFLOOD presence in the unified config-schema registry."""

    def test_lisflood_registered_in_config_schemas(self):
        import symfluence.core.config.models.model_configs  # noqa: F401 - triggers registration
        from symfluence.core.registries import R

        assert "LISFLOOD" in R.config_schemas

    def test_lisflood_schema_is_lisflood_config(self):
        import symfluence.core.config.models.model_configs  # noqa: F401 - triggers registration
        from symfluence.core.registries import R
        from symfluence.models.lisflood.config_schema import LisfloodConfig

        assert R.config_schemas.get("LISFLOOD") is LisfloodConfig

    def test_model_config_resolves_lisflood(self):
        from symfluence.core.config.models.model_configs import ModelConfig

        cfg = ModelConfig(HYDROLOGICAL_MODEL="LISFLOOD")
        # Resolved via __getattr__ against model_specific (no static field anymore).
        assert type(cfg.lisflood).__name__ == "LisfloodConfig"
        assert "lisflood" in cfg.model_specific
