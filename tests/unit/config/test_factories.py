"""
Unit tests for configuration factory methods.

Tests the three factory methods for creating SymfluenceConfig instances:
- from_file(): Load from YAML with 5-layer hierarchy
- from_preset(): Load from named preset
- from_minimal(): Create minimal config with smart defaults
"""

import os
import tempfile
from pathlib import Path

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import ConfigurationError


class TestFromMinimalFactory:
    """Test from_minimal() factory method"""

    def test_minimal_config_creation(self):
        """Test creating minimal config with required overrides"""
        config = SymfluenceConfig.from_minimal(
            domain_name='test_basin',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # Verify basic fields
        assert config.domain.name == 'test_basin'
        assert config.model.hydrological_model == 'SUMMA'
        assert config.domain.time_start == '2020-01-01 00:00'
        assert config.domain.time_end == '2020-12-31 23:00'

        # Verify defaults were applied
        assert config.forcing.dataset == 'ERA5'  # Default forcing
        assert config.domain.experiment_id == 'run_1'  # Default experiment ID

    def test_minimal_config_with_additional_overrides(self):
        """Test minimal config with additional overrides"""
        config = SymfluenceConfig.from_minimal(
            domain_name='test_basin',
            model='FUSE',
            forcing_dataset='NLDAS',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00',
            POUR_POINT_COORDS='40.5/-111.0',
            NUM_PROCESSES=8
        )

        assert config.domain.name == 'test_basin'
        assert config.model.hydrological_model == 'FUSE'
        assert config.forcing.dataset == 'NLDAS'
        assert config.domain.pour_point_coords == '40.5/-111.0'
        assert config.system.num_processes == 8

    def test_minimal_config_missing_required_fields(self):
        """Test that minimal config raises error if required fields missing"""
        # Missing EXPERIMENT_TIME_START and EXPERIMENT_TIME_END
        with pytest.raises(ConfigurationError, match="Missing required fields"):
            SymfluenceConfig.from_minimal(
                domain_name='test_basin',
                model='SUMMA'
            )

    def test_minimal_config_model_specific_defaults(self):
        """Test that model-specific defaults are applied"""
        summa_config = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # SUMMA should have its defaults
        assert summa_config.model.summa is not None
        assert summa_config.model.summa.exe == 'summa_sundials.exe'

        fuse_config = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='FUSE',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # FUSE should have its defaults
        assert fuse_config.model.fuse is not None
        assert fuse_config.model.fuse.exe == 'fuse.exe'


class TestFromFileFactory:
    """Test from_file() factory method"""

    def create_temp_config(self, content: str) -> Path:
        """Helper to create temporary config file"""
        fd, path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return Path(path)

    def test_from_file_basic(self):
        """Test loading config from YAML file"""
        yaml_content = """
SYMFLUENCE_DATA_DIR: /data
SYMFLUENCE_CODE_DIR: /code
DOMAIN_NAME: test_basin
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
SUB_GRID_DISCRETIZATION: lumped
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
"""
        config_path = self.create_temp_config(yaml_content)

        try:
            config = SymfluenceConfig.from_file(config_path)

            assert config.domain.name == 'test_basin'
            assert config.domain.experiment_id == 'run_1'
            assert config.model.hydrological_model == 'SUMMA'
            assert config.forcing.dataset == 'ERA5'
        finally:
            config_path.unlink()

    def test_from_file_with_overrides(self):
        """Test loading config with CLI overrides"""
        yaml_content = """
DOMAIN_NAME: test_basin
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
SUB_GRID_DISCRETIZATION: lumped
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
NUM_PROCESSES: 1
"""
        config_path = self.create_temp_config(yaml_content)

        try:
            config = SymfluenceConfig.from_file(
                config_path,
                overrides={'NUM_PROCESSES': 8, 'DEBUG_MODE': True}
            )

            # Override should take precedence
            assert config.system.num_processes == 8
            assert config.system.debug_mode is True

            # File values should still be present
            assert config.domain.name == 'test_basin'
        finally:
            config_path.unlink()

    def test_from_file_missing_file(self):
        """Test that from_file raises error for missing file"""
        with pytest.raises(FileNotFoundError):
            SymfluenceConfig.from_file(Path('/nonexistent/config.yaml'))

    def test_from_file_environment_variables(self, monkeypatch):
        """Test that environment variables override file values"""
        yaml_content = """
DOMAIN_NAME: test_basin
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
SUB_GRID_DISCRETIZATION: lumped
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
NUM_PROCESSES: 1
"""
        config_path = self.create_temp_config(yaml_content)

        try:
            # Set environment variable
            monkeypatch.setenv('SYMFLUENCE_NUM_PROCESSES', '16')

            config = SymfluenceConfig.from_file(config_path, use_env=True)

            # Environment variable should override file value
            assert config.system.num_processes == 16
        finally:
            config_path.unlink()


class TestFromPresetFactory:
    """Test from_preset() factory method"""

    def test_from_preset_basic(self):
        """Test loading config from preset with required overrides"""
        # Presets don't include all required fields - user must provide them
        try:
            config = SymfluenceConfig.from_preset(
                'fuse-basic',
                DOMAIN_NAME='test_basin',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00'
            )

            # Should have FUSE model configured
            assert 'FUSE' in config.model.hydrological_model or config.model.hydrological_model == 'FUSE'

            # Should have all required fields
            assert config.domain.name == 'test_basin'
            assert config.domain.time_start == '2020-01-01 00:00'
            assert config.domain.time_end == '2020-12-31 23:00'

        except (ConfigurationError, ValueError) as e:
            if "not found" in str(e).lower():
                pytest.skip("Preset 'fuse-basic' not available")
            else:
                raise

    def test_from_preset_with_overrides(self):
        """Test loading preset with overrides"""
        try:
            config = SymfluenceConfig.from_preset(
                'fuse-basic',
                DOMAIN_NAME='custom_basin',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00',
                NUM_PROCESSES=16
            )

            # Overrides should take precedence
            assert config.domain.name == 'custom_basin'
            assert config.system.num_processes == 16

        except (ConfigurationError, ValueError) as e:
            if "not found" in str(e).lower():
                pytest.skip("Preset 'fuse-basic' not available")
            else:
                raise

    def test_from_preset_invalid_preset(self):
        """Test that invalid preset name raises error"""
        with pytest.raises(ConfigurationError, match="not found"):
            SymfluenceConfig.from_preset('nonexistent_preset')


class TestFactoryRoundTrip:
    """Test round-trip compatibility between factories and to_dict()"""

    def test_minimal_to_dict_round_trip(self):
        """Test that minimal config can be converted to dict and back"""
        config1 = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # Convert to flat dict
        flat = config1.to_dict(flatten=True)

        # Verify dict-like access works
        assert flat['DOMAIN_NAME'] == 'test'
        assert flat['HYDROLOGICAL_MODEL'] == 'SUMMA'

        # Verify bracket access works
        assert config1['DOMAIN_NAME'] == flat['DOMAIN_NAME']
        assert config1['EXPERIMENT_ID'] == flat['EXPERIMENT_ID']


class TestFlatNestedParity:
    """Verify flat and nested configs resolve to the same Pydantic defaults."""

    def create_temp_config(self, content: str) -> Path:
        """Helper to create temporary config file"""
        fd, path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return Path(path)

    def test_flat_nested_defaults_match(self):
        """Flat and nested configs with same required fields produce identical defaults.

        This is the core invariant: after removing ConfigDefaults pre-seeding,
        both paths should resolve optional fields to the same Pydantic defaults.
        """
        # Flat format config (uppercase keys)
        flat_yaml = """
SYMFLUENCE_DATA_DIR: /data
SYMFLUENCE_CODE_DIR: /code
DOMAIN_NAME: parity_test
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
SUB_GRID_DISCRETIZATION: grus
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
"""
        # Nested format config (hierarchical)
        nested_yaml = """
system:
  data_dir: /data
  code_dir: /code
domain:
  name: parity_test
  experiment_id: run_1
  time_start: "2020-01-01 00:00"
  time_end: "2020-12-31 23:00"
  definition_method: lumped
  discretization: grus
model:
  hydrological_model: SUMMA
forcing:
  dataset: ERA5
"""
        flat_path = self.create_temp_config(flat_yaml)
        nested_path = self.create_temp_config(nested_yaml)

        try:
            flat_config = SymfluenceConfig.from_file(flat_path)
            nested_config = SymfluenceConfig.from_file(nested_path)

            flat_dict = flat_config.to_dict(flatten=True)
            nested_dict = nested_config.to_dict(flatten=True)

            # All optional fields that exist in both should have the same values
            all_keys = set(flat_dict.keys()) | set(nested_dict.keys())
            mismatches = {}
            for key in all_keys:
                if key in flat_dict and key in nested_dict:
                    if flat_dict[key] != nested_dict[key]:
                        mismatches[key] = (flat_dict[key], nested_dict[key])

            assert not mismatches, (
                f"Flat/nested config defaults diverge on {len(mismatches)} keys:\n"
                + "\n".join(
                    f"  {k}: flat={v[0]!r}, nested={v[1]!r}"
                    for k, v in sorted(mismatches.items())
                )
            )
        finally:
            flat_path.unlink()
            nested_path.unlink()

    def test_lapse_rate_is_positive(self):
        """LAPSE_RATE default should be positive 0.0065, not the old negative value."""
        config = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )
        assert config.forcing.lapse_rate == 0.0065

    def test_flat_config_uses_pydantic_defaults(self):
        """Flat config loading should use Pydantic defaults, not old ConfigDefaults."""
        flat_yaml = """
SYMFLUENCE_DATA_DIR: /data
SYMFLUENCE_CODE_DIR: /code
DOMAIN_NAME: default_test
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
SUB_GRID_DISCRETIZATION: grus
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
"""
        config_path = self.create_temp_config(flat_yaml)

        try:
            config = SymfluenceConfig.from_file(config_path)

            # These were the 8 conflict values — all should match Pydantic now
            assert config.forcing.lapse_rate == 0.0065  # was -0.0065 in ConfigDefaults
            assert config.domain.delineation.stream_threshold == 5000.0  # was 1000
            assert config.domain.elevation_band_size == 200.0  # was 400
            assert config.domain.delineation.move_outlets_max_distance == 200.0  # was 1000
            assert config.system.random_seed is None  # was 42
            assert config.domain.delineation.drop_analysis_num_thresholds == 10  # was 20
        finally:
            config_path.unlink()


class TestDefaultPathResolution:
    """Test default path resolution edge cases."""

    def test_relative_code_dir_resolves_to_parent_data_dir(self, monkeypatch, tmp_path):
        """Relative code_dir values should not resolve to ./SYMFLUENCE_data."""
        from symfluence.core.config.factories import _resolve_default_data_dir

        monkeypatch.delenv('SYMFLUENCE_DATA_DIR', raising=False)
        monkeypatch.delenv('SYMFLUENCE_DATA', raising=False)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.chdir(repo_dir)

        resolved = Path(_resolve_default_data_dir("."))
        assert resolved == tmp_path / "SYMFLUENCE_data"
