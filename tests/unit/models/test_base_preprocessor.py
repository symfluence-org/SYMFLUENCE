"""
Unit tests for BaseModelPreProcessor.

Tests the base class functionality shared across all model preprocessors.
"""
from __future__ import annotations

import shutil
from unittest.mock import Mock

import pytest

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor


class ConcretePreProcessor(BaseModelPreProcessor):
    """Concrete implementation for testing."""

    MODEL_NAME = "TEST"

    def run_preprocessing(self):
        pass


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    return {
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'EXPERIMENT_ID': 'test_experiment',
        'EXPERIMENT_TIME_START': '2020-01-01',
        'EXPERIMENT_TIME_END': '2020-01-02',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'GR',  # must be a registered model (validated at load)
        'DOMAIN_NAME': 'test_domain',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'CATCHMENT_PATH': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'RIVER_NETWORK_SHP_PATH': 'default',
        'RIVER_NETWORK_SHP_NAME': 'default'
    }


@pytest.fixture
def logger():
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def preprocessor(config, logger, tmp_path):
    """Create a concrete preprocessor instance."""
    # Create necessary directory structure (using data/ prefix for forcing)
    project_dir = tmp_path / 'domain_test_domain'
    (project_dir / 'settings' / 'TEST').mkdir(parents=True, exist_ok=True)
    (project_dir / 'data' / 'forcing' / 'TEST_input').mkdir(parents=True, exist_ok=True)

    return ConcretePreProcessor(config, logger)


class TestBaseModelPreProcessorInitialization:
    """Test initialization and basic properties."""

    def test_initialization(self, preprocessor, config, tmp_path):
        """Test basic initialization."""
        # Check that all keys in the input config are present and correct in the result
        actual_config = preprocessor.config.to_dict(flatten=True)
        for key, value in config.items():
            assert actual_config[key] == value

        assert preprocessor.domain_name == 'test_domain'
        assert preprocessor.data_dir == tmp_path
        assert preprocessor.project_dir == tmp_path / 'domain_test_domain'
        assert preprocessor.model_name == 'TEST'

    def test_setup_dir_creation(self, preprocessor, tmp_path):
        """Test that setup_dir is set correctly."""
        expected_setup_dir = tmp_path / 'domain_test_domain' / 'settings' / 'TEST'
        assert preprocessor.setup_dir == expected_setup_dir

    def test_forcing_dir_creation(self, preprocessor, tmp_path):
        """Test that forcing_dir is set correctly."""
        expected_forcing_dir = tmp_path / 'domain_test_domain' / 'data' / 'forcing' / 'TEST_input'
        assert preprocessor.forcing_dir == expected_forcing_dir

    def test_forcing_basin_path(self, preprocessor, tmp_path):
        """Test that forcing_basin_path is set correctly."""
        expected_path = tmp_path / 'domain_test_domain' / 'data' / 'forcing' / 'basin_averaged_data'
        assert preprocessor.forcing_basin_path == expected_path


class TestPathResolution:
    """Test path resolution methods."""

    def test_get_default_path_with_default(self, preprocessor):
        """Test _get_default_path with 'default' value."""
        result = preprocessor._get_default_path('SOME_PATH', 'some/subpath')
        expected = preprocessor.project_dir / 'some/subpath'
        assert result == expected

    def test_get_default_path_with_none(self, preprocessor):
        """Test _get_default_path with None value."""
        result = preprocessor._get_default_path('NONEXISTENT_KEY', 'fallback/path')
        expected = preprocessor.project_dir / 'fallback/path'
        assert result == expected

    def test_get_default_path_with_custom(self, config, logger, tmp_path):
        """Test _get_default_path with custom path."""
        custom_path = tmp_path / 'custom' / 'path'
        config['CUSTOM_PATH'] = str(custom_path)
        # Re-initialize since config is frozen in preprocessor
        proc = ConcretePreProcessor(config, logger)
        result = proc._get_default_path('CUSTOM_PATH', 'default/path')
        assert result == custom_path

    def test_get_catchment_path_default(self, preprocessor):
        """Test get_catchment_path with defaults.

        Note: The path structure includes domain_definition_method and experiment_id
        for uniqueness: shapefiles/catchment/{method}/{experiment_id}/{filename}.shp
        """
        result = preprocessor.get_catchment_path()
        # New path structure includes definition method and experiment id
        expected = (preprocessor.project_dir / 'shapefiles' / 'catchment' /
                   'lumped' / 'test_experiment' / 'test_domain_HRUs_lumped.shp')
        assert result == expected

    def test_get_river_network_path_default(self, preprocessor):
        """Test get_river_network_path with defaults."""
        result = preprocessor.get_river_network_path()
        # Since domain_definition_method is 'lumped', the suffix is 'lumped'
        expected = preprocessor.project_dir / 'shapefiles' / 'river_network' / 'test_domain_riverNetwork_lumped.shp'
        assert result == expected


class TestDirectoryCreation:
    """Test directory creation methods."""

    def test_create_directories_basic(self, preprocessor, tmp_path):
        """Test basic directory creation."""
        # Remove directories if they exist
        if preprocessor.setup_dir.exists():
            shutil.rmtree(preprocessor.setup_dir)
        if preprocessor.forcing_dir.exists():
            shutil.rmtree(preprocessor.forcing_dir)

        preprocessor.create_directories()

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_dir.exists()

    def test_create_directories_with_additional(self, preprocessor, tmp_path):
        """Test directory creation with additional directories."""
        additional_dir = preprocessor.project_dir / 'extra' / 'directory'

        preprocessor.create_directories(additional_dirs=[additional_dir])

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_dir.exists()
        assert additional_dir.exists()

    def test_create_directories_idempotent(self, preprocessor):
        """Test that create_directories can be called multiple times."""
        preprocessor.create_directories()
        preprocessor.create_directories()  # Should not raise error

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_dir.exists()


class TestFilePathResolution:
    """Test file path resolution."""

    def test_get_file_path_with_defaults(self, preprocessor):
        """Test _get_base_file_path with all defaults."""
        result = preprocessor._get_base_file_path(
            'test_file',
            'FILE_PATH',
            'FILE_NAME',
            'default.txt'
        )

        # When path is default/None, uses project_dir
        assert result == preprocessor.project_dir / 'default.txt'

    def test_get_file_path_with_custom_path(self, config, logger, tmp_path):
        """Test _get_base_file_path with custom path."""
        custom_dir = tmp_path / 'custom'
        custom_dir.mkdir(exist_ok=True)
        config['FILE_PATH'] = str(custom_dir)
        config['FILE_NAME'] = 'custom_file.txt'

        # Re-initialize since config is frozen
        proc = ConcretePreProcessor(config, logger)

        result = proc._get_base_file_path(
            'test_file',
            'FILE_PATH',
            'FILE_NAME',
            'default.txt'
        )

        assert result == custom_dir / 'custom_file.txt'


class TestHelperMethods:
    """Test helper methods."""

    def test_is_lumped_true(self, preprocessor):
        """Test _is_lumped returns True for lumped configuration."""
        assert preprocessor._is_lumped() is True

    def test_is_lumped_false(self, config, logger):
        """Test _is_lumped returns False for distributed configuration."""
        config['DOMAIN_DEFINITION_METHOD'] = 'delineate'
        proc = ConcretePreProcessor(config, logger)
        assert proc._is_lumped() is False


class TestCopyBaseSettings:
    """Test base settings copying."""

    def test_copy_base_settings_directory_not_exist(self, preprocessor, tmp_path, logger):
        """Test copy_base_settings handles missing source directory gracefully."""
        non_existent = tmp_path / 'does_not_exist'

        preprocessor.copy_base_settings(source_dir=non_existent)

        # Should log warning but not raise error
        assert logger.warning.called

    def test_copy_base_settings_with_source_dir(self, preprocessor, tmp_path):
        """Test copy_base_settings with valid source directory."""
        source_dir = tmp_path / 'source'
        source_dir.mkdir(exist_ok=True)
        (source_dir / 'test_file.txt').write_text('test content')

        preprocessor.copy_base_settings(source_dir=source_dir)

        # Check if file was copied
        dest_file = preprocessor.setup_dir / 'test_file.txt'
        assert dest_file.exists()
        assert dest_file.read_text() == 'test content'

    def test_copy_base_settings_with_patterns(self, preprocessor, tmp_path):
        """Test copy_base_settings with file patterns."""
        source_dir = tmp_path / 'source'
        source_dir.mkdir(exist_ok=True)
        (source_dir / 'test.txt').write_text('text file')
        (source_dir / 'test.nc').write_text('netcdf file')
        (source_dir / 'ignore.dat').write_text('ignore this')

        # Clear setup dir
        if preprocessor.setup_dir.exists():
            import shutil
            shutil.rmtree(preprocessor.setup_dir)
        preprocessor.setup_dir.mkdir(parents=True, exist_ok=True)

        preprocessor.copy_base_settings(
            source_dir=source_dir,
            file_patterns=['*.txt', '*.nc']
        )

        # Check that only .txt and .nc files were copied
        assert (preprocessor.setup_dir / 'test.txt').exists()
        assert (preprocessor.setup_dir / 'test.nc').exists()
        assert not (preprocessor.setup_dir / 'ignore.dat').exists()


class TestAbstractMethods:
    """Test abstract method requirements."""

    def test_requires_model_name(self, config, logger):
        """Test that MODEL_NAME or _get_model_name() must be provided."""
        class IncompletePreProcessor(BaseModelPreProcessor):
            def run_preprocessing(self):
                pass
            # Missing MODEL_NAME and _get_model_name override

        with pytest.raises(NotImplementedError):
            IncompletePreProcessor(config, logger)

    def test_requires_run_preprocessing(self, config, logger):
        """Test that run_preprocessing must be implemented."""
        class IncompletePreProcessor(BaseModelPreProcessor):
            MODEL_NAME = "TEST"
            # Missing run_preprocessing

        with pytest.raises(TypeError):
            IncompletePreProcessor(config, logger)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestGetSimulationDates:
    """Base ``_get_simulation_dates`` (shared by the per-model preprocessors)."""

    def test_parses_config_strings_to_datetime(self, preprocessor):
        from datetime import datetime

        start, end = preprocessor._get_simulation_dates()
        assert (start, end) == (datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert type(start) is datetime and type(end) is datetime

    def test_keeps_time_of_day(self, config, logger, tmp_path):
        from datetime import datetime

        config = {**config, 'EXPERIMENT_TIME_START': '2020-01-01 01:00',
                  'EXPERIMENT_TIME_END': '2020-01-02 23:00'}
        (tmp_path / 'domain_test_domain' / 'settings' / 'TEST').mkdir(parents=True, exist_ok=True)
        pp = ConcretePreProcessor(config, logger)
        assert pp._get_simulation_dates() == (datetime(2020, 1, 1, 1), datetime(2020, 1, 2, 23))

    def test_defaults_used_when_unset(self, preprocessor, monkeypatch):
        from datetime import datetime

        monkeypatch.setattr(preprocessor, '_get_config_value',
                            lambda accessor, default=None, dict_key=None: default)
        monkeypatch.setattr(preprocessor, 'SIMULATION_DATE_DEFAULTS', ('2002-01-01', '2009-12-31'))
        assert preprocessor._get_simulation_dates() == (datetime(2002, 1, 1), datetime(2009, 12, 31))

    def test_unset_without_default_raises(self, preprocessor, monkeypatch):
        monkeypatch.setattr(preprocessor, '_get_config_value',
                            lambda accessor, default=None, dict_key=None: default)
        with pytest.raises(ValueError):
            preprocessor._get_simulation_dates()

    def test_model_overrides_pass_their_defaults(self, monkeypatch):
        """PRMS/WRF-Hydro and GSFLOW/WATFLOOD keep their historical defaults."""
        from datetime import datetime

        from symfluence.models.gsflow.preprocessor import GSFLOWPreProcessor
        from symfluence.models.prms.preprocessor import PRMSPreProcessor
        from symfluence.models.watflood.preprocessor import WATFLOODPreProcessor
        from symfluence.models.wrfhydro.preprocessor import WRFHydroPreProcessor

        expected = {
            PRMSPreProcessor: (datetime(2000, 1, 1), datetime(2001, 12, 31)),
            WRFHydroPreProcessor: (datetime(2000, 1, 1), datetime(2001, 12, 31)),
            GSFLOWPreProcessor: (datetime(2002, 1, 1), datetime(2009, 12, 31)),
            WATFLOODPreProcessor: (datetime(2002, 1, 1), datetime(2009, 12, 31)),
        }
        for cls, dates in expected.items():
            pp = cls.__new__(cls)
            monkeypatch.setattr(pp, '_get_config_value',
                                lambda accessor, default=None, dict_key=None: default, raising=False)
            assert pp._get_simulation_dates() == dates, cls.__name__
