"""
Unit tests for configuration validation.

Tests the Pydantic validators in SymfluenceConfig model.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import ConfigurationError


class TestBasicValidation:
    """Test basic field type and presence validation"""

    def test_required_fields_present(self):
        """Test that configuration with all required fields validates"""
        config = {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }
        model = SymfluenceConfig(**config)
        assert model.DOMAIN_NAME == 'test_domain'
        assert model.EXPERIMENT_ID == 'exp_001'

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError"""
        config = {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            # Missing DOMAIN_NAME
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }
        with pytest.raises(ValidationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'DOMAIN_NAME' in str(exc_info.value)

    def test_invalid_literal_value_raises_error(self):
        """Test that invalid literal value raises ValidationError"""
        config = {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'invalid_method',  # Invalid literal
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }
        with pytest.raises(ValidationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'DOMAIN_DEFINITION_METHOD' in str(exc_info.value)

    def test_optional_fields_with_defaults(self):
        """Test that optional fields use their defaults"""
        config = {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }
        model = SymfluenceConfig(**config)
        assert model.NUM_PROCESSES == 1
        assert model.LOG_LEVEL == 'INFO'
        assert model.STREAM_THRESHOLD == 5000.0
        assert model.PET_METHOD == 'oudin'


class TestFieldValidators:
    """Test individual field validators"""

    def test_path_expansion(self):
        """Test that paths are expanded and resolved"""
        config = self._get_minimal_config()
        config['SYMFLUENCE_DATA_DIR'] = '~/test_data'
        model = SymfluenceConfig(**config)
        assert isinstance(model.SYMFLUENCE_DATA_DIR, Path)
        assert not str(model.SYMFLUENCE_DATA_DIR).startswith('~')

    def test_hydrological_model_list_to_string(self):
        """Test that model list is converted to comma-separated string"""
        config = self._get_minimal_config()
        config['HYDROLOGICAL_MODEL'] = ['SUMMA', 'FUSE', 'GR']
        model = SymfluenceConfig(**config)
        assert model.HYDROLOGICAL_MODEL == 'SUMMA,FUSE,GR'

    def test_hydrological_model_string_unchanged(self):
        """Test that model string remains unchanged"""
        config = self._get_minimal_config()
        config['HYDROLOGICAL_MODEL'] = 'SUMMA,FUSE'
        model = SymfluenceConfig(**config)
        assert model.HYDROLOGICAL_MODEL == 'SUMMA,FUSE'

    def test_list_fields_comma_separated_to_list(self):
        """Test that comma-separated strings become lists"""
        config = self._get_minimal_config()
        config['OPTIMIZATION_METHODS'] = 'iteration,emulation'
        config['EVALUATION_DATA'] = 'streamflow,snow'
        model = SymfluenceConfig(**config)
        assert model.OPTIMIZATION_METHODS == ['iteration', 'emulation']
        assert model.EVALUATION_DATA == ['streamflow', 'snow']

    def test_list_fields_already_list_unchanged(self):
        """Test that list fields stay as lists"""
        config = self._get_minimal_config()
        config['NEX_MODELS'] = ['ACCESS-CM2', 'GFDL-ESM4']
        model = SymfluenceConfig(**config)
        assert model.NEX_MODELS == ['ACCESS-CM2', 'GFDL-ESM4']

    def test_positive_threshold_validation(self):
        """Test that thresholds must be non-negative"""
        config = self._get_minimal_config()
        config['STREAM_THRESHOLD'] = -100.0  # Invalid
        with pytest.raises(ValidationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'non-negative' in str(exc_info.value).lower()

    def test_positive_integer_validation(self):
        """Test that certain integers must be positive"""
        config = self._get_minimal_config()
        config['NUM_PROCESSES'] = 0  # Invalid, must be >= 1
        with pytest.raises(ValidationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'at least 1' in str(exc_info.value).lower()

    @staticmethod
    def _get_minimal_config():
        """Helper to get minimal valid configuration"""
        return {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }


class TestCrossFieldValidation:
    """Test cross-field validators (model_validator)"""

    def test_time_period_start_before_end(self):
        """Test that experiment start must be before end"""
        config = self._get_minimal_config()
        config['EXPERIMENT_TIME_START'] = '2020-12-31 23:00'
        config['EXPERIMENT_TIME_END'] = '2020-01-01 00:00'  # Before start!
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'must be before' in str(exc_info.value)

    def test_calibration_period_within_experiment(self):
        """Test that calibration period must be within experiment period"""
        config = self._get_minimal_config()
        config['EXPERIMENT_TIME_START'] = '2020-06-01 00:00'
        config['EXPERIMENT_TIME_END'] = '2020-08-31 23:00'
        config['CALIBRATION_PERIOD'] = '2020-01-01, 2020-03-31'  # Before experiment!
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'within' in str(exc_info.value).lower()

    def test_evaluation_period_within_experiment(self):
        """Test that evaluation period must be within experiment period"""
        config = self._get_minimal_config()
        config['EXPERIMENT_TIME_START'] = '2020-01-01 00:00'
        config['EXPERIMENT_TIME_END'] = '2020-06-30 23:00'
        config['EVALUATION_PERIOD'] = '2020-07-01, 2020-12-31'  # After experiment!
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'within' in str(exc_info.value).lower()

    def test_valid_time_periods(self):
        """Test that valid time periods pass validation"""
        config = self._get_minimal_config()
        config['EXPERIMENT_TIME_START'] = '2020-01-01 00:00'
        config['EXPERIMENT_TIME_END'] = '2020-12-31 23:00'
        config['CALIBRATION_PERIOD'] = '2020-01-01, 2020-06-30'
        config['EVALUATION_PERIOD'] = '2020-07-01, 2020-12-31'
        model = SymfluenceConfig(**config)
        assert model.CALIBRATION_PERIOD == '2020-01-01, 2020-06-30'
        assert model.EVALUATION_PERIOD == '2020-07-01, 2020-12-31'

    def test_pour_point_coords_valid(self):
        """Test that valid pour point coordinates pass"""
        config = self._get_minimal_config()
        config['POUR_POINT_COORDS'] = '51.1722/-115.5717'
        model = SymfluenceConfig(**config)
        assert model.POUR_POINT_COORDS == '51.1722/-115.5717'

    def test_pour_point_coords_invalid_latitude(self):
        """Test that invalid latitude is rejected"""
        config = self._get_minimal_config()
        config['POUR_POINT_COORDS'] = '91.5/-115.5'  # Lat > 90
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'out of range' in str(exc_info.value)

    def test_pour_point_coords_invalid_longitude(self):
        """Test that invalid longitude is rejected"""
        config = self._get_minimal_config()
        config['POUR_POINT_COORDS'] = '45.0/-185.0'  # Lon < -180
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'out of range' in str(exc_info.value)

    def test_pour_point_coords_invalid_format(self):
        """Test that invalid coordinate format is rejected"""
        config = self._get_minimal_config()
        config['POUR_POINT_COORDS'] = 'invalid_format'
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'format' in str(exc_info.value).lower()

    def test_bounding_box_coords_valid(self):
        """Test that valid bounding box passes"""
        config = self._get_minimal_config()
        config['BOUNDING_BOX_COORDS'] = '51.76/-116.55/50.95/-115.5'  # north/west/south/east
        model = SymfluenceConfig(**config)
        assert model.BOUNDING_BOX_COORDS == '51.76/-116.55/50.95/-115.5'

    def test_bounding_box_coords_invalid_south_north(self):
        """Test that south must be less than north"""
        config = self._get_minimal_config()
        config['BOUNDING_BOX_COORDS'] = '50.0/-116.0/52.0/-115.0'  # south > north!
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'south' in str(exc_info.value).lower() and 'north' in str(exc_info.value).lower()

    def test_bounding_box_coords_invalid_latitude_range(self):
        """Test that latitude must be in valid range"""
        config = self._get_minimal_config()
        config['BOUNDING_BOX_COORDS'] = '95.0/-116.0/50.0/-115.0'  # north > 90
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'latitude' in str(exc_info.value).lower()

    def test_bounding_box_coords_invalid_format(self):
        """Test that invalid bounding box format is rejected"""
        config = self._get_minimal_config()
        config['BOUNDING_BOX_COORDS'] = 'invalid/format'
        with pytest.raises(ConfigurationError) as exc_info:
            SymfluenceConfig(**config)
        assert 'format' in str(exc_info.value).lower()

    @staticmethod
    def _get_minimal_config():
        """Helper to get minimal valid configuration"""
        return {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }


class TestHelperMethods:
    """Test helper methods in SymfluenceConfig"""

    def test_parse_period(self):
        """Test _parse_period helper"""
        config = self._get_minimal_config()
        model = SymfluenceConfig(**config)
        start, end = model._parse_period('2020-01-01, 2020-12-31')
        assert start.year == 2020
        assert start.month == 1
        assert end.year == 2020
        assert end.month == 12

    def test_parse_models_from_string(self):
        """Test _parse_models from comma-separated string"""
        config = self._get_minimal_config()
        config['HYDROLOGICAL_MODEL'] = 'summa, fuse, gr'
        model = SymfluenceConfig(**config)
        models = model._parse_models()
        assert models == ['SUMMA', 'FUSE', 'GR']

    def test_parse_models_from_list(self):
        """Test _parse_models from list"""
        config = self._get_minimal_config()
        config['HYDROLOGICAL_MODEL'] = ['SUMMA', 'FUSE']
        model = SymfluenceConfig(**config)
        # Note: validator converts list to string first
        assert model.HYDROLOGICAL_MODEL == 'SUMMA,FUSE'
        models = model._parse_models()
        assert models == ['SUMMA', 'FUSE']

    @staticmethod
    def _get_minimal_config():
        """Helper to get minimal valid configuration"""
        return {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }
