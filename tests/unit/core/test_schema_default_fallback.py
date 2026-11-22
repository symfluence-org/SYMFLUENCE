"""resolve_config_value falls back to the schema's default, not a restated one."""
from __future__ import annotations

import pytest

import symfluence.models  # noqa: F401 - registers the model schemas
from symfluence.core.config.schema_defaults import NO_SCHEMA_DEFAULT, schema_default
from symfluence.core.mixins.config import ConfigMixin, resolve_config_value


def test_schema_default_by_flat_key():
    assert schema_default('SUMMA_TIMEOUT') == 7200            # model schema
    assert schema_default('OPTIMIZATION_METRIC') == 'KGE'     # framework section
    assert schema_default('DATA_ACCESS') == 'cloud'           # enum-free plain value
    assert schema_default('CALIBRATION_PERIOD') is None       # declared, unset
    assert schema_default('DOMAIN_NAME') is NO_SCHEMA_DEFAULT  # required
    assert schema_default('NOT_A_KEY') is NO_SCHEMA_DEFAULT


def test_partial_dict_gets_the_schema_default():
    config = {'DOMAIN_NAME': 'x'}
    assert resolve_config_value(config, lambda: config.model.summa.timeout, dict_key='SUMMA_TIMEOUT') == 7200
    assert resolve_config_value(config, lambda: config.domain.calibration_period, dict_key='CALIBRATION_PERIOD') is None
    assert resolve_config_value(config, lambda: config.nope, dict_key='NOT_A_KEY') is None


def test_configured_value_wins():
    config = {'SUMMA_TIMEOUT': 60}
    assert resolve_config_value(config, lambda: config.model.summa.timeout, dict_key='SUMMA_TIMEOUT') == 60


@pytest.mark.parametrize('default', [None, 0, '', 300])
def test_explicit_default_is_still_honoured(default):
    assert resolve_config_value({}, lambda: None, default, 'SUMMA_TIMEOUT') == default


def test_no_key_no_default_is_none():
    assert resolve_config_value({}, lambda: None) is None


def test_mixin_method_falls_back_the_same_way():
    class Reader(ConfigMixin):
        pass

    reader = Reader()
    reader._config = {}
    assert reader._get_config_value(lambda: reader.config.model.vic.timeout, dict_key='VIC_TIMEOUT') == 7200
    assert reader._get_config_value(lambda: reader.config.model.vic.timeout, default=5, dict_key='VIC_TIMEOUT') == 5
