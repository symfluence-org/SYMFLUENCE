"""Regression tests for config-loading/validation fixes from the core audit."""
from __future__ import annotations

import logging

import pytest
import yaml

from symfluence.core.config.config_loader import _load_env_overrides
from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import ConfigurationError, ConfigValidationError

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _minimal(**extra):
    base = dict(
        domain_name='audit', model='SUMMA',
        time_start='2010-01-01 00:00', time_end='2012-12-31 23:00',
    )
    base.update(extra)
    return SymfluenceConfig.from_minimal(**base)


def _flat_yaml(tmp_path, **extra):
    data = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path), 'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'audit', 'EXPERIMENT_ID': 'run_1',
        'EXPERIMENT_TIME_START': '2010-01-01 00:00', 'EXPERIMENT_TIME_END': '2012-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped', 'SUB_GRID_DISCRETIZATION': 'GRUs',
        'HYDROLOGICAL_MODEL': 'SUMMA', 'FORCING_DATASET': 'ERA5',
        'POUR_POINT_COORDS': '51.0/-115.0',
    }
    data.update(extra)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data))
    return path


# --- 6. environment overrides ------------------------------------------------


class TestEnvOverrides:
    def test_string_fields_stay_strings(self, monkeypatch):
        monkeypatch.setenv('SYMFLUENCE_STATION_ID', '01013500')
        monkeypatch.setenv('SYMFLUENCE_EXPERIMENT_ID', '2024')
        monkeypatch.setenv('SYMFLUENCE_CALIBRATION_PERIOD', '2010-01-01, 2010-12-31')
        env = _load_env_overrides()
        assert env['STATION_ID'] == '01013500'
        assert env['EXPERIMENT_ID'] == '2024'
        assert env['CALIBRATION_PERIOD'] == '2010-01-01, 2010-12-31'

    def test_typed_fields_coerced(self, monkeypatch):
        monkeypatch.setenv('SYMFLUENCE_NUM_PROCESSES', '4')
        monkeypatch.setenv('SYMFLUENCE_STOP_ON_ERROR', 'no')
        monkeypatch.setenv('SYMFLUENCE_OPTIMIZATION_METHODS', 'iteration, data_assimilation')
        env = _load_env_overrides()
        assert env['NUM_PROCESSES'] == 4
        assert env['STOP_ON_ERROR'] is False
        assert env['OPTIMIZATION_METHODS'] == ['iteration', 'data_assimilation']

    @pytest.mark.parametrize('name', [
        'SYMFLUENCE_DEBUG', 'SYMFLUENCE_NO_FAULTHANDLER', 'SYMFLUENCE_PROFILE',
        'SYMFLUENCE_EVAL_WATCHDOG_SECONDS', 'SYMFLUENCE_PYTHON', 'SYMFLUENCE_STRICT_CONFIG',
    ])
    def test_control_vars_are_not_config(self, monkeypatch, name):
        monkeypatch.setenv(name, '1')
        stripped = name[len('SYMFLUENCE_'):]
        assert stripped not in _load_env_overrides()

    def test_control_vars_do_not_break_strict_load(self, monkeypatch, tmp_path):
        monkeypatch.setenv('SYMFLUENCE_EVAL_WATCHDOG_SECONDS', '30')
        monkeypatch.setenv('SYMFLUENCE_PYTHON', '/usr/bin/python3')
        monkeypatch.setenv('SYMFLUENCE_STRICT_CONFIG', 'true')
        monkeypatch.setenv('SYMFLUENCE_STATION_ID', '01013500')
        cfg = SymfluenceConfig.from_file(_flat_yaml(tmp_path))
        assert cfg.evaluation.streamflow.station_id == '01013500'


# --- 7. secrets masking + _metadata reload ------------------------------------


def test_mask_sensitive_config_recursive():
    from symfluence.project.logging_manager import MASKED_VALUE, mask_sensitive_config

    cfg = {
        'DB_PASSWORD': 'x', 'cds_api_key': 'k', 'GITHUB_TOKEN': 't', 'EARTHDATA_PASSWD': 'p',
        'my_Credentials': 'c', 'APIKEY': 'a', 'DOMAIN_NAME': 'keep', 'EMPTY_SECRET': None,
        'nested': {'client_secret': 's', 'items': [{'access_token': 'z'}], 'ok': 1},
    }
    out = mask_sensitive_config(cfg)
    for key in ('DB_PASSWORD', 'cds_api_key', 'GITHUB_TOKEN', 'EARTHDATA_PASSWD', 'my_Credentials', 'APIKEY'):
        assert out[key] == MASKED_VALUE, key
    assert out['DOMAIN_NAME'] == 'keep'
    assert out['EMPTY_SECRET'] is None
    assert out['nested']['client_secret'] == MASKED_VALUE
    assert out['nested']['items'][0]['access_token'] == MASKED_VALUE
    assert out['nested']['ok'] == 1
    assert cfg['DB_PASSWORD'] == 'x'  # input untouched


def test_archived_config_with_metadata_reloads_strict(tmp_path, monkeypatch):
    monkeypatch.setenv('SYMFLUENCE_STRICT_CONFIG', 'true')
    path = _flat_yaml(tmp_path, _metadata={'logged_at': 'now', 'symfluence_version': '0.9'})
    cfg = SymfluenceConfig.from_file(path, use_env=False)
    assert cfg.domain.name == 'audit'


# --- 8. nested unknown-key validation -----------------------------------------


def _nested_yaml(tmp_path, domain_extra=None):
    data = {
        'system': {'data_dir': str(tmp_path), 'code_dir': str(tmp_path)},
        'domain': {
            'name': 'audit', 'experiment_id': 'run_1',
            'time_start': '2010-01-01 00:00', 'time_end': '2010-12-31 23:00',
            'definition_method': 'lumped', 'discretization': 'GRUs',
            'pour_point_coords': '51.0/-115.0', **(domain_extra or {}),
        },
        'forcing': {'dataset': 'ERA5'},
        'model': {'hydrological_model': 'SUMMA'},
    }
    path = tmp_path / 'nested.yaml'
    path.write_text(yaml.safe_dump(data))
    return path


def test_nested_unknown_key_warns_with_suggestion(tmp_path, caplog):
    path = _nested_yaml(tmp_path, {'download_land_cover': True})
    with caplog.at_level(logging.WARNING):
        SymfluenceConfig.from_file(path, use_env=False)
    assert 'domain.download_land_cover' in caplog.text
    assert "domain.download_landcover" in caplog.text


def test_nested_unknown_key_strict_raises(tmp_path, monkeypatch):
    monkeypatch.setenv('SYMFLUENCE_STRICT_CONFIG', '1')
    with pytest.raises(ConfigValidationError, match='domain.download_land_cover'):
        SymfluenceConfig.from_file(_nested_yaml(tmp_path, {'download_land_cover': True}), use_env=False)


def test_shipped_nested_template_has_no_unknown_keys():
    from importlib.resources import files

    from symfluence.core.config.factories import _normalize_nested_config
    from symfluence.core.config.key_validation import find_unknown_nested_keys

    path = files('symfluence.resources.config_templates') / 'config_template_comprehensive_nested.yaml'
    data = yaml.safe_load(path.read_text())
    assert find_unknown_nested_keys(_normalize_nested_config(data)) == []


# --- 9. validation gaps ---------------------------------------------------------


@pytest.mark.parametrize('extra, match', [
    ({'CALIBRATION_PERIOD': '2012-01-01, 2010-01-01'}, 'reversed'),
    ({'EVALUATION_PERIOD': '2012-06-01, 2011-01-01'}, 'reversed'),
    ({'SPINUP_PERIOD': 'garbage'}, 'SPINUP_PERIOD'),
    ({'CALIBRATION_PERIOD': '2009-01-01, 2010-12-31'}, 'must be within'),
    ({'OPTIMIZATION_METHODS': ['iteraton']}, "did you mean 'iteration'"),
    ({'SUB_GRID_DISCRETIZATION': 'elevatoin'}, "did you mean 'elevation'"),
    ({'COUPLING_MODE': 'dcoupla'}, "did you mean 'dcoupler'"),
    ({'ROUTING_MODEL': 'mizuroot'}, 'ROUTING_MODEL'),
    ({'HYDROLOGICAL_MODEL': 'SUMA'}, "did you mean 'SUMMA'"),
    ({'HYDROLOGICAL_MODEL': 'SUMMA-MODFLOW'}, 'GROUNDWATER_MODEL'),
])
def test_invalid_values_rejected(tmp_path, extra, match):
    with pytest.raises((ConfigurationError, ValueError), match=match):
        SymfluenceConfig.from_file(_flat_yaml(tmp_path, **extra), use_env=False)


def test_valid_enum_values_accepted():
    cfg = _minimal(
        OPTIMIZATION_METHODS=['Iteration'], SUB_GRID_DISCRETIZATION='elevation,aspect',
        COUPLING_MODE='DCOUPLER', ROUTING_MODEL='mizuRoute', SPINUP_PERIOD='2010-01-01, 2010-06-30',
    )
    assert cfg.optimization.methods == ['iteration']
    assert cfg.model.coupling_mode == 'dcoupler'
    assert cfg.model.routing_model == 'mizuRoute'
    for routing in ('none', 'troute', 'T-ROUTE', 'dRoute'):
        _minimal(ROUTING_MODEL=routing)
    for disc in ('lumped', 'GRUs'):
        _minimal(SUB_GRID_DISCRETIZATION=disc)


def test_unknown_analysis_only_warns(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _minimal(ANALYSES=['benchmark'])
    assert cfg.evaluation.analyses == ['benchmark']
    assert "did you mean 'benchmarking'" in caplog.text


# --- 4. model name normalisation ---------------------------------------------


def test_model_names_normalized_upper():
    cfg = _minimal(HYDROLOGICAL_MODEL='summa, fuse')
    assert cfg.model.hydrological_model == 'SUMMA,FUSE'


def test_lowercase_summa_still_routes(tmp_path):
    from symfluence.project.model_manager import ModelManager

    cfg = _minimal(HYDROLOGICAL_MODEL='summa', DOMAIN_DEFINITION_METHOD='semidistributed',
                   SYMFLUENCE_DATA_DIR=str(tmp_path), ROUTING_MODEL='mizuRoute')
    workflow = ModelManager(cfg, logging.getLogger('t'))._resolve_model_workflow()
    assert workflow[0] == 'SUMMA'
    assert 'MIZUROUTE' in workflow


# --- 10. error location + resolve_path ------------------------------------------


def test_validation_error_shows_full_location(tmp_path):
    with pytest.raises(ConfigurationError) as excinfo:
        SymfluenceConfig.from_file(_flat_yaml(tmp_path, SUB_GRID_DISCRETIZATION='elevatoin'), use_env=False)
    assert 'domain.discretization (SUB_GRID_DISCRETIZATION)' in str(excinfo.value)


@pytest.mark.parametrize('value', ['', 'default', 'Default', 'DEFAULT', '  default  '])
def test_resolve_path_unset_spellings(tmp_path, value):
    from symfluence.core.path_resolver import resolve_path

    resolved = resolve_path({'X_PATH': value}, 'X_PATH', tmp_path, 'some/sub')
    assert resolved == tmp_path / 'some/sub'


def test_resolve_path_expands_user_and_vars(tmp_path, monkeypatch):
    from symfluence.core.path_resolver import resolve_path

    monkeypatch.setenv('MY_ROOT', str(tmp_path))
    assert resolve_path({'X': '$MY_ROOT/a'}, 'X', tmp_path, 'd') == tmp_path / 'a'
    monkeypatch.setenv('HOME', str(tmp_path))
    assert resolve_path({'X': '~/b'}, 'X', tmp_path, 'd') == tmp_path / 'b'
