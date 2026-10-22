"""Verify basin bounds survive typed configuration and reach calibration."""
from __future__ import annotations

import logging

import yaml

from symfluence.core.config.models.root import SymfluenceConfig
from symfluence.models.crhm.calibration.parameter_manager import CRHMParameterManager


def test_typed_drainage_bounds_reach_parameter_manager(tmp_path):
    data = {
        'system': {'data_dir': str(tmp_path)},
        'domain': {'name': 'bounds_test', 'experiment_id': 'bounds_test',
                   'time_start': '2020-01-01', 'time_end': '2021-01-01',
                   'definition_method': 'lumped', 'discretization': 'elevation'},
        'model': {'hydrological_model': 'CRHM', 'crhm': {
            'params_to_calibrate': 'soil_gw_K,gw_K'}},
    }
    overrides = {'soil_gw_K': {'min': .01, 'max': 50., 'transform': 'log'}}
    data['model']['crhm']['param_bounds'] = overrides
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data))
    config = SymfluenceConfig.from_file(path)
    assert config.model.crhm.param_bounds == overrides
    assert config.to_dict()['CRHM_PARAM_BOUNDS'] == overrides
    for view in [config, config.to_dict()]:
        manager = CRHMParameterManager(view, logging.getLogger('bounds'), tmp_path)
        assert manager.param_bounds['soil_gw_K'] == overrides['soil_gw_K']
        # Native drainage is mm/day, so candidates above 1 must be reachable.
        assert manager.param_bounds['gw_K']['max'] >= 50
