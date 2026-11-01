"""MHM_ROUTING_CASE selects mRM processCase(8); streamflow_celerity is calibratable."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.mhm.config_schema import MHMConfig
from symfluence.models.mhm.parameters import DEFAULT_PARAMS, PARAM_BOUNDS
from symfluence.models.mhm.preprocessor import MHMPreProcessor


def _preprocessor(**cfg):
    pre = SimpleNamespace(config=None)
    pre._get_config_value = lambda accessor, default=None, dict_key=None: cfg.get(dict_key, default)
    return pre


def test_muskingum_is_the_default():
    assert MHMPreProcessor._routing_case(_preprocessor()) == 1
    assert MHMConfig().routing_case == 1


def test_adaptive_routing_selected():
    assert MHMPreProcessor._routing_case(_preprocessor(MHM_ROUTING_CASE=2)) == 2


def test_unsupported_routing_case_refused():
    with pytest.raises(ConfigurationError, match='MHM_ROUTING_CASE'):
        MHMPreProcessor._routing_case(_preprocessor(MHM_ROUTING_CASE=3))


def test_namelist_template_reads_the_routing_case():
    import inspect
    src = inspect.getsource(MHMPreProcessor._generate_mhm_namelist)
    assert 'processCase(8)  = {self._routing_case()}' in src


def test_celerity_is_calibratable_within_the_namelist_range():
    b = PARAM_BOUNDS['streamflow_celerity']
    assert 0.1 <= b['min'] < DEFAULT_PARAMS['streamflow_celerity'] < b['max'] <= 15.0
