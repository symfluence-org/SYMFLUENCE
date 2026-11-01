"""Config options that used to point at non-existent fields (always the default)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.ngen.config_schema import NGENConfig
from symfluence.models.rhessys.config_schema import RHESSysConfig
from symfluence.models.wflow.config_schema import WflowConfig
from symfluence.models.wflow.preprocessor import WflowPreProcessor

pytestmark = pytest.mark.unit


def test_new_fields_keep_previous_defaults():
    assert RHESSysConfig().std_scale == 1.0
    assert RHESSysConfig().use_subsurface_gw is True
    assert NGENConfig().forcing_provider is None
    assert WflowConfig().soil_layer_thickness == [100, 300, 800]


def test_new_fields_read_their_aliases():
    assert RHESSysConfig(RHESSYS_STD_SCALE=0.0, RHESSYS_USE_SUBSURFACE_GW=False).std_scale == 0.0
    assert RHESSysConfig(RHESSYS_USE_SUBSURFACE_GW=False).use_subsurface_gw is False
    assert NGENConfig(NGEN_FORCING_PROVIDER="NetCDF").forcing_provider == "NetCDF"
    assert WflowConfig(WFLOW_SOIL_LAYER_THICKNESS=[50, 150]).soil_layer_thickness == [50, 150]


def _wflow_stub(config):
    stub = SimpleNamespace(config=config)
    stub._get_config_value = lambda acc, default=None, dict_key=None: resolve_config_value(
        config, acc, default, dict_key)
    return stub


@pytest.mark.parametrize("config,expected", [
    ({}, [100, 300, 800]),                                    # default, ints as before
    ({"WFLOW_SOIL_LAYER_THICKNESS": [50.0, 150.0]}, [50, 150]),  # dict config honoured
    ({"WFLOW_SOIL_LAYER_THICKNESS": [50.5, 150]}, [50.5, 150]),
])
def test_wflow_soil_layers_single_source(config, expected):
    layers = WflowPreProcessor._soil_layer_thickness(_wflow_stub(config))
    assert layers == expected
    # The TOML line is rendered from the same list.
    assert f"soil_layer__thickness = {layers}" == f"soil_layer__thickness = {expected}"
