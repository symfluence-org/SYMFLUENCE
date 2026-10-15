"""Reject silent native bypass and preserve unrelated/ice parameters."""
from __future__ import annotations

import json

import pytest

from symfluence.models.mesh.preprocessing.frozen_soil import prepare_frozen_soil, validate_frozen_soil


@pytest.fixture
def inputs(tmp_path):
    (tmp_path/'MESH_input_run_options.ini').write_text('FROZENSOILINFILFLAG 1\n2009 1 0 0\n2011 1 0 0\n')
    (tmp_path/'MESH_parameters_CLASS.ini').write_text('30 30 30 14 3xSAND\n-4 -4 -4 14 3xSAND\n30 30 30 14 3xSAND\n')
    (tmp_path/'spatial_units.json').write_text(json.dumps({'units': [{'modis': 16}, {'modis': 15}, {'modis': 17}]}))
    (tmp_path/'MESH_parameters_hydrology.ini').write_text('2.0\nGRU class independent hydrologic parameters\n---\n1\nWrchrgIni 0.0\nGRU class dependent hydrologic parameters\n---\n2\nZSNL .1 .2 .3\nIWF 1 1 0\nRCHARG .2\nFRZTH .1\n')
    return tmp_path


@pytest.fixture
def parameters():
    return dict(soil_porosity=.45, soil_depth=.35, surface_saturation=.75,
                ice_lens_temperature=-2, infiltration_coefficient=2, opportunity_time=0)


def test_counted_writer_and_preservation(inputs, parameters):
    with pytest.raises(ValueError, match='SOIL_POR_MAX'):
        validate_frozen_soil(inputs)
    prepare_frozen_soil(inputs, parameters)
    p=inputs/'MESH_parameters_hydrology.ini';s=p.read_text()
    assert 'FRZC 2 0 0' in s
    assert 'T0_ACC 0 0 0' in s
    assert 'WrchrgIni 0.0' in s and 'ZSNL .1 .2 .3' in s and 'IWF 1 1 0' in s
    assert 'RCHARG' not in s and 'FRZTH' not in s
    assert '6 # counted native parameters' in s and '3 # counted native parameters' in s
    prepare_frozen_soil(inputs, parameters)
    assert p.read_text()==s
    prepare_frozen_soil(inputs)


@pytest.mark.parametrize('old,new', [('SOIL_DEPTH 0.35','SOIL_DEPTH 0'), ('FRZC 2 0 0','FRZC 0 0 0'), ('FRZC 2 0 0','FRZC 2 2 0'), ('T0_ACC 0 0 0','T0_ACC 0'), ('S0 0.75','S0 nan')])
def test_invalid_counted_values_fail(inputs, parameters, old, new):
    prepare_frozen_soil(inputs,parameters)
    p=inputs/'MESH_parameters_hydrology.ini';p.write_text(p.read_text().replace(old,new))
    with pytest.raises(ValueError):validate_frozen_soil(inputs)


def test_off_does_not_rewrite(inputs):
    p=inputs/'MESH_input_run_options.ini';p.write_text(p.read_text().replace('FLAG 1','FLAG 0'))
    hydro=inputs/'MESH_parameters_hydrology.ini';before=hydro.read_bytes();prepare_frozen_soil(inputs)
    assert hydro.read_bytes()==before


def test_unsupported_flag_fails(inputs):
    p=inputs/'MESH_input_run_options.ini';p.write_text(p.read_text().replace('FLAG 1','FLAG 2'))
    with pytest.raises(ValueError,match='supports'):prepare_frozen_soil(inputs)


def test_explicit_config_rejects_partial_or_nonfinite_assumptions(inputs, parameters):
    original=(inputs/'MESH_parameters_hydrology.ini').read_bytes()
    broken=dict(parameters);broken.pop('soil_depth')
    with pytest.raises(ValueError):prepare_frozen_soil(inputs,broken)
    assert (inputs/'MESH_parameters_hydrology.ini').read_bytes()==original
    broken=dict(parameters,soil_depth=float('inf'))
    with pytest.raises(ValueError):prepare_frozen_soil(inputs,broken)
    assert (inputs/'MESH_parameters_hydrology.ini').read_bytes()==original
