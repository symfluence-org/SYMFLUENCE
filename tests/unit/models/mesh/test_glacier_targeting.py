from __future__ import annotations

import logging

import pytest

from symfluence.models.mesh.calibration.glacier_parameters import update_gru_arrays
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.mesh.preprocessing.class_file_manager import CLASSFileManager
from symfluence.models.mesh.preprocessing.glacier_writer import ice_block


def test_surface_specific_class_updates(mesh_config, tmp_path):
    land = CLASSFileManager._get_default_class_block()
    glacier = ice_block(land, mid=201, albedo_visible=.4, albedo_infrared=.25, temperature_c=-3)
    path = tmp_path/'MESH_parameters_CLASS.ini'
    path.write_text('\n'.join(land+glacier)+'\n')
    manager = MESHParameterManager(mesh_config, logging.getLogger(__name__), tmp_path)
    assert manager._update_class_file(path, {'KSAT': 10, 'LAMX': 2, 'ICE_ALBEDO_VISIBLE': .5})
    lines = path.read_text().splitlines()
    for old, new in zip(glacier, lines[15:]):
        if '5xALVC' in old:
            assert float(new.split()[4]) == .5
        else:
            assert old == new
    assert lines[2] == land[2]  # Land visible albedo unchanged.
    assert manager._read_param_from_file('ICE_ALBEDO_VISIBLE') == .5


def test_counted_arrays_and_refreezing_temperature():
    original = 'GRU class dependent hydrologic parameters\n1 # Number\nZSNL .1 .2\nFRZTH .1 # unread\n'
    result = update_gru_arrays(original, {'ZSNL': .3, 'FREZTH': -5}, [False, True])
    assert '2 # Number' in result
    assert 'ZSNL 0.3 .2' in result
    assert 'FREZTH -2.0 -5' in result
    with pytest.raises(ValueError, match='not in the counted'):
        update_gru_arrays(original, {'FRZTH': .2}, [False, True])


def test_glacier_bounds_are_explicit(mesh_config, tmp_path):
    manager = MESHParameterManager(mesh_config, logging.getLogger(__name__), tmp_path)
    manager.mesh_params = ['ICE_ALBEDO_VISIBLE']
    with pytest.raises(ValueError, match='explicit MESH_PARAM_BOUNDS'):
        manager._load_parameter_bounds()


def test_glacier_hydrology_without_ice_is_rejected():
    with pytest.raises(ValueError, match='native ice GRU'):
        update_gru_arrays('GRU class dependent hydrologic parameters\n0\n', {'SWELIM': 100}, [False])


def test_land_snow_parameter_leaves_water_proxy_and_ice_unchanged():
    from symfluence.models.mesh.calibration.glacier_parameters import update_gru_arrays

    text = '# GRU class dependent hydrologic parameters\n1\nZSNL 0.1 0.1 0.1\n'
    changed = update_gru_arrays(text, {'ZSNL': .3}, [False, False, True], [True, False, False])
    assert 'ZSNL 0.3 0.1 0.1' in changed
