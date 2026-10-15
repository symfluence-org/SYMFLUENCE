from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from netCDF4 import Dataset

from symfluence.models.mesh.calibration.forcing_scaling import (
    SOURCE_RECORD,
    apply_forcing_corrections,
    read_applied,
)
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.mesh.preprocessing.class_file_manager import CLASSFileManager
from symfluence.models.mesh.preprocessing.glacier_writer import ice_block


def _forcing(path):
    with Dataset(path, 'w') as nc:
        nc.createDimension('time', 4)
        nc.createDimension('subbasin', 2)
        for name, value in {'PRE': 1e-4, 'TA': 270.0, 'QA': 0.003}.items():
            var = nc.createVariable(name, 'f4', ('time', 'subbasin'))
            var[:] = np.full((4, 2), value, dtype='f4')


def _read(path, name):
    with Dataset(path) as nc:
        return np.asarray(nc.variables[name][:])


def test_corrections_replace_link_and_never_touch_shared_source(tmp_path):
    shared = tmp_path/'shared.nc'
    _forcing(shared)
    run = tmp_path/'run'
    run.mkdir()
    (run/'MESH_forcing_safe.nc').symlink_to(shared)

    apply_forcing_corrections(run, {'PRECIP_MULT': 1.25, 'TEMP_OFFSET': .5})
    assert not (run/'MESH_forcing_safe.nc').is_symlink()
    np.testing.assert_allclose(_read(run/'MESH_forcing_safe.nc', 'PRE'), 1.25e-4, rtol=1e-6)
    np.testing.assert_allclose(_read(run/'MESH_forcing_safe.nc', 'TA'), 270.5, rtol=1e-7)
    np.testing.assert_allclose(_read(run/'MESH_forcing_safe.nc', 'QA'), .003, rtol=1e-7)
    np.testing.assert_allclose(_read(shared, 'PRE'), 1e-4, rtol=1e-7)

    # The next trial scales the source, not the previous trial's output, and
    # corrections it does not name return to neutral.
    apply_forcing_corrections(run, {'PRECIP_MULT': 1.1})
    np.testing.assert_allclose(_read(run/'MESH_forcing_safe.nc', 'PRE'), 1.1e-4, rtol=1e-6)
    np.testing.assert_allclose(_read(run/'MESH_forcing_safe.nc', 'TA'), 270.0, rtol=1e-7)
    assert read_applied(run) == {'PRECIP_MULT': 1.1, 'TEMP_OFFSET': 0.0}


def test_regular_file_is_preserved_as_unscaled_source(tmp_path):
    _forcing(tmp_path/'MESH_forcing_safe.nc')
    apply_forcing_corrections(tmp_path, {'PRECIP_MULT': 2.0})
    apply_forcing_corrections(tmp_path, {'PRECIP_MULT': 1.5})
    np.testing.assert_allclose(_read(tmp_path/'MESH_forcing_safe.unscaled.nc', 'PRE'), 1e-4, rtol=1e-7)
    np.testing.assert_allclose(_read(tmp_path/'MESH_forcing_safe.nc', 'PRE'), 1.5e-4, rtol=1e-6)


def test_changed_source_and_invalid_values_are_rejected(tmp_path):
    shared = tmp_path/'shared.nc'
    _forcing(shared)
    run = tmp_path/'run'
    run.mkdir()
    (run/'MESH_forcing_safe.nc').symlink_to(shared)
    with pytest.raises(ValueError, match='positive'):
        apply_forcing_corrections(run, {'PRECIP_MULT': 0.})
    with pytest.raises(ValueError, match='Unknown'):
        apply_forcing_corrections(run, {'RAIN_MULT': 1.})
    apply_forcing_corrections(run, {'PRECIP_MULT': 1.2})
    record = json.loads((run/SOURCE_RECORD).read_text())
    assert record['source'] == str(shared.resolve())
    with Dataset(shared, 'r+') as nc:
        nc.variables['PRE'][:] = 0.
    with pytest.raises(ValueError, match='changed since'):
        apply_forcing_corrections(run, {'PRECIP_MULT': 1.2})


def test_manager_routes_forcing_and_rejects_unmapped_names(mesh_config, tmp_path):
    _forcing(tmp_path/'MESH_forcing_safe.nc')
    manager = MESHParameterManager(mesh_config, logging.getLogger(__name__), tmp_path)
    assert manager.update_mesh_params({'PRECIP_MULT': 1.3})
    assert manager._read_param_from_file('PRECIP_MULT') == 1.3
    assert not manager.update_mesh_params({'NOT_A_MESH_PARAMETER': 1.})


def test_texture_targets_land_layers_only(mesh_config, tmp_path):
    land = CLASSFileManager._get_default_class_block()
    glacier = ice_block(land, mid=201, albedo_visible=.4, albedo_infrared=.25, temperature_c=-3)
    path = tmp_path/'MESH_parameters_CLASS.ini'
    path.write_text('\n'.join(land+land+glacier)+'\n')
    (tmp_path/'spatial_units.json').write_text(json.dumps({'units': [
        {'surface': 'land', 'modis': 16}, {'surface': 'land', 'modis': 17}, {'surface': 'ice', 'modis': 15}]}))
    manager = MESHParameterManager(mesh_config, logging.getLogger(__name__), tmp_path)
    assert manager._update_class_file(path, {'SAND': 85., 'CLAY': 3., 'DRN': .5})
    rows = [line for line in path.read_text().splitlines() if 'xSAND' in line or 'xCLAY' in line]
    sand, clay = rows[0].split(), rows[1].split()
    n = len([p for p in land if 'xSAND' in p][0].split()) - 4
    assert [float(v) for v in sand[:n]] == [85.]*n and [float(v) for v in clay[:n]] == [3.]*n
    assert rows[2] == [p for p in land if 'xSAND' in p][0]  # mapped water keeps template
    assert rows[4].split()[0] == '-4'                       # ice untouched
    assert manager._read_param_from_file('SAND') == 85.
    with pytest.raises(ValueError, match='exceeds 100'):
        manager._update_texture(path.read_text().splitlines(keepends=True),
                                manager._detect_all_gru_blocks_from_lines(path.read_text().splitlines()),
                                [False, False, True], {'SAND': 90., 'CLAY': 20.}, tmp_path)


def test_small_baseflow_coefficient_is_not_rounded_to_zero(mesh_config, tmp_path):
    hydro = tmp_path/'MESH_parameters_hydrology.ini'
    hydro.write_text('2.0: header\nPWR    2.370\nFLZ    1.000E-03\n')
    manager = MESHParameterManager(mesh_config, logging.getLogger(__name__), tmp_path)
    assert manager._update_ini_file(hydro, {'FLZ': 3.2e-7, 'PWR': 1.5})
    values = {line.split()[0]: float(line.split()[1]) for line in hydro.read_text().splitlines()[1:]}
    assert values == {'PWR': 1.5, 'FLZ': 3.2e-7}


def test_texture_feasibility_caps_clay(mesh_config, tmp_path):
    manager = MESHParameterManager(mesh_config, logging.getLogger(__name__), tmp_path)
    assert manager._validate_parameter_feasibility({'SAND': 92., 'CLAY': 14.}) == {'SAND': 92., 'CLAY': 8.}
    assert manager._validate_parameter_feasibility({'SAND': 60., 'CLAY': 14.}) == {'SAND': 60., 'CLAY': 14.}
