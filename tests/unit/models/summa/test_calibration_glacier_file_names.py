"""SUMMA calibration writes the files the file manager points SUMMA at.

A glacier setup with a hand-written ``fileManager_glac.txt`` (naming
``attributes_glac.nc`` / ``coldState_glac.nc``) and default
``SETTINGS_SUMMA_ATTRIBUTES`` / ``_COLDSTATE`` had the calibration writers work
on ``attributes.nc`` / ``coldState.nc``: soil depths went into a cold state
SUMMA never opened, and where only the ``_glac`` files had been copied the
trial-parameter writer failed with "Attributes file not found".
"""
from __future__ import annotations

import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from symfluence.core.exceptions import FileOperationError
from symfluence.models.summa.calibration.optimizer_mixin import SUMMAOptimizerMixin
from symfluence.models.summa.calibration.parameter_files import (
    read_file_manager_names,
    resolve_settings_file_names,
)
from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
from symfluence.models.summa.calibration.worker_impl import _apply_parameters_worker

pytestmark = [pytest.mark.unit, pytest.mark.quick]

LOG = logging.getLogger('test_summa_glacier_file_names')
DEPTHS = np.array([0.025, 0.075, 0.15, 0.25, 0.5])
GLACIER_FM = 'fileManager_glac.txt'


def _attributes(path: Path, n_hru: int) -> None:
    with nc.Dataset(path, 'w') as ds:
        ds.createDimension('hru', n_hru)
        ds.createDimension('gru', 1)
        ds.createVariable('hruId', 'i4', ('hru',))[:] = np.arange(1, n_hru + 1)
        ds.createVariable('gruId', 'i4', ('gru',))[:] = [1]


def _coldstate(path: Path, n_hru: int) -> None:
    with nc.Dataset(path, 'w') as ds:
        ds.createDimension('hru', n_hru)
        ds.createDimension('midToto', len(DEPTHS))
        ds.createDimension('ifcToto', len(DEPTHS) + 1)
        ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))[:] = np.tile(DEPTHS[:, None], (1, n_hru))
        ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))[:] = np.tile(
            np.r_[0, DEPTHS.cumsum()][:, None], (1, n_hru))


def _file_manager(path: Path, attributes: str, coldstate: str, trialparams: str = 'trialParams.nc') -> None:
    path.write_text(
        "controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n"
        "simStartTime         '2009-10-01 00:00'\n"
        f"settingsPath         '{path.parent.as_posix()}/'\n"
        f"initConditionFile    '{coldstate}'   ! glacier domains\n"
        f"attributeFile        '{attributes}'\n"
        f"trialParamFile       '{trialparams}'\n"
        "forcingListFile      'forcingFileList.txt'\n",
        encoding='utf-8',
    )


def _glacier_settings(tmp_path: Path, with_plain_files: bool = True) -> Path:
    """Settings dir of a glacier setup: 3 HRUs with glaciers, 2 in the plain files."""
    settings = tmp_path / 'settings' / 'SUMMA'
    settings.mkdir(parents=True)
    _attributes(settings / 'attributes_glac.nc', 3)
    _coldstate(settings / 'coldState_glac.nc', 3)
    if with_plain_files:
        _attributes(settings / 'attributes.nc', 2)
        _coldstate(settings / 'coldState.nc', 2)
    _file_manager(settings / GLACIER_FM, 'attributes_glac.nc', 'coldState_glac.nc')
    (settings / 'localParamInfo.txt').write_text("k_soil | 1.0d-5 | 1.0d-7 | 1.0d-3\n")
    (settings / 'basinParamInfo.txt').write_text("routingGammaShape | 2.5 | 2.0 | 3.0\n")
    return settings


def _depths(path: Path) -> np.ndarray:
    with nc.Dataset(path) as ds:
        return np.asarray(ds['mLayerDepth'][:, 0]).copy()


def _n_hru(path: Path) -> int:
    with nc.Dataset(path) as ds:
        return ds.dimensions['hru'].size


# config as in the report: glacier file manager, default attributes/coldState names
CONFIG = {'SETTINGS_SUMMA_FILEMANAGER': GLACIER_FM, 'PARAMS_TO_CALIBRATE': 'k_soil',
          'BASIN_PARAMS_TO_CALIBRATE': '', 'CALIBRATE_DEPTH': True}
PARAMS = {'k_soil': np.array([2e-5]), 'total_mult': np.array([2.0]), 'shape_factor': np.array([1.0])}


def test_file_manager_names_are_read(tmp_path):
    settings = _glacier_settings(tmp_path)
    assert read_file_manager_names(settings / GLACIER_FM) == {
        'attributes': 'attributes_glac.nc', 'coldstate': 'coldState_glac.nc', 'trialparams': 'trialParams.nc'}
    assert read_file_manager_names(settings / 'missing.txt') == {}


def test_names_come_from_the_file_manager_when_it_is_there(tmp_path):
    settings = _glacier_settings(tmp_path)
    assert resolve_settings_file_names(CONFIG) == ('attributes.nc', 'coldState.nc', 'trialParams.nc')
    assert resolve_settings_file_names(CONFIG, settings) == (
        'attributes_glac.nc', 'coldState_glac.nc', 'trialParams.nc')
    # no such file manager in the directory: config only
    assert resolve_settings_file_names({}, settings) == ('attributes.nc', 'coldState.nc', 'trialParams.nc')
    # a file manager that leaves an entry out: config fills it in
    (settings / 'partial.txt').write_text("attributeFile 'a.nc'\n")
    assert resolve_settings_file_names(
        {'SETTINGS_SUMMA_FILEMANAGER': 'partial.txt', 'SETTINGS_SUMMA_COLDSTATE': 'c.nc'}, settings,
    ) == ('a.nc', 'c.nc', 'trialParams.nc')


@pytest.mark.parametrize('with_plain_files', [True, False])
def test_worker_writes_the_files_the_glacier_file_manager_names(tmp_path, with_plain_files):
    settings = _glacier_settings(tmp_path, with_plain_files)
    debug = {'files_checked': [], 'errors': []}
    task = {'config': CONFIG, 'original_depths': DEPTHS.tolist()}

    assert _apply_parameters_worker(PARAMS, task, settings, LOG, debug), debug

    np.testing.assert_allclose(_depths(settings / 'coldState_glac.nc'), DEPTHS * 2.0)
    assert _n_hru(settings / 'trialParams.nc') == 3  # laid out on the glacier attributes
    if with_plain_files:
        np.testing.assert_allclose(_depths(settings / 'coldState.nc'), DEPTHS)  # not SUMMA's file


@pytest.mark.parametrize('with_plain_files', [True, False])
def test_manager_writes_the_files_the_glacier_file_manager_names(tmp_path, with_plain_files):
    settings = _glacier_settings(tmp_path, with_plain_files)
    manager = SUMMAParameterManager(CONFIG, LOG, settings)
    assert manager.attr_file_path == settings / 'attributes_glac.nc'

    assert manager.update_model_files(PARAMS)

    np.testing.assert_allclose(_depths(settings / 'coldState_glac.nc'), DEPTHS * 2.0)
    assert _n_hru(settings / 'trialParams.nc') == 3
    assert len(manager.get_initial_parameters()['k_soil']) == 3


def test_configured_glacier_names_with_standard_file_manager(tmp_path):
    """File manager and config agree on the ``_glac`` files (fileManager.txt, explicit geometry)."""
    settings = _glacier_settings(tmp_path)
    (settings / GLACIER_FM).rename(settings / 'fileManager.txt')
    config = {'SETTINGS_SUMMA_ATTRIBUTES': 'attributes_glac.nc', 'SETTINGS_SUMMA_COLDSTATE': 'coldState_glac.nc',
              'PARAMS_TO_CALIBRATE': 'k_soil', 'CALIBRATE_DEPTH': True}
    assert resolve_settings_file_names(config, settings) == resolve_settings_file_names(config)

    debug = {'files_checked': [], 'errors': []}
    assert _apply_parameters_worker(
        PARAMS, {'config': config, 'original_depths': DEPTHS.tolist()}, settings, LOG, debug), debug
    np.testing.assert_allclose(_depths(settings / 'coldState_glac.nc'), DEPTHS * 2.0)
    assert _n_hru(settings / 'trialParams.nc') == 3


def _copy(tmp_path: Path, config: dict) -> set:
    class Optimizer(SUMMAOptimizerMixin):
        pass

    opt = Optimizer()
    opt.project_dir = tmp_path
    opt.optimization_settings_dir = tmp_path / 'opt'
    opt.optimization_settings_dir.mkdir()
    opt.logger = LOG
    opt._config = None
    opt.config_dict = dict(config)
    opt._copy_mizuroute_settings = lambda: None
    opt._copy_summa_settings_files()
    return {p.name for p in opt.optimization_settings_dir.iterdir()}


def test_settings_copy_follows_the_file_manager(tmp_path):
    settings = _glacier_settings(tmp_path)
    for name in ('modelDecisions.txt', 'outputControl.txt'):
        (settings / name).write_text('x')

    copied = _copy(tmp_path, CONFIG)
    assert {'attributes_glac.nc', 'coldState_glac.nc', GLACIER_FM} <= copied
    assert not {'attributes.nc', 'coldState.nc'} & copied

    # a 'glac' file manager that names the plain files gets the plain files
    _file_manager(settings / GLACIER_FM, 'attributes.nc', 'coldState.nc')
    (tmp_path / 'opt').rename(tmp_path / 'opt_first')
    copied = _copy(tmp_path, CONFIG)
    assert {'attributes.nc', 'coldState.nc'} <= copied
    assert not {'attributes_glac.nc', 'coldState_glac.nc'} & copied


def test_settings_copy_names_the_missing_attributes_file(tmp_path):
    settings = _glacier_settings(tmp_path)
    for name in ('modelDecisions.txt', 'outputControl.txt'):
        (settings / name).write_text('x')
    (settings / 'attributes_glac.nc').unlink()
    with pytest.raises(FileOperationError, match='attributes_glac.nc'):
        _copy(tmp_path, CONFIG)


def test_depth_defaults_match_the_calibrated_depth_parameters(tmp_path):
    settings = _glacier_settings(tmp_path)

    only_multiplier = SUMMAParameterManager(
        {**CONFIG, 'CALIBRATE_DEPTH': False, 'PARAMS_TO_CALIBRATE': 'k_soil,total_soil_depth_multiplier'},
        LOG, settings)
    defaults = only_multiplier.get_initial_parameters()
    assert set(defaults) == set(only_multiplier.all_param_names) == {'k_soil', 'total_soil_depth_multiplier'}
    np.testing.assert_allclose(defaults['total_soil_depth_multiplier'], [1.0])
    # ... so the search starts at the model's own soil depth, not at mid-range (2.55x)
    start = only_multiplier.denormalize_parameters(only_multiplier.normalize_parameters(defaults))
    np.testing.assert_allclose(np.ravel(start['total_soil_depth_multiplier'])[0], 1.0)

    classic = SUMMAParameterManager(CONFIG, LOG, settings)
    defaults = classic.get_initial_parameters()
    assert set(defaults) == {'k_soil', 'total_mult', 'shape_factor'}
    np.testing.assert_allclose([defaults['total_mult'][0], defaults['shape_factor'][0]], [1.0, 1.0])
