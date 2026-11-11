"""One set of SUMMA calibration file writers, shared by the manager and the worker.

Regression tests for three defects of the previously separate copies:
pseudo-parameters written into trialParams.nc as HRU variables, a KeyError when
only ``total_soil_depth_multiplier`` was calibrated, and trial parameters
written to hard-coded file names that SUMMA does not read when
``SETTINGS_SUMMA_ATTRIBUTES`` / ``_COLDSTATE`` / ``_TRIALPARAMS`` are set.
"""
from __future__ import annotations

import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from symfluence.models.summa.calibration.optimizer_mixin import SUMMAOptimizerMixin
from symfluence.models.summa.calibration.parameter_files import (
    calculate_soil_depths,
    resolve_settings_file_names,
    select_trial_params,
)
from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
from symfluence.models.summa.calibration.worker_impl import _apply_parameters_worker

pytestmark = [pytest.mark.unit, pytest.mark.quick]

LOG = logging.getLogger('test_summa_parameter_files')
HRU_IDS = [11, 12, 21, 22]
GRU_IDS = [1, 2]
DEPTHS = np.array([0.025, 0.075, 0.15, 0.25, 0.5])
PSEUDO = {'total_mult', 'shape_factor', 'total_soil_depth_multiplier', 'velo', 'diff', 'precip_mult'}
CUSTOM = {
    'SETTINGS_SUMMA_ATTRIBUTES': 'my_attributes.nc',
    'SETTINGS_SUMMA_COLDSTATE': 'my_coldState.nc',
    'SETTINGS_SUMMA_TRIALPARAMS': 'my_trialParams.nc',
}


def _settings(tmp_path: Path, attributes='attributes.nc', coldstate='coldState.nc') -> Path:
    settings = tmp_path / 'settings' / 'SUMMA'
    settings.mkdir(parents=True)
    with nc.Dataset(settings / attributes, 'w') as ds:
        ds.createDimension('hru', len(HRU_IDS))
        ds.createDimension('gru', len(GRU_IDS))
        ds.createVariable('hruId', 'i4', ('hru',))[:] = HRU_IDS
        ds.createVariable('gruId', 'i4', ('gru',))[:] = GRU_IDS
    with nc.Dataset(settings / coldstate, 'w') as ds:
        ds.createDimension('hru', len(HRU_IDS))
        ds.createDimension('midToto', len(DEPTHS))
        ds.createDimension('ifcToto', len(DEPTHS) + 1)
        ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))[:] = np.tile(DEPTHS[:, None], (1, len(HRU_IDS)))
        ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))[:] = np.tile(
            np.r_[0, DEPTHS.cumsum()][:, None], (1, len(HRU_IDS)))
    (settings / 'localParamInfo.txt').write_text(
        "k_soil | 1.0d-5 | 1.0d-7 | 1.0d-3\ntheta_sat | 0.45 | 0.30 | 0.60\n")
    (settings / 'basinParamInfo.txt').write_text("routingGammaShape | 2.5 | 2.0 | 3.0\n")
    mizu = settings.parent / 'mizuRoute'
    mizu.mkdir()
    (mizu / 'param.nml.default').write_text("&HSLOPE\n  fshape = 2.5\n  tscale = 86400\n/\n&IRF_UH\n  velo = 1.5\n  diff = 800.0\n/\n")
    return settings


def _trial(path: Path) -> dict:
    with nc.Dataset(path) as ds:
        return {name: (var.dimensions, np.asarray(var[:]).copy()) for name, var in ds.variables.items()}


def _layer_depths(path: Path) -> np.ndarray:
    with nc.Dataset(path) as ds:
        return np.asarray(ds['mLayerDepth'][:, 0]).copy()


def _task(config: dict, settings: Path) -> dict:
    return {'config': config, 'original_depths': DEPTHS.tolist(),
            'mizuroute_settings_dir': str(settings.parent / 'mizuRoute')}


def _debug() -> dict:
    return {'files_checked': [], 'errors': []}


# ---------------------------------------------------------------------------
# configured file names
# ---------------------------------------------------------------------------

def test_file_names_default_to_the_standard_ones():
    assert resolve_settings_file_names({}) == ('attributes.nc', 'coldState.nc', 'trialParams.nc')
    assert resolve_settings_file_names(None) == ('attributes.nc', 'coldState.nc', 'trialParams.nc')
    assert resolve_settings_file_names(CUSTOM) == ('my_attributes.nc', 'my_coldState.nc', 'my_trialParams.nc')


def test_worker_writes_the_configured_files(tmp_path):
    settings = _settings(tmp_path, attributes='my_attributes.nc', coldstate='my_coldState.nc')
    config = {**CUSTOM, 'PARAMS_TO_CALIBRATE': 'k_soil', 'CALIBRATE_DEPTH': True}
    params = {'k_soil': np.full(4, 2e-5), 'total_mult': np.array([2.0]), 'shape_factor': np.array([1.0])}
    debug = _debug()

    assert _apply_parameters_worker(params, _task(config, settings), settings, LOG, debug), debug

    assert not (settings / 'trialParams.nc').exists()
    np.testing.assert_allclose(_trial(settings / 'my_trialParams.nc')['k_soil'][1], 2e-5)
    np.testing.assert_allclose(_layer_depths(settings / 'my_coldState.nc'), DEPTHS * 2.0)


def test_manager_writes_the_configured_files(tmp_path):
    settings = _settings(tmp_path, attributes='my_attributes.nc', coldstate='my_coldState.nc')
    config = {**CUSTOM, 'PARAMS_TO_CALIBRATE': 'k_soil', 'BASIN_PARAMS_TO_CALIBRATE': '', 'CALIBRATE_DEPTH': True}
    manager = SUMMAParameterManager(config, LOG, settings)

    assert manager.update_model_files(
        {'k_soil': np.full(4, 2e-5), 'total_mult': np.array([2.0]), 'shape_factor': np.array([1.0])})

    assert not (settings / 'trialParams.nc').exists()
    np.testing.assert_allclose(_trial(settings / 'my_trialParams.nc')['k_soil'][1], 2e-5)
    np.testing.assert_allclose(_layer_depths(settings / 'my_coldState.nc'), DEPTHS * 2.0)


def test_optimizer_copies_the_configured_files(tmp_path):
    settings = _settings(tmp_path, attributes='my_attributes.nc', coldstate='my_coldState.nc')
    for name in ('modelDecisions.txt', 'outputControl.txt', 'fileManager.txt', 'my_trialParams.nc'):
        (settings / name).write_text('x')

    class Optimizer(SUMMAOptimizerMixin):
        pass

    opt = Optimizer()
    opt.project_dir = tmp_path
    opt.optimization_settings_dir = tmp_path / 'opt'
    opt.optimization_settings_dir.mkdir()
    opt.logger = LOG
    opt._config = None
    opt.config_dict = dict(CUSTOM)
    opt._copy_mizuroute_settings = lambda: None

    opt._copy_summa_settings_files()

    copied = {p.name for p in opt.optimization_settings_dir.iterdir()}
    assert {'my_attributes.nc', 'my_coldState.nc', 'my_trialParams.nc'} <= copied


# ---------------------------------------------------------------------------
# pseudo-parameters stay out of trialParams
# ---------------------------------------------------------------------------

def test_select_trial_params_drops_every_pseudo_parameter():
    params = {name: 1.0 for name in PSEUDO} | {'k_soil': 1e-5, 'mann_n': 0.03, 'tscale': 3600}
    assert set(select_trial_params(params)) == {'k_soil'}


@pytest.mark.parametrize('flags', [
    {'CALIBRATE_DEPTH': True, 'CALIBRATE_MIZUROUTE': True, 'MIZUROUTE_PARAMS_TO_CALIBRATE': 'velo,diff'},
    {},  # pseudo-parameters present although their calibration flags are off
])
def test_worker_never_writes_pseudo_parameters(tmp_path, flags):
    settings = _settings(tmp_path)
    config = {'PARAMS_TO_CALIBRATE': 'k_soil', **flags}
    params = {'k_soil': np.full(4, 2e-5), 'total_mult': np.array([1.2]), 'shape_factor': np.array([1.1]),
              'velo': 2.0, 'diff': 500.0}
    debug = _debug()

    assert _apply_parameters_worker(params, _task(config, settings), settings, LOG, debug), debug

    assert set(_trial(settings / 'trialParams.nc')) == {'hruId', 'gruId', 'k_soil'}


def test_manager_never_writes_pseudo_parameters(tmp_path):
    """The path taken when regionalisation is active (worker -> manager.update_model_files)."""
    settings = _settings(tmp_path)
    config = {'PARAMS_TO_CALIBRATE': 'k_soil', 'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaShape',
              'CALIBRATE_DEPTH': True, 'CALIBRATE_MIZUROUTE': True, 'MIZUROUTE_PARAMS_TO_CALIBRATE': 'velo,diff'}
    manager = SUMMAParameterManager(config, LOG, settings)

    assert manager.update_model_files({
        'k_soil': np.full(4, 2e-5), 'routingGammaShape': np.array([2.7]),
        'total_mult': np.array([1.2]), 'shape_factor': np.array([1.1]), 'velo': 2.0, 'diff': 500.0})

    trial = _trial(settings / 'trialParams.nc')
    assert set(trial) == {'hruId', 'gruId', 'k_soil', 'routingGammaShape'}
    assert trial['routingGammaShape'][0] == ('gru',)
    assert 'velo = 2.000000' in (settings.parent / 'mizuRoute' / 'param.nml.default').read_text()


# ---------------------------------------------------------------------------
# soil depth
# ---------------------------------------------------------------------------

def test_manager_accepts_total_soil_depth_multiplier_alone(tmp_path):
    settings = _settings(tmp_path)
    config = {'PARAMS_TO_CALIBRATE': 'k_soil,total_soil_depth_multiplier', 'BASIN_PARAMS_TO_CALIBRATE': ''}
    manager = SUMMAParameterManager(config, LOG, settings)
    assert manager.depth_params == ['total_soil_depth_multiplier']

    assert manager.update_model_files(
        {'k_soil': np.full(4, 2e-5), 'total_soil_depth_multiplier': np.array([1.5])})

    np.testing.assert_allclose(_layer_depths(settings / 'coldState.nc'), DEPTHS * 1.5)
    assert set(_trial(settings / 'trialParams.nc')) == {'hruId', 'gruId', 'k_soil'}


def test_worker_accepts_total_soil_depth_multiplier_alone(tmp_path):
    settings = _settings(tmp_path)
    config = {'PARAMS_TO_CALIBRATE': 'k_soil,total_soil_depth_multiplier'}
    params = {'k_soil': np.full(4, 2e-5), 'total_soil_depth_multiplier': np.array([1.5])}
    debug = _debug()

    assert _apply_parameters_worker(params, _task(config, settings), settings, LOG, debug), debug

    np.testing.assert_allclose(_layer_depths(settings / 'coldState.nc'), DEPTHS * 1.5)


def test_soil_depth_shape_keeps_total_depth_and_orders_layers():
    deeper = calculate_soil_depths(np.ones(5), 1.0, 3.0)
    shallower = calculate_soil_depths(np.ones(5), 1.0, 1 / 3.0)
    np.testing.assert_allclose(deeper.sum(), 5.0)
    np.testing.assert_allclose(deeper, shallower[::-1])
    assert np.all(np.diff(deeper) > 0)
    np.testing.assert_allclose(calculate_soil_depths(DEPTHS, 2.0, 1.0), DEPTHS * 2.0)


def test_manager_and_worker_write_the_same_files(tmp_path):
    config = {'PARAMS_TO_CALIBRATE': 'k_soil,theta_sat', 'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaShape',
              'CALIBRATE_DEPTH': True, 'CALIBRATE_MIZUROUTE': True, 'MIZUROUTE_PARAMS_TO_CALIBRATE': 'velo,diff'}
    params = {'k_soil': np.array([1e-5, 2e-5, 3e-5, 4e-5]), 'theta_sat': np.array([0.5]),
              'routingGammaShape': np.array([2.7]), 'total_mult': np.array([1.3]),
              'shape_factor': np.array([1.7]), 'velo': 2.25, 'diff': 1234.5}

    via_worker = _settings(tmp_path / 'worker')
    debug = _debug()
    assert _apply_parameters_worker(params, _task(config, via_worker), via_worker, LOG, debug), debug

    via_manager = _settings(tmp_path / 'manager')
    assert SUMMAParameterManager(config, LOG, via_manager).update_model_files(params)

    worker_trial, manager_trial = _trial(via_worker / 'trialParams.nc'), _trial(via_manager / 'trialParams.nc')
    assert set(worker_trial) == set(manager_trial)
    for name, (dims, values) in worker_trial.items():
        assert manager_trial[name][0] == dims
        np.testing.assert_array_equal(manager_trial[name][1], values)
    np.testing.assert_array_equal(_layer_depths(via_worker / 'coldState.nc'),
                                  _layer_depths(via_manager / 'coldState.nc'))
    assert ((via_worker.parent / 'mizuRoute' / 'param.nml.default').read_text()
            == (via_manager.parent / 'mizuRoute' / 'param.nml.default').read_text())


def test_regionalised_worker_uses_task_depths_not_rescaled_coldstate(tmp_path):
    """A manager built inside a worker must not take already-rescaled layers as the originals."""
    from symfluence.models.summa.calibration.worker import SUMMAWorker

    settings = _settings(tmp_path)
    with nc.Dataset(settings / 'coldState.nc', 'r+') as ds:
        ds['mLayerDepth'][:] = np.asarray(ds['mLayerDepth'][:]) * 3.0  # left by an earlier trial

    worker = SUMMAWorker({}, LOG)
    worker._cached_pm = SUMMAParameterManager(
        {'PARAMS_TO_CALIBRATE': 'k_soil', 'BASIN_PARAMS_TO_CALIBRATE': '', 'CALIBRATE_DEPTH': True}, LOG, settings)
    seen = {}

    def record(params):
        seen['depths'] = np.asarray(worker._cached_pm.original_depths).copy()
        return True

    worker._cached_pm.update_model_files = record
    assert worker.apply_parameters({'k_soil_a': 1.0}, settings, original_depths=DEPTHS.tolist())
    np.testing.assert_allclose(seen['depths'], DEPTHS)
