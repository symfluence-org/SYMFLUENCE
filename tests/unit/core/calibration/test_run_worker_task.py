"""Tests for the shared module-level calibration worker entry point.

``run_worker_task`` replaces ~20 hand-copied ``_evaluate_<model>_parameters_worker``
bodies. These tests pin its behaviour and the per-model variants each thin
wrapper passes, and check the wrappers stay picklable, importable module-level
functions (MPI strategies import them by ``__module__``/``__name__``).
"""
from __future__ import annotations

import importlib
import os
import pickle
import signal
from unittest.mock import MagicMock

import pytest

from symfluence.core.calibration.workers import base_worker
from symfluence.core.calibration.workers.base_worker import (
    MACOS_MALLOC_QUIET_ENV,
    BaseWorker,
    WorkerResult,
    WorkerTask,
    run_worker_task,
)
from symfluence.core.constants import ModelDefaults

ENV_KEYS = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NETCDF_DISABLE_LOCKING', 'HDF5_USE_FILE_LOCKING',
    'MallocStackLogging', 'HDF5_DISABLE_VERSION_CHECK',
)


class _OkWorker(BaseWorker):
    """Minimal worker that records how it was built and returns a fixed score."""

    instances: list = []

    def __init__(self, config=None, logger=None):
        self.built_with_config = config
        _OkWorker.instances.append(self)

    def evaluate(self, task: WorkerTask) -> WorkerResult:  # type: ignore[override]
        return WorkerResult(individual_id=task.individual_id, params=task.params, score=0.75)

    def apply_parameters(self, params, settings_dir, **kwargs):  # pragma: no cover
        return True

    def run_model(self, config, settings_dir, output_dir, **kwargs):  # pragma: no cover
        return True

    def calculate_metrics(self, output_dir, config, **kwargs):  # pragma: no cover
        return {}


class _BoomWorker(_OkWorker):
    def evaluate(self, task: WorkerTask) -> WorkerResult:  # type: ignore[override]
        raise RuntimeError('model blew up')


@pytest.fixture
def task_data(tmp_path):
    return {
        'individual_id': 7,
        'params': {'a': 1.0},
        'proc_id': 3,
        'config': {'DOMAIN_NAME': 'x'},
        'proc_settings_dir': str(tmp_path),
        'proc_output_dir': str(tmp_path),
    }


@pytest.fixture
def isolated(monkeypatch):
    """Clear worker env vars and stub signal/sleep so the test process is untouched."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    sig = MagicMock()
    sleep = MagicMock()
    uniform = MagicMock(return_value=0.0)
    monkeypatch.setattr(signal, 'signal', sig)
    monkeypatch.setattr(base_worker.time, 'sleep', sleep)
    monkeypatch.setattr('random.uniform', uniform)
    _OkWorker.instances.clear()
    return {'signal': sig, 'sleep': sleep, 'uniform': uniform}


class TestRunWorkerTask:
    def test_isolation_sets_env_signals_and_jitter(self, isolated, task_data):
        run_worker_task(_OkWorker, task_data, model_name='X')

        assert os.environ['OMP_NUM_THREADS'] == '1'
        assert os.environ['MKL_NUM_THREADS'] == '1'
        assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
        # Adopted from BaseWorker.setup_worker_isolation.
        assert os.environ['NETCDF_DISABLE_LOCKING'] == '1'
        assert os.environ['HDF5_USE_FILE_LOCKING'] == 'FALSE'
        assert 'MallocStackLogging' not in os.environ
        installed = {c.args[0] for c in isolated['signal'].call_args_list}
        assert installed == {signal.SIGTERM, signal.SIGINT}
        isolated['uniform'].assert_called_once_with(0.1, 0.5)
        isolated['sleep'].assert_called_once()

    def test_extra_env_and_custom_jitter(self, isolated, task_data):
        run_worker_task(
            _OkWorker, task_data,
            extra_env={**MACOS_MALLOC_QUIET_ENV, 'HDF5_DISABLE_VERSION_CHECK': '1'},
            start_jitter=(0.1, 0.8),
        )
        assert os.environ['MallocStackLogging'] == '0'
        assert os.environ['HDF5_DISABLE_VERSION_CHECK'] == '1'
        isolated['uniform'].assert_called_once_with(0.1, 0.8)

    def test_no_jitter(self, isolated, task_data):
        run_worker_task(_OkWorker, task_data, start_jitter=None)
        isolated['sleep'].assert_not_called()
        assert os.environ['HDF5_USE_FILE_LOCKING'] == 'FALSE'

    def test_signal_value_error_is_tolerated(self, isolated, task_data):
        isolated['signal'].side_effect = ValueError('not main thread')
        result = run_worker_task(_OkWorker, task_data)
        assert result['score'] == 0.75

    def test_no_isolation_touches_nothing(self, isolated, task_data):
        run_worker_task(_OkWorker, task_data, isolate=False)
        for key in ENV_KEYS:
            assert key not in os.environ
        isolated['signal'].assert_not_called()
        isolated['sleep'].assert_not_called()

    def test_success_returns_legacy_dict(self, isolated, task_data):
        result = run_worker_task(_OkWorker, task_data)
        assert result['individual_id'] == 7
        assert result['params'] == {'a': 1.0}
        assert result['score'] == 0.75
        assert not result.get('error')
        assert _OkWorker.instances[-1].built_with_config == {'DOMAIN_NAME': 'x'}

    def test_pass_config_false_builds_without_config(self, isolated, task_data):
        run_worker_task(_OkWorker, task_data, isolate=False, pass_config=False)
        assert _OkWorker.instances[-1].built_with_config is None

    def test_exception_becomes_penalty_dict(self, isolated, task_data):
        result = run_worker_task(_BoomWorker, task_data, model_name='VIC')
        assert result['individual_id'] == 7
        assert result['params'] == {'a': 1.0}
        assert result['proc_id'] == 3
        assert result['score'] == ModelDefaults.PENALTY_SCORE
        assert result['error'].startswith('VIC worker exception: model blew up\n')
        assert 'Traceback' in result['error']

    def test_error_prefix_override(self, isolated, task_data):
        result = run_worker_task(
            _BoomWorker, task_data, error_prefix='Critical HYPE worker exception'
        )
        assert result['error'].startswith('Critical HYPE worker exception: model blew up')

    def test_penalty_defaults_for_missing_keys(self, isolated):
        result = run_worker_task(_BoomWorker, {}, model_name='X')
        assert result['individual_id'] == -1
        assert result['proc_id'] == -1
        assert result['params'] == {}

    def test_catch_exceptions_false_propagates(self, isolated, task_data):
        with pytest.raises(RuntimeError, match='model blew up'):
            run_worker_task(_BoomWorker, task_data, isolate=False, catch_exceptions=False)


_MALLOC = {'extra_env': MACOS_MALLOC_QUIET_ENV}
_BARE = {'isolate': False, 'catch_exceptions': False}

# (module, entry function, worker class, kwargs passed to run_worker_task)
DELEGATES = [
    ('symfluence.models.clm.calibration.worker', '_evaluate_clm_parameters_worker', 'CLMWorker', {'model_name': 'CLM', **_MALLOC}),
    ('symfluence.models.gsflow.calibration.worker', '_evaluate_gsflow_parameters_worker', 'GSFLOWWorker', {'model_name': 'GSFLOW', **_MALLOC}),
    ('symfluence.models.mhm.calibration.worker', '_evaluate_mhm_parameters_worker', 'MHMWorker', {'model_name': 'mHM', **_MALLOC}),
    ('symfluence.models.noahmp.calibration.worker', '_evaluate_noahmp_parameters_worker', 'NoahMPWorker', {'model_name': 'Noah-MP', **_MALLOC}),
    ('symfluence.models.prms.calibration.worker', '_evaluate_prms_parameters_worker', 'PRMSWorker', {'model_name': 'PRMS', **_MALLOC}),
    ('symfluence.models.vic.calibration.worker', '_evaluate_vic_parameters_worker', 'VICWorker', {'model_name': 'VIC', **_MALLOC}),
    ('symfluence.models.watflood.calibration.worker', '_evaluate_watflood_parameters_worker', 'WATFLOODWorker', {'model_name': 'WATFLOOD', **_MALLOC}),
    ('symfluence.models.wflow.calibration.worker', '_evaluate_wflow_parameters_worker', 'WflowWorker', {'model_name': 'Wflow', **_MALLOC}),
    ('symfluence.models.wrfhydro.calibration.worker', '_evaluate_wrfhydro_parameters_worker', 'WRFHydroWorker', {'model_name': 'WRF-Hydro', **_MALLOC}),
    ('symfluence.models.crhm.calibration.worker', '_evaluate_crhm_parameters_worker', 'CRHMWorker', {'model_name': 'CRHM'}),
    ('symfluence.models.swat.calibration.worker', '_evaluate_swat_parameters_worker', 'SWATWorker', {'model_name': 'SWAT'}),
    ('symfluence.models.hype.calibration.worker', '_evaluate_hype_parameters_worker', 'HYPEWorker', {'error_prefix': 'Critical HYPE worker exception'}),
    ('symfluence.models.rhessys.calibration.worker', '_evaluate_rhessys_parameters_worker', 'RHESSysWorker', {
        'extra_env': {'HDF5_DISABLE_VERSION_CHECK': '1'}, 'start_jitter': (0.1, 0.8),
        'error_prefix': 'Critical RHESSys worker exception'}),
    ('symfluence.core.calibration.coupled.worker', '_evaluate_coupled_parameters_worker', 'CoupledModelWorker', {
        'start_jitter': None, 'error_prefix': 'coupled worker exception'}),
    ('symfluence.models.pihm.calibration.worker', '_evaluate_pihm_parameters_worker', 'PIHMWorker', {'model_name': 'PIHM', 'isolate': False}),
    ('symfluence.models.clmparflow.calibration.worker', '_evaluate_clmparflow_parameters_worker', 'CLMParFlowWorker', _BARE),
    ('symfluence.models.fuse.calibration.worker', '_evaluate_fuse_parameters_worker', 'FUSEWorker', _BARE),
    ('symfluence.models.gr.calibration.worker', '_evaluate_gr_parameters_worker', 'GRWorker', _BARE),
    ('symfluence.models.ngen.calibration.worker', '_evaluate_ngen_parameters_worker', 'NgenWorker', _BARE),
    ('symfluence.models.parflow.calibration.worker', '_evaluate_parflow_parameters_worker', 'ParFlowWorker', _BARE),
    ('symfluence.models.modflow.calibration.worker', '_evaluate_coupled_gw_worker', 'CoupledGWWorker', _BARE),
    ('symfluence.models.mesh.calibration.worker', '_evaluate_mesh_parameters_worker', 'MESHWorker', {**_BARE, 'pass_config': False}),
]


@pytest.mark.parametrize('module_name,func_name,cls_name,kwargs', DELEGATES, ids=[d[1] for d in DELEGATES])
class TestModelEntryFunctions:
    def test_delegates_with_model_variant(self, monkeypatch, module_name, func_name, cls_name, kwargs):
        module = importlib.import_module(module_name)
        fake = MagicMock(return_value={'score': 1.0})
        monkeypatch.setattr(module, 'run_worker_task', fake)
        task = {'individual_id': 1}

        assert getattr(module, func_name)(task) == {'score': 1.0}
        fake.assert_called_once_with(getattr(module, cls_name), task, **kwargs)

    def test_picklable_module_level_function(self, module_name, func_name, cls_name, kwargs):
        fn = getattr(importlib.import_module(module_name), func_name)
        # MPI strategies re-import the function from these two attributes.
        assert fn.__module__ == module_name
        assert fn.__name__ == func_name
        assert pickle.loads(pickle.dumps(fn)) is fn
