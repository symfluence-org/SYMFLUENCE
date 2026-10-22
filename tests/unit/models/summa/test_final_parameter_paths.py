from __future__ import annotations

import logging
from unittest.mock import patch

from symfluence.models.summa.calibration.worker import SUMMAWorker


def test_final_parameters_resolve_sibling_routing_settings(tmp_path):
    worker = SUMMAWorker({}, logging.getLogger('test'))
    settings = tmp_path / 'settings/SUMMA'
    def apply(params, task, actual_settings, logger, debug):
        assert actual_settings == settings
        assert task['mizuroute_settings_dir'] == str(settings.parent / 'mizuRoute')
        assert task['params'] == {'velo': 2.}
        return True
    with patch('symfluence.models.summa.calibration.worker_impl._apply_parameters_worker', side_effect=apply):
        assert worker.apply_parameters({'velo': 2.}, settings, config={'CALIBRATE_MIZUROUTE': True})


def test_explicit_trial_routing_path_is_preserved(tmp_path):
    worker = SUMMAWorker({}, logging.getLogger('test'))
    def apply(params, task, *args):
        assert task['mizuroute_settings_dir'] == '/explicit/routing'
        return True
    with patch('symfluence.models.summa.calibration.worker_impl._apply_parameters_worker', side_effect=apply):
        assert worker.apply_parameters({'velo': 2.}, tmp_path, task_data={'mizuroute_settings_dir':'/explicit/routing'})


def test_final_evaluation_passes_original_soil_depths(tmp_path):
    # Without original depths the worker skips the depth update, so the final
    # run used uncalibrated soil depths (Tungnaa lumped: KGE 0.742 -> 0.732).
    from types import SimpleNamespace

    import numpy as np

    from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer

    seen = {}

    def apply(params, task, *args):
        seen.update(task)
        return True
    optimizer = SimpleNamespace(
        worker=SUMMAWorker({}, logging.getLogger('test')), optimization_settings_dir=tmp_path, config={},
        param_manager=SimpleNamespace(original_depths=np.array([0.2, 0.3, 0.5])), logger=logging.getLogger('test'))
    with patch('symfluence.models.summa.calibration.worker_impl._apply_parameters_worker', side_effect=apply):
        assert BaseModelOptimizer._apply_best_parameters_for_final(optimizer, {'total_mult': 0.5})
    assert seen['original_depths'] == [0.2, 0.3, 0.5]
