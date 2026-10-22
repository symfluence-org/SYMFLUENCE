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
