from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.models.summa.calibration.worker import SUMMAWorker


def test_final_evaluation_passes_original_soil_depths(tmp_path):
    # Without original depths the worker skips the depth update, so the final
    # run used uncalibrated soil depths (Tungnaa lumped: KGE 0.742 -> 0.732).
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
