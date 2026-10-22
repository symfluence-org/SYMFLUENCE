from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer


def test_improving_checkpoint_persists_current_best(tmp_path):
    state = {'history': [], 'best': {'score': 0.7615, 'params': {'x': 9}}}
    def record(i, score, params, **kwargs):
        state['history'].append({'iteration': i, 'score': score, 'params': params})
    def update(score, params, i):
        if score > state['best']['score']:
            state['best'] = {'iteration': i, 'score': score, 'params': params}
    fake = SimpleNamespace(
        _get_config_value=lambda *a, **k: 10,
        get_crash_stats=lambda: {'crash_count': 0, 'crash_rate': 0},
        record_iteration=record, update_best=update,
        save_results=lambda *a, **k: (tmp_path/'history.json').write_text(json.dumps(state['history'])),
        save_best_params=lambda *a: (tmp_path/'best.json').write_text(json.dumps(state['best'])),
        logger=logging.getLogger('test'), log_initial_population=None,
        results_dir=tmp_path, experiment_id='test',
    )
    callbacks, _ = BaseModelOptimizer._build_algorithm_callbacks(fake, 'DDS')
    callbacks['record_iteration'](10, .770855, {'x': 10})
    callbacks['update_best'](.770855, {'x': 10}, 10)
    last = json.loads((tmp_path/'history.json').read_text())[-1]
    best = json.loads((tmp_path/'best.json').read_text())
    assert last == best
    callbacks['record_iteration'](20, .770855, {'x': 10})
    callbacks['update_best'](.770855, {'x': 10}, 20)
    assert json.loads((tmp_path/'best.json').read_text()) == best
