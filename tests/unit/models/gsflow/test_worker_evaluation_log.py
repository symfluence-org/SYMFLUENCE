"""GSFLOW calibration worker per-run evaluation log."""
from __future__ import annotations

import json
import logging

from symfluence.models.gsflow.calibration.worker import GSFLOWWorker


def test_failed_run_is_logged_with_params(tmp_path):
    log = tmp_path / 'evaluations.jsonl'
    config = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'GSFLOW_INSTALL_PATH': str(tmp_path / 'missing'),
              'GSFLOW_EVALUATION_LOG': str(log)}
    worker = GSFLOWWorker(config, logging.getLogger('test'))
    worker._trial_params = {'K': 2.0}
    assert not worker.run_model(config, tmp_path / 'settings', tmp_path / 'out')
    worker.run_model(config, tmp_path / 'settings', tmp_path / 'out')
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 2 and records[0]['ok'] is False and records[0]['params'] == {'K': 2.0}
    assert records[0]['seconds'] >= 0 and records[0]['run_dir'].endswith('settings')


def test_no_log_without_setting(tmp_path):
    config = {'SYMFLUENCE_DATA_DIR': str(tmp_path), 'GSFLOW_INSTALL_PATH': str(tmp_path / 'missing')}
    GSFLOWWorker(config, logging.getLogger('test')).run_model(config, tmp_path / 's', tmp_path / 'o')
    assert not list(tmp_path.glob('*.jsonl'))
