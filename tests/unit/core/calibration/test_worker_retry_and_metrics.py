"""Workers: what counts as transient, and which metrics the in-memory worker produces."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from symfluence.core import logging_utils
from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerResult, WorkerTask
from symfluence.core.calibration.workers.inmemory_worker import InMemoryModelWorker


class _CrashingWorker(BaseWorker):
    """Model 'crashes': no output file, so metric extraction cannot open it."""

    def __init__(self, message, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.runs = 0

    def apply_parameters(self, params, settings_dir, **kwargs):
        return True

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        return True

    def calculate_metrics(self, output_dir, config, **kwargs):
        return {}

    def _evaluate_once(self, task):
        self.runs += 1
        raise IOError(self.message)


def _task(tmp_path, config=None):
    return WorkerTask(1, {'m': 1.3}, 0, config or {}, tmp_path, tmp_path)


@pytest.mark.parametrize('message', [
    "[Errno 2] No such file or directory: 'run_1_timestep.nc'",
    "[Errno 13] Permission denied: 'output.nc'",
])
def test_missing_output_is_not_retried(tmp_path, message):
    """A crashed parameter set must not be rerun WORKER_MAX_RETRIES more times."""
    worker = _CrashingWorker(message, config={'WORKER_BASE_DELAY': 0})
    with pytest.raises(IOError):
        worker._evaluate_with_retry(_task(tmp_path))
    assert worker.runs == 1


def test_stale_file_handle_is_still_retried(tmp_path, monkeypatch):
    monkeypatch.setattr('symfluence.core.calibration.workers.base_worker.time.sleep', lambda _s: None)
    worker = _CrashingWorker('[Errno 116] Stale file handle', config={})
    with pytest.raises(IOError):
        worker._evaluate_with_retry(_task(tmp_path))
    assert worker.runs == worker.max_retries + 1


class _Toy(InMemoryModelWorker):
    def _get_model_name(self):
        return 'TOY'

    def _get_forcing_subdir(self):
        return 'x'

    def _get_forcing_variable_map(self):
        return {}

    def _load_forcing(self, task=None):
        return True

    def _run_simulation(self, forcing, params, **kwargs):
        return self._obs0 * params['m']

    def _get_warmup_days_config(self):
        return 0

    def initialize(self, task=None):
        rng = np.random.default_rng(0)
        self._obs0 = rng.gamma(2.0, 2.0, 400)
        self._observations = self._obs0.copy()
        self._time_index = pd.date_range('2000-01-01', periods=400)
        self._forcing = {}
        self._initialized = True
        return True


def _evaluate(tmp_path, metric, **extra):
    config = {'OPTIMIZATION_METRIC': metric, **extra}
    worker = _Toy(config=config)
    worker.initialize()
    result = worker.evaluate(_task(tmp_path, config))
    assert isinstance(result, WorkerResult)
    return worker, result


def test_inmemory_worker_scores_minimise_metrics(tmp_path):
    """RMSE used to be absent from the metrics dict -> every trial = penalty."""
    worker, result = _evaluate(tmp_path, 'RMSE')
    expected_rmse = float(np.sqrt(np.mean((0.3 * worker._obs0) ** 2)))
    assert result.metrics['rmse'] == pytest.approx(expected_rmse, rel=1e-6)
    assert result.score == pytest.approx(-expected_rmse, rel=1e-6)

    _, pbias = _evaluate(tmp_path, 'PBIAS')
    assert 'pbias' in pbias.metrics
    assert pbias.score == pytest.approx(-abs(pbias.metrics['pbias']))
    assert pbias.score > worker.penalty_score


def test_inmemory_worker_scores_the_configured_transformed_metric(tmp_path):
    """KGE_LOG used to be silently replaced by plain KGE."""
    _, kge = _evaluate(tmp_path, 'KGE')
    _, kge_log = _evaluate(tmp_path, 'KGE_LOG')
    assert 'kge_log' in kge_log.metrics
    assert kge_log.score == pytest.approx(kge_log.metrics['kge_log'])
    assert kge_log.score != pytest.approx(kge.score)
    # kge/nse are still always reported
    assert {'kge', 'nse'} <= set(kge_log.metrics)


def test_inmemory_worker_composite_components_are_computed(tmp_path):
    _, result = _evaluate(tmp_path, 'COMPOSITE', COMPOSITE_METRIC={'KGE': 0.5, 'KGE_LOG': 0.5})
    assert 'kge_log' in result.metrics
    expected = 0.5 * result.metrics['kge'] + 0.5 * result.metrics['kge_log']
    assert result.score == pytest.approx(expected)


def test_loss_for_a_minimise_metric_is_not_inverted(tmp_path):
    worker = _Toy(config={'OPTIMIZATION_METRIC': 'RMSE'})
    worker.initialize()
    near = worker._evaluate_loss({'m': 1.05}, 'rmse')
    far = worker._evaluate_loss({'m': 1.9}, 'rmse')
    assert near < far  # lower loss for the better parameter set


def test_fallback_to_another_metric_warns_once(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(logging_utils, '_log_once_seen', set())
    worker = _CrashingWorker('unused', config={})
    metrics = {'kge': 0.7, 'nse': 0.6}
    config = {'OPTIMIZATION_METRIC': 'KGE_LOG'}
    with caplog.at_level(logging.WARNING):
        first = worker._extract_primary_score(metrics, config)
        second = worker._extract_primary_score(metrics, config)
    assert first == second == pytest.approx(0.7)
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "scoring on 'kge' instead" in r.getMessage()]
    assert len(warnings) == 1
