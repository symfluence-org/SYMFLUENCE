"""StreamflowWorker: the shared scoring tail behind model-specific readers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.core.calibration.workers.streamflow_worker import StreamflowWorker
from symfluence.core.exceptions import ModelExecutionError

DAYS = pd.date_range('2001-01-01', '2002-12-31', freq='D')
OBS = 20 + 10 * np.sin(np.arange(len(DAYS)) / 58.0)


class _Worker(StreamflowWorker):
    """Reader returns (or raises) whatever the test hands it."""

    def __init__(self, sim, extra=None, **kwargs):
        super().__init__(**kwargs)
        self.sim, self.extra, self.seen = sim, extra, {}

    def apply_parameters(self, params, settings_dir, **kwargs):
        return True

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        return True

    def read_simulated_streamflow(self, output_dir, config, **kwargs):
        self.seen['read'] = kwargs
        if isinstance(self.sim, Exception):
            raise self.sim
        return self.sim

    def additional_metrics(self, output_dir, config, *, project_dir, **kwargs):
        self.seen['project_dir'] = project_dir
        if isinstance(self.extra, Exception):
            raise self.extra
        return self.extra or {}


@pytest.fixture
def config(tmp_path):
    obs_dir = tmp_path / 'domain_x' / 'observations' / 'streamflow' / 'preprocessed'
    obs_dir.mkdir(parents=True)
    pd.DataFrame({'datetime': DAYS, 'discharge_cms': OBS}).to_csv(
        obs_dir / 'x_streamflow_processed.csv', index=False)
    return {'DOMAIN_NAME': 'x', 'SYMFLUENCE_DATA_DIR': str(tmp_path)}


def _sim(scale=1.0, index=DAYS):
    return pd.Series(OBS[:len(index)] * scale, index=index)


def test_scores_reader_output_against_observations(tmp_path, config):
    worker = _Worker(_sim(), config=config)
    metrics = worker.calculate_metrics(tmp_path, config, sim_dir='run', settings_dir='s')
    assert metrics['kge'] == pytest.approx(1.0)
    assert metrics['nse'] == pytest.approx(1.0)
    assert worker.seen['read'] == {'sim_dir': 'run', 'settings_dir': 's'}
    assert worker.seen['project_dir'] == tmp_path / 'domain_x'


def test_calibration_period_limits_the_scored_window(tmp_path, config):
    # Perfect in 2001, doubled in 2002: only the window decides the score.
    sim = _sim()
    sim[sim.index.year == 2002] *= 2
    worker = _Worker(sim, config=config)
    in_2001 = dict(config, CALIBRATION_PERIOD='2001-01-01, 2001-12-31')
    in_2002 = dict(config, CALIBRATION_PERIOD='2002-01-01, 2002-12-31')
    assert worker.calculate_metrics(tmp_path, in_2001)['kge'] == pytest.approx(1.0)
    assert worker.calculate_metrics(tmp_path, in_2002)['kge'] < 0.5
    # An explicit period (evaluation scoring of a full-period run) wins.
    assert worker.calculate_metrics(
        tmp_path, in_2001, period='2002-01-01, 2002-12-31')['kge'] < 0.5


def test_requested_objective_is_computed(tmp_path, config):
    cfg = dict(config, OPTIMIZATION_METRIC='RMSE')
    metrics = _Worker(_sim(1.1), config=cfg).calculate_metrics(tmp_path, cfg)
    assert {'rmse', 'kge', 'nse'} <= set(metrics)
    assert metrics['rmse'] == pytest.approx(float(np.sqrt(np.mean((OBS * 0.1) ** 2))))


def test_additional_metrics_are_merged(tmp_path, config):
    metrics = _Worker(_sim(), extra={'snow_score': 0.4}, config=config).calculate_metrics(tmp_path, config)
    assert metrics['snow_score'] == 0.4 and metrics['kge'] == pytest.approx(1.0)


@pytest.mark.parametrize('sim, reason', [
    (None, 'No streamflow data'),
    (pd.Series(dtype=float), 'No streamflow data'),
    (ModelExecutionError('No statvar files found'), 'No statvar files found'),
    (OSError('disk went away'), 'disk went away'),
    (_sim(index=pd.date_range('1980-01-01', periods=40, freq='D')), 'Insufficient overlapping data'),
])
def test_unscorable_trial_is_a_penalty_with_a_reason(tmp_path, config, sim, reason):
    cfg = dict(config, PENALTY_SCORE=-5.0)
    metrics = _Worker(sim, config=cfg).calculate_metrics(tmp_path, cfg)
    assert metrics['kge'] == -5.0
    assert reason in metrics['error']


def test_missing_observations_is_a_penalty(tmp_path, config):
    cfg = dict(config, DOMAIN_NAME='elsewhere')
    metrics = _Worker(_sim(), config=cfg).calculate_metrics(tmp_path, cfg)
    assert metrics == {'kge': _Worker.DEFAULT_PENALTY_SCORE, 'error': 'No observations'}


def test_failing_additional_metric_fails_the_trial(tmp_path, config):
    worker = _Worker(_sim(), extra=ModelExecutionError('No statvar for cryosphere constraints'), config=config)
    metrics = worker.calculate_metrics(tmp_path, config)
    assert metrics == {'kge': worker.penalty_score, 'error': 'No statvar for cryosphere constraints'}


def test_reader_is_required():
    assert StreamflowWorker.__abstractmethods__ >= {'read_simulated_streamflow'}
