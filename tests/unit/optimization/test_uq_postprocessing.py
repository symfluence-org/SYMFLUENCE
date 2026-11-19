# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Persisted samples, predictive quantiles and their scores after DREAM / GLUE."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.optimizers.uncertainty import (
    UncertaintyPostprocessor,
    extract_uq_samples,
)
from symfluence.core.mixins.config import resolve_config_value

pytestmark = pytest.mark.unit

INDEX = pd.date_range('2001-01-01', '2002-12-31', freq='D')
BASE = 5 + 4 * np.sin(np.arange(INDEX.size) / 20.0) ** 2
OBS = pd.Series(BASE * 0.5 * np.exp(np.random.default_rng(0).normal(0, 0.1, INDEX.size)), index=INDEX)


def _fake_optimizer(tmp_path, **cfg):
    config = {'CALIBRATION_PERIOD': '2001-01-01, 2001-12-31',
              'EVALUATION_PERIOD': '2002-01-01, 2002-12-31', 'RANDOM_SEED': 3, **cfg}
    out_dir = tmp_path / 'final_evaluation'

    def run_final_evaluation(params):
        out_dir.mkdir(parents=True, exist_ok=True)
        if params['scale'] > 0.95:  # one member "crashes"
            return None
        (out_dir / 'q.csv').write_text(str(params['scale']))
        return {'success': True}

    target = SimpleNamespace(
        get_simulation_files=lambda d: list(d.glob('q.csv')),
        extract_simulated_data=lambda files: pd.Series(
            float(files[0].read_text()) * BASE, index=INDEX),
        _load_observed_data=lambda: OBS,
    )
    pm = SimpleNamespace(
        all_param_names=['scale'],
        denormalize_parameters=lambda x: {'scale': float(x[0])},
    )
    return SimpleNamespace(
        config=config, logger=logging.getLogger('test_uq'), results_dir=tmp_path,
        param_manager=pm, calibration_target=target, random_seed=3,
        target_metric='LOGLIK_BC_AR1', run_final_evaluation=run_final_evaluation,
        _final_eval_daily_series=BaseModelOptimizer._final_eval_daily_series,
        _get_config_value=lambda acc, default=None, dict_key=None: resolve_config_value(
            config, acc, dict_key=dict_key),
    )


def _dream_result(n=200, n_chains=4):
    rng = np.random.default_rng(1)
    samples = np.clip(rng.normal(0.5, 0.03, (n, 1)), 0, 1)
    samples[0, 0] = 0.99  # the crashing member
    return {
        'posterior_samples': samples,
        'posterior_log_likelihood': rng.normal(-100, 1, n),
        'posterior_fitness': rng.normal(-100, 1, n),
        'n_chains': n_chains, 'likelihood': 'formal',
    }


def test_dream_writes_samples_quantiles_and_scores(tmp_path):
    opt = _fake_optimizer(tmp_path, UQ_PREDICTIVE_RUNS=200, UQ_RESIDUAL_DRAWS=5)
    written = UncertaintyPostprocessor(opt).run(_dream_result(), 'dream')
    assert set(written) == {'samples', 'quantiles', 'scores'}

    ds = xr.open_dataset(written['samples'])
    assert ds.sizes['sample'] == 200 and list(ds['parameter'].values) == ['scale']
    assert ds.attrs['kind'] == 'posterior'
    assert set(ds['chain'].values) == {0, 1, 2, 3}
    assert ds.sizes['member'] == 199  # crashed member dropped, never read stale
    assert 0 not in ds['member_sample'].values
    ds.close()

    table = pd.read_csv(written['quantiles'], index_col=0, parse_dates=True)
    for band in ('parameter', 'total'):
        cols = [f'{band}_q{q:g}' for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
        assert set(cols) <= set(table.columns)
        assert (np.diff(table[cols].to_numpy(), axis=1) >= -1e-9).all()

    scores = json.loads(written['scores'].read_text())['scores']
    for band in ('parameter', 'total'):
        assert set(scores[band]) == {'Calib', 'Eval'}
        assert set(scores[band]['Calib']) == {'coverage_90', 'width_90', 'crps', 'n_obs'}
    # Adding the residual error model widens the band and covers far more
    assert scores['total']['Calib']['width_90'] > scores['parameter']['Calib']['width_90']
    assert scores['total']['Eval']['coverage_90'] > 0.6
    assert scores['total']['Eval']['coverage_90'] > scores['parameter']['Eval']['coverage_90']


def test_glue_uses_likelihood_weights(tmp_path):
    opt = _fake_optimizer(tmp_path, UQ_PREDICTIVE_RUNS=10)
    samples = np.linspace(0.3, 0.7, 10)[:, None]
    weights = np.zeros(10)
    weights[2] = 1.0  # all GLUE weight on scale = 0.389
    result = {'behavioral_samples': samples, 'behavioral_fitness': np.linspace(0.1, 0.9, 10),
              'behavioral_likelihoods': weights}
    written = UncertaintyPostprocessor(opt).run(result, 'glue')
    table = pd.read_csv(written['quantiles'], index_col=0)
    assert 'total_q0.5' not in table.columns  # no error model for GLUE
    np.testing.assert_allclose(table['parameter_q0.05'], samples[2, 0] * BASE)
    np.testing.assert_allclose(table['parameter_q0.95'], samples[2, 0] * BASE)
    ds = xr.open_dataset(written['samples'])
    np.testing.assert_allclose(ds['weight'].values, weights)
    ds.close()


def test_predictive_runs_disabled_still_saves_samples(tmp_path):
    opt = _fake_optimizer(tmp_path, UQ_PREDICTIVE_RUNS=0)
    written = UncertaintyPostprocessor(opt).run(_dream_result(), 'dream')
    assert set(written) == {'samples'}
    assert not (tmp_path / 'final_evaluation').exists()


def test_no_samples_is_a_no_op(tmp_path):
    assert extract_uq_samples({'best_params': {}}) is None
    assert UncertaintyPostprocessor(_fake_optimizer(tmp_path)).run({}, 'dds') == {}
