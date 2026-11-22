"""Split-sample design: period checks, unified spin-up, TEST_PERIOD, benchmarks.

Covers the rules documented in docs/source/calibration.rst ("Split-sample
design"): problems warn (never raise), the full-record fallback is flagged in
the metrics dict, spin-up is one window, the test period is scored only in the
final evaluation, and benchmarks are fitted on the calibration period and
scored on exactly the configured periods.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.config.models import root as root_module
from symfluence.core.metrics.benchmarks import benchmark_series, skill_score
from symfluence.core.modeling.evaluators.base import ModelEvaluator
from symfluence.core.modeling.utilities.periods import (
    periods_overlap,
    spinup_window,
    split_sample_warnings,
)

LOGGER = logging.getLogger("test_split_sample")


@pytest.fixture(autouse=True)
def _reset_warn_once():
    root_module._WARNED_MESSAGES.clear()
    ModelEvaluator._FALLBACK_WARNED.clear()
    yield
    root_module._WARNED_MESSAGES.clear()
    ModelEvaluator._FALLBACK_WARNED.clear()


# --- pure period helpers ----------------------------------------------------

def test_adjacent_periods_do_not_overlap_but_shared_day_does():
    assert not periods_overlap(('2000-01-01', '2005-12-31'), ('2006-01-01', '2008-12-31'))
    assert periods_overlap(('2000-01-01', '2005-12-31'), ('2005-12-31', '2008-12-31'))
    assert periods_overlap(('2000-01-01', '2010-12-31'), ('2003-01-01', '2004-12-31'))


def test_spinup_window_maps_legacy_years_onto_same_window():
    assert spinup_window(None, 0, '2000-01-01') is None
    w = spinup_window(None, 2, '2000-01-01')
    assert w == (pd.Timestamp('2000-01-01'), pd.Timestamp('2002-01-01'))
    w = spinup_window('2000-01-01, 2000-12-31', 0, '2000-01-01')
    assert w == (pd.Timestamp('2000-01-01'), pd.Timestamp('2001-01-01'))
    # both set: the later end wins
    w = spinup_window('2000-01-01, 2000-12-31', 2, '2000-01-01')
    assert w[1] == pd.Timestamp('2002-01-01')
    # garbage is ignored, not raised
    assert spinup_window('not a period', 'x', None) is None


def test_split_sample_warnings_each_problem():
    clean = split_sample_warnings(
        calibration='2002-01-01, 2005-12-31', evaluation='2006-01-01, 2008-12-31',
        test='2009-01-01, 2010-12-31', spinup='2000-01-01, 2001-12-31',
        simulation_start='2000-01-01', calibrating=True,
    )
    assert clean == []

    overlap = split_sample_warnings(
        calibration='2002-01-01, 2006-12-31', evaluation='2006-01-01, 2008-12-31',
    )
    assert len(overlap) == 1 and 'CALIBRATION_PERIOD' in overlap[0] and 'EVALUATION_PERIOD' in overlap[0]

    test_overlap = split_sample_warnings(
        calibration='2002-01-01, 2005-12-31', evaluation='2006-01-01, 2008-12-31',
        test='2008-06-01, 2010-12-31',
    )
    assert len(test_overlap) == 1 and 'TEST_PERIOD' in test_overlap[0]

    spin = split_sample_warnings(
        calibration='2001-01-01, 2005-12-31', spinup='2000-01-01, 2001-12-31',
    )
    assert len(spin) == 1 and 'Spin-up' in spin[0]

    legacy_spin = split_sample_warnings(
        calibration='2001-01-01, 2005-12-31', spinup_years=2, simulation_start='2000-01-01',
    )
    assert len(legacy_spin) == 1 and 'Spin-up' in legacy_spin[0]

    missing = split_sample_warnings(evaluation='2006-01-01, 2008-12-31')
    assert any('CALIBRATION_PERIOD is not set' in m for m in missing)
    assert split_sample_warnings() == []  # plain simulation: nothing to say
    assert split_sample_warnings(calibrating=True)  # calibrating without a period


# --- config validation warns, never raises ----------------------------------

def _config(**extra):
    base = dict(
        domain_name='split', model='SUMMA',
        EXPERIMENT_TIME_START='2000-01-01 00:00', EXPERIMENT_TIME_END='2010-12-31 23:00',
    )
    base.update(extra)
    return SymfluenceConfig.from_minimal(**base)


def test_config_overlap_warns_and_still_loads(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _config(
            CALIBRATION_PERIOD='2002-01-01, 2006-12-31',
            EVALUATION_PERIOD='2006-01-01, 2008-12-31',
        )
    assert cfg.domain.calibration_period == '2002-01-01, 2006-12-31'
    assert 'Split-sample design' in caplog.text and 'overlaps' in caplog.text


def test_config_test_period_parsed_and_overlap_warned(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _config(
            CALIBRATION_PERIOD='2002-01-01, 2005-12-31',
            EVALUATION_PERIOD='2006-01-01, 2008-12-31',
            TEST_PERIOD='2008-01-01, 2010-12-31',
        )
    assert cfg.domain.test_period == '2008-01-01, 2010-12-31'
    assert 'TEST_PERIOD' in caplog.text


def test_config_spinup_overlap_and_missing_calibration_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _config(SPINUP_PERIOD='2000-01-01, 2002-12-31',
                CALIBRATION_PERIOD='2002-01-01, 2005-12-31')
    assert 'Spin-up' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _config(CALIBRATION_PERIOD=None, EVALUATION_PERIOD='2006-01-01, 2008-12-31')
    assert 'CALIBRATION_PERIOD is not set' in caplog.text


def test_config_legacy_spinup_years_deprecated(caplog):
    with caplog.at_level(logging.WARNING), pytest.warns(DeprecationWarning):
        _config(EVALUATION_SPINUP_YEARS=1, CALIBRATION_PERIOD='2002-01-01, 2005-12-31')
    assert 'EVALUATION_SPINUP_YEARS is deprecated' in caplog.text


# --- evaluator --------------------------------------------------------------

class _Eval(ModelEvaluator):
    def __init__(self, config):
        super().__init__(config, Path("."), LOGGER)

    def get_simulation_files(self, sim_dir):
        return []

    def extract_simulated_data(self, sim_files, **kwargs):
        return pd.Series(dtype=float)

    def get_observed_data_path(self):
        return Path("obs.csv")

    def needs_routing(self):
        return False

    def _get_observed_data_column(self, columns):
        return None


def _cfg(**kw):
    base = {
        'CALIBRATION_PERIOD': '',
        'EVALUATION_PERIOD': '',
        'TEST_PERIOD': '',
        'SPINUP_PERIOD': '',
        'CALIBRATION_TIMESTEP': 'native',
        'EVALUATION_SPINUP_YEARS': 0,
    }
    base.update(kw)
    return base


def _series(start='2000-01-01', end='2010-12-31', seed=0):
    idx = pd.date_range(start, end, freq='D')
    rng = np.random.default_rng(seed)
    doy = idx.dayofyear.values
    obs = pd.Series(5 + 3 * np.sin(2 * np.pi * doy / 365.25) + rng.gamma(2, 0.5, len(idx)), index=idx)
    sim = obs * 0.9 + rng.normal(0, 0.3, len(idx))
    return obs, sim


def test_full_record_fallback_warns_and_flags(caplog):
    obs, sim = _series()
    ev = _Eval(_cfg(EVALUATION_PERIOD='2006-01-01, 2008-12-31'))
    with caplog.at_level(logging.WARNING):
        m = ev.calculate_metrics(sim, obs, calibration_only=True)
    assert 'KGE' in m
    assert 'CALIBRATION_PERIOD is not set' in m['period_warning']
    assert 'SPLIT-SAMPLE WARNING' in caplog.text


def test_no_period_warning_with_calibration_period():
    obs, sim = _series()
    ev = _Eval(_cfg(CALIBRATION_PERIOD='2001-01-01, 2005-12-31'))
    m = ev.calculate_metrics(sim, obs)
    assert 'period_warning' not in m and 'Calib_KGE' in m


def test_test_period_scored_only_in_final_evaluation():
    obs, sim = _series()
    ev = _Eval(_cfg(
        CALIBRATION_PERIOD='2001-01-01, 2005-12-31',
        EVALUATION_PERIOD='2006-01-01, 2008-12-31',
        TEST_PERIOD='2009-01-01, 2010-12-31',
    ))
    during = ev.calculate_metrics(sim, obs, calibration_only=True)
    assert not any(k.startswith(('Eval_', 'Test_')) for k in during)
    final = ev.calculate_metrics(sim, obs, calibration_only=False)
    assert 'Test_KGE' in final and 'Eval_KGE' in final
    # Test score is computed on the test window only
    expected = ev._calculate_period_metrics(obs, sim, ev.test_period, 'Test')['Test_KGE']
    assert final['Test_KGE'] == pytest.approx(expected)
    # Unprefixed (objective) metrics are still the calibration ones
    assert final['KGE'] == final['Calib_KGE']


def test_spinup_period_is_excluded_from_scores():
    obs, sim = _series()
    sim = sim.copy()
    sim['2000-01-01':'2001-12-31'] = 1000.0  # garbage during spin-up
    ev_spin = _Eval(_cfg(SPINUP_PERIOD='2000-01-01, 2001-12-31'))
    ev_none = _Eval(_cfg())
    full_spin = ev_spin._calculate_period_metrics(obs, sim, (None, None), '')
    full_none = ev_none._calculate_period_metrics(obs, sim, (None, None), '')
    assert full_spin['KGE'] > 0.5 > full_none['KGE']


def test_legacy_spinup_years_still_honoured():
    obs, sim = _series()
    sim = sim.copy()
    sim['2000-01-01':'2001-12-31'] = 1000.0
    ev = _Eval(_cfg(EVALUATION_SPINUP_YEARS=2))
    sim_a, obs_a = ev.align_series(sim, obs)
    assert sim_a.index[0] == pd.Timestamp('2002-01-01')


def test_benchmark_skill_on_configured_periods():
    obs, sim = _series()
    ev = _Eval(_cfg(
        CALIBRATION_PERIOD='2001-01-01, 2005-12-31',
        EVALUATION_PERIOD='2006-01-01, 2008-12-31',
        TEST_PERIOD='2009-01-01, 2010-12-31',
    ))
    m = ev.calculate_metrics(sim, obs, calibration_only=False)
    # Mean-flow benchmark fitted on calibration: KGE = 1 - sqrt(2) there
    assert m['Calib_KGE_benchmark_mean_flow'] == pytest.approx(1 - np.sqrt(2), abs=1e-6)
    for prefix in ('Calib', 'Eval', 'Test'):
        for name in ('mean_flow', 'climatology', 'persistence'):
            bm = m[f'{prefix}_KGE_benchmark_{name}']
            assert np.isfinite(bm)
            assert m[f'{prefix}_KGE_skill_{name}'] == pytest.approx(
                (m[f'{prefix}_KGE'] - bm) / (1 - bm)
            )
    # Not computed during optimisation
    during = ev.calculate_metrics(sim, obs, calibration_only=True)
    assert not any('benchmark' in k for k in during)


def test_benchmarks_fitted_on_calibration_only():
    idx = pd.date_range('2000-01-01', '2003-12-31', freq='D')
    obs = pd.Series(np.where(idx.year < 2002, 1.0, 100.0), index=idx)
    bms = benchmark_series(obs, (pd.Timestamp('2000-01-01'), pd.Timestamp('2001-12-31')))
    assert (bms['mean_flow'] == 1.0).all()
    assert (bms['climatology'] == 1.0).all()
    assert bms['persistence'].iloc[0] != bms['persistence'].iloc[0]  # NaN first step
    assert bms['persistence'].iloc[1] == 1.0


def test_skill_score():
    assert skill_score(1.0, -0.41) == pytest.approx(1.0)
    assert skill_score(-0.41, -0.41) == pytest.approx(0.0)
    assert skill_score(0.2, 0.6) < 0
    assert np.isnan(skill_score(0.5, 1.0))
    assert np.isnan(skill_score(None, 0.1))


# --- hydrobm benchmarker masks ----------------------------------------------

def _benchmarker(cfg):
    from symfluence.evaluation.benchmarking import Benchmarker
    bm = Benchmarker.__new__(Benchmarker)
    bm._config = cfg
    bm.logger = LOGGER
    return bm


def test_benchmarker_uses_configured_periods():
    cfg = _config(
        SPINUP_PERIOD='2000-01-01, 2000-12-31',
        CALIBRATION_PERIOD='2001-01-01, 2004-12-31',
        EVALUATION_PERIOD='2005-01-01, 2007-12-31',
        TEST_PERIOD='2008-01-01, 2010-12-31',
    )
    idx = pd.date_range('2000-01-01', '2010-12-31', freq='D')
    masks, warning = _benchmarker(cfg)._configured_masks(idx)
    assert warning is None
    assert idx[masks['cal']][0] == pd.Timestamp('2001-01-01')
    assert idx[masks['cal']][-1] == pd.Timestamp('2004-12-31')
    assert idx[masks['val']][0] == pd.Timestamp('2005-01-01')
    assert idx[masks['test']][-1] == pd.Timestamp('2010-12-31')
    assert not (masks['cal'] & masks['val']).any()


def test_benchmarker_median_split_fallback_warns():
    cfg = _config()
    idx = pd.date_range('2000-01-01', '2009-12-31', freq='D')
    masks, warning = _benchmarker(cfg)._configured_masks(idx)
    assert warning and 'median year' in warning
    assert 'test' not in masks


# --- final-evaluation reporting ---------------------------------------------

def test_results_saver_separates_test_and_tolerates_warning(tmp_path, caplog):
    from symfluence.core.calibration.optimizers.final_evaluation.results_saver import (
        FinalResultsSaver,
    )
    metrics = {
        'KGE': 0.8, 'Calib_KGE': 0.8, 'Eval_KGE': 0.7, 'Test_KGE': 0.6,
        'period_warning': 'CALIBRATION_PERIOD is not set',
    }
    calib = FinalResultsSaver.extract_period_metrics(metrics, 'Calib')
    test = FinalResultsSaver.extract_period_metrics(metrics, 'Test')
    assert 'Test_KGE' not in calib and 'period_warning' not in calib
    assert test == {'KGE': 0.6}
    saver = FinalResultsSaver(tmp_path, 'exp', 'dom', LOGGER)
    with caplog.at_level(logging.INFO):
        saver.log_results(calib, {'KGE': 0.7, 'note': 'text'}, test, metrics['period_warning'])
    assert 'TEST PERIOD PERFORMANCE' in caplog.text and 'WARNING: CALIBRATION_PERIOD' in caplog.text
    out = saver.save_results(
        {'calibration_metrics': calib, 'evaluation_metrics': {}, 'test_metrics': test,
         'period_warning': metrics['period_warning']}, 'DDS')
    import json
    saved = json.loads(out.read_text())
    assert saved['test_metrics'] == {'KGE': 0.6} and saved['period_warning']
