"""Regression tests for evaluator time alignment and aggregation.

Covers: spin-up anchoring, inclusive period end for sub-daily data, native
timestep alignment of hourly sim vs daily obs, timestep inference with gaps,
rate aggregation for ET, ET daily_sum gaps, GRACE anomaly KGE, and the
``target=`` override used by multivariate calibration.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symfluence.core.modeling.evaluators.base import ModelEvaluator
from symfluence.core.modeling.evaluators.et import ETEvaluator
from symfluence.core.modeling.evaluators.groundwater import GroundwaterEvaluator
from symfluence.core.modeling.evaluators.snow import SnowEvaluator
from symfluence.core.modeling.evaluators.soil_moisture import SoilMoistureEvaluator
from symfluence.core.modeling.evaluators.streamflow import StreamflowEvaluator
from symfluence.core.modeling.evaluators.tws import TWSEvaluator

LOGGER = logging.getLogger("test_evaluation_alignment")


class _Eval(ModelEvaluator):
    def __init__(self, config, flux=False, rates=False, **kw):
        self._flux = flux
        self._rates = rates
        super().__init__(config, Path("."), LOGGER, **kw)

    @property
    def variable_type(self):
        return 'flux' if self._flux else 'state'

    @property
    def values_are_rates(self):
        return self._rates

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
        'CALIBRATION_TIMESTEP': 'native',
        'EVALUATION_SPINUP_YEARS': 0,
    }
    base.update(kw)
    return base


# --- spin-up (fix 4) -------------------------------------------------------

def test_spinup_counted_from_simulation_start_not_each_period():
    idx = pd.date_range('2000-01-01', '2005-12-31', freq='D')
    rng = np.random.default_rng(0)
    obs = pd.Series(rng.gamma(2, 1, len(idx)) + 1, index=idx)
    sim = obs * 1.1
    ev = _Eval(_cfg(EVALUATION_SPINUP_YEARS=2,
                    CALIBRATION_PERIOD='2000-01-01, 2003-12-31',
                    EVALUATION_PERIOD='2004-01-01, 2005-12-31'))

    # Spy on the aligned length via the metrics call
    seen = {}
    orig = ev._calculate_performance_metrics

    def spy(o, s):
        seen.setdefault('n', []).append(len(o))
        return orig(o, s)

    ev._calculate_performance_metrics = spy
    ev._calculate_period_metrics(obs, sim, ev.calibration_period, 'Calib')
    ev._calculate_period_metrics(obs, sim, ev.evaluation_period, 'Eval')
    calib_n, eval_n = seen['n']
    # Calib loses 2000-2001 (spin-up); Eval keeps both its years.
    assert calib_n == len(pd.date_range('2002-01-01', '2003-12-31'))
    assert eval_n == len(pd.date_range('2004-01-01', '2005-12-31'))


def test_spinup_anchored_to_sim_start_when_obs_start_later():
    idx = pd.date_range('2000-01-01', '2004-12-31', freq='D')
    sim = pd.Series(np.arange(len(idx), dtype=float) + 1, index=idx)
    obs = sim['2001-06-01':] * 1.0
    ev = _Eval(_cfg(EVALUATION_SPINUP_YEARS=1))
    s, o = ev.align_series(sim, obs)
    # Cutoff is 2001-01-01 (sim start + 1 y), so obs from 2001-06-01 all kept.
    assert o.index.min() == pd.Timestamp('2001-06-01')


# --- inclusive period end (fix 5) ------------------------------------------

def test_hourly_period_end_covers_whole_last_day():
    idx = pd.date_range('2010-01-01', '2010-01-10 23:00', freq='h')
    rng = np.random.default_rng(1)
    obs = pd.Series(rng.random(len(idx)) + 1, index=idx)
    sim = obs + 0.01
    ev = _Eval(_cfg(CALIBRATION_PERIOD='2010-01-01, 2010-01-10'))
    seen = {}
    orig = ev._calculate_performance_metrics
    ev._calculate_performance_metrics = lambda o, s: (seen.update(n=len(o), last=o.index.max()), orig(o, s))[1]
    ev._calculate_period_metrics(obs, sim, ev.calibration_period, 'Calib')
    assert seen['n'] == 240
    assert seen['last'] == pd.Timestamp('2010-01-10 23:00')


def test_explicit_period_end_time_is_respected():
    mask = ModelEvaluator._period_mask(
        pd.date_range('2010-01-01', periods=48, freq='h'),
        (pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-01 12:00')),
    )
    assert mask.sum() == 13


# --- native hourly sim vs daily obs (fix 6) --------------------------------

def test_native_hourly_sim_aggregated_to_daily_obs():
    hours = pd.date_range('2010-01-01', '2010-03-31 23:00', freq='h')
    # Strong diurnal cycle: value at 00:00 is far from the daily mean.
    diurnal = 5 * np.sin(2 * np.pi * (hours.hour - 6) / 24)
    daily_signal = np.repeat(np.linspace(10, 20, len(hours) // 24), 24)
    sim = pd.Series(daily_signal + diurnal, index=hours)
    obs = sim.resample('D').mean()
    ev = _Eval(_cfg(CALIBRATION_PERIOD='2010-01-01, 2010-03-31'))
    m = ev._calculate_period_metrics(obs, sim, ev.calibration_period, 'Calib')
    assert m['Calib_KGE'] == pytest.approx(1.0, abs=1e-9)
    assert m['Calib_RMSE'] == pytest.approx(0.0, abs=1e-9)


def test_native_sparse_snapshot_obs_not_aggregated():
    days = pd.date_range('2010-01-01', '2010-12-31', freq='D')
    sim = pd.Series(np.linspace(0, 100, len(days)), index=days)
    obs = sim.iloc[::30]
    ev = _Eval(_cfg())
    out = ev._aggregate_to_observation_steps(sim, obs)
    assert out is sim


# --- timestep inference with gaps (fix 7) ----------------------------------

def test_infer_timestep_ignores_single_gap():
    idx = pd.date_range('2010-01-01', periods=100, freq='h').delete(1)
    assert ModelEvaluator._infer_timestep(idx) == pd.Timedelta(hours=1)


def test_daily_resample_with_missing_second_obs():
    idx = pd.date_range('2010-01-01', '2010-01-31', freq='D').delete(1)
    obs = pd.Series(np.arange(len(idx), dtype=float) + 1, index=idx)
    ev = _Eval(_cfg(CALIBRATION_TIMESTEP='daily'))
    out = ev._resample_to_timestep(obs, 'daily')
    pd.testing.assert_series_equal(out, obs)


# --- rate aggregation (fix 3) ----------------------------------------------

def test_daily_aggregation_of_rates_uses_mean():
    hours = pd.date_range('2010-01-01', periods=48, freq='h')
    et = pd.Series(2.0, index=hours)  # mm/day at every hour
    ev = _Eval(_cfg(), flux=True, rates=True)
    out = ev._resample_to_timestep(et, 'daily')
    assert (out == 2.0).all()


def test_daily_aggregation_of_amounts_uses_sum_and_keeps_gaps():
    hours = pd.date_range('2010-01-01', periods=72, freq='h')
    p = pd.Series(0.5, index=hours)
    p = p.drop(p['2010-01-02'].index[1:])  # one stamp left on day 2
    ev = _Eval(_cfg(), flux=True, rates=False)
    out = ev._resample_to_timestep(p, 'daily')
    assert out['2010-01-01'] == pytest.approx(12.0)


def test_et_evaluator_declares_rates():
    ev = ETEvaluator(_cfg(), Path("."), LOGGER)
    assert ev.variable_type == 'flux'
    assert ev._aggregation_method() == 'mean'


def test_et_daily_sum_does_not_turn_gaps_into_zero(tmp_path):
    idx = pd.date_range('2010-01-01', '2010-01-05 23:30', freq='30min')
    vals = pd.Series(0.1, index=idx)
    vals['2010-01-03'] = np.nan  # whole day missing
    df = pd.DataFrame({'et_mm_day': vals})
    path = tmp_path / 'et.csv'
    df.to_csv(path)
    ev = ETEvaluator(_cfg(ET_TEMPORAL_AGGREGATION='daily_sum', ET_OBS_SOURCE='mod16'),
                     tmp_path, LOGGER)
    ev.get_observed_data_path = lambda: path
    ev._get_observed_data_column = lambda cols: 'et_mm_day'
    out = ev._load_observed_data()
    assert pd.Timestamp('2010-01-03') not in out.index
    assert len(out) == 4


# --- GRACE anomaly KGE (fix 1) ---------------------------------------------

def test_tws_anomaly_kge_independent_of_mean_noise():
    months = pd.date_range('2004-01-01', periods=120, freq='MS')
    seasonal = 100 * np.sin(2 * np.pi * np.arange(120) / 12)
    ev = TWSEvaluator(_cfg(), Path("."), LOGGER)
    kges = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        obs = pd.Series(seasonal + rng.normal(0, 1, 120), index=months)
        sim = pd.Series(seasonal + rng.normal(0, 1, 120), index=months)
        obs, sim = obs - obs.mean(), sim - sim.mean()
        m = ev._calculate_anomaly_metrics(obs, sim)
        assert np.isnan(m['beta'])
        assert np.isnan(m['PBIAS'])
        kges.append(m['KGE'])
    assert min(kges) > 0.97


def test_tws_calculate_metrics_uses_anomaly_kge(tmp_path):
    days = pd.date_range('2004-01-01', '2013-12-31', freq='D')
    storage = 500 + 100 * np.sin(2 * np.pi * days.dayofyear / 365.25)
    sim = pd.Series(storage, index=days)
    monthly = sim.resample('MS').mean()
    rng = np.random.default_rng(3)
    obs = monthly - monthly.mean() + rng.normal(0, 0.5, len(monthly))
    path = tmp_path / 'grace.csv'
    pd.DataFrame({'grace_jpl_anomaly': obs}).to_csv(path)
    ev = TWSEvaluator(_cfg(), tmp_path, LOGGER)
    ev.get_simulation_files = lambda d: [Path('x.nc')]
    ev.extract_simulated_data = lambda files, **kw: sim
    ev.get_observed_data_path = lambda: path
    ev._get_observed_data_column = lambda cols: 'grace_jpl_anomaly'
    m = ev.calculate_metrics(tmp_path)
    assert m['KGE'] > 0.95
    assert np.isnan(m['beta'])


# --- multivariate target= (fix 2) ------------------------------------------

@pytest.mark.parametrize("cls,var,expected", [
    (ETEvaluator, 'ET', 'et'),
    (ETEvaluator, 'latent_heat', 'latent_heat'),
    (TWSEvaluator, 'TWS', 'stor_grace'),
    (TWSEvaluator, 'stor_mb', 'stor_mb'),
    (SoilMoistureEvaluator, 'sm_smap', 'sm_smap'),
    (GroundwaterEvaluator, 'gw_depth', 'gw_depth'),
    (SnowEvaluator, 'swe', 'swe'),
])
def test_evaluators_accept_target_kwarg(cls, var, expected):
    ev = cls(_cfg(), Path("."), LOGGER, target=var)
    assert ev.optimization_target == expected


def test_streamflow_evaluator_accepts_target_kwarg():
    StreamflowEvaluator(_cfg(), Path("."), LOGGER, target='STREAMFLOW')


def test_registry_get_evaluator_with_target():
    from symfluence.core.modeling.evaluation_registry import EvaluationRegistry
    for var in ('ET', 'TWS', 'SOIL_MOISTURE', 'GROUNDWATER', 'SNOW', 'STREAMFLOW'):
        ev = EvaluationRegistry.get_evaluator(var, _cfg(), LOGGER, Path("."), target=var)
        assert ev is not None
