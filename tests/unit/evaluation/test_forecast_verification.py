# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Forecast verification by lead time, on synthetic forecasts built from the observations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import EvaluationError
from symfluence.evaluation.forecast_verification import (
    cluster_events,
    contingency,
    doy_climatology,
    scores,
    verify_forecasts,
)

pytestmark = pytest.mark.unit

START, END = pd.Timestamp('2020-05-01'), pd.Timestamp('2020-10-01')


def _observed() -> pd.Series:
    """Hourly means stamped at the end of each hour: seasonal wave + daily cycle + one flood."""
    t = pd.date_range(START + pd.Timedelta(hours=1), END, freq='h')
    h = (t - START) / pd.Timedelta(hours=1)
    q = 100 + 30 * np.sin(2 * np.pi * h / (24 * 150)) + 10 * np.sin(2 * np.pi * (t.hour - 15) / 24)
    q = pd.Series(q, index=t, name='q')
    q[(t > '2020-07-10') & (t <= '2020-07-13')] += 200
    return q


def _three_hourly(obs: pd.Series) -> pd.Series:
    return obs.resample('3h', closed='right', label='right').mean()


def _forecasts(obs: pd.Series, model: str, values: pd.Series, step: str, stamp: str = 'end') -> pd.DataFrame:
    rows = []
    span = pd.Timedelta(step)
    for init in pd.date_range('2020-05-05', '2020-09-25', freq='D'):
        ends = pd.date_range(init + span, init + pd.Timedelta(days=3), freq=step)
        stamps = ends - span if stamp == 'start' else ends
        rows.append(pd.DataFrame({'model': model, 'init': init, 'time': stamps,
                                  'q': values.reindex(ends).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope='module')
def result():
    obs = _observed()
    q3 = _three_hourly(obs)
    daily_end = obs.resample('D', closed='right', label='right').mean()
    fc = pd.concat([
        _forecasts(obs, 'perfect', q3, '3h'),
        _forecasts(obs, 'plus5', q3 + 5, '3h'),
        _forecasts(obs, 'scaled', q3 * 1.2, '3h'),
        _forecasts(obs, 'late', q3.shift(1), '3h'),                        # one step late
        _forecasts(obs, 'daily', daily_end, '1D', stamp='start'),
    ])
    return verify_forecasts(fc, obs, stamps={'daily': 'start'},
                            multi_model_means={'mean_pp': ('perfect', 'plus5')})


def _row(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, kw
    return sel.iloc[0]


def test_perfect_forecast_scores(result):
    for d in (1, 2, 3):
        r = _row(result.lead_scores, model='perfect', variant='raw', lead_day=d)
        assert r['bias'] == pytest.approx(0, abs=1e-9) and r['nse'] == pytest.approx(1)
        assert r['skill_vs_persistence'] == pytest.approx(1)
        assert r['n'] == 8 * 144                                           # 144 issue days x 8 steps


def test_offset_removes_a_constant_error(result):
    raw = _row(result.lead_scores, model='plus5', variant='raw', lead_day=2)
    off = _row(result.lead_scores, model='plus5', variant='offset', lead_day=2)
    assert raw['bias'] == pytest.approx(5) and off['rmse'] == pytest.approx(0, abs=1e-9)
    assert result.pairs['plus5']['err_at_issue'].dropna().to_numpy() == pytest.approx(-5)


def test_start_stamped_daily_model_lines_up(result):
    r = _row(result.lead_scores, model='daily', variant='raw', lead_day=1)
    assert r['rmse'] == pytest.approx(0, abs=1e-9)
    pairs = result.pairs['daily']
    first = pairs[pairs['init'] == pd.Timestamp('2020-05-05')]
    assert list(first['lead_day']) == [1, 2, 3]
    assert first['time'].iloc[0] == pd.Timestamp('2020-05-05')


def test_persistence_is_the_interval_ending_at_issue(result):
    p = result.pairs['perfect']
    q3 = _three_hourly(_observed())
    init = pd.Timestamp('2020-06-01')
    assert p.loc[p['init'] == init, 'persistence'].eq(q3[init]).all()


def test_daily_scores_monthly_bias_and_means(result):
    ds = result.daily_scores
    assert _row(ds, model='mean_pp', variant='raw', lead_day=1)['bias'] == pytest.approx(2.5)
    assert {'persistence', 'climatology'} <= set(ds.loc[ds['model'] == 'benchmark', 'variant'])
    mb = result.monthly_bias
    assert mb.loc[mb['model'] == 'scaled', 'ratio'].to_numpy() == pytest.approx(1.2)
    assert mb.loc[mb['model'] == 'perfect', 'ratio'].to_numpy() == pytest.approx(1.0)


def test_high_flows_and_events(result):
    c = _row(result.contingency, model='perfect', variant='raw', lead_day=3)
    assert c['hits'] > 0 and c['pod'] == 1 and c['far'] == 0
    ev = result.events
    flood = ev[ev['peak_day'].astype(str).str.startswith('2020-07-1')]
    assert len(flood) == 1
    assert flood['perfect_warned_lead_days'].iloc[0] == 3
    assert result.info['events'] == len(ev)


def test_timing_finds_the_lag(result):
    t = result.timing
    late = t[t['model'] == 'late'].set_index('shift_model_by_h')['corr_of_daily_cycle']
    assert late.idxmax() == -3.0
    perfect = t[t['model'] == 'perfect'].set_index('shift_model_by_h')['corr_of_daily_cycle']
    assert perfect.idxmax() == 0.0
    assert 'daily' not in set(t['model'])


def test_write(tmp_path, result):
    out = result.write(tmp_path / 'verify')
    assert (out / 'lead_scores.csv').exists() and (out / 'pairs_perfect.csv').exists()
    assert '## lead_scores' in (out / 'report.md').read_text(encoding='utf-8')


def test_helpers():
    o = pd.Series([1.0, 2.0, 3.0, np.nan])
    s = scores(o + 1, o, {'zero': pd.Series([0.0, 0.0, 0.0, 0.0])})
    assert s['n'] == 3 and s['bias'] == 1 and s['skill_vs_zero'] == pytest.approx(1 - 1 / (14 / 3))
    c = contingency(pd.Series([5.0, 0.0, 5.0]), pd.Series([5.0, 5.0, 0.0]), 1.0)
    assert (c['hits'], c['misses'], c['false_alarms']) == (1, 1, 1)
    days = pd.date_range('2020-01-01', periods=10, freq='D')
    ev = cluster_events(pd.Series([0, 9, 0, 0, 9, 0, 0, 0, 0, 9.0], index=days), 5, max_gap_days=3)
    assert [len(e) for e in ev] == [2, 1]
    clim = doy_climatology(pd.Series(1.0, index=pd.date_range('2019-01-01', '2020-12-31', freq='D')))
    assert len(clim) == 365 and (clim == 1.0).all()


def test_rejects_bad_input():
    obs = _observed()
    with pytest.raises(EvaluationError):
        verify_forecasts(pd.DataFrame({'model': [], 'init': []}), obs)
    fc = _forecasts(obs, 'm', _three_hourly(obs), '3h')
    with pytest.raises(EvaluationError):
        verify_forecasts(fc, obs, stamps={'m': 'middle'})
    with pytest.raises(EvaluationError):
        verify_forecasts(fc, obs.resample('D').mean())                   # observed coarser than the model
