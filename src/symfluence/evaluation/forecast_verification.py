# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Verification of discharge forecasts (hindcasts) by lead time.

Inputs are a tidy forecast table and an observed series; nothing is assumed
about files or directory layout.

* ``forecasts``: columns ``model``, ``init`` (issue time), ``time`` (stamp of
  the forecast interval) and ``q``.  Each model has a fixed interval length
  (inferred from its stamps, or given) and stamps its intervals at their
  ``'end'`` (``(t - dt, t]``, most models) or ``'start'`` (``[t, t + dt)``).
* ``observed``: discharge on a regular step no coarser than any model's,
  each value the mean over the sampling interval ending (``obs_stamp='end'``)
  or starting (``'start'``) at its stamp.

Leads count to the END of each model interval, so lead day 1 is the first
24 h after issue whatever the stamping.  Benchmarks, all known at issue time:

* persistence: observed mean of the last model interval before issue, held;
* climatology: observed daily mean for the day of year (+-7 days), over a
  chosen climatology period (ideally outside the verified period);
* offset: the forecast plus the error of the forecast issued ``offset_lag``
  earlier at the interval ending at issue (observed minus forecast, held).
  It is the simplest updating an operator can run without data assimilation.

Results (:class:`ForecastVerification`):

* ``lead_scores``: bias, MAE, RMSE, NSE and MSE skill vs persistence and
  climatology per model, variant (raw / offset / persistence / climatology)
  and lead day, at each model's own step;
* ``daily_scores``: the same on daily means, for each model and multi-model
  mean, over the (issue, day) pairs every member has;
* ``monthly_bias``: forecast / observed volume by calendar month;
* ``contingency``: hits, misses, false alarms, POD, FAR and CSI of daily means
  above a high-flow threshold (a quantile of observed daily means);
* ``events``: clusters of observed days above the threshold, with the
  forecast peak-day value per lead day and the longest lead day at which any
  forecast put an event day above the threshold (warned lead days);
* ``timing``: correlation of the model's and the observed summer daily cycle
  (anomalies from a centred daily mean) at lead day 1, for shifts of the model
  by whole steps (sub-daily models only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError

DAY = pd.Timedelta(days=1)
_EPS = pd.Timedelta(minutes=1)
REQUIRED_COLUMNS = ('model', 'init', 'time', 'q')


@dataclass
class ForecastVerification:
    """Tables produced by :func:`verify_forecasts` (see the module docstring)."""

    pairs: Dict[str, pd.DataFrame]
    lead_scores: pd.DataFrame
    daily_scores: pd.DataFrame
    monthly_bias: pd.DataFrame
    contingency: pd.DataFrame
    events: pd.DataFrame
    timing: pd.DataFrame
    info: Dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir: Union[str, Path]) -> Path:
        """Write every table as CSV plus a Markdown summary ``report.md``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for model, df in self.pairs.items():
            df.to_csv(out / f'pairs_{model}.csv', index=False)
        tables = {'lead_scores': self.lead_scores, 'daily_scores': self.daily_scores,
                  'monthly_bias': self.monthly_bias, 'contingency': self.contingency,
                  'events': self.events, 'timing': self.timing}
        for name, df in tables.items():
            df.to_csv(out / f'{name}.csv', index=False)
        lines = ['# Forecast verification', '']
        lines += [f'- {k}: {v}' for k, v in self.info.items()] + ['']
        for name, df in tables.items():
            if len(df):
                lines += [f'## {name}', '', _markdown(df.round(3)), '']
        with open(out / 'report.md', 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(lines))
        return out


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def scores(forecast: pd.Series, observed: pd.Series,
           references: Optional[Mapping[str, pd.Series]] = None) -> Dict[str, float]:
    """Bias, MAE, RMSE, NSE and MSE skill vs each reference, on pairs where all are present.

    ``skill_vs_<name> = 1 - MSE(forecast) / MSE(reference)``.
    """
    references = references or {}
    ok = forecast.notna() & observed.notna()
    for r in references.values():
        ok &= r.notna()
    f, o = forecast[ok].astype(float), observed[ok].astype(float)
    e = f - o
    n = int(ok.sum())
    out: Dict[str, float] = {'n': n}
    if n == 0:
        out.update(bias=np.nan, mae=np.nan, rmse=np.nan, nse=np.nan)
        out.update({f'skill_vs_{k}': np.nan for k in references})
        return out
    mse = float((e ** 2).mean())
    var = float(((o - o.mean()) ** 2).sum())
    out.update(bias=float(e.mean()), mae=float(e.abs().mean()), rmse=float(np.sqrt(mse)),
               nse=1 - float((e ** 2).sum()) / var if var > 0 else np.nan)
    for k, r in references.items():
        ref_mse = float(((r[ok].astype(float) - o) ** 2).mean())
        out[f'skill_vs_{k}'] = 1 - mse / ref_mse if ref_mse > 0 else np.nan
    return out


def contingency(forecast: pd.Series, observed: pd.Series, threshold: float) -> Dict[str, float]:
    """Hits, misses, false alarms, POD, FAR and CSI of exceeding ``threshold``."""
    ok = forecast.notna() & observed.notna()
    ob, fc = observed[ok] > threshold, forecast[ok] > threshold
    h, m, f = int((ob & fc).sum()), int((ob & ~fc).sum()), int((~ob & fc).sum())
    return {'hits': h, 'misses': m, 'false_alarms': f,
            'pod': h / (h + m) if h + m else np.nan,
            'far': f / (h + f) if h + f else np.nan,
            'csi': h / (h + m + f) if h + m + f else np.nan}


def doy_climatology(daily: pd.Series, half_window: int = 7) -> pd.Series:
    """Mean by day of year (366 folded into 365), smoothed by a centred circular window."""
    d = daily.dropna()
    doy = d.groupby(np.minimum(d.index.dayofyear, 365)).mean().reindex(range(1, 366))
    ext = pd.concat([doy.iloc[-half_window:], doy, doy.iloc[:half_window]])
    smooth = ext.rolling(2 * half_window + 1, center=True, min_periods=1).mean()
    return pd.Series(smooth.iloc[half_window:half_window + 365].to_numpy(), index=doy.index)


def cluster_events(daily_obs: pd.Series, threshold: float, max_gap_days: int = 3) -> List[List[pd.Timestamp]]:
    """Days above ``threshold`` grouped into events; days at most ``max_gap_days`` apart join one event."""
    events: List[List[pd.Timestamp]] = []
    for t in daily_obs[daily_obs > threshold].index:
        if events and (t - events[-1][-1]).days <= max_gap_days:
            events[-1].append(t)
        else:
            events.append([t])
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _markdown(df: pd.DataFrame) -> str:
    rows = [list(map(str, df.columns)), ['---'] * df.shape[1]]
    rows += [['' if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
    return '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows)


def _step_of(index: Iterable[pd.Timestamp]) -> pd.Timedelta:
    t = pd.DatetimeIndex(sorted(set(index)))
    if len(t) < 2:
        raise EvaluationError('cannot infer a time step from fewer than two stamps; pass it explicitly')
    return pd.Timedelta(int(np.median(np.diff(t.asi8))), unit='ns')


def _obs_end_stamped(observed: pd.Series, obs_stamp: str) -> Tuple[pd.Series, pd.Timedelta]:
    obs = observed.dropna().sort_index()
    obs = obs[~obs.index.duplicated(keep='last')]
    step = _step_of(obs.index)
    if obs_stamp == 'start':
        obs = obs.copy()
        obs.index = obs.index + step
    elif obs_stamp != 'end':
        raise EvaluationError(f"obs_stamp must be 'start' or 'end', got {obs_stamp!r}")
    return obs.astype(float), step


def _interval_means(obs_end: pd.Series, obs_step: pd.Timedelta, step: pd.Timedelta,
                    origin: pd.Timestamp, min_coverage: float) -> pd.Series:
    """Means of end-stamped observations over intervals of ``step``, labelled by interval end."""
    if obs_step > step:
        raise EvaluationError(f'observed step {obs_step} is coarser than a model step {step}')
    g = obs_end.resample(step, closed='right', label='right', origin=origin)
    mean, count = g.mean(), g.count()
    mean[count < min_coverage * (step / obs_step)] = np.nan
    return mean


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_forecasts(
    forecasts: pd.DataFrame,
    observed: pd.Series,
    *,
    stamps: Optional[Mapping[str, str]] = None,
    steps: Optional[Mapping[str, Union[str, pd.Timedelta]]] = None,
    obs_stamp: str = 'end',
    max_lead_days: int = 3,
    climatology_period: Optional[Tuple[int, int]] = None,
    high_flow_quantile: float = 0.95,
    high_flow_threshold: Optional[float] = None,
    multi_model_means: Optional[Mapping[str, Sequence[str]]] = None,
    min_coverage: float = 0.8,
    offset_lag: Union[str, pd.Timedelta] = '1D',
    event_gap_days: int = 3,
    timing_months: Sequence[int] = (6, 7, 8, 9),
    timing_max_shift: int = 3,
) -> ForecastVerification:
    """Verify forecasts by lead time against an observed series.

    Args:
        forecasts: Tidy table with columns ``model, init, time, q``.
        observed: Observed discharge on a regular step (see ``obs_stamp``).
        stamps: Per model ``'end'`` (default) or ``'start'`` interval stamping.
        steps: Per model interval length; inferred from the stamps if absent.
        obs_stamp: Whether observed values are stamped at the end or start of
            their sampling interval.
        max_lead_days: Lead days to verify.
        climatology_period: (first, last) year for climatology and the
            high-flow threshold; all observed years if ``None``.
        high_flow_quantile: Quantile of observed daily means used as the
            high-flow threshold (ignored if ``high_flow_threshold`` is given).
        multi_model_means: Name -> member models, averaged on daily means over
            the (issue, day) pairs all members have.
        min_coverage: Share of observed values needed for an interval mean.
        offset_lag: Age of the forecast whose error at issue is added.
        event_gap_days: Days above threshold at most this far apart are one event.
        timing_months: Calendar months used for the daily-cycle timing.
        timing_max_shift: Largest shift (in model steps) tried for timing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in forecasts]
    if missing:
        raise EvaluationError(f'forecast table lacks columns {missing}')
    fc = forecasts.loc[:, list(REQUIRED_COLUMNS)].copy()
    fc['init'] = pd.to_datetime(fc['init'])
    fc['time'] = pd.to_datetime(fc['time'])
    fc['q'] = pd.to_numeric(fc['q'], errors='coerce')
    stamps = dict(stamps or {})
    lag = pd.Timedelta(offset_lag)
    obs_end, obs_step = _obs_end_stamped(observed, obs_stamp)

    # Observed daily means (start-stamped by date) for climatology, thresholds and daily scores
    obs_daily = _interval_means(obs_end, obs_step, DAY, obs_end.index.min().normalize(), min_coverage)
    obs_daily.index = obs_daily.index - DAY
    clim_src = obs_daily
    if climatology_period is not None:
        y0, y1 = climatology_period
        clim_src = obs_daily[(obs_daily.index.year >= y0) & (obs_daily.index.year <= y1)]
    if clim_src.dropna().empty:
        raise EvaluationError('no observed daily means in the climatology period')
    clim = doy_climatology(clim_src)
    threshold = (float(high_flow_threshold) if high_flow_threshold is not None
                 else float(clim_src.quantile(high_flow_quantile)))

    pairs: Dict[str, pd.DataFrame] = {}
    lead_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    daily_fc: Dict[str, pd.DataFrame] = {}

    for model, df in fc.groupby('model', sort=False):
        model = str(model)
        stamp = stamps.get(model, 'end')
        if stamp not in ('start', 'end'):
            raise EvaluationError(f"stamp for {model} must be 'start' or 'end', got {stamp!r}")
        step = pd.Timedelta(steps[model]) if steps and model in steps else _step_of(df['time'])
        if DAY % step != pd.Timedelta(0):
            raise EvaluationError(f'{model}: step {step} does not divide a day')
        df = df.sort_values(['init', 'time']).reset_index(drop=True)
        df['end'] = df['time'] + (step if stamp == 'start' else pd.Timedelta(0))
        df['lead_h'] = (df['end'] - df['init']) / pd.Timedelta(hours=1)
        df = df[df['lead_h'] > 0].copy()
        df['lead_day'] = np.ceil(df['lead_h'] / 24).astype(int)
        all_ends = pd.DatetimeIndex(df['end'])
        obs_int = _interval_means(obs_end, obs_step, step, all_ends.min(), min_coverage)
        df['obs'] = obs_int.reindex(df['end']).to_numpy()

        # Benchmarks known at issue: the observed interval ending at issue, and the error of the
        # forecast issued `lag` earlier for that same interval
        inits = pd.DatetimeIndex(df['init'].unique())
        at_issue = obs_int.reindex(inits)
        by_key = df.set_index(['init', 'end'])['q']
        fc_lookup = by_key[~by_key.index.duplicated()]
        prev_fc = pd.Series([fc_lookup.get((i - lag, i), np.nan) for i in inits], index=inits)
        err0 = at_issue - prev_fc
        df['persistence'] = df['init'].map(at_issue).to_numpy()
        df['err_at_issue'] = df['init'].map(err0).to_numpy()
        df['offset'] = df['q'] + df['err_at_issue']
        df['day'] = (df['end'] - _EPS).dt.normalize()
        df['climatology'] = clim.reindex(np.minimum(df['day'].dt.dayofyear, 365)).to_numpy()
        df = df[df['lead_day'] <= max_lead_days]
        pairs[model] = df.reset_index(drop=True)

        for d in range(1, max_lead_days + 1):
            s = df[df['lead_day'] == d]
            refs = {'persistence': s['persistence'], 'climatology': s['climatology']}
            for variant in ('raw', 'offset', 'persistence', 'climatology'):
                col = 'q' if variant == 'raw' else variant
                lead_rows.append({'model': model, 'variant': variant, 'lead_day': d,
                                  **scores(s[col], s['obs'], refs)})

        # Daily means of complete forecast days
        per_day = int(DAY / step)
        g = df.groupby(['init', 'day'])
        dd = pd.DataFrame({'q': g['q'].mean(), 'n': g['q'].count()}).reset_index()
        dd = dd[dd['n'] == per_day].drop(columns='n')
        dd['lead_day'] = np.ceil((dd['day'] + DAY - dd['init']) / DAY).astype(int)
        dd['offset'] = dd['q'] + dd['init'].map(err0).to_numpy()
        dd['obs'] = obs_daily.reindex(dd['day']).to_numpy()
        daily_fc[model] = dd[dd['lead_day'] <= max_lead_days].reset_index(drop=True)

        if step < DAY:
            timing_rows += _timing(model, df, obs_int, step, timing_months, timing_max_shift)

    # Multi-model means on daily means over common (init, day)
    for name, members in (multi_model_means or {}).items():
        if not all(m in daily_fc for m in members):
            continue
        parts = [daily_fc[m].set_index(['init', 'day']) for m in members]
        common = parts[0].index
        for p in parts[1:]:
            common = common.intersection(p.index)
        mean = sum(p.loc[common, ['q', 'offset']] for p in parts) / len(parts)
        daily_fc[name] = mean.join(parts[0].loc[common, ['lead_day', 'obs']]).reset_index()

    daily_rows, monthly_rows, cont_rows = [], [], []
    keys = None
    for dd in daily_fc.values():
        k = set(zip(dd['init'], dd['day']))
        keys = k if keys is None else keys & k
    keys = keys or set()
    first = True
    for name, dd in daily_fc.items():
        dd = dd[[(i, d) in keys for i, d in zip(dd['init'], dd['day'])]].copy()
        dd['persistence'] = obs_daily.reindex(dd['init'].dt.floor('D') - DAY).to_numpy()
        dd['climatology'] = clim.reindex(np.minimum(dd['day'].dt.dayofyear, 365)).to_numpy()
        for d in range(1, max_lead_days + 1):
            s = dd[dd['lead_day'] == d]
            refs = {'persistence': s['persistence'], 'climatology': s['climatology']}
            daily_rows.append({'model': name, 'variant': 'raw', 'lead_day': d, **scores(s['q'], s['obs'], refs)})
            daily_rows.append({'model': name, 'variant': 'offset', 'lead_day': d,
                               **scores(s['offset'], s['obs'], refs)})
            if first:
                for b in ('persistence', 'climatology'):
                    daily_rows.append({'model': 'benchmark', 'variant': b, 'lead_day': d,
                                       **scores(s[b], s['obs'], refs)})
            for variant, col in (('raw', 'q'), ('offset', 'offset')):
                cont_rows.append({'model': name, 'variant': variant, 'lead_day': d,
                                  **contingency(s[col], s['obs'], threshold)})
        first = False
        ok = dd.dropna(subset=['q', 'obs'])
        for month, s in ok.groupby(ok['day'].dt.month):
            monthly_rows.append({'model': name, 'month': int(month), 'n': len(s),
                                 'obs_mean': s['obs'].mean(), 'fc_mean': s['q'].mean(),
                                 'ratio': s['q'].sum() / s['obs'].sum(),
                                 'offset_ratio': s['offset'].sum() / s['obs'].sum()})

    events = _events(daily_fc, obs_daily, threshold, event_gap_days, max_lead_days)
    info = {'high_flow_threshold': threshold,
            'high_flow_quantile': None if high_flow_threshold is not None else high_flow_quantile,
            'climatology_period': climatology_period, 'max_lead_days': max_lead_days,
            'offset_lag': str(lag), 'events': len(events),
            'issue_times': {m: int(d['init'].nunique()) for m, d in daily_fc.items()},
            'multi_model_means': {k: list(v) for k, v in (multi_model_means or {}).items()}}
    return ForecastVerification(
        pairs=pairs, lead_scores=pd.DataFrame(lead_rows), daily_scores=pd.DataFrame(daily_rows),
        monthly_bias=pd.DataFrame(monthly_rows), contingency=pd.DataFrame(cont_rows),
        events=events, timing=pd.DataFrame(timing_rows), info=info)


def _timing(model: str, df: pd.DataFrame, obs_int: pd.Series, step: pd.Timedelta,
            months: Sequence[int], max_shift: int) -> List[Dict[str, Any]]:
    """Correlation of daily-cycle anomalies (lead day 1) for model shifts of -max..max steps."""
    sub = df[(df['lead_day'] == 1) & df['end'].dt.month.isin(list(months))]
    sub = sub.sort_values(['end', 'init']).drop_duplicates('end').set_index('end')
    per_day = int(DAY / step)
    if len(sub) < 4 * per_day:
        return []
    full = pd.date_range(sub.index.min(), sub.index.max(), freq=step)
    m = sub['q'].reindex(full)
    o = obs_int.reindex(full)

    def anomaly(x: pd.Series) -> pd.Series:
        return x - x.rolling(per_day, center=True, min_periods=max(1, int(0.75 * per_day))).mean()

    am, ao = anomaly(m), anomaly(o)
    hours = step / pd.Timedelta(hours=1)
    return [{'model': model, 'shift_model_by_h': lag * hours, 'corr_of_daily_cycle': am.shift(lag).corr(ao)}
            for lag in range(-max_shift, max_shift + 1)]


def _events(daily_fc: Mapping[str, pd.DataFrame], obs_daily: pd.Series, threshold: float,
            gap: int, max_lead_days: int) -> pd.DataFrame:
    if not daily_fc:
        return pd.DataFrame()
    days = pd.DatetimeIndex(pd.concat([d['day'] for d in daily_fc.values()]).drop_duplicates().sort_values())
    od = obs_daily.reindex(days).dropna()
    rows = []
    for ev in cluster_events(od, threshold, gap):
        peak_day = max(ev, key=lambda t: od[t])
        row: Dict[str, Any] = {'start': ev[0].date(), 'end': ev[-1].date(), 'peak_day': peak_day.date(),
                               'obs_peak_daily': float(od[peak_day])}
        for name, dd in daily_fc.items():
            for d in range(1, max_lead_days + 1):
                s = dd[(dd['day'] == peak_day) & (dd['lead_day'] == d)]
                row[f'{name}_d{d}'] = float(s['q'].iloc[0]) if len(s) else np.nan
            hits = dd[dd['day'].isin(ev) & (dd['q'] > threshold)]
            row[f'{name}_warned_lead_days'] = int(hits['lead_day'].max()) if len(hits) else 0
        rows.append(row)
    return pd.DataFrame(rows)
