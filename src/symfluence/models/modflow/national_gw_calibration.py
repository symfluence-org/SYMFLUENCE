# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration of the national MODFLOW 6 groundwater parameters (docs/adr/0014-national-groundwater-calibration.md).

The HYPE calibration (ADR 0013) held groundwater fixed at literature K and SY. Spring-fed rivers (lava, high baseflow)
then scored a median KGE of 0.2 against 0.65 elsewhere: too flashy and in places far too much water. Here the
groundwater parameters are fitted with HYPE's pass-1 recharge and fast flow held fixed:

    Q_gauge(month) = HYPE pass-1 flow at the gauge + sum of MODFLOW discharge over the gauge's upstream units

MODFLOW runs in monthly stress periods (one steady state on the mean recharge, then one step per month), a few
minutes per run instead of hours for the daily model; the chosen set is checked with the daily coupled run.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

from . import national_model as nm

logger = logging.getLogger(__name__)


@dataclass
class GwParam:
    name: str          # k_<group> (log10 m/d), sy_<group>, river_leakance / seepage_leakance (log10)
    lo: float
    hi: float
    x0: float


GW_PARAMS = [
    GwParam('k_holocene_lava', 0.5, 3.5, 2.0), GwParam('k_holocene_sediment', -1.0, 2.5, 1.0),
    GwParam('k_upper_pleistocene', -2.5, 1.5, 0.0), GwParam('k_lower_pleistocene', -3.5, 1.0, -1.0),
    GwParam('k_tertiary', -4.5, 0.0, -2.0), GwParam('k_intrusive', -4.5, 0.0, -2.0),
    GwParam('sy_holocene_lava', 0.02, 0.30, 0.10), GwParam('sy_holocene_sediment', 0.03, 0.30, 0.15),
    GwParam('sy_upper_pleistocene', 0.005, 0.20, 0.05),
    GwParam('river_leakance', -1.0, 2.5, 1.0), GwParam('seepage_leakance', -3.0, 0.5, -1.0),
]


def to_params(x, params: list = GW_PARAMS) -> nm.NationalParams:
    p = nm.NationalParams()
    for q, v in zip(params, x):
        kind, rest = q.name.split('_', 1)
        if kind == 'k':
            p.k[rest] = float(10 ** v)
        elif kind == 'sy':
            p.sy[rest] = float(v)
        else:
            setattr(p, q.name, float(10 ** v))
    return p


# ---------------------------------------------------------------- monthly MODFLOW
def write_monthly(sim: Path, grid: nm.NationalGrid, attributes: pd.DataFrame, params: nm.NationalParams,
                  recharge: pd.DataFrame) -> None:
    """MODFLOW with a steady state on the mean recharge, then one stress period per month (recharge m/d per unit,
    rows = month starts)."""
    sim = Path(sim)
    nm.write_simulation(sim, grid, attributes, params, n_days=0, save_heads='all')
    days = np.array([pd.Timestamp(m).days_in_month for m in recharge.index], float)
    (sim / 'gwf.tdis').write_text(
        'BEGIN OPTIONS\n  TIME_UNITS DAYS\nEND OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  NPER {1 + len(days)}\nEND DIMENSIONS\n\nBEGIN PERIODDATA\n  1.0 1 1.0\n'
        + ''.join(f'  {d:.1f} 1 1.0\n' for d in days) + 'END PERIODDATA\n')
    units = sorted(grid.overlap.unit.unique())
    missing = [u for u in units if u not in recharge.columns]
    if missing:
        raise ValidationError(f'recharge lacks {len(missing)} units, e.g. {missing[:5]}')
    vals = recharge[units].astype(float).clip(lower=0.0).to_numpy()
    times = np.concatenate([[1.0], 1.0 + np.cumsum(days)])            # start of each month, then the end
    rowfmt = ' '.join(['%.4e'] * len(units))
    body = ['  0.0 ' + rowfmt % tuple(vals.mean(axis=0))]
    body += [f'  {t:.1f} ' + rowfmt % tuple(r) for t, r in zip(times[:-1], vals)]
    body.append(f'  {times[-1]:.1f} ' + rowfmt % tuple(vals[-1]))
    with open(sim / nm.RECHARGE_TS, 'w') as fh:
        fh.write('BEGIN ATTRIBUTES\n  NAMES ' + ' '.join(f'u{u}' for u in units) + '\n  METHODS '
                 + ' '.join(['STEPWISE'] * len(units)) + '\nEND ATTRIBUTES\n\nBEGIN TIMESERIES\n')
        fh.write('\n'.join(body) + '\nEND TIMESERIES\n')


def monthly_discharge(sim: Path, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Groundwater discharge per unit (m3/s) at the end of each month (drain table x saved heads)."""
    from scipy import sparse
    sim = Path(sim)
    dr = pd.read_csv(sim / nm.DRAIN_FILE)
    ncol = int(np.load(sim / nm.GRID_FILE)['ncol'])
    cell = (dr.row.to_numpy() - 1) * ncol + dr.col.to_numpy() - 1
    units = np.sort(dr.unit.unique())
    col = pd.Series(np.arange(len(units)), index=units)
    A = sparse.csr_matrix((dr.cond.to_numpy(), (col[dr.unit].to_numpy(), np.arange(len(dr)))),
                          shape=(len(units), len(dr)))
    elev = dr.elev.to_numpy()
    rows = []
    for totim, h in nm.read_heads(sim / 'gwf.hds'):
        if totim <= 1.0 + 1e-9:
            continue
        rows.append(A @ np.maximum(h.ravel()[cell] - elev, 0.0) / 86400.0)
    if len(rows) != len(months):
        raise ValidationError(f'{len(rows)} head records for {len(months)} months')
    return pd.DataFrame(np.array(rows), index=months, columns=units)


# ---------------------------------------------------------------- objective
def kge(s: np.ndarray, o: np.ndarray) -> float:
    m = np.isfinite(s) & np.isfinite(o)
    s, o = s[m], o[m]
    if len(o) < 24 or o.std() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))


def gauge_scores(gw: pd.DataFrame, fast: pd.DataFrame, obs: pd.DataFrame, upstream: dict, win: slice) -> pd.DataFrame:
    """Per gauge: monthly KGE and KGE of log flows of fast flow + upstream groundwater discharge."""
    out = {}
    for g, ups in upstream.items():
        u = [x for x in ups if x in gw.columns]
        q = fast[g] + (gw[u].sum(axis=1) if u else 0.0)
        o = obs[g]
        s, oo = q.loc[win].to_numpy(float), o.loc[win].to_numpy(float)
        eps = 0.01 * np.nanmean(oo)
        out[g] = dict(kge=kge(s, oo), kge_log=kge(np.log(s + eps), np.log(oo + eps)),
                      beta=float(np.nanmean(s[np.isfinite(oo)]) / np.nanmean(oo)))
    return pd.DataFrame(out).T


def objective(sc: pd.DataFrame, gauges: pd.DataFrame) -> float:
    """0.5 x median over the calibration gauges + 0.5 x median over the spring-fed ones, each gauge scored by the
    mean of its monthly KGE and log-KGE (gauges flagged `exclude` left out)."""
    g = gauges.set_index('gauge')
    use = g[(g.set == 'cal') & ~g.exclude].index
    v = sc.reindex(use)[['kge', 'kge_log']].mean(axis=1)
    spring = g.loc[use, 'regime'] == 'spring-fed'
    a, b = v.median(), v[spring.to_numpy()].median()
    return float(0.5 * a + 0.5 * b) if np.isfinite(a) and np.isfinite(b) else -10.0


# ---------------------------------------------------------------- driver
@dataclass
class GwSetup:
    data: Path                  # prepared inputs (prepare_inputs)
    root: Path                  # run directories + journal
    mf6_exe: Path
    params: list = field(default_factory=lambda: list(GW_PARAMS))
    eval_start: str = '2000-10-01'
    timeout_s: float = 2400.0         # a run slower than this scores worst (slow = extreme parameters)

    def load(self):
        d = Path(self.data)
        self.grid = nm.NationalGrid.load(d / 'MF6_grid')
        self.att = pd.read_csv(d / 'attributes.csv', index_col=0)
        self.recharge = pd.read_csv(d / 'recharge_monthly.csv', index_col=0, parse_dates=True)
        self.recharge.columns = self.recharge.columns.astype(int)
        self.fast = pd.read_csv(d / 'fast_monthly.csv', index_col=0, parse_dates=True)
        self.obs = pd.read_csv(d / 'obs_monthly.csv', index_col=0, parse_dates=True)
        self.fast.columns = self.fast.columns.astype(int)
        self.obs.columns = self.obs.columns.astype(int)
        self.gauges = pd.read_csv(d / 'gauges.csv')
        self.upstream = {int(k): v for k, v in json.loads((d / 'upstream.json').read_text()).items()}
        return self


@dataclass
class GwEval:
    x: list
    score: float
    scores: dict
    seconds: float
    tag: str


def evaluate(setup: GwSetup, x, tag: str, keep: bool = False) -> GwEval:
    t0 = time.time()
    sim = Path(setup.root) / tag
    shutil.rmtree(sim, ignore_errors=True)
    try:
        write_monthly(sim, setup.grid, setup.att, to_params(x, setup.params), setup.recharge)
        with open(sim / 'mf6.log', 'w') as fh:
            r = subprocess.run([str(setup.mf6_exe)], cwd=sim, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=setup.timeout_s)
        lst = (sim / 'mfsim.lst').read_text(errors='replace')
        if r.returncode != 0 or 'Normal termination' not in lst:
            raise ValidationError(f'MODFLOW failed ({r.returncode})')
        gw = monthly_discharge(sim, setup.recharge.index)
        sc = gauge_scores(gw, setup.fast, setup.obs, setup.upstream, slice(pd.Timestamp(setup.eval_start), None))
        s = objective(sc, setup.gauges)
        if keep:
            gw.to_csv(sim / 'gw_monthly.csv.gz', float_format='%.5g')
    except Exception as exc:
        logger.warning('%s failed: %s', tag, exc)
        return GwEval(list(map(float, x)), -10.0, {}, time.time() - t0, tag)
    finally:
        if not keep:
            shutil.rmtree(sim, ignore_errors=True)
    return GwEval(list(map(float, x)), s, {int(k): {kk: float(vv) for kk, vv in r.items()} for k, r in sc.iterrows()},
                  time.time() - t0, tag)


def calibrate(setup: GwSetup, n_iter: int, batch: int, seed: int = 2) -> GwEval:
    """Asynchronous parallel DDS: `batch` workers stay busy; whenever a run finishes, a new candidate is drawn around
    the current best (DDS neighbourhood size follows the evaluation count, n_iter x batch evaluations in all).
    Journal and resume as in national_calibration."""
    from symfluence.models.hype.national_calibration import dds_candidates
    lo = np.array([p.lo for p in setup.params])
    hi = np.array([p.hi for p in setup.params])
    rng = np.random.default_rng(seed + 1000 * _journal_len(setup))
    journal = Path(setup.root) / 'journal.jsonl'
    Path(setup.root).mkdir(parents=True, exist_ok=True)
    best, done = None, 0
    if journal.exists():
        for line in journal.read_text().splitlines():
            e = GwEval(**json.loads(line))
            done += 1
            if best is None or e.score > best.score:
                best = e
    if best is None:
        best = evaluate(setup, np.array([p.x0 for p in setup.params]), 'e0000')
        with journal.open('a') as fh:
            fh.write(json.dumps(best.__dict__) + '\n')
        done = 1
    total = n_iter * batch
    nxt = done
    with ThreadPoolExecutor(max_workers=batch) as pool:
        futs = set()

        def submit(k):
            c = dds_candidates(np.array(best.x), lo, hi, 1 + k // batch, n_iter, 1, rng)[0]
            futs.add(pool.submit(evaluate, setup, c, f'e{k:04d}'))

        while nxt < total and len(futs) < batch:
            submit(nxt)
            nxt += 1
        while futs:
            fin, _ = wait(futs, return_when=FIRST_COMPLETED)
            for f in fin:
                futs.discard(f)
                e = f.result()
                with journal.open('a') as fh:
                    fh.write(json.dumps(e.__dict__) + '\n')
                if e.score > best.score:
                    best = e
                    logger.info('%s: new best %.3f', e.tag, best.score)
                if nxt < total:
                    submit(nxt)
                    nxt += 1
    return best


def _journal_len(setup: GwSetup) -> int:
    j = Path(setup.root) / 'journal.jsonl'
    return sum(1 for _ in j.open()) if j.exists() else 0
