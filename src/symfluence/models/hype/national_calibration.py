# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration of the national HYPE model against LamaH-Ice gauges and the national glacier mass balance
(docs/adr/0013-national-calibration.md).

* Gauges: LamaH-Ice river gauges whose reference catchment matches the fabric (area_validation include, not an
  effective catchment) and whose upstream area has no transfer endpoint or reservoir. They are split into a
  calibration and a spatial validation set (every third gauge by area rank is held out).
* Sub-model: the calibration runs on the union of the gauges' upstream areas and every glacier subbasin, written
  as a self-contained HYPE settings directory (GeoData, classes, forcing columns and dates, point sources, lakes,
  glaciers all subset), so HYPE reads and writes only what the objective needs. Groundwater returns from the
  latest MODFLOW run enter as fixed point sources (the second pass of the two-pass coupling).
* Glacier mass balance: the run carries ice between subbasins (national_glacier.IceFlux) and the objective
  penalises the departure of the simulated national ice-mass change from the surface balance implied by
  Adalgeirsdottir et al. (2020) for 1994/95-2009/10 (GLACIER_MB_GT_YR; the window must lie in that period).
* Snow: the simulated mean annual maximum SWE on glacier-free units is compared with the CARRA 1991-2020
  normals in the fabric (`snow_ref.csv` in the template); the Tungnaa parameters melted snow from -2.6 C and kept
  about a tenth of CARRA's highland snowpack.
* Search: parallel DDS (Tolson & Shoemaker 2007) with a batch of candidates per iteration.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Surface mass balance target for 1994/95-2009/10: total -11.6 +/- 0.8 Gt/yr (Adalgeirsdottir et al. 2020, Sect. 4)
# minus the non-surface balance HYPE does not simulate (geothermal, dissipation, calving: Vatnajokull about
# -0.20 m w.e. over 7,700 km2, Langjokull -0.055 over 835 km2, Hofsjokull -0.067 over 810 km2; about -1.6 Gt/yr).
GLACIER_MB_GT_YR = (-10.0, 1.0)
GLACDENS = 0.85                         # HYPE glacdens: water equivalent of a m3 of glacier ice (Gt per km3)
# Per-ice-cap surface mass balance (glaciological, m w.e./yr): VÍ glaciers API /glacier/mass_balance (CC BY 4.0),
# annual ba by hydrological year; uncertainty of the annual glaciological balance (Adalgeirsdottir et al. 2020).
CAP_SIGMA = {'Vatnajökull': 0.10, 'Langjökull': 0.07, 'Hofsjökull': 0.07}


def cap_balance(out: Path, caps: pd.DataFrame, geodata_subids: np.ndarray, y0: int, y1: int) -> dict:
    """Simulated surface mass balance per ice cap (m w.e./yr per hydrological year y0..y1) from the 1 October
    states: {cap: {year: ba}}. `caps` has subid, cap, glac_km2. Ice flow is within ice basins, which do not cross
    ice caps, so the cap totals are not affected by the redistribution."""
    from symfluence.models.hype.national_glacier import read_glacvol
    c = caps.set_index('subid').reindex(geodata_subids)
    vol = {}
    for f in Path(out).glob('era*_*/state_save*1001.txt'):
        v = pd.Series(read_glacvol(f), index=geodata_subids)
        vol[int(f.name[10:14])] = v.groupby(c.cap.to_numpy()).sum()
    area = c.groupby('cap').glac_km2.sum()
    res = {}
    for y in range(y0, y1 + 1):
        if y in vol and y - 1 in vol:
            d = (vol[y] - vol[y - 1]) / 1e9 * GLACDENS * 1000.0 / area.reindex(vol[y].index)   # Gt -> m w.e.
            for cap, b in d.items():
                res.setdefault(cap, {})[y] = float(b)
    return res


def cap_penalty(sim: dict, obs: dict, sigma: dict = CAP_SIGMA) -> tuple[float, dict]:
    """Mean over the measured ice caps of |mean sim - mean obs| / (3 sigma), over the years both have."""
    terms, diag = [], {}
    for cap, s in sigma.items():
        ys = sorted(set(sim.get(cap, {})) & {int(y) for y in obs.get(cap, {})})
        if not ys:
            continue
        ms = float(np.mean([sim[cap][y] for y in ys]))
        mo = float(np.mean([obs[cap][str(y)] if str(y) in obs[cap] else obs[cap][y] for y in ys]))
        terms.append(abs(ms - mo) / (3 * s))
        diag[cap] = {'sim': ms, 'obs': mo, 'n': len(ys)}
    return (float(np.mean(terms)) if terms else np.nan), diag


# ---------------------------------------------------------------- gauges
def upstream_sets(ids: np.ndarray, toid: np.ndarray) -> callable:
    up: dict[int, list] = {}
    for a, t in zip(ids.astype(int), toid.astype(int)):
        up.setdefault(t, []).append(a)

    def ups(u: int) -> set:
        out, stack = set(), [int(u)]
        while stack:
            x = stack.pop()
            if x not in out:
                out.add(x)
                stack += up.get(x, [])
        return out
    return ups


def select_gauges(link: pd.DataFrame, divides: pd.DataFrame, transfers: pd.DataFrame,
                  lakes: pd.DataFrame) -> pd.DataFrame:
    """Unregulated LamaH river gauges with a matching catchment; column `set` is 'cal' or 'val'."""
    ups = upstream_sets(divides.id.to_numpy(), divides.toid.to_numpy())
    g = link[(link.observation_type == 'river') & (link.area_validation == 'include')
             & (link.effective_catchment.astype(str) != 'True') & link.unit_id.notna()].copy()
    ends = set(pd.concat([transfers.from_unit, transfers.to_unit]).dropna().astype(int))
    res = set(lakes.loc[lakes.reservoir.astype(str) == 'True', 'unit_id'].dropna().astype(int))
    g['upstream'] = g.unit_id.astype(int).map(ups)
    g['regulated'] = g.upstream.map(lambda s: bool(s & (ends | res)))
    g = g[~g.regulated].sort_values('area_fabric_km2').reset_index(drop=True)
    g['set'] = np.where(np.arange(len(g)) % 3 == 1, 'val', 'cal')
    return g


def read_lamah(root: Path, path: str) -> pd.Series:
    d = pd.read_csv(Path(root) / 'D_gauges' / path.split('D_gauges/')[-1], sep=';')
    q = pd.Series(d.qobs.to_numpy(float), pd.to_datetime(dict(year=d.YYYY, month=d.MM, day=d.DD)))
    return q.where(q >= 0)


# ---------------------------------------------------------------- sub-model
def _subset_table(path: Path, out: Path, col: str, keep: set, sep='\t') -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_csv(path, sep=sep)
    d = d[d[col].astype(int).isin(keep)]
    d.to_csv(out, sep=sep, index=False)
    return d


def _subset_series(src: Path, dst: Path, cols: list, start, end) -> None:
    """Copy a HYPE time-series file keeping the given column ids and dates. The id row is the first header row,
    or the second when the first holds variable names (PSDailySeries: DATE/flow, then psid/ids)."""
    want = set(str(int(c)) for c in cols)
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    with open(src) as fh, open(dst, 'w') as out:
        heads = [fh.readline()]
        second = fh.readline()
        first_data = None
        if second.split('\t')[0].strip().lower() in ('psid', 'subid', 'id'):
            heads.append(second)
        else:
            first_data = second
        ids = heads[-1].rstrip('\n').split('\t')
        idx = [0] + [i for i, c in enumerate(ids) if i and c.strip().split('.')[0] in want]
        for h in heads:
            f = h.rstrip('\n').split('\t')
            out.write('\t'.join(f[i] for i in idx) + '\n')
        for line in itertools.chain([first_data] if first_data else [], fh):
            d = line[:10]
            t = pd.Timestamp(d)
            if t < a:
                continue
            if t > b:
                break
            f = line.rstrip('\n').split('\t')
            out.write('\t'.join(f[i] for i in idx) + '\n')


def build_submodel(src: Path, dst: Path, units: set, start, end, gauges: pd.DataFrame, lamah_root: Path) -> dict:
    """Self-contained HYPE settings for `units` over [start, end], with Qobs at the calibration gauges."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    gd = pd.read_csv(src / 'GeoData.txt', sep='\t')
    gd = gd[gd.subid.astype(int).isin(units)].copy()
    gd.loc[~gd.maindown.astype(int).isin(units), 'maindown'] = 0
    gd.to_csv(dst / 'GeoData.txt', sep='\t', index=False)
    keep = set(gd.subid.astype(int))
    for f in ('GeoClass.txt', 'par.txt', 'info.txt'):
        shutil.copy2(src / f, dst / f)
    _subset_table(src / 'GlacierData.txt', dst / 'GlacierData.txt', 'subid', keep)
    _subset_table(src / 'LakeData.txt', dst / 'LakeData.txt', 'subid', keep)
    fk = _subset_table(src / 'ForcKey.txt', dst / 'ForcKey.txt', 'subid', keep)
    if (src / 'BranchData.txt').exists():
        b = pd.read_csv(src / 'BranchData.txt', sep='\t')
        b = b[b.sourceid.astype(int).isin(keep) & (b.branchid.astype(int).isin(keep) | (b.branchid == 0))]
        if len(b):
            b.to_csv(dst / 'BranchData.txt', sep='\t', index=False)
    if (src / 'national_transfers.csv').exists():
        t = pd.read_csv(src / 'national_transfers.csv')
        t[t.from_unit.astype(int).isin(keep) & t.to_unit.astype(int).isin(keep)].to_csv(
            dst / 'national_transfers.csv', index=False)
    ps = _subset_table(src / 'PointSourceData.txt', dst / 'PointSourceData.txt', 'subid', keep)
    if ps is not None:
        _subset_series(src / 'PSDailySeries.txt', dst / 'PSDailySeries.txt', ps.psid.astype(int).tolist(), start, end)
    stations = (fk.stationid if fk is not None and 'stationid' in fk else gd.subid).astype(int).tolist()
    for f in sorted(src.glob('*obs.txt')):
        if f.name == 'Qobs.txt':
            continue
        _subset_series(f.resolve(), dst / f.name, stations, start, end)
    # observed discharge at the gauges inside the sub-model
    days = pd.date_range(start, end)
    q = {}
    for _, g in gauges.iterrows():
        if int(g.unit_id) in keep:
            q[int(g.unit_id)] = read_lamah(lamah_root, g.ts_daily).reindex(days)
    qo = pd.DataFrame(q, index=days).fillna(-9999)
    qo.index.name = 'DATE'
    qo.to_csv(dst / 'Qobs.txt', sep='\t', date_format='%Y-%m-%d', float_format='%.3f')
    (dst / 'filedir.txt').write_text(f'{dst}/\n')
    return dict(subbasins=len(gd), gauges=len(q), start=str(start), end=str(end))


# ---------------------------------------------------------------- parameters
@dataclass
class Param:
    name: str
    lo: float
    hi: float
    mode: str = 'mult'          # 'mult': scale the base row; 'add': add to it; 'set': set every value
    col: int | None = None      # only this value of the row (0-based), e.g. one soil type; applied after the
                                # whole-row parameters listed before it, so it acts relative to them

    def apply(self, base: list, x: float) -> list:
        op = {'mult': lambda v: v * x, 'add': lambda v: v + x}.get(self.mode, lambda v: x)
        if self.col is None:
            return [op(v) for v in base]
        out = list(base)
        out[self.col] = op(out[self.col])
        return out


DEFAULT_PARAMS = [
    Param('cmlt', 0.5, 2.0), Param('ttmp', -1.0, 4.0, 'add'), Param('cmrad', 0.3, 2.0),
    Param('glaccmlt', 2.0, 12.0, 'set'), Param('glacttmp', -2.0, 2.0, 'set'), Param('glaccmrad', 0.0, 1.0, 'set'),
    Param('tcalt', 0.3, 0.9, 'set'), Param('pcusnow', 0.0, 0.5, 'set'),
    Param('cevp', 0.5, 2.0), Param('wcfc', 0.5, 3.5), Param('wcep', 0.25, 1.8),   # wcwp+wcfc+wcep <= 0.96
    Param('rrcs1', 0.3, 3.0), Param('rrcs2', 0.3, 3.0), Param('srrcs', 0.3, 2.0),
    Param('mperc1', 0.2, 3.0), Param('mperc2', 0.2, 3.0),
    Param('rivvel', 0.5, 10.0, 'set'), Param('damp', 0.05, 0.9, 'set'),
]
# Lava soil type (national_lava, soil type 4 = column 3), relative to the whole-row values: faster percolation to
# the aquifer, slower lateral soil runoff, more infiltration into frozen ground. 1.0 reproduces the unsplit model.
LAVA_PARAMS = [
    Param('mperc1', 1.0, 20.0, col=3), Param('mperc2', 1.0, 20.0, col=3),
    Param('rrcs1', 0.05, 1.0, col=3), Param('rrcs2', 0.05, 1.0, col=3),
    Param('bfroznsoil', 1.0, 20.0, col=3),     # frozen-soil infiltration capacity (soil_proc.f90): higher = more
]


def read_par(path: Path) -> list:
    return Path(path).read_text().splitlines()


def write_par(lines: list, params: list, x: np.ndarray, out: Path) -> None:
    rows = list(lines)
    names = [r.split('\t')[0].split()[0] if r.strip() else '' for r in rows]
    for p, v in zip(params, x):
        if p.name in names:
            i = names.index(p.name)
            f = rows[i].split('\t')
            vals, rest = [], []
            for tok in f[1:]:
                try:
                    vals.append(float(tok))
                except ValueError:
                    rest.append(tok)
            rows[i] = '\t'.join([p.name] + [repr(float(a)) for a in p.apply(vals, v)] + rest)
        elif p.mode == 'set':
            rows.append(f'{p.name}\t{float(v)!r}')
        else:
            raise ValidationError(f'{p.name} not in par.txt')
    Path(out).write_text('\n'.join(rows) + '\n')


# ---------------------------------------------------------------- objective
def kge_log(sim: np.ndarray, obs: np.ndarray) -> float:
    """KGE of log flows (offset 1 % of the mean observed flow): weights low flows and baseflow."""
    eps = 0.01 * np.nanmean(obs)
    return kge(np.log(np.asarray(sim, float) + eps), np.log(np.asarray(obs, float) + eps))


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 365:
        return np.nan
    s, o = sim[m], obs[m]
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def read_cout(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep='\t', skiprows=1, index_col=0, parse_dates=True)
    d.columns = [int(float(c)) for c in d.columns]
    return d.where(d > -9998)


def gw_return(aqin: pd.DataFrame, ref_file: Path, geodata: pd.DataFrame, gauges, tau_days: float) -> pd.DataFrame:
    """Surrogate for MODFLOW's response to a change in recharge during HYPE calibration (ADR 0014).

    The groundwater point sources are fixed: MODFLOW run on a reference recharge (`ref_file`: dates, subids, values
    m3/d). A parameter set that sends more (or less) water to the aquifer would otherwise lose (or gain) it. The
    difference AQIN - reference, summed over each gauge's upstream units, passes through a linear reservoir with
    time constant `tau_days` and is added to the gauge's flow (m3/s). Local return only: the coupled rerun
    replaces it with MODFLOW."""
    z = np.load(ref_file)
    ref = pd.DataFrame(z['values'], index=pd.DatetimeIndex(z['dates']), columns=z['subids'].astype(int))
    ups = upstream_sets(geodata.subid.to_numpy(int), geodata.maindown.to_numpy(int))
    d = aqin.sub(ref.reindex(index=aqin.index, columns=aqin.columns), fill_value=0.0) / 86400.0
    a = float(np.exp(-1.0 / tau_days))
    out = {}
    for g in gauges:
        x = d[[u for u in ups(int(g)) if u in d.columns]].sum(axis=1).to_numpy(float)
        y = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = a * acc + (1.0 - a) * v
            y[i] = acc
        out[int(g)] = y
    return pd.DataFrame(out, index=aqin.index)


def glacier_rate(out: Path, y0: int, y1: int) -> float:
    """Mean national ice-mass change (Gt/yr) over hydrological years y0..y1 (ending 30 September), from the
    1 October states the eras saved (before redistribution, which conserves the total)."""
    from symfluence.models.hype.national_glacier import read_glacvol
    tot = {}
    for f in Path(out).glob('era*_*/state_save*1001.txt'):
        tot[int(f.name[10:14])] = read_glacvol(f).sum() / 1e9
    ch = [tot[y] - tot[y - 1] for y in range(y0, y1 + 1) if y in tot and y - 1 in tot]
    return float(np.mean(ch) * GLACDENS) if ch else np.nan


@dataclass
class Evaluation:
    x: list
    score: float
    kge_median: float
    kge: dict
    glacier_gt_yr: float
    seconds: float
    run: str = ''
    meta: dict = field(default_factory=dict)


def snow_error(snow_file: Path, ref_file: Path, win: slice) -> dict:
    """Simulated snow against the CARRA 1991-2020 normals in the fabric (clim_swe_annual_max_mean,
    clim_snow_days_yr) on glacier-free units: median |ln(sim/CARRA)| of the mean annual maximum SWE and the
    median relative error of snow days (SWE > 1 mm), hydrological years (October-September) in `win`."""
    if not Path(snow_file).exists() or not Path(ref_file).exists():
        return {}
    ref = pd.read_csv(ref_file).set_index('subid')
    sn = read_cout(snow_file).loc[win]
    sn = sn[[c for c in sn.columns if c in ref.index]]
    hy = sn.index.year + (sn.index.month >= 10)
    swe_max = sn.groupby(hy).max().mean()
    days = (sn > 1.0).groupby(hy).sum().mean()
    r = ref.loc[swe_max.index]
    lr = np.log(np.maximum(swe_max, 1.0) / r.swe_max)
    return dict(log_swe_max=float(np.median(np.abs(lr))), swe_max_ratio_median=float(np.exp(np.median(lr))),
                snow_days_rel=float(np.median((days - r.snow_days) / r.snow_days)), units=int(len(r)))


def score(kges: dict, glacier_gt_yr: float, snow_log_err: float = np.nan, w_glacier: float = 0.5,
          w_snow: float = 0.25) -> float:
    """Objective to maximise: median KGE over calibration gauges minus glacier mass-balance and snow penalties."""
    vals = np.array([v for v in kges.values() if np.isfinite(v)])
    med = float(np.median(vals)) if len(vals) else -10.0
    target, sd = GLACIER_MB_GT_YR
    pen = w_glacier * abs(glacier_gt_yr - target) / (3 * sd) if np.isfinite(glacier_gt_yr) else 10.0
    if np.isfinite(snow_log_err):
        pen += w_snow * snow_log_err
    return med - pen


# ---------------------------------------------------------------- DDS
def dds_candidates(best: np.ndarray, lo: np.ndarray, hi: np.ndarray, it: int, n_it: int, k: int,
                   rng: np.random.Generator, r: float = 0.2) -> np.ndarray:
    d = len(best)
    p = max(1.0 - np.log(it + 1) / np.log(n_it + 1), 1.0 / d)
    out = np.empty((k, d))
    for j in range(k):
        sel = rng.random(d) < p
        if not sel.any():
            sel[rng.integers(d)] = True
        x = best.copy()
        x[sel] += r * (hi[sel] - lo[sel]) * rng.standard_normal(sel.sum())
        x = np.where(x < lo, lo + (lo - x), x)
        x = np.where(x > hi, hi - (x - hi), x)
        out[j] = np.clip(x, lo, hi)
    return out


# ---------------------------------------------------------------- evaluation and driver
@dataclass
class CalibSetup:
    template: Path              # sub-model settings from build_submodel
    root: Path                  # run directories
    hype_exe: Path
    spinup_start: str
    cal_start: str
    end: str
    gauges: pd.DataFrame        # with unit_id, set
    params: list
    mb_years: tuple = (1996, 2009)   # hydrological years (ending 30 Sep) used for the glacier rate (inside 1995-2010)
    objective: str = 'median'        # 'median': median KGE; 'mixed': per gauge mean(KGE, log-KGE), half the median
                                     # over the calibration gauges + half over the spring-fed ones (ADR 0014)
    spring: frozenset = frozenset()  # unit ids of spring-fed gauges
    exclude: frozenset = frozenset() # unit ids left out of the objective
    cap_obs: dict | None = None      # {cap: {year: ba m w.e.}}; with template/glacier_caps.csv adds a per-cap term
    gw_tau_days: float | None = None # with template/aqin_ref.npz: MODFLOW surrogate for recharge changes (gw_return)
    w_caps: float = 0.25             # global parameters cannot fix opposite cap errors (forcing); keep it moderate


def _link_template(template: Path, run: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    for f in template.iterdir():
        if f.is_file() and f.name not in ('par.txt', 'info.txt', 'BranchData.txt', 'filedir.txt'):
            t = run / f.name
            if not t.exists():
                os.link(f, t)
    for f in ('info.txt', 'BranchData.txt'):
        if (template / f).exists():
            shutil.copy2(template / f, run / f)
    (run / 'filedir.txt').write_text(f'{run}/\n')


def evaluate(setup: CalibSetup, x: np.ndarray, tag: str, keep: bool = False) -> Evaluation:
    from symfluence.models.hype.national_glacier import IceFlux
    from symfluence.models.modflow.national_coupling import run_eras, set_info
    t0 = time.time()
    run = Path(setup.root) / tag
    if run.exists():
        shutil.rmtree(run)
    _link_template(Path(setup.template), run)
    write_par(read_par(Path(setup.template) / 'par.txt'), setup.params, x, run / 'par.txt')
    vars_ = 'cout\tsnow\taqin' if setup.gw_tau_days else 'cout\tsnow'
    set_info(run, {'timeoutput variable': vars_, 'timeoutput meanperiod': 1, 'timeoutput decimals': 3,
                   'submodel': 'n', 'modeloption glacierini': 1})
    out = run / 'out'
    try:
        log = run_eras(Path(setup.hype_exe), run, out, setup.spinup_start, setup.end, IceFlux.from_settings(run))
    except Exception as exc:      # a parameter set that crashes HYPE scores worst
        logger.warning('%s failed: %s', tag, exc)
        return Evaluation(list(map(float, x)), -10.0, np.nan, {}, np.nan, time.time() - t0, str(run))
    ev = score_run(setup, run, x, time.time() - t0)
    if not keep:
        shutil.rmtree(run, ignore_errors=True)
    return ev


def score_run(setup: CalibSetup, run: Path, x, seconds: float = np.nan) -> Evaluation:
    out = Path(run) / 'out'
    cout = read_cout(out / 'timeCOUT.txt')
    ref = Path(setup.template) / 'aqin_ref.npz'
    if setup.gw_tau_days and ref.exists() and (out / 'timeAQIN.txt').exists():
        geo = pd.read_csv(run / 'GeoData.txt', sep='\t', usecols=['subid', 'maindown'])
        gauged = [int(c) for c in pd.read_csv(run / 'Qobs.txt', sep='\t', nrows=0).columns[1:]]
        cout = cout.add(gw_return(read_cout(out / 'timeAQIN.txt'), ref, geo, gauged, setup.gw_tau_days),
                        fill_value=0.0).clip(lower=0.0)
    qobs = pd.read_csv(run / 'Qobs.txt', sep='\t', index_col=0, parse_dates=True)
    qobs.columns = [int(c) for c in qobs.columns]
    qobs = qobs.where(qobs > -9998)
    win = slice(pd.Timestamp(setup.cal_start), pd.Timestamp(setup.end))
    kges, klog = {}, {}
    for u in qobs.columns:
        if u in cout.columns:
            s_, o_ = cout.loc[win, u].to_numpy(float), qobs.loc[win, u].to_numpy(float)
            kges[u], klog[u] = kge(s_, o_), kge_log(s_, o_)
    gt = glacier_rate(out, *setup.mb_years)
    snow = snow_error(out / 'timeSNOW.txt', Path(setup.template) / 'snow_ref.csv', win)
    cal = set(setup.gauges.loc[setup.gauges.set == 'cal', 'unit_id'].astype(int)) - set(setup.exclude)
    kc = {u: v for u, v in kges.items() if u in cal}
    vals = [v for v in kc.values() if np.isfinite(v)]
    if setup.objective == 'mixed':
        mixed = {u: np.nanmean([kges[u], klog[u]]) for u in kc}
        a = np.nanmedian(list(mixed.values()))
        b = np.nanmedian([v for u, v in mixed.items() if u in setup.spring])
        sc = score({0: 0.5 * a + 0.5 * b}, gt, snow.get('log_swe_max', np.nan))
    else:
        sc = score(kc, gt, snow.get('log_swe_max', np.nan))
    meta = dict(snow, kge_log={int(k): float(v) for k, v in klog.items()})
    caps_file = Path(setup.template) / 'glacier_caps.csv'
    if setup.cap_obs and caps_file.exists():
        sub = pd.read_csv(run / 'GeoData.txt', sep='\t', usecols=['subid']).subid.to_numpy(int)
        pen, diag = cap_penalty(cap_balance(out, pd.read_csv(caps_file), sub, *setup.mb_years), setup.cap_obs)
        sc = sc - setup.w_caps * pen if np.isfinite(pen) else sc - 10.0
        meta['caps'] = diag
    return Evaluation(list(map(float, x)), sc, float(np.median(vals)) if vals else np.nan,
                      {int(k): float(v) for k, v in kges.items()}, gt, seconds, str(run), meta)


def calibrate(setup: CalibSetup, n_iter: int, batch: int, seed: int = 1, x0: np.ndarray | None = None,
              journal: Path | None = None) -> Evaluation:
    """Parallel DDS: each iteration evaluates `batch` candidates around the best in parallel."""
    lo = np.array([p.lo for p in setup.params])
    hi = np.array([p.hi for p in setup.params])
    rng = np.random.default_rng(seed)
    journal = Path(journal or Path(setup.root) / 'journal.jsonl')
    best = None
    if journal.exists():          # resume
        for line in journal.read_text().splitlines():
            e = Evaluation(**json.loads(line))
            if best is None or e.score > best.score:
                best = e
    start_it = sum(1 for _ in journal.open()) // batch if journal.exists() else 0
    if best is None:
        base = x0 if x0 is not None else np.array([1.0 if p.mode == 'mult' else 0.0 if p.mode == 'add'
                                                    else (p.lo + p.hi) / 2 for p in setup.params])
        best = evaluate(setup, base, 'it0000_0')
        with journal.open('a') as fh:
            fh.write(json.dumps(best.__dict__) + '\n')
    with ThreadPoolExecutor(max_workers=batch) as pool:      # workers wait on HYPE subprocesses
        for it in range(max(start_it, 1), n_iter + 1):
            cands = dds_candidates(np.array(best.x), lo, hi, it, n_iter, batch, rng)
            futs = [pool.submit(evaluate, setup, c, f'it{it:04d}_{j}') for j, c in enumerate(cands)]
            for f in futs:
                e = f.result()
                with journal.open('a') as fh:
                    fh.write(json.dumps(e.__dict__) + '\n')
                if e.score > best.score:
                    best = e
            logger.info('iteration %d: best %.3f (median KGE %.3f, glacier %.2f Gt/yr)', it, best.score,
                        best.kge_median, best.glacier_gt_yr)
    return best
