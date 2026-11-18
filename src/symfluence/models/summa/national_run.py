# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Run national SUMMA (ice and land runs, GRU blocks in parallel) and route the GRU runoff with dRoute (ADR 0015).

Chunks of whole water years: each chunk writes its forcing (``national.write_forcing``), runs every GRU block of
both runs from the previous chunk's restart, then deletes the forcing (about 0.8 GB per year). Setup costs about
0.5 s per HRU (the enthalpy lookup tables of ``nrgConserv enthalpyForm``), so chunks are several years long.

    python national_run.py summa <fabric_dir> <domain_dir> <carra_3h_dir> <name> <wy0> <wy1> [--procs N] [--chunk Y]
    python national_run.py route <fabric_dir> <domain_dir> <name>
    python national_run.py chunk - <domain_dir> <name> <wy0> <wy1> --exe .. --init-ice .. --init-land ..
        (server side of national_remote.py: forcing already in place)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import national as N

logger = logging.getLogger(__name__)
EXE = Path(__file__).resolve().parents[5] / 'SYMFLUENCE_data' / 'installs' / 'summa-glacier' / 'bin' / 'summa.exe'
# relative cost per GRU for block balancing: HRUs, ice HRUs carry a second (glacier) domain
ICE_COST = 2.0


def blocks(hrus: pd.DataFrame, run: str, n: int) -> list:
    """Contiguous GRU blocks (1-based start, count) of about equal cost (HRUs, ice x ICE_COST)."""
    h = hrus[hrus.run == run]
    cost = h.groupby('gruId', sort=False).size().to_numpy() * (ICE_COST if run == 'ice' else 1.0)
    edges = np.searchsorted(np.cumsum(cost), np.linspace(0, cost.sum(), n + 1)[1:-1])
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(cost)]
    return [(int(s) + 1, int(e - s)) for s, e in zip(starts, ends) if e > s]


def run_chunk(domain: Path, run: str, tag: str, start: str, end: str, forcing: list, out: Path, init: str,
              procs: int, exe: Path = EXE, niceness: int = 0) -> list:
    """Run every GRU block of one run over [start, end] (SUMMA time strings, 3-h period ends) from `init`,
    writing a restart at the end (-r e)."""
    hrus = pd.read_csv(domain / 'settings' / 'SUMMA' / 'hrus.csv')
    flist = domain / 'settings' / 'SUMMA' / run / f'forcing_{tag}.txt'
    flist.write_text(''.join(f'{f}\n' for f in forcing))
    out.mkdir(parents=True, exist_ok=True)
    fm = N.write_filemanager(domain, run, start, end, flist.name, out, f'{run}_{tag}', init=init)
    jobs = blocks(hrus, run, procs)
    nice = ['nice', '-n', str(niceness)] if niceness else []

    def one(job):
        s, c = job
        log = out / f'{run}_{tag}_G{s:05d}.log'
        t = time.time()
        # blocks open the same settings files at once: HDF5 file locking made most of 9 fail to open
        # trialParams.nc ("netcdf-file_open")
        env = dict(os.environ, HDF5_USE_FILE_LOCKING='FALSE')
        with open(log, 'w') as f:
            r = subprocess.run(nice + [str(exe), '-g', str(s), str(c), '-r', 'e', '-m', str(fm)], stdout=f,
                               stderr=subprocess.STDOUT, cwd=out, env=env)
        return s, c, r.returncode, time.time() - t, log
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(one, jobs))
    bad = [r for r in res if r[2] != 0 or 'finished simulation successfully' not in r[4].read_text()]
    if bad:
        raise RuntimeError(f'{run} {tag}: {len(bad)} blocks failed, e.g. {bad[0][4]}')
    logger.info('%s %s: %d blocks, slowest %.0f s', run, tag, len(res), max(r[3] for r in res))
    return res


def gru_runoff(out: Path) -> pd.DataFrame:
    """Daily GRU runoff (m/s, averageRoutedRunoff) from every *_day.nc block file in out, [day, gru]."""
    import xarray as xr
    parts = []
    for f in sorted(out.glob('*_day.nc')):
        with xr.open_dataset(f) as d:
            parts.append(pd.DataFrame(d.averageRoutedRunoff.values, index=pd.DatetimeIndex(d.time.values),
                                      columns=d.gruId.values.astype(int)))
    df = pd.concat(parts, axis=1)
    df = df.T.groupby(level=0).first().T          # chunks: same GRU in several files -> columns per GRU
    return df.sort_index()


def route(fabric_dir: Path, runoff: pd.DataFrame, dt: float = 3600.0, enable_groundwater: bool = False):
    """Route daily GRU runoff [day, unit] (m/s) through the fabric network with dRoute's Muskingum-Cunge router
    (transfers by commissioning year, level-pool lakes). Returns daily mean discharge (m3/s) [day, unit] and the
    mass balance."""
    import droute
    import netCDF4
    from droute.network_adapter import DRouteNetworkAdapter
    d = Path(fabric_dir) / 'droute'
    year0 = runoff.index[0].year + (runoff.index[0].dayofyear - 1) / 365.2425
    net, cfg, summary = DRouteNetworkAdapter().load_hydrofabric(d, sim_start_year=year0,
                                                                enable_groundwater=enable_groundwater)
    with netCDF4.Dataset(d / 'topology.nc') as nc:
        seg = nc['segId'][:].astype(int)
        area = nc['area'][:].astype(float)
    missing = set(seg) - set(runoff.columns)
    if missing:
        raise ValueError(f'{len(missing)} units have no runoff, e.g. {sorted(missing)[:5]}')
    q_in = runoff[seg].to_numpy() * area                        # m3/s per reach, reach order = topology order
    c = droute.RouterConfig()
    c.dt = dt
    r = droute.MuskingumCungeRouter(net, c)
    sub = int(round(86400 / dt))
    out = np.zeros_like(q_in)
    for k in range(len(q_in)):
        acc = np.zeros(len(seg))
        for _ in range(sub):
            r.set_lateral_inflows(np.ascontiguousarray(q_in[k]))
            r.route_timestep()
            acc += np.asarray(r.get_all_discharges())
        out[k] = acc / sub
    return pd.DataFrame(out, index=runoff.index, columns=seg), r.mass_balance().as_dict(), summary


def run_both(domain: Path, tag: str, start: str, end: str, forcing: dict, out: Path, init: dict, procs: dict,
             exe: Path = EXE, niceness: int = 0) -> dict:
    """Run the ice and land runs side by side over [start, end] and merge each run's block restarts into
    settings/SUMMA/<run>/restart_<run>_<tag>.nc; returns {run: {wall_s, slowest_s, blocks, restart}}."""
    def one(run):
        t = time.time()
        res = run_chunk(domain, run, tag, start, end, forcing[run], out / run, init[run], procs[run], exe=exe,
                        niceness=niceness)
        name = merge_restarts(out / run, domain / 'settings' / 'SUMMA' / run, f'{run}_{tag}')
        return run, dict(wall_s=round(time.time() - t), slowest_s=round(max(r[3] for r in res)), blocks=len(res),
                         restart=name)
    with ThreadPoolExecutor(len(forcing)) as ex:
        return dict(ex.map(one, list(forcing)))


def chunk_main(a) -> int:
    """Server side of `national_remote`: one chunk of water years whose forcing is already in forcing/SUMMA.
    Writes <sim>/done_<mark>.json (or failed_<mark>.txt) and deletes the chunk's forcing; <mark> is <tag> for
    both runs, <run>_<tag> for one (--runs ice)."""
    c0, c1 = int(a.rest[1]), int(a.rest[2])
    sim = a.domain / 'simulations' / a.rest[0] / 'SUMMA'
    sim.mkdir(parents=True, exist_ok=True)
    runs = a.runs.split(',')
    tag = f'WY{c0}-{c1}'
    mark = tag if len(runs) > 1 else f'{runs[0]}_{tag}'
    forcing = {r: [f'{r}_WY{wy}.nc' for wy in range(c0, c1 + 1)] for r in runs}
    init = {'ice': a.init_ice, 'land': a.init_land}
    procs = {'ice': a.procs_ice, 'land': a.procs_land}
    try:
        res = run_both(a.domain, tag, f'{c0 - 1}-10-01 03:00', f'{c1}-10-01 00:00', forcing, sim, init, procs,
                       exe=Path(a.exe) if a.exe else EXE, niceness=a.nice)
    except Exception as e:  # noqa: BLE001  (reported to the orchestrator through the marker file)
        (sim / f'failed_{mark}.txt').write_text(repr(e))
        raise
    (sim / f'done_{mark}.json').write_text(json.dumps(res, indent=1))
    for r in runs:
        for f in forcing[r]:
            (a.domain / 'forcing' / 'SUMMA' / f).unlink(missing_ok=True)
    return 0


def host_status(sample_s: float = 3.0) -> dict:
    """For `national_remote`: CPU count, processes other than summa.exe busy right now (>10% of a core over
    sample_s, each counted as a whole core: MPI ranks pinned to shared cores show 20% but want a core of their own;
    the ps %CPU is a lifetime average) and the running `national_run.py chunk` jobs."""
    def ticks():
        out = {}
        for d in Path('/proc').iterdir():
            if d.name.isdigit():
                try:
                    st = (d / 'stat').read_text()
                    comm = st[st.index('(') + 1:st.rindex(')')]
                    f = st[st.rindex(')') + 2:].split()
                    out[d.name] = (comm, int(f[11]) + int(f[12]))
                except (OSError, ValueError):
                    pass
        return out
    t0 = ticks()
    time.sleep(sample_s)
    t1 = ticks()
    hz = os.sysconf('SC_CLK_TCK')
    busy = sum(1 for k, (comm, v) in t1.items()
               if k in t0 and comm != 'summa.exe' and (v - t0[k][1]) / hz / sample_s > 0.1)
    jobs = []
    for d in Path('/proc').iterdir():
        if d.name.isdigit() and d.name != str(os.getpid()):
            try:
                cmd = (d / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            cmd = [c.decode(errors='replace') for c in cmd if c]
            if any(c.endswith('national_run.py') for c in cmd) and 'chunk' in cmd:
                jobs.append(cmd[cmd.index('chunk'):])
    return dict(ncpu=os.cpu_count(), busy_other=busy, jobs=jobs)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['summa', 'route', 'chunk', 'hoststatus'])
    ap.add_argument('fabric', type=Path)
    ap.add_argument('domain', type=Path)
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=5)
    ap.add_argument('--runs', default='ice,land')
    ap.add_argument('--keep-forcing', action='store_true')
    ap.add_argument('--nice', type=int, default=0)
    ap.add_argument('--exe')
    ap.add_argument('--init-ice', default='coldState.nc')
    ap.add_argument('--init-land', default='coldState.nc')
    ap.add_argument('--procs-ice', type=int, default=10)
    ap.add_argument('--procs-land', type=int, default=13)
    a = ap.parse_args()
    if a.cmd == 'hoststatus':
        print(json.dumps(host_status()))
        return 0
    if a.cmd == 'chunk':
        # chunk <domain> <name> <wy0> <wy1>  (fabric argument unused: pass '-')
        return chunk_main(a)
    if a.cmd == 'summa':
        h3, name, wy0, wy1 = Path(a.rest[0]), a.rest[1], int(a.rest[2]), int(a.rest[3])
        sim = a.domain / 'simulations' / name / 'SUMMA'
        init = {run: 'coldState.nc' for run in a.runs.split(',')}
        for c0 in range(wy0, wy1 + 1, a.chunk):
            c1 = min(c0 + a.chunk - 1, wy1)
            for wy in range(c0, c1 + 1):
                N.write_forcing(a.fabric, a.domain, h3, wy, runs=tuple(init))
            for run in init:
                run_chunk(a.domain, run, f'WY{c0}-{c1}', f'{c0 - 1}-10-01 03:00', f'{c1}-10-01 00:00',
                          [f'{run}_WY{wy}.nc' for wy in range(c0, c1 + 1)], sim / run, init[run], a.procs, niceness=a.nice)
                # restarts: one file per block; SUMMA reads one initial-condition file, so merge them
                init[run] = merge_restarts(sim / run, a.domain / 'settings' / 'SUMMA' / run, f'{run}_WY{c0}-{c1}')
            if not a.keep_forcing:
                for wy in range(c0, c1 + 1):
                    for run in init:
                        (a.domain / 'forcing' / 'SUMMA' / f'{run}_WY{wy}.nc').unlink(missing_ok=True)
    else:
        name = a.rest[0]
        sim = a.domain / 'simulations' / name
        ro = pd.concat([gru_runoff(sim / 'SUMMA' / run) for run in ('ice', 'land')], axis=1)
        q, mb, summary = route(a.fabric, ro)
        (sim / 'dRoute').mkdir(parents=True, exist_ok=True)
        q.to_parquet(sim / 'dRoute' / 'discharge_daily.parquet') if _has_pyarrow() else \
            q.to_csv(sim / 'dRoute' / 'discharge_daily.csv.gz')
        (sim / 'dRoute' / 'mass_balance.json').write_text(json.dumps(dict(mass_balance=mb, network=summary),
                                                                      indent=2, default=str))
        print(json.dumps(mb, indent=2))
    return 0


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def fix_layer_heights(ds, tol: float = 1e-3):
    """Repair (hru, dom) layer stacks whose heights disagree with their depths. The glacier branch writes domains
    whose area went to 0 (retreated glacier: DOMarea 0, DOMelev -9999) with snow layers of zero depth and broken
    interface heights; a continuous run skips them, but SUMMA's restart check (check_icond: "mis-match between layer
    depth and layer height") stops on them. Zero-depth snow layers are dropped (nSnow, SWE, snow depth to 0) and
    heights are rebuilt from the depths: iLayerHeight[0] = -snow depth, mLayerHeight[k+1] = middle of layer k."""
    mid = [v for v in ds.data_vars if ds[v].dims[:1] == ('midToto',)]
    a = {v: ds[v].values.copy() for v in mid + ['iLayerHeight', 'mLayerHeight', 'nSnow', 'nSoil', 'nGlce']
         + [v for v in ('scalarSWE', 'scalarSnowDepth') if v in ds]}
    dep, ih, mh = a['mLayerDepth'], a['iLayerHeight'], a['mLayerHeight']
    ffill = 9.969209968386869e36
    fixed = dropped = 0
    for h in range(dep.shape[1]):
        for k in range(dep.shape[2]):
            ns, n = int(a['nSnow'][h, k]), int(a['nSnow'][h, k]) + int(a['nSoil'][h, k]) + int(a['nGlce'][h, k])
            if ns < 0 or n <= 0 or n > dep.shape[0]:
                continue
            if np.allclose(np.diff(ih[:n + 1, h, k]), dep[:n, h, k], atol=tol):
                continue
            if ns and np.all(dep[:ns, h, k] == 0):
                for v in mid:
                    a[v][:n - ns, h, k] = a[v][ns:n, h, k]
                    a[v][n - ns:n, h, k] = ffill
                n -= ns
                ns = 0
                a['nSnow'][h, k] = 0
                for v in ('scalarSWE', 'scalarSnowDepth'):
                    if v in a:
                        a[v][..., h, k] = 0.
                dropped += 1
            top = -dep[:ns, h, k].sum()
            ih[:, h, k] = ffill
            ih[0, h, k] = top
            ih[1:n + 1, h, k] = top + np.cumsum(dep[:n, h, k])
            mh[1:, h, k] = ffill
            mh[1:n + 1, h, k] = 0.5 * (ih[:n, h, k] + ih[1:n + 1, h, k])
            fixed += 1
    if fixed:
        logger.info('restart: rebuilt layer heights of %d (hru, dom) stacks, dropped zero-depth snow in %d',
                    fixed, dropped)
    for v, x in a.items():
        ds[v] = (ds[v].dims, x, ds[v].attrs)
    return ds


def merge_restarts(out: Path, settings: Path, prefix: str) -> str:
    """Join the block restart files (<prefix>_restart_*_G*.nc) along hru/gru into settings/restart_<prefix>.nc,
    in attributes order; returns its name for the next chunk's fileManager."""
    import xarray as xr
    files = sorted(out.glob(f'{prefix}_restart_*G*.nc'))
    if not files:
        raise FileNotFoundError(f'no restart files for {prefix} in {out}')
    parts = [xr.open_dataset(f, mask_and_scale=False) for f in files]
    # glacier restarts size ygrid/xgrid (and possibly layer dims) to each block's largest GRU, filling the rest
    # with the netCDF default fill: pad every non-concatenation dimension to the largest block's (or the cold
    # state's grid) size with that same fill
    target = {}
    for p in parts:
        for d, n in p.sizes.items():
            if d not in ('hru', 'gru'):
                target[d] = max(target.get(d, 0), n)
    cold = Path(settings) / 'coldState.nc'
    if cold.exists() and 'ygrid' in target:
        with xr.open_dataset(cold) as c:
            for d in ('ygrid', 'xgrid'):
                target[d] = max(target[d], c.sizes.get(d, 0))

    def pad(p):
        out = {}
        for v in p.data_vars:
            da = p[v]
            w = {d: (0, target[d] - da.sizes[d]) for d in da.dims if d in target and da.sizes[d] < target[d]}
            if w:
                fill = 9.969209968386869e36 if da.dtype.kind == 'f' else -2147483647
                da = da.pad(w, constant_values=fill)
            out[v] = da
        return xr.Dataset(out, attrs=p.attrs)
    parts = [pad(p) for p in parts]
    hru_vars = [v for v in parts[0].data_vars if 'hru' in parts[0][v].dims]
    gru_vars = [v for v in parts[0].data_vars if 'gru' in parts[0][v].dims and 'hru' not in parts[0][v].dims]
    other = [v for v in parts[0].data_vars if v not in hru_vars + gru_vars]
    merged = xr.merge([xr.concat([p[hru_vars] for p in parts], 'hru'),
                       xr.concat([p[gru_vars] for p in parts], 'gru'), parts[0][other]])
    if 'iLayerHeight' in merged:
        merged = fix_layer_heights(merged)
    name = f'restart_{prefix}.nc'
    merged.to_netcdf(settings / name)
    for p in parts:
        p.close()
    return name


if __name__ == '__main__':
    raise SystemExit(main())
