# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""National ParFlow-CLM (whole Iceland, 1 km) from the landslíkan hydrofabric (ADR 0016).

Basin mode (``basin/``) is one catchment with one outlet; the country has some 1,400 sea outlets. This module
builds the grid and inputs for the whole island and reuses basin mode's key database, CLM driver files and
PFB writer:

* Grid: national MF6's 1 km grid (``modflow.national_model.build_grid``, on the fabric's 25 m raster); a cell is
  active when at least half of it is land, islands under ``MIN_ISLAND`` cells are dropped.
* Slopes: D4 priority flood seeded from every coastal cell, each draining across its sea face, on the block mean
  of the fabric's conditioned surface (``dem_filled.tif``: its subglacial hydraulic potential keeps the fabric's
  ice divides). Water leaves the land as overland flow at the coast (OverlandFlow BC on the top patch).
* Subsurface: 15 terrain-following layers (100 m, the MF6 aquifer thickness; basin mode's 14 plus 30 m); aquifer K from the geology groups with the
  national groundwater calibration (``calibration/gw_v1/best.json``, MF6 K in m/d -> m/h); soil K uniform; glacier
  cells get ``glacier_bed_k`` in the CLM layers. Coast and base are no-flow in this first version (submarine
  groundwater discharge was ~20 mm/yr in the MF6 model).
* Vegetation: IGBP class of each cell's majority unit's dominant land class, snow/ice where the cell is >= 50 %
  glacier.
* Forcing: each cell takes its majority unit's CARRA weights (as SUMMA and HYPE), with temperature lapsed to the
  cell elevation, pressure hypsometric, specific humidity capped at saturation and longwave scaled with T^4.
  ``dt_hours`` 1 repeats each 3-h value over its three hours (as the Tungnaá forecast does); 3 uses the 3-h
  values directly (a third of the steps and forcing volume).
* Segments of whole days chain by restart (pressure + CLM restart files), written and run one at a time; the
  forcing is deleted after each segment (about 300 MB per simulated day at 1 h).

    python -m symfluence.models.clmparflow.national grid <fabric_dir> <domain_dir>
    python -m symfluence.models.clmparflow.national segment <fabric_dir> <domain_dir> <carra_3h_dir> <name> \
        <start> <end> [--P 4 --Q 2 --dt 1 --prev <dir>] [--run]
    python -m symfluence.models.clmparflow.national spinup <fabric_dir> <domain_dir> <name> <years> \
        [--P 4 --Q 4 --prev <dir> --max-step 240] [--run]     # Richards only, MF6 mean recharge
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.models.modflow import national_model as nm
from symfluence.models.modflow.national_gw_calibration import to_params

from .basin import case, pfio
from .basin.grid import BasinGrid

logger = logging.getLogger(__name__)

MIN_ISLAND = 4
MIN_SLOPE = 1e-4
CRS = 'EPSG:3057'
GLACIER_LAND, WATER_LAND = 1, (2, 9)
LAND_TO_IGBP = {1: 15, 2: 17, 3: 16, 4: 7, 5: 10, 6: 11, 7: 4, 8: 12, 9: 17}
LAPSE_K_PER_M, G, RD = -0.0065, 9.80665, 287.05
CARRA_NX = 301
PARFLOW_DIR = Path(__file__).resolve().parents[5] / 'SYMFLUENCE_data' / 'installs' / 'clmparflow_glacier'


def _block_mean(path: Path, f: int, nr: int, nc: int, valid=lambda a: np.isfinite(a)) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1).astype('f8')
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
    a = np.pad(a, ((0, nr * f - a.shape[0]), (0, nc * f - a.shape[1])), constant_values=np.nan)
    ok = valid(a)
    s_ = np.where(ok, a, 0.0).reshape(nr, f, nc, f).sum((1, 3))
    n_ = ok.reshape(nr, f, nc, f).sum((1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        return s_ / n_


def coastal_slopes(zc_in: np.ndarray, mask: np.ndarray, dx: float, min_slope: float = MIN_SLOPE):
    """D4 priority flood from all coastal cells (north-up). Returns (sx, sy, conditioned z) north-up, ParFlow
    signs (Sx > 0 -> flow to -x, Sy > 0 -> flow to -y = south; row 0 is north)."""
    ny, nx = mask.shape
    nb = ((1, 0), (-1, 0), (0, 1), (0, -1))
    zc = np.where(mask, np.nan_to_num(zc_in, nan=0.0), np.nan)
    down = np.zeros((ny, nx, 2), int)
    seen = np.zeros_like(mask)
    queue = []
    for r, c in zip(*np.nonzero(mask)):
        for dr, dc in nb:
            q = (r + dr, c + dc)
            if not (0 <= q[0] < ny and 0 <= q[1] < nx) or not mask[q]:
                down[r, c] = (dr, dc)
                zc[r, c] = max(zc[r, c], 0.0)
                seen[r, c] = True
                heapq.heappush(queue, (zc[r, c], r, c))
                break
    eps = 1e-3
    while queue:
        zz, r, c = heapq.heappop(queue)
        for dr, dc in nb:
            q = (r + dr, c + dc)
            if 0 <= q[0] < ny and 0 <= q[1] < nx and mask[q] and not seen[q]:
                seen[q] = True
                zc[q] = max(zc[q], zz + eps)
                down[q] = (-dr, -dc)
                heapq.heappush(queue, (zc[q], q[0], q[1]))
    drop = np.zeros((ny, nx))
    for r, c in zip(*np.nonzero(mask)):
        dr, dc = down[r, c]
        rr, cc = r + dr, c + dc
        inside = 0 <= rr < ny and 0 <= cc < nx and mask[rr, cc]
        drop[r, c] = (zc[r, c] - (zc[rr, cc] if inside else 0.0)) / dx
    s = np.maximum(drop, min_slope)
    sx = np.where(mask & (down[..., 1] == -1), s, 0.) - np.where(mask & (down[..., 1] == 1), s, 0.)
    sy = np.where(mask & (down[..., 0] == 1), s, 0.) - np.where(mask & (down[..., 0] == -1), s, 0.)
    return sx, sy, zc


def build(fabric_dir: Path, raw_dem: Path, attributes: pd.DataFrame, hru_classes: pd.DataFrame,
          gw_x: list, dx: float = 1000.0) -> tuple[BasinGrid, dict]:
    """National grid (BasinGrid, arrays south-up as basin mode expects) and the extra national fields."""
    from scipy import ndimage
    fabric_dir = Path(fabric_dir)
    g = nm.build_grid(fabric_dir, raw_dem, dx)
    f = int(round(dx / 25.0))
    nr, nc = g.nrow, g.ncol
    mask = g.land_frac >= 0.5
    lab, n = ndimage.label(mask)
    size = np.bincount(lab.ravel())
    mask &= size[lab] >= MIN_ISLAND
    mask[lab == 0] = False
    zf = _block_mean(fabric_dir / 'dem_filled.tif', f, nr, nc, lambda a: np.isfinite(a) & (a > -1000))
    ice_frac = _block_mean(fabric_dir / 'glacier.tif', f, nr, nc, lambda a: np.isfinite(a) & (a <= 1))
    lake_frac = _block_mean(fabric_dir / 'lake.tif', f, nr, nc, lambda a: np.isfinite(a) & (a <= 1))
    sx, sy, zc = coastal_slopes(np.where(np.isfinite(zf), zf, g.top), mask, dx)
    # majority unit per cell (overlap table, 1-based row/col)
    ov = g.overlap.sort_values('area_m2', ascending=False).drop_duplicates(['row', 'col'])
    unit_nu = np.zeros((nr, nc), np.int64)
    unit_nu[ov.row.to_numpy() - 1, ov.col.to_numpy() - 1] = ov.unit.to_numpy()
    units = np.unique(unit_nu[mask])
    # dominant non-water land class per unit -> IGBP
    lc = hru_classes[~hru_classes.land.isin(WATER_LAND + (GLACIER_LAND,))]
    lc = lc.groupby(['id', 'land']).area_km2.sum().reset_index()
    dom = lc.loc[lc.groupby('id').area_km2.idxmax()].set_index('id').land
    igbp = np.vectorize(lambda u: LAND_TO_IGBP.get(int(dom.get(u, 3)), 16))(unit_nu)
    ice = mask & (np.nan_to_num(ice_frac) >= 0.5)
    igbp = np.where(ice, 15, igbp)
    params = to_params(gw_x)
    k_md = nm._cell_array(g, nm.unit_property(attributes, params.k, log=True), log=True)
    flip = lambda a: a[::-1].copy()
    idx = np.full((nr, nc), -1, np.int64)
    pos = {u: i for i, u in enumerate(units)}
    idx[mask] = [pos[u] for u in unit_nu[mask]]
    left, top = g.origin
    grid = BasinGrid(mask=flip(mask), z=flip(np.where(mask, g.top, np.nan)), zc=flip(zc), sx=flip(sx), sy=flip(sy),
                     unit=flip(idx), ice=flip(ice), icefrac=flip(np.nan_to_num(ice_frac)), x0=left, y0=top - nr * dx,
                     dx=dx, crs=CRS, units=[str(u) for u in units],
                     info=dict(kind='national', cells_active=int(mask.sum()), nrow=nr, ncol=nc,
                               islands_dropped=int((size[1:] < MIN_ISLAND).sum()), gw_params=params.k),
                     lake=flip(mask & (np.nan_to_num(lake_frac) >= 0.5)))
    extra = dict(igbp=flip(igbp), aquifer_k_mh=flip(np.where(mask, k_md / 24.0, 0.0)))
    return grid, extra


def save(grid: BasinGrid, extra: dict, d: Path) -> None:
    grid.save(d)
    np.savez_compressed(d / 'national_extra.npz', **extra)


def load(d: Path) -> tuple[BasinGrid, dict]:
    z = np.load(d / 'national_extra.npz')
    return BasinGrid.load(d), {k: z[k] for k in z.files}


def permeability(grid: BasinGrid, extra: dict, params: case.BasinParameters) -> np.ndarray:
    nz = params.nz
    K = np.repeat(np.where(grid.mask, extra['aquifer_k_mh'], params.aquifer_k)[None], nz, axis=0)
    K[nz - 10:] = params.soil_k
    if params.glacier_bed_k is not None:
        K[nz - 10:, grid.ice] = params.glacier_bed_k
    return K


def write_static(run: Path, grid: BasinGrid, extra: dict, params: case.BasinParameters, parflow_dir: Path,
                 P: int, Q: int) -> None:
    import subprocess
    dx = grid.dx
    pfio.write_pfb(run / 'slope_x.pfb', grid.sx[None], dx=dx, dy=dx, P=P, Q=Q)
    pfio.write_pfb(run / 'slope_y.pfb', grid.sy[None], dx=dx, dy=dx, P=P, Q=Q)
    pfio.write_pfb(run / 'perm.pfb', permeability(grid, extra, params), dx=dx, dy=dx, P=P, Q=Q)
    if case.lake_active(grid, params):
        pfio.write_pfb(run / 'mannings.pfb', case.mannings(grid, params)[None], dx=dx, dy=dx, P=P, Q=Q)
    pfio.write_pfb(run / 'mask.pfb', grid.mask[None].astype(float), dx=dx, dy=dx)
    r = subprocess.run([str(Path(parflow_dir) / 'bin' / 'pfmask-to-pfsol'), '--mask', 'mask.pfb', '--pfsol',
                        'basin.pfsol', '--bottom-patch-label', '2', '--side-patch-label', '3', '--z-bottom', '0',
                        '--z-top', str(float(params.nz)), '--dx', str(dx), '--dy', str(dx)],
                       cwd=run, capture_output=True, text=True)
    if r.returncode != 0 or not (run / 'basin.pfsol').exists():
        raise RuntimeError(f'pfmask-to-pfsol failed: {r.stdout}{r.stderr}')


def write_vegm(run: Path, grid: BasinGrid, extra: dict, ranks: int) -> None:
    from pyproj import Transformer
    ny, nx = grid.shape
    xc, yc = grid.cell_centres()
    lon, lat = Transformer.from_crs(grid.crs, 4326, always_xy=True).transform(xc, yc)
    head = [' x  y  lat    lon    sand clay color  fractional coverage of grid by vegetation class (Must/Should Add to 1.0)',
            '       (Deg)\t (Deg)  (%/100)   index  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18']
    rows = []
    zeros = ['0.0'] * 18
    for j in range(ny):
        for i in range(nx):
            fr = zeros.copy()
            fr[int(extra['igbp'][j, i]) - 1] = '1.0'
            rows.append(f'{i + 1:4d}{j + 1:4d}{lat[j, i]:9.3f}{lon[j, i]:9.3f}  0.65 0.07   4  ' + ' '.join(fr))
    (run / 'drv_vegm.dat').write_text('\n'.join(head + rows) + '\n')
    for r in range(ranks):
        for name in ('drv_vegm.dat',):
            dst = run / f'{name}.{r}'
            dst.unlink(missing_ok=True)
            os.symlink(name, dst)


def cell_forcing_matrix(grid: BasinGrid, weights: pd.DataFrame, cells: np.ndarray):
    """Sparse [n_active, n_archive_cells]: each active cell takes its unit's forcing_carra weights."""
    from scipy import sparse
    act = np.flatnonzero(grid.mask.ravel())
    u_of_cell = np.array(grid.units, dtype=np.int64)[grid.unit.ravel()[act]]
    col = pd.Series(np.arange(len(cells)), index=cells)
    w = weights[weights.unit_id.isin(set(u_of_cell))]
    units = np.unique(u_of_cell)
    row = pd.Series(np.arange(len(units)), index=units)
    flat = w.carra_j.to_numpy() * CARRA_NX + w.carra_i.to_numpy()
    W = sparse.csr_matrix((w.weight.to_numpy(), (row[w.unit_id].to_numpy(), col[flat].to_numpy())),
                          shape=(len(units), len(cells)))
    U = sparse.csr_matrix((np.ones(len(act)), (np.arange(len(act)), row[u_of_cell].to_numpy())),
                          shape=(len(act), len(units)))
    return (U @ W).tocsr(), act


def write_forcing(out: Path, grid: BasinGrid, h3_dir: Path, weights: pd.DataFrame, zs: np.ndarray, start: str,
                  end: str, dt_hours: int, P: int, Q: int, precip_scale: float = 1.0) -> int:
    """CLM 3-D met files, one per day per variable, for [start, end) (00Z days). Returns the number of steps."""
    sys.path.insert(0, str(Path(h3_dir).resolve().parents[3] / 'src'))
    from landslikan.carra import read_3h
    end_, cells, x = read_3h(start, end)
    M, act = cell_forcing_matrix(grid, weights, cells)
    z0 = M @ zs[cells]
    dz = np.nan_to_num(grid.z.ravel()[act]) - z0
    ta = (M @ x['t'].T).T + 273.15
    tb = ta + LAPSE_K_PER_M * dz
    ps = (M @ x['ps'].T).T * 100.0 * np.exp(-G * dz / (RD * 0.5 * (ta + tb)))
    es = 611.2 * np.exp(17.67 * (tb - 273.15) / (tb - 29.65))
    q = np.minimum((M @ np.nan_to_num(x['q'], nan=float(np.nanmedian(x['q']))).T).T, 0.622 * es / (ps - 0.378 * es))
    met = {'DSWR': np.clip((M @ x['sw'].T).T, 0, None), 'DLWR': (M @ x['lw'].T).T * (tb / ta) ** 4,
           'APCP': np.clip((M @ x['p'].T).T, 0, None) / 3600.0 * precip_scale, 'Temp': tb,
           'UGRD': np.clip((M @ x['u10'].T).T, 0.1, None), 'Press': ps, 'SPFH': q}
    rep = 3 // dt_hours
    steps_day = 24 // dt_hours
    n = len(end_) * rep
    if n % steps_day:
        raise ValueError(f'{start}..{end}: {n} steps is not whole days')
    out.mkdir(parents=True, exist_ok=True)
    ny, nx = grid.shape
    fill_col = 0
    for var in case.MET_VARIABLES:
        v = met.get(var)
        full = np.empty((n, ny * nx))
        if v is None:                                   # VGRD: wind speed is all in UGRD
            full[:] = 0.0
        else:
            v = np.repeat(v, rep, axis=0)
            if not np.isfinite(v).all():
                raise ValueError(f'non-finite {var}')
            full[:] = v[:, [fill_col]]                  # inactive cells: any finite value
            full[:, act] = v
        full = full.reshape(n, ny, nx)
        for s in range(0, n, steps_day):
            pfio.write_pfb(out / f'CARRA.{var}.{s + 1:06d}_to_{s + steps_day:06d}.pfb', full[s:s + steps_day],
                           dx=grid.dx, dy=grid.dx, P=P, Q=Q)
    return n


def read_mf6_heads(path: Path, nrow: int, ncol: int) -> np.ndarray:
    """Last head array of a one-layer MF6 binary head file (north-up [nrow, ncol])."""
    raw = Path(path).read_bytes()
    rec = 4 + 4 + 8 + 8 + 16 + 4 + 4 + 4 + nrow * ncol * 8
    off = (len(raw) // rec - 1) * rec
    return np.frombuffer(raw, '<f8', count=nrow * ncol, offset=off + 52).reshape(nrow, ncol)


def ic_from_mf6(grid: BasinGrid, heads_nu: np.ndarray, params: case.BasinParameters, K: np.ndarray | None = None,
                recharge_mh: np.ndarray | None = None) -> np.ndarray:
    """Initial pressure head [z, y, x] (z = 0 at the base) from MF6 steady-state heads on the same grid: the water
    table at min(h, surface), hydrostatic below it, the same linear profile (suction) above it, capped at the
    steady-infiltration suction (`flux_suction`) when the cell K [z, y, x] and recharge (m/h, south-up) are given."""
    h = heads_nu[::-1]
    zs = np.nan_to_num(grid.z)
    # the water table at least one top layer below the surface: with it at the surface, 19 % of cells start with
    # 5 cm ponded (top-cell psi > 0), which at the basin Manning's n flows at km/h and the first step cannot solve
    top = params.dz_top_down[0]
    wt = np.where(grid.mask & (np.abs(h) < 1e20), np.minimum(h, zs - top), zs - params.initial_water_table_depth)
    dz = np.asarray(params.dz_top_down)
    depth = np.cumsum(dz) - dz / 2                       # cell centres below the surface, top layer first
    psi = wt[None] - (zs[None] - depth[:, None, None])
    psi = np.clip(psi, -sum(dz), None)[::-1]            # bottom layer first
    if K is not None and recharge_mh is not None:
        psi = np.maximum(psi, flux_suction(K, recharge_mh[None], params))
    return np.where(grid.mask[None], psi, 0.0)


def flux_suction(K: np.ndarray, flux: np.ndarray, params: case.BasinParameters) -> np.ndarray:
    """Pressure head (< 0) at which van Genuchten–Mualem K(psi) equals `flux` (unit-gradient infiltration): the
    moisture above a water table under steady recharge. Hydrostatic suction above a deep MF6 water table (to
    -97 m, kr ~1e-12) gives ParFlow an infiltration front into dry soil that the first step cannot solve."""
    a, n = params.vg_alpha, params.vg_n
    m = 1 - 1 / n
    ps = -np.logspace(-3, 2, 400)
    se = (1 + (a * -ps) ** n) ** -m
    kr = se ** .5 * (1 - (1 - se ** (1 / m)) ** m) ** 2          # decreasing with suction
    ratio = np.clip(np.maximum(flux, 1e-9) / np.maximum(K, 1e-12), kr[-1], 1.0)
    return np.interp(np.log(ratio), np.log(kr[::-1]), ps[::-1])


def mf6_recharge(mf6_dir: Path, nrow: int, ncol: int) -> np.ndarray:
    """Recharge (m/h, north-up [nrow, ncol]) of the MF6 steady state: RCH rows (layer row col unit mult) times the
    unit's rate at t = 0 of gwf.rch.ts (the long-term mean, m/d)."""
    d = Path(mf6_dir)
    ts = (d / 'gwf.rch.ts').read_text()
    names = ts.split('NAMES', 1)[1].split('\n', 1)[0].split()
    first = ts.split('BEGIN TIMESERIES', 1)[1].strip().split('\n', 1)[0].split()
    rate = dict(zip(names, map(float, first[1:])))
    body = (d / 'gwf.rch').read_text().split('BEGIN PERIOD', 1)[1].split('END PERIOD', 1)[0].split('\n')[1:]
    r = np.zeros((nrow, ncol))
    for line in body:
        f = line.split()
        if len(f) == 5:
            r[int(f[1]) - 1, int(f[2]) - 1] += rate[f[3]] * float(f[4])
    return r / 24.0


def write_spinup(run: Path, name: str, grid: BasinGrid, extra: dict, params: case.BasinParameters,
                 recharge_mh: np.ndarray, years: float, P: int, Q: int, parflow_dir: Path = PARFLOW_DIR,
                 ic_pressure: np.ndarray | None = None, prev: Path | None = None, max_step_h: float = 240.0,
                 dump_h: float = 8760.0) -> dict:
    """Stage 1 of the spin-up (ADR 0016): Richards only (no CLM), the MF6 mean recharge as a source in the top
    layer (south-up recharge_mh / top dz, 1/h), overland flow on, growing steps up to `max_step_h`, a pressure
    dump every `dump_h` hours. Starts from `ic_pressure` (MF6 state) or the last pressure of `prev`."""
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    t0 = time.time()
    write_static(run, grid, extra, params, parflow_dir, P, Q)
    nz = params.nz
    pme = np.zeros((nz,) + grid.shape)
    pme[nz - 1] = np.where(grid.mask, recharge_mh, 0.0) / params.dz_top_down[0]
    pfio.write_pfb(run / 'pme.pfb', pme, dx=grid.dx, dy=grid.dx, P=P, Q=Q)
    if prev is not None:
        last = sorted(prev.glob(f'{prev.name}.out.press.*.pfb'))[-1]
        shutil.copy(last, run / 'ic_press.pfb')
        shutil.copy(str(last) + '.dist', run / 'ic_press.pfb.dist')
    else:
        pfio.write_pfb(run / 'ic_press.pfb', ic_pressure, dx=grid.dx, dy=grid.dx, P=P, Q=Q)
    hours = float(round(years * 8760))
    k = case.parflow_keys(grid, params, name, int(hours), P, Q, dump_hours=int(dump_h), ic_file='ic_press.pfb')
    k = {a: b for a, b in k.items() if not a.startswith('Solver.CLM.')}
    k.update({'Solver.LSM': 'none', 'Solver.PrintCLM': False, 'Solver.EvapTransFile': True,
              'Solver.EvapTrans.FileName': 'pme.pfb', 'TimingInfo.StopTime': hours,
              'TimingInfo.DumpInterval': float(dump_h), 'TimeStep.Type': 'Growth', 'TimeStep.InitialStep': 1.0,
              'TimeStep.GrowthFactor': 1.2, 'TimeStep.MaxStep': float(max_step_h), 'TimeStep.MinStep': 1e-3})
    k.pop('TimeStep.Value', None)
    pfio.write_pfidb(run / f'{name}.pfidb', k)
    m = grid.mask
    meta = dict(seg=name, stage='richards_spinup', years=years, hours=hours, max_step_h=max_step_h, dump_h=dump_h,
                prev=prev.name if prev else None, P=P, Q=Q, active=int(m.sum()),
                recharge_mm_yr=round(float(recharge_mh[m].mean() * 8760 * 1000), 1), params=params.record(),
                write_s=round(time.time() - t0))
    (run / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
    return meta


def national_parameters(**kw) -> case.BasinParameters:
    """Basin-mode parameters with a 100 m column (the MF6 aquifer thickness: 15 layers, the top 10 CLM's) and the
    Tungnaá hand-tuned glacier bed K and Manning's n; aquifer K comes from the geology grid instead."""
    d = dict(dz_top_down=[.1, .1, .2, .2, .4, .6, .8, 1., 1.6, 2., 5., 10., 20., 28., 30.], glacier_bed_k=0.001,
             manning=6e-6, precip_scale=1.0)
    d.update(kw)
    return case.BasinParameters(**d)


def write_segment(run: Path, name: str, grid: BasinGrid, extra: dict, params: case.BasinParameters, h3_dir: Path,
                  weights: pd.DataFrame, zs: np.ndarray, start: str, end: str, P: int, Q: int, dt_hours: int = 1,
                  prev: Path | None = None, parflow_dir: Path = PARFLOW_DIR,
                  ic_pressure: np.ndarray | None = None) -> dict:
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    t0 = time.time()
    write_static(run, grid, extra, params, parflow_dir, P, Q)
    steps = write_forcing(run / 'forcing', grid, h3_dir, weights, zs, start, end, dt_hours, P, Q,
                          params.precip_scale)
    hours = steps * dt_hours
    ic = None
    if prev is not None:
        last = sorted(prev.glob(f'{prev.name}.out.press.*.pfb'))[-1]
        shutil.copy(last, run / 'ic_press.pfb')
        shutil.copy(str(last) + '.dist', run / 'ic_press.pfb.dist')
        ic = 'ic_press.pfb'
        for r in range(P * Q):
            shutil.copy(prev / f'clm.rst.00000.{r}', run / f'clm.rst.00000.{r}')
    elif ic_pressure is not None:
        pfio.write_pfb(run / 'ic_press.pfb', ic_pressure, dx=grid.dx, dy=grid.dx, P=P, Q=Q)
        ic = 'ic_press.pfb'
    # basin mode's CLM files (dates, restart flags, per-rank drv_clmin), then the national vegetation map
    case.write_clm_files(run, grid, params, start, end, restart=prev is not None, ranks=P * Q)
    write_vegm(run, grid, extra, P * Q)
    k = case.parflow_keys(grid, params, name, hours, P, Q, dump_hours=24, ic_file=ic)
    k.update({'TimeStep.Value': float(dt_hours), 'Solver.CLM.MetFileNT': 24 // dt_hours,
              'Solver.CLM.CLMDumpInterval': 24 // dt_hours})
    pfio.write_pfidb(run / f'{name}.pfidb', k)
    meta = dict(seg=name, start=start, end=end, hours=hours, dt_hours=dt_hours, prev=prev.name if prev else None,
                P=P, Q=Q, nx=grid.shape[1], ny=grid.shape[0], active=int(grid.mask.sum()),
                params=params.record(), write_s=round(time.time() - t0))
    (run / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
    return meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['grid', 'segment', 'spinup'])
    ap.add_argument('fabric', type=Path)
    ap.add_argument('domain', type=Path)
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--P', type=int, default=4)
    ap.add_argument('--Q', type=int, default=2)
    ap.add_argument('--dt', type=int, default=1, choices=[1, 3])
    ap.add_argument('--prev', type=Path)
    ap.add_argument('--max-step', type=float, default=240.0)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    gdir = a.domain / 'settings' / 'PARFLOW' / 'grid_1km'
    if a.cmd == 'grid':
        cal = a.domain / 'calibration' / 'gw_v1'
        att = pd.read_csv(cal / 'attributes.csv').set_index('id')
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models' / 'summa'))
        from symfluence.models.summa.national import _table
        hc = _table(a.fabric, 'hru_classes')
        x = json.loads((cal / 'best.json').read_text())['x']
        raw = Path(a.fabric).resolve().parents[2] / 'data' / 'raw' / 'rasters' / 'IslandsDEMv1.0_25x25m_isn93_zmasl.tif'
        grid, extra = build(a.fabric, raw, att, hc, x)
        save(grid, extra, gdir)
        print(json.dumps(grid.info, indent=1, default=str))
    elif a.cmd == 'spinup':
        # spinup <fabric> <domain> <name> <years> [--prev <dir>] [--max-step 240] [--run]
        name, years = a.rest[0], float(a.rest[1])
        grid, extra = load(gdir)
        params = national_parameters()
        mf6 = a.domain / 'settings' / 'PARFLOW' / 'mf6_steady'
        nrow, ncol = grid.info['nrow'], grid.info['ncol']
        rch = mf6_recharge(mf6, nrow, ncol)[::-1]
        ic = None if a.prev else ic_from_mf6(grid, read_mf6_heads(mf6 / 'gwf.hds', nrow, ncol), params,
                                             K=permeability(grid, extra, params), recharge_mh=rch)
        run = a.domain / 'simulations' / 'parflow_national' / name
        meta = write_spinup(run, name, grid, extra, params, rch, years, a.P, a.Q, ic_pressure=ic, prev=a.prev,
                            max_step_h=a.max_step)
        print(json.dumps(meta, indent=1, default=str))
        if a.run:
            t = time.time()
            rc, out = case.run_parflow(run, name, PARFLOW_DIR, a.P * a.Q)
            (run / 'run.log').write_text(out)
            print('exit', rc, 'seconds', round(time.time() - t))
            print(out[-1500:])
    else:
        from symfluence.models.summa.national import _table, carra_orography
        h3, name, start, end = Path(a.rest[0]), a.rest[1], a.rest[2], a.rest[3]
        grid, extra = load(gdir)
        w = _table(a.fabric, 'forcing_carra')
        zs = carra_orography(h3.parent / 'monthly')
        params = national_parameters()
        run = a.domain / 'simulations' / 'parflow_national' / name
        ic = None
        hds = a.domain / 'settings' / 'PARFLOW' / 'mf6_steady' / 'gwf.hds'
        if a.prev is None and hds.exists():
            ic = ic_from_mf6(grid, read_mf6_heads(hds, grid.info['nrow'], grid.info['ncol']), params)
        meta = write_segment(run, name, grid, extra, params, h3, w, zs, start, end, a.P, a.Q, a.dt, a.prev,
                             ic_pressure=ic)
        print(json.dumps(meta, indent=1, default=str))
        if a.run:
            t = time.time()
            rc, out = case.run_parflow(run, name, PARFLOW_DIR, a.P * a.Q)
            (run / 'run.log').write_text(out)
            print('exit', rc, 'seconds', round(time.time() - t))
            print(out[-1500:])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
