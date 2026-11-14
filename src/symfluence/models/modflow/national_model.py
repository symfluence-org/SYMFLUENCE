# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Whole-country MODFLOW 6 groundwater model on a delivered hydrofabric (landslíkan, Iceland).

Built from the fabric's rasters by cell counting, not polygon overlay (10,000 units x 100,000 cells):

* Grid: square cells (default 1 km) aligned with the fabric raster; a cell is active when any fabric unit covers
  it. TOP = mean DEM over the land part. One convertible layer, BOTM = min(TOP, streambed) - thickness.
* K and SY per cell: area-weighted (log-mean for K) over the cell's units, each unit's value from its geology-group
  fractions (NÍ 1:600k groups in the fabric attributes). Group values are the calibration parameters.
* RCH: one time series per unit (HYPE subid); entry (cell, unit) with multiplier = overlap area / cell area, so
  every unit's recharge volume is conserved exactly.
* DRN ``RIV``: per (cell, unit) stream length (cells with upstream area >= stream threshold) at the mean stream
  DEM minus the channel depth. DRN ``SEEP``: per (cell, unit) land area at the cell top. Both carry the unit id
  as BOUNDNAME. Discharge per unit is computed exactly from the saved heads and the drain table (``drains.csv``),
  q = C * max(h - z, 0); boundname observations re-resolve every stress period and cost more than the solve.
* GHB ``SEA``: active cells touching the sea, stage 0 m.
* TDIS: period 1 steady state at the mean recharge, then one transient period of daily time steps.

See docs/adr/0012-national-hype-mf6.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ConfigurationError, ValidationError

GEO_GROUPS = ('holocene_lava', 'holocene_sediment', 'upper_pleistocene', 'lower_pleistocene', 'tertiary', 'intrusive')
UNCLASSIFIED_GEOLOGY = 'upper_pleistocene'     # glaciers and gaps in the 1:600k map (mostly young volcanics)
GRID_FILE, OVERLAP_FILE, RIVER_FILE = 'national_grid.npz', 'unit_cell_overlap.csv', 'river_drains.csv'
DRAIN_FILE = 'drains.csv'          # every DRN entry: package, cell, unit, elevation, conductance (as written)
RECHARGE_TS = 'gwf.rch.ts'


@dataclass
class NationalParams:
    """Groundwater parameters (m, days). K and SY per geology group."""
    k: Dict[str, float] = field(default_factory=lambda: {
        'holocene_lava': 100.0, 'holocene_sediment': 10.0, 'upper_pleistocene': 1.0,
        'lower_pleistocene': 0.1, 'tertiary': 0.01, 'intrusive': 0.01})
    sy: Dict[str, float] = field(default_factory=lambda: {
        'holocene_lava': 0.10, 'holocene_sediment': 0.15, 'upper_pleistocene': 0.05,
        'lower_pleistocene': 0.02, 'tertiary': 0.01, 'intrusive': 0.01})
    ss: float = 1e-5
    thickness: float = 100.0
    channel_depth: float = 1.0
    river_leakance: float = 10.0       # m2/d per m of stream
    seepage_leakance: float = 0.1      # 1/d per m2 of land
    sea_conductance: float = 1000.0    # m2/d per coastal cell

    def validate(self) -> None:
        for name in ('k', 'sy'):
            d = getattr(self, name)
            if set(d) != set(GEO_GROUPS) or not all(v > 0 for v in d.values()):
                raise ConfigurationError(f'{name} needs a positive value for each of {GEO_GROUPS}')
        for name in ('ss', 'thickness', 'river_leakance', 'seepage_leakance', 'sea_conductance'):
            if not getattr(self, name) > 0:
                raise ConfigurationError(f'{name} must be positive')


@dataclass
class NationalGrid:
    nrow: int
    ncol: int
    cell_size: float
    origin: tuple            # (x left, y top), EPSG:3057
    active: np.ndarray       # bool (nrow, ncol)
    top: np.ndarray          # mean DEM over land, NaN where inactive
    land_frac: np.ndarray
    coast: np.ndarray        # bool: active cell touching the sea
    overlap: pd.DataFrame    # row, col (1-based), unit, area_m2
    river: pd.DataFrame      # row, col, unit, length_m, strtop

    def save(self, d: Path) -> None:
        np.savez_compressed(d / GRID_FILE, nrow=self.nrow, ncol=self.ncol, cell_size=self.cell_size,
                            origin=np.array(self.origin), active=self.active, top=self.top,
                            land_frac=self.land_frac, coast=self.coast)
        self.overlap.to_csv(d / OVERLAP_FILE, index=False)
        self.river.to_csv(d / RIVER_FILE, index=False)

    @classmethod
    def load(cls, d: Path) -> 'NationalGrid':
        z = np.load(Path(d) / GRID_FILE)
        return cls(int(z['nrow']), int(z['ncol']), float(z['cell_size']), tuple(z['origin']), z['active'], z['top'],
                   z['land_frac'], z['coast'], pd.read_csv(Path(d) / OVERLAP_FILE), pd.read_csv(Path(d) / RIVER_FILE))


def build_grid(fabric_dir: Path, dem_path: Path, cell_size: float = 1000.0,
               stream_min_area_km2: float = 1.0) -> NationalGrid:
    """National grid from ``units.tif``, ``upa_km2.tif`` (fabric) and the DEM (same 25 m grid)."""
    import rasterio
    fabric_dir = Path(fabric_dir)
    with rasterio.open(fabric_dir / 'units.tif') as s:
        units, tr = s.read(1), s.transform
    res = abs(tr.a)
    f = int(round(cell_size / res))
    if abs(f * res - cell_size) > 1e-6:
        raise ConfigurationError(f'cell size {cell_size} is not a multiple of the fabric resolution {res}')
    nr, nc = -(-units.shape[0] // f), -(-units.shape[1] // f)
    with rasterio.open(dem_path) as s:
        if s.transform != tr or s.shape != units.shape:
            raise ValidationError(f'DEM {dem_path} is not on the fabric grid')
        dem = s.read(1).astype('float64')
        nd = s.nodata
    with rasterio.open(fabric_dir / 'upa_km2.tif') as s:
        stream = s.read(1) >= stream_min_area_km2
    land = units > 0
    dem = np.where(land & np.isfinite(dem) & ((dem != nd) if nd is not None else True) & (dem > -1000), dem, np.nan)
    rr, cc = np.nonzero(land)
    cell = (rr // f) * nc + (cc // f)
    u = units[rr, cc].astype(np.int64)
    n_cells = nr * nc
    cnt = np.bincount(cell, minlength=n_cells)
    z = dem[rr, cc]
    okz = np.isfinite(z)
    zsum = np.bincount(cell[okz], weights=z[okz], minlength=n_cells)
    zcnt = np.bincount(cell[okz], minlength=n_cells)
    with np.errstate(invalid='ignore', divide='ignore'):
        top = (zsum / zcnt).reshape(nr, nc)
    land_frac = (cnt / (f * f)).reshape(nr, nc)
    active = land_frac > 0
    top = np.where(active, np.where(np.isfinite(top), top, 0.0), np.nan)
    # (cell, unit) overlaps
    key = cell * (int(u.max()) + 1) + u
    k, n = np.unique(key, return_counts=True)
    c_, u_ = np.divmod(k, int(u.max()) + 1)
    overlap = pd.DataFrame({'row': c_ // nc + 1, 'col': c_ % nc + 1, 'unit': u_, 'area_m2': n * res * res})
    # streams per (cell, unit)
    s = stream[rr, cc]
    ks, ns = np.unique(key[s], return_counts=True)
    zs = pd.Series(z[s]).groupby(key[s]).mean().reindex(ks).to_numpy()
    c2, u2 = np.divmod(ks, int(u.max()) + 1)
    river = pd.DataFrame({'row': c2 // nc + 1, 'col': c2 % nc + 1, 'unit': u2, 'length_m': ns * res, 'zmean': zs})
    river = river[np.isfinite(river.zmean)]
    # coast: active cell with an inactive 4-neighbour inside the grid, or partly sea
    pad = np.pad(active, 1, constant_values=False)
    nb_inactive = ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    coast = active & (nb_inactive | (land_frac < 0.999))
    return NationalGrid(nr, nc, cell_size, (tr.c, tr.f), active, top, land_frac, coast, overlap, river)


def unit_property(attributes: pd.DataFrame, values: Dict[str, float], log: bool) -> pd.Series:
    """Per-unit property from geology-group fractions (unclassified area -> UNCLASSIFIED_GEOLOGY)."""
    F = pd.DataFrame({g: attributes.get(f'geo_{g}_frac', 0.0) for g in GEO_GROUPS}, index=attributes.index).fillna(0.0)
    F[UNCLASSIFIED_GEOLOGY] += (1.0 - F.sum(axis=1)).clip(lower=0.0)
    v = np.array([values[g] for g in GEO_GROUPS])
    return pd.Series(np.exp(F.to_numpy() @ np.log(v)) if log else F.to_numpy() @ v, index=attributes.index)


def _cell_array(grid: NationalGrid, per_unit: pd.Series, log: bool) -> np.ndarray:
    ov = grid.overlap
    w = ov.area_m2.to_numpy()
    x = per_unit.reindex(ov.unit).to_numpy()
    x = np.log(x) if log else x
    idx = (ov.row.to_numpy() - 1) * grid.ncol + ov.col.to_numpy() - 1
    s = np.bincount(idx, weights=w * x, minlength=grid.nrow * grid.ncol)
    a = np.bincount(idx, weights=w, minlength=grid.nrow * grid.ncol)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (s / a).reshape(grid.nrow, grid.ncol)
    out = np.exp(out) if log else out
    return np.where(grid.active, np.nan_to_num(out, nan=float(np.nanmedian(out))), 0.0)


def _array(a: np.ndarray, fmt: str) -> str:
    rowfmt = ' '.join([fmt] * a.shape[1])
    return '\n'.join('    ' + rowfmt % tuple(r) for r in a)


def _list_pkg(path: Path, lines: list, options: str = '  SAVE_FLOWS\n') -> None:
    Path(path).write_text(f'BEGIN OPTIONS\n{options}END OPTIONS\n\nBEGIN DIMENSIONS\n  MAXBOUND {len(lines)}\n'
                          'END DIMENSIONS\n\nBEGIN PERIOD 1\n' + '\n'.join(lines) + '\nEND PERIOD 1\n')


def write_packages(sim_dir: Path, grid: NationalGrid, attributes: pd.DataFrame, params: NationalParams) -> None:
    """Parameter-dependent packages: DIS, IC, NPF, STO, RCH (entries), DRN RIV/SEEP (+OBS), GHB SEA."""
    params.validate()
    d = Path(sim_dir)
    top = np.where(grid.active, grid.top, 0.0)
    riv = grid.river.copy()
    riv['strtop'] = riv.zmean - params.channel_depth
    base = top.copy()
    np.minimum.at(base, (riv.row.to_numpy() - 1, riv.col.to_numpy() - 1), riv.strtop.to_numpy())
    bot = base - params.thickness
    bot = np.where(grid.coast, np.minimum(bot, -10.0), bot)   # the sea boundary (stage 0) must lie inside the aquifer
    idomain = grid.active.astype(int)
    (d / 'gwf.dis').write_text(
        'BEGIN OPTIONS\n  LENGTH_UNITS METERS\n'
        f'  XORIGIN {grid.origin[0]:.3f}\n  YORIGIN {grid.origin[1] - grid.nrow * grid.cell_size:.3f}\n'
        'END OPTIONS\n\n'
        f'BEGIN DIMENSIONS\n  NLAY 1\n  NROW {grid.nrow}\n  NCOL {grid.ncol}\nEND DIMENSIONS\n\n'
        f'BEGIN GRIDDATA\n  DELR\n    CONSTANT {grid.cell_size}\n  DELC\n    CONSTANT {grid.cell_size}\n'
        f'  TOP\n    INTERNAL\n{_array(top, "%.2f")}\n  BOTM\n    INTERNAL\n{_array(bot, "%.2f")}\n'
        f'  IDOMAIN\n    INTERNAL\n{_array(idomain, "%d")}\nEND GRIDDATA\n')
    (d / 'gwf.ic').write_text(f'BEGIN GRIDDATA\n  STRT\n    INTERNAL\n{_array(top - 0.1 * params.thickness, "%.2f")}\n'
                              'END GRIDDATA\n')
    att = attributes
    k = _cell_array(grid, unit_property(att, params.k, log=True), log=True)
    sy = _cell_array(grid, unit_property(att, params.sy, log=False), log=False)
    (d / 'gwf.npf').write_text('BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\nBEGIN GRIDDATA\n  ICELLTYPE\n    CONSTANT 1\n'
                               f'  K\n    INTERNAL\n{_array(k, "%.6g")}\nEND GRIDDATA\n')
    (d / 'gwf.sto').write_text(
        'BEGIN OPTIONS\n  SAVE_FLOWS\nEND OPTIONS\n\nBEGIN GRIDDATA\n  ICONVERT\n    CONSTANT 1\n'
        f'  SS\n    CONSTANT {params.ss:.6g}\n  SY\n    INTERNAL\n{_array(sy, "%.5g")}\nEND GRIDDATA\n\n'
        'BEGIN PERIOD 1\n  STEADY-STATE\nEND PERIOD 1\n\nBEGIN PERIOD 2\n  TRANSIENT\nEND PERIOD 2\n')
    ca = grid.cell_size ** 2
    ov = grid.overlap
    rch = [f'  1 {r} {c} u{u} {a / ca:.8g}' for r, c, u, a in ov[['row', 'col', 'unit', 'area_m2']].itertuples(index=False)]
    _list_pkg(d / 'gwf.rch', rch, f'  AUXILIARY mult\n  AUXMULTNAME mult\n  TS6 FILEIN {RECHARGE_TS}\n')
    riv_lines = [f'  1 {r} {c} {s:.2f} {params.river_leakance * L:.6g} u{u}'
                 for r, c, u, L, s in riv[['row', 'col', 'unit', 'length_m', 'strtop']].itertuples(index=False)]
    _list_pkg(d / 'gwf.riv.drn', riv_lines, '  BOUNDNAMES\n  SAVE_FLOWS\n')
    seep = [f'  1 {r} {c} {top[r - 1, c - 1]:.2f} {params.seepage_leakance * a:.6g} u{u}'
            for r, c, u, a in ov[['row', 'col', 'unit', 'area_m2']].itertuples(index=False)]
    _list_pkg(d / 'gwf.seep.drn', seep, '  BOUNDNAMES\n  SAVE_FLOWS\n')
    drains = pd.concat([
        pd.DataFrame({'pkg': 'riv', 'row': riv.row, 'col': riv.col, 'unit': riv.unit, 'elev': riv.strtop.round(2),
                      'cond': (params.river_leakance * riv.length_m).map(lambda v: float(f'{v:.6g}'))}),
        pd.DataFrame({'pkg': 'seep', 'row': ov.row, 'col': ov.col, 'unit': ov.unit,
                      'elev': top[ov.row.to_numpy() - 1, ov.col.to_numpy() - 1].round(2),
                      'cond': (params.seepage_leakance * ov.area_m2).map(lambda v: float(f'{v:.6g}'))})], ignore_index=True)
    drains.to_csv(d / DRAIN_FILE, index=False)
    cr, cc = np.nonzero(grid.coast)
    sea = [f'  1 {r + 1} {c + 1} 0.0 {params.sea_conductance:.6g}' for r, c in zip(cr, cc)]
    _list_pkg(d / 'gwf.sea.ghb', sea)
    (d / 'national_params.json').write_text(json.dumps(asdict(params), indent=1))


def write_simulation(sim_dir: Path, grid: NationalGrid, attributes: pd.DataFrame, params: NationalParams,
                     n_days: int, save_heads: str = 'all') -> None:
    d = Path(sim_dir)
    d.mkdir(parents=True, exist_ok=True)
    grid.save(d)
    (d / 'mfsim.nam').write_text(
        'BEGIN OPTIONS\nEND OPTIONS\n\nBEGIN TIMING\n  TDIS6 gwf.tdis\nEND TIMING\n\n'
        'BEGIN MODELS\n  GWF6 gwf.nam gwf\nEND MODELS\n\nBEGIN EXCHANGES\nEND EXCHANGES\n\n'
        'BEGIN SOLUTIONGROUP 1\n  IMS6 gwf.ims gwf\nEND SOLUTIONGROUP 1\n')
    set_simulation_length(d, n_days)
    (d / 'gwf.nam').write_text(
        'BEGIN OPTIONS\n  NEWTON UNDER_RELAXATION\nEND OPTIONS\n\nBEGIN PACKAGES\n  DIS6 gwf.dis\n  IC6 gwf.ic\n'
        '  NPF6 gwf.npf\n  STO6 gwf.sto\n  RCH6 gwf.rch RCH\n  DRN6 gwf.riv.drn RIV\n  DRN6 gwf.seep.drn SEEP\n'
        '  GHB6 gwf.sea.ghb SEA\n  OC6 gwf.oc\nEND PACKAGES\n')
    head_save = {'none': '', 'last': '  SAVE HEAD LAST\n', 'all': '  SAVE HEAD ALL\n'}[save_heads]
    (d / 'gwf.oc').write_text('BEGIN OPTIONS\n  HEAD FILEOUT gwf.hds\n  BUDGETCSV FILEOUT gwf.bud.csv\nEND OPTIONS\n\n'
                              f'BEGIN PERIOD 1\n{head_save}  PRINT BUDGET LAST\nEND PERIOD 1\n')
    (d / 'gwf.ims').write_text(
        'BEGIN OPTIONS\n  PRINT_OPTION SUMMARY\n  COMPLEXITY COMPLEX\nEND OPTIONS\n\n'
        'BEGIN NONLINEAR\n  OUTER_DVCLOSE 1.0e-2\n  OUTER_MAXIMUM 300\nEND NONLINEAR\n\n'
        'BEGIN LINEAR\n  LINEAR_ACCELERATION BICGSTAB\n  INNER_DVCLOSE 1.0e-3\n  INNER_RCLOSE 1.0\n'
        '  INNER_MAXIMUM 500\nEND LINEAR\n')
    write_packages(d, grid, attributes, params)


def set_simulation_length(sim_dir: Path, n_days: int) -> None:
    """Period 1: one-day steady state. Period 2: ``n_days`` daily time steps (0 = steady state only).

    One transient period, not one per day: recharge varies through the time series, and package data are read
    once instead of every day."""
    periods = '  1.0 1 1.0\n' + (f'  {float(n_days)} {int(n_days)} 1.0\n' if n_days > 0 else '')
    Path(sim_dir, 'gwf.tdis').write_text(
        f'BEGIN OPTIONS\n  TIME_UNITS DAYS\nEND OPTIONS\n\nBEGIN DIMENSIONS\n  NPER {1 + (n_days > 0)}\nEND DIMENSIONS\n\n'
        'BEGIN PERIODDATA\n' + periods + 'END PERIODDATA\n')


def write_recharge_timeseries(sim_dir: Path, recharge: pd.DataFrame, units) -> pd.DatetimeIndex:
    """Recharge (m/d, daily rows, one column per unit). Time 0 (steady state) = record mean."""
    units = list(units)
    missing = [u for u in units if u not in recharge.columns]
    if missing:
        raise ValidationError(f'Recharge lacks {len(missing)} units, e.g. {missing[:5]}')
    rec = recharge[units].astype(float).clip(lower=0.0)
    vals = rec.to_numpy()
    rowfmt = ' '.join(['%.4e'] * len(units))
    body = ['  0.0 ' + rowfmt % tuple(vals.mean(axis=0))]
    body += [f'  {i + 1}.0 ' + rowfmt % tuple(r) for i, r in enumerate(vals)]
    body.append(f'  {len(vals) + 1}.0 ' + rowfmt % tuple(vals[-1]))
    with open(Path(sim_dir) / RECHARGE_TS, 'w') as fh:
        fh.write('BEGIN ATTRIBUTES\n  NAMES ' + ' '.join(f'u{u}' for u in units) + '\n  METHODS '
                 + ' '.join(['STEPWISE'] * len(units)) + '\nEND ATTRIBUTES\n\nBEGIN TIMESERIES\n')
        fh.write('\n'.join(body) + '\nEND TIMESERIES\n')
    return pd.DatetimeIndex(rec.index)


def read_heads(path: Path):
    """Iterate (totim, head array [nrow, ncol]) over an MF6 binary head file (one layer)."""
    hdr = np.dtype([('kstp', '<i4'), ('kper', '<i4'), ('pertim', '<f8'), ('totim', '<f8'), ('text', 'S16'),
                    ('ncol', '<i4'), ('nrow', '<i4'), ('ilay', '<i4')])
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(hdr.itemsize)
            if len(b) < hdr.itemsize:
                return
            h = np.frombuffer(b, hdr)[0]
            n = int(h['ncol']) * int(h['nrow'])
            a = np.frombuffer(fh.read(8 * n), '<f8').reshape(int(h['nrow']), int(h['ncol']))
            yield float(h['totim']), a


def discharge_per_unit(sim_dir: Path, start_date, by_package: bool = False) -> pd.DataFrame:
    """Daily groundwater discharge per unit (m3/s) through the river and seepage drains, transient steps only.

    Exact for MF6 drains without scaling options: q = C * max(h - z, 0) per entry, from the saved heads and the
    drain table written with the packages; columns are unit ids."""
    from scipy import sparse
    d = Path(sim_dir)
    dr = pd.read_csv(d / DRAIN_FILE)
    grid = np.load(d / GRID_FILE)
    ncol = int(grid['ncol'])
    cell = (dr.row.to_numpy() - 1) * ncol + dr.col.to_numpy() - 1
    units = np.sort(dr.unit.unique())
    col_of = pd.Series(np.arange(len(units)), index=units)
    keys = ['riv', 'seep'] if by_package else [None]
    mats = {}
    for k in keys:
        m = (dr.pkg == k).to_numpy() if k else np.ones(len(dr), bool)
        mats[k] = sparse.csr_matrix((dr.cond.to_numpy()[m], (col_of[dr.unit[m]].to_numpy(), np.flatnonzero(m))),
                                    shape=(len(units), len(dr)))
    elev = dr.elev.to_numpy()
    rows, times = {k: [] for k in keys}, []
    for totim, h in read_heads(d / 'gwf.hds'):
        if totim <= 1.0 + 1e-9:
            continue
        excess = np.maximum(h.ravel()[cell] - elev, 0.0)
        for k in keys:
            rows[k].append(mats[k] @ excess / 86400.0)
        times.append(totim)
    idx = pd.Timestamp(start_date).normalize() + pd.to_timedelta(np.round(np.array(times) - 2.0).astype(int), unit='D')
    out = {k: pd.DataFrame(np.array(v), index=idx, columns=units) for k, v in rows.items()}
    return out if by_package else out[None]
