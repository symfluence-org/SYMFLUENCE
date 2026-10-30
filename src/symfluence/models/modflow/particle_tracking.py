# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Groundwater travel times for a basin-mode MODFLOW 6 model with MODPATH 7.

The finished basin simulation is copied and rerun as one steady-state period at
the record-mean recharge (time 0 of the recharge series), saving heads and
cell-by-cell flows. MODPATH 7 then tracks one particle per active cell forward
from the water table (local z 0.99 of the saturated thickness, draped to the top
active cell) until it reaches a discharge boundary, with the final steady state
extended indefinitely.

Each particle is weighted by its cell's steady recharge (m3/d), so recharge-weighted
statistics describe the travel time of the water, not of the cells. The end cell
is classed as ``lake`` (lake GHB cell), ``river`` (river drain cell) or
``seepage`` (any other discharge).

Travel time scales linearly with effective porosity, which MODFLOW does not
constrain; it defaults to the model's specific yield.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ModelExecutionError, ValidationError

from .basin_model import GRID_FILE, PARAMS_FILE, RECHARGE_TS, WEIGHTS_FILE, MF6BasinGrid, _ts_name

ENDPOINT_FILE = 'mp.endpoint'
# MODPATH 7.2 endpoint status (MPath7.f90): 2 boundary face, 3 weak sink, 5 no exit possible (strong sink) are
# normal endings at a discharge cell; 1 still active, 6 stop zone, 7 inactive cell, 8/9 invalid/undefined are not.
DISCHARGED = (2, 3, 5)
DAYS_PER_YEAR = 365.25
# MODPATH 7 endpoint record (cell-based, structured grid): 26 columns
ENDPOINT_COLUMNS = ['sequence', 'group', 'particle_id', 'status', 'time0', 'time', 'node0', 'layer0', 'xloc0',
                    'yloc0', 'zloc0', 'x0', 'y0', 'z0', 'zone0', 'face0', 'node', 'layer', 'xloc', 'yloc', 'zloc',
                    'x', 'y', 'z', 'zone', 'face']


def _steady_state_copy(sim_dir: Path, work_dir: Path) -> None:
    """Copy the basin simulation and cut it to the steady-state period, saving heads and flows."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(sim_dir).iterdir():
        if f.is_file() and not f.name.endswith(('.hds', '.cbc', '.lst', '.csv')):
            shutil.copy2(f, work_dir / f.name)
    (work_dir / 'gwf.tdis').write_text(
        'BEGIN OPTIONS\n  TIME_UNITS DAYS\nEND OPTIONS\n\nBEGIN DIMENSIONS\n  NPER 1\nEND DIMENSIONS\n\n'
        'BEGIN PERIODDATA\n  1.0 1 1.0\nEND PERIODDATA\n')
    (work_dir / 'gwf.oc').write_text(
        'BEGIN OPTIONS\n  HEAD FILEOUT gwf.hds\n  BUDGET FILEOUT gwf.cbc\nEND OPTIONS\n\n'
        'BEGIN PERIOD 1\n  SAVE HEAD ALL\n  SAVE BUDGET ALL\nEND PERIOD 1\n')


def _write_modpath(work_dir: Path, grid: MF6BasinGrid, porosity: float) -> int:
    """MODPATH 7 name, basic and simulation files; returns the particle count."""
    d = Path(work_dir)
    (d / 'mp.mpnam').write_text('MPBAS      mp.mpbas\nGRBDIS     gwf.dis.grb\nTDIS       gwf.tdis\n'
                                'HEAD       gwf.hds\nBUDGET     gwf.cbc\n')
    # Recharge enters through the top face; drains and the lake take water from the whole cell
    (d / 'mp.mpbas').write_text(
        # MODFLOW 6 grids: no HNOFLO/HDRY, LAYTYP or IBOUND (MODPATH 7.2 takes them from the grid file)
        '# MODPATH basic data\n1\nRCH\n6\n'
        f'CONSTANT {porosity:.6g}\n')
    rows, cols = np.nonzero(grid.active)
    particles = [f'1 {r + 1} {c + 1} 0.5 0.5 0.99 0.0 1' for r, c in zip(rows, cols)]
    (d / 'mp.mpsim').write_text(
        '# MODPATH simulation: forward endpoints from the water table, steady state extended\n'
        'mp.mpnam\nmp.mplst\n'
        '1 1 2 1 0 0\n'          # endpoint run, forward, stop at weak sinks, pass weak sources, no budget, no trace
        f'{ENDPOINT_FILE}\n'
        '0\n'                    # no budget-report cells
        '1\n0.0\n'               # reference time option 1: time 0
        '2\n'                    # stop time option 2: extend the final steady-state period
        '1\n1\n'                 # no zones, no retardation
        '1\nwater_table\n1\n0.0\nINTERNAL\n'
        f'1\n1\n{len(particles)} 0\n' + '\n'.join(particles) + '\n')
    return len(particles)


def read_endpoints(path: Path) -> pd.DataFrame:
    """MODPATH 7 endpoint file -> DataFrame (one row per particle)."""
    lines = Path(path).read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper() == 'END HEADER') + 1
    except StopIteration as e:
        raise ValidationError(f'{path} is not a MODPATH 7 endpoint file') from e
    rows = [line.split() for line in lines[start:] if line.strip()]
    bad = [r for r in rows if len(r) != len(ENDPOINT_COLUMNS)]
    if bad:
        raise ValidationError(f'{path}: {len(bad)} endpoint records do not have {len(ENDPOINT_COLUMNS)} columns')
    df = pd.DataFrame(rows, columns=ENDPOINT_COLUMNS).apply(pd.to_numeric)
    return df


def cell_recharge(sim_dir: Path, grid: MF6BasinGrid) -> np.ndarray:
    """Steady (record-mean) recharge per cell (m3/d), with the reduced-zone factor applied."""
    d = Path(sim_dir)
    w = pd.read_csv(d / WEIGHTS_FILE)
    names, mean = None, None
    for line in (d / RECHARGE_TS).read_text().splitlines():
        t = line.split()
        if t[:1] == ['NAMES']:
            names = t[1:]
        elif names and len(t) == len(names) + 1 and t[0] in ('0.0', '0', '0.00'):
            mean = dict(zip(names, map(float, t[1:])))
            break
    if mean is None:
        raise ValidationError(f'No time-0 (record mean) row in {d / RECHARGE_TS}')
    params = json.loads((d / PARAMS_FILE).read_text()) if (d / PARAMS_FILE).exists() else {}
    f = float(params.get('zone_recharge_factor', 1.0))
    out = np.zeros((grid.nrow, grid.ncol))
    for x in w.itertuples():
        keep = f if x.zone_id in grid.reduced_zones else 1.0
        out[int(x.row) - 1, int(x.col) - 1] += mean[_ts_name(x.zone_id)] * x.mult * keep * grid.cell_area
    return out


def _run(exe: str, args: list, cwd: Path, timeout: int) -> None:
    r = subprocess.run([exe, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or 'Error' in r.stdout[-500:]:
        raise ModelExecutionError(f'{Path(exe).name} failed in {cwd}:\n{(r.stdout + r.stderr)[-2000:]}')


def track_steady_state(sim_dir: Path, work_dir: Path, mf6: str, mp7: str,
                       porosity: Optional[float] = None, timeout: int = 3600) -> pd.DataFrame:
    """Steady-state forward particle tracking for a finished basin-mode simulation.

    Returns one row per particle: start row/col, end row/col, travel time (days
    and years), MODPATH status, end class (lake/river/seepage) and the start
    cell's steady recharge (m3/d) as the weight. Also written to
    ``work_dir/travel_times.csv``.
    """
    sim_dir, work_dir = Path(sim_dir), Path(work_dir)
    if not (sim_dir / GRID_FILE).exists():
        raise ValidationError(f'{sim_dir} is not a basin-mode MODFLOW simulation')
    grid = MF6BasinGrid.from_json(sim_dir / GRID_FILE)
    if porosity is None:
        params = json.loads((sim_dir / PARAMS_FILE).read_text()) if (sim_dir / PARAMS_FILE).exists() else {}
        porosity = float(params.get('sy', 0.1))
    _steady_state_copy(sim_dir, work_dir)
    _run(mf6, [], work_dir, timeout)
    n = _write_modpath(work_dir, grid, porosity)
    _run(mp7, ['mp.mpsim'], work_dir, timeout)
    ep = read_endpoints(work_dir / ENDPOINT_FILE)
    if len(ep) != n:
        raise ValidationError(f'MODPATH returned {len(ep)} endpoints for {n} particles')
    ncell = grid.nrow * grid.ncol

    def rc(node):
        k = (node.to_numpy(int) - 1) % ncell
        return k // grid.ncol + 1, k % grid.ncol + 1

    r0, c0 = rc(ep.node0)
    r1, c1 = rc(ep.node)
    lake = grid.lake_frac if grid.has_lake else np.zeros((grid.nrow, grid.ncol))
    river = np.zeros((grid.nrow, grid.ncol), bool)
    river[grid.river.row.to_numpy(int) - 1, grid.river.col.to_numpy(int) - 1] = True
    end_class = np.where(lake[r1 - 1, c1 - 1] >= 0.5, 'lake',
                         np.where(river[r1 - 1, c1 - 1] & (lake[r1 - 1, c1 - 1] < 0.5), 'river', 'seepage'))
    rech = cell_recharge(sim_dir, grid)
    out = pd.DataFrame({'row0': r0, 'col0': c0, 'row': r1, 'col': c1,
                        'travel_days': ep.time - ep.time0, 'status': ep.status,
                        'discharged': ep.status.isin(DISCHARGED), 'end': end_class,
                        'recharge_m3d': rech[r0 - 1, c0 - 1], 'x0': ep.x0, 'y0': ep.y0, 'x': ep.x, 'y': ep.y})
    out['travel_years'] = out.travel_days / DAYS_PER_YEAR
    out.attrs['porosity'] = porosity
    out.to_csv(work_dir / 'travel_times.csv', index=False)
    return out


def weighted_quantiles(tt: pd.DataFrame, q=(0.1, 0.25, 0.5, 0.75, 0.9), by: Optional[str] = None) -> pd.DataFrame:
    """Recharge-weighted travel-time quantiles (years) of discharged particles, overall or per column ``by``."""
    def one(g):
        g = g[(g.recharge_m3d > 0) & g.discharged].sort_values('travel_years')
        if g.empty:
            return pd.Series({f'q{int(p * 100)}': np.nan for p in q} | {'recharge_m3d': 0.0})
        cw = g.recharge_m3d.cumsum().to_numpy() / g.recharge_m3d.sum()
        vals = {f'q{int(p * 100)}': float(np.interp(p, cw, g.travel_years.to_numpy())) for p in q}
        mean = float(np.average(g.travel_years, weights=g.recharge_m3d))
        return pd.Series(vals | {'mean': mean, 'recharge_m3d': float(g.recharge_m3d.sum())})
    if by is None:
        return one(tt).to_frame('all').T
    return tt.groupby(by)[[c for c in tt.columns if c != by]].apply(one)
