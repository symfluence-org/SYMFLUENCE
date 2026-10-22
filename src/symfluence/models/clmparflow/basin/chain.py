# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Basin-mode run chain: spin-up cycles over a few water years, then the production years continuously.

Spin-up repeats the spin-up water years (rewinding the calendar, carrying the state) until total subsurface plus
ponded storage changes by less than ``drift_tol`` of the cycle's precipitation, at most ``max_cycles`` times.
Each segment's daily outlet discharge, storage and glacier balance go to ``<runs>/<segment>/summary.json``;
progress to ``<runs>/status.json``. Segments whose summary exists are skipped, so the chain is resumable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ConfigurationError, ModelExecutionError

from . import pfio
from .case import BasinParameters, run_segment
from .grid import BasinGrid


def _clm_files(run: Path, name: str) -> List[Path]:
    return sorted(run.glob(f'{name}.out.clm_output.*.C.pfb'), key=lambda f: int(f.name.split('.')[-3]))


def glacier_ledger(run: Path, grid: BasinGrid) -> Dict[str, float]:
    """Ice melt and ice-column deficit (sublimation beyond supply) summed over the segment, in mm per glacier cell.

    Read from the per-rank ``glacier_ice.txt.<rank>`` logs of the patched CLM (istep, melt, deficit, infiltration
    per land-ice tile and step). Fortran writes very small numbers without the exponent letter; those are zero.
    """
    n = int(grid.ice.sum())
    logs = sorted(run.glob('glacier_ice.txt.*'))
    if not n or not logs:
        return dict(ice_melt_mm=float('nan'), ice_deficit_mm=float('nan'), glacier_cells=n)
    melt = deficit = 0.0
    lines = 0
    for log in logs:
        for line in log.read_text().splitlines():
            fields = line.split()[1:3]
            m, d = (float(x) if ('E' in x or x.lstrip('-').isdigit()) else 0.0 for x in fields)
            melt += m
            deficit += d
            lines += 1
    return dict(ice_melt_mm=melt / n, ice_deficit_mm=deficit / n, glacier_cells=n, log_lines=lines)


def summarise_segment(run: Path, grid: BasinGrid) -> Dict:
    """Daily outlet discharge (m3/s), storage at both ends (m3), end SWE and glacier ledger of a finished segment."""
    name = run.name
    meta = json.loads((run / 'meta.json').read_text())
    hours = meta['hours']
    dz = np.array(meta['params']['dz_top_down'][::-1])[:, None, None]
    cell = grid.dx * grid.dx
    phi = pfio.read_pfb(run / f'{name}.out.porosity.pfb')
    ss = pfio.read_pfb(run / f'{name}.out.specific_storage.pfb')

    def files(var: str) -> List[Path]:  # numbered by dump index or by hour depending on the run; order matters
        return sorted(run.glob(f'{name}.out.{var}.*.pfb'), key=lambda f: int(f.name.split('.')[-2]))

    press, satur, osum = files('press'), files('satur'), files('overlandsum')

    def storage(i: int):
        p, s = pfio.read_pfb(press[i]), pfio.read_pfb(satur[i])
        sub = ((s * phi + ss * s * p) * dz * cell * grid.mask[None]).sum()
        pond = (np.maximum(p[-1], 0) * cell * grid.mask).sum()
        return float(sub), float(pond)

    days = pd.date_range(meta['start'], periods=hours // 24, freq='D')
    if not len(osum) == len(days) == len(press) - 1:
        raise ModelExecutionError(f'{name}: {len(osum)} outflow and {len(press)} pressure dumps for {len(days)} days')
    q = [float(pfio.read_pfb(f).sum()) / 86400 for f in osum]
    s0, p0 = storage(0)
    s1, p1 = storage(-1)
    clm = pfio.read_pfb(_clm_files(run, name)[-1])
    out = dict(seg=name, start=meta['start'], end=meta['end'],
               outlet_m3s=dict(zip([d.strftime('%Y-%m-%d') for d in days], q)),
               sub0=s0, sub1=s1, pond0=p0, pond1=p1, swe_end_m3=float((clm[10] * grid.mask).sum() / 1000 * cell),
               glacier_swe_end_mm=float(clm[10][grid.ice].mean()) if grid.ice.any() else float('nan'),
               mean_q_m3s=float(np.mean(q)), run_s=meta.get('run_s'), **glacier_ledger(run, grid))
    (run / 'summary.json').write_text(json.dumps(out))
    return out


def basin_precipitation(grid: BasinGrid, forcing_file: Path, start: str, end: str, precip_name: str,
                        unit_variable: str, precip_scale: float) -> float:
    """Basin precipitation volume (m3) over [start, end); each active cell takes its unit's column (mm/s)."""
    from .case import open_forcing

    counts = np.bincount(grid.unit[grid.mask], minlength=len(grid.units))
    with open_forcing(forcing_file) as ds:
        ids = [str(u) for u in ds[unit_variable].values]
        total = ds[precip_name].transpose('time', ...).sel(time=slice(start, pd.Timestamp(end) - pd.Timedelta(hours=1))).sum('time').values
    return float(sum(total[ids.index(u)] * 3600 / 1000 * grid.dx * grid.dx * n
                     for u, n in zip(grid.units, counts) if n) * precip_scale)


@dataclass
class ChainSettings:
    """Water years of the spin-up and production segments, the drift test and the process grid."""

    spinup_years: List[int] = field(default_factory=lambda: [2010, 2011, 2012])
    production_years: List[int] = field(default_factory=lambda: list(range(2013, 2024)))
    drift_tol: float = 0.05
    max_cycles: int = 3
    initial_state: Optional[Path] = None  # finished segment to start the spin-up from (warm start)
    P: int = 1
    Q: int = 1
    forcing_unit_variable: str = 'unit_id'
    forcing_names: Dict[str, str] = field(default_factory=dict)
    mpirun: Optional[str] = None
    timeout: Optional[int] = None


def run_chain(runs: Path, grid: BasinGrid, params: BasinParameters, parflow_dir: Path, forcing_file: Path,
              settings: ChainSettings, log: Callable[[str], None] = print) -> pd.Series:
    """Run (or resume) the chain; returns the daily outlet discharge of the production years."""
    from .case import DEFAULT_FORCING_NAMES

    runs.mkdir(parents=True, exist_ok=True)
    status_file = runs / 'status.json'

    def status(**kw) -> None:
        s = json.loads(status_file.read_text()) if status_file.exists() else {}
        s.update(kw, updated=time.strftime('%Y-%m-%d %H:%M'))
        status_file.write_text(json.dumps(s, indent=1))

    def segment(name: str, wy: int, prev: Optional[Path]) -> Dict:
        run = runs / name
        if (run / 'summary.json').exists():
            return json.loads((run / 'summary.json').read_text())
        status(current=name)
        log(f'ParFlow-CLM basin segment {name} (WY{wy})')
        run_segment(run, name, grid, params, parflow_dir, forcing_file, f'{wy - 1}-10-01', f'{wy}-10-01', prev,
                    settings.P, settings.Q, mpirun=settings.mpirun, timeout=settings.timeout,
                    forcing_unit_variable=settings.forcing_unit_variable, forcing_names=settings.forcing_names)
        s = summarise_segment(run, grid)
        status(**{f'done_{name}': dict(mean_q_m3s=round(s['mean_q_m3s'], 1),
                                       dsub_hm3=round((s['sub1'] - s['sub0']) / 1e6, 1),
                                       run_min=round((s['run_s'] or 0) / 60))})
        return s

    precip_name = {**DEFAULT_FORCING_NAMES, **settings.forcing_names}['APCP']
    pc = sum(basin_precipitation(grid, forcing_file, f'{y - 1}-10-01', f'{y}-10-01', precip_name,
                                 settings.forcing_unit_variable, params.precip_scale) for y in settings.spinup_years)
    if not settings.spinup_years:
        raise ConfigurationError('basin mode needs at least one spin-up water year')
    prev = settings.initial_state
    cycle = 0
    for cycle in range(1, settings.max_cycles + 1):
        cycle_segments = []
        for wy in settings.spinup_years:
            name = f's{cycle}_wy{wy}'
            cycle_segments.append(segment(name, wy, prev))
            prev = runs / name
        first, last = cycle_segments[0], cycle_segments[-1]
        drift =(last['sub1'] + last['pond1'] - first['sub0'] - first['pond0']) / pc
        status(**{f'cycle{cycle}_drift_frac_P': round(drift, 4)})
        log(f'ParFlow-CLM spin-up cycle {cycle}: storage drift {100 * drift:+.1f}% of precipitation')
        if abs(drift) < settings.drift_tol:
            break
    status(spinup_cycles=cycle)
    q = []
    for wy in settings.production_years:
        name = f'p_wy{wy}'
        q.append(pd.Series(segment(name, wy, prev)['outlet_m3s']))
        prev = runs / name
    status(current='done')
    out = pd.concat(q)
    out.index = pd.to_datetime(out.index)
    return out.sort_index().rename('discharge_cms')


def glacier_mass_balance(runs: Path, grid: BasinGrid, years: List[int], first_prev: Optional[Path] = None) -> pd.Series:
    """Glacier-wide net balance per water year (mm w.e.): change of glacier-cell SWE minus ice melt and deficit."""
    rows = {}
    prev_swe = None
    if first_prev is not None:
        prev_swe = json.loads((first_prev / 'summary.json').read_text())['glacier_swe_end_mm']
    for wy in years:
        s = json.loads((runs / f'p_wy{wy}' / 'summary.json').read_text())
        if prev_swe is not None:
            rows[wy] = s['glacier_swe_end_mm'] - prev_swe - s['ice_melt_mm'] - s['ice_deficit_mm']
        prev_swe = s['glacier_swe_end_mm']
    return pd.Series(rows, name='glacier_net_balance_mm', dtype=float)
