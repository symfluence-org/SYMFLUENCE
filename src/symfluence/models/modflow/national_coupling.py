# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Two-pass HYPE + MODFLOW 6 coupling for a whole-country fabric (docs/adr/0012-national-hype-mf6.md).

Pass 1: HYPE with deep percolation leaving each subbasin (AQIN). MODFLOW 6 takes it as recharge per unit and
returns groundwater discharge per unit (river and seepage drains). Pass 2: HYPE again with that discharge as daily
main-river point sources, so it is routed through lakes, branches and the river network. Point sources do not
change soil water, so pass 2's recharge equals pass 1's and the exchange is exact for one-way coupling.

Transfers have commissioning years and HYPE's BranchData is static, so pass 2 runs as consecutive eras
(``branch_eras``), each with the transfers active that year, carrying the full model state across era boundaries.
Pass 1 runs straight through with every transfer: branches move river water only, so recharge does not depend on them.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

from symfluence.models.hype.national_fabric import branch_eras
from symfluence.models.hype.national_glacier import IceFlux, read_glacvol, write_glacvol

from . import national_model as nm
from .land_adapters import get_land_adapter

logger = logging.getLogger(__name__)


@dataclass
class TwoPassPaths:
    hype_settings: Path
    hype_exe: Path
    mf6_exe: Path
    results: Path              # results/pass1, results/pass2, results/mf6


def _run(cmd: list, cwd: Path, log: Path) -> None:
    with open(log, 'w') as fh:
        r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise ValidationError(f'{cmd[0]} failed ({r.returncode}); see {log}')


def _check_mf6(sim: Path) -> None:
    lst = (sim / 'mfsim.lst').read_text(errors='replace')
    if 'Normal termination' not in lst:
        raise ValidationError(f'MODFLOW did not terminate normally in {sim}')
    failed = (sim / 'gwf.lst').read_text(errors='replace').count('FAILED TO MEET SOLVER CONVERGENCE')
    if failed:
        raise ValidationError(f'MODFLOW: {failed} time steps failed to converge in {sim}')


def run_two_pass(paths: TwoPassPaths, grid: nm.NationalGrid, attributes: pd.DataFrame,
                 params: nm.NationalParams, ice: 'IceFlux | None' = None,
                 pass1_outputs: tuple | None = ('COUT', 'AQIN')) -> dict:
    """Pass 1 (HYPE, deep drainage leaves the model) -> MODFLOW 6 on its recharge -> pass 2 (HYPE with the MODFLOW
    discharge as point sources). With ``ice`` both HYPE passes run as eras split on the annual date with ice flow
    (run_eras); pass 1 then also follows the transfer eras. ``pass1_outputs`` limits pass 1's time series to what the
    coupling needs (None keeps the settings' list)."""
    hype = get_land_adapter('HYPE')
    s, out = Path(paths.hype_settings), Path(paths.results)
    timing = {}

    t = time.time()
    hype.clear_lake_flux(s)                 # no point sources in pass 1
    hype.enable_recharge_export(s)
    full_outputs = next((l for l in _info_lines(s) if l.split()[:2] == ['timeoutput', 'variable']), None)
    if pass1_outputs and full_outputs:
        set_info(s, {'timeoutput variable': '\t'.join(pass1_outputs)})
    p1 = out / 'pass1'
    hype.set_output_dir(s, p1)
    eras1 = None
    try:
        if ice is None:
            _run([str(paths.hype_exe), f'{s}/'], s, p1 / 'hype.log')
        else:
            info = {l.split()[0]: l.split()[1] for l in _info_lines(s) if len(l.split()) >= 2}
            eras1 = run_eras(paths.hype_exe, s, p1, pd.Timestamp(info['bdate']), pd.Timestamp(info['edate']), ice)
    finally:
        if pass1_outputs and full_outputs:
            set_info(s, {'timeoutput variable': '\t'.join(full_outputs.split()[2:])})
    timing['hype_pass1_s'] = time.time() - t

    t = time.time()
    rec = hype.read_recharge(p1, s)
    sim = out / 'mf6'
    nm.write_simulation(sim, grid, attributes, params, n_days=len(rec))
    nm.write_recharge_timeseries(sim, rec, sorted(grid.overlap.unit.unique()))
    _run([str(paths.mf6_exe)], sim, sim / 'mf6.log')
    _check_mf6(sim)
    gw = nm.discharge_per_unit(sim, rec.index[0])
    timing['mf6_s'] = time.time() - t

    t = time.time()
    hype.route_groundwater(s, gw.reindex(rec.index).fillna(0.0))
    p2 = out / 'pass2'
    eras = run_eras(paths.hype_exe, s, p2, rec.index[0], rec.index[-1], ice)
    timing['hype_pass2_s'] = time.time() - t

    report = dict(timing=timing, eras_pass1=eras1, eras=eras, balance=water_balance(s, p1, p2, sim, gw))
    (out / 'two_pass.json').write_text(json.dumps(report, indent=1, default=float))
    return report


def _info_lines(settings: Path) -> list:
    return (Path(settings) / 'info.txt').read_text().splitlines()


def set_info(settings: Path, values: dict) -> None:
    """Set info.txt entries; a key may have several words ('modeloption glacierini'). None removes the entry."""
    lines = _info_lines(settings)
    for key, val in values.items():
        n = len(key.split())
        lines = [l for l in lines if l.split()[:n] != key.split()]
        if val is not None:
            lines.append(f'{key}\t{val}')
    (Path(settings) / 'info.txt').write_text('\n'.join(lines) + '\n')


def run_eras(hype_exe: Path, settings: Path, out: Path, start, end, ice: 'IceFlux | None' = None) -> list:
    """Run HYPE over [start, end] as consecutive branch eras with the state carried across.

    Each era writes its BranchData, starts from the previous era's ``state_save<date>.txt`` (copied from its
    result directory into the model directory, where HYPE reads it) and saves its own at the next era's start.
    Glaciers are initialised from GlacierData only in the first era (glacierini 1), later eras take the saved
    state (glacierini 0). The last era also saves the state for the day after ``end`` (``state_save<date>.txt`` in
    its result directory), the starting point of a continuation or a live forecast. With ``ice`` (national_glacier.
    IceFlux) eras are also split on its annual date and each ice basin's volume change over the year is
    redistributed by delta-h in the state handed to the next segment (ice flow between subbasins). Era outputs are concatenated into ``out``; settings files are restored afterwards."""
    s, out = Path(settings), Path(out)
    tr_file = s / 'national_transfers.csv'
    if not tr_file.exists():
        raise ValidationError(f'{tr_file} missing: rebuild the HYPE inputs with national_fabric.write_inputs')
    eras = branch_eras(pd.read_csv(tr_file), start, end)
    if ice is not None:
        eras = _split_annual(eras, ice.month, ice.day)
    saved = {f: (s / f).read_bytes() for f in ('info.txt', 'BranchData.txt') if (s / f).exists()}
    log = []
    v_year = None
    try:
        for k, (a, b, branch, ids) in enumerate(eras):
            nxt = b + pd.Timedelta(days=1)
            if len(branch):
                branch.to_csv(s / 'BranchData.txt', sep='\t', index=False, float_format='%.6g')
            elif (s / 'BranchData.txt').exists():
                (s / 'BranchData.txt').unlink()
            d = out / f'era{k:03d}_{a:%Y%m}'
            d.mkdir(parents=True, exist_ok=True)
            set_info(s, {'bdate': f'{a:%Y-%m-%d}', 'cdate': f'{a:%Y-%m-%d}', 'edate': f'{b:%Y-%m-%d}',
                         'resultdir': f'{d}/', 'instate': 'y' if k else 'n',
                         'outstatedate': f'{nxt:%Y-%m-%d}',
                         'modeloption glacierini': 1 if k == 0 else 0})
            t = time.time()
            _run([str(hype_exe), f'{s}/'], s, d / 'hype.log')
            rec = dict(start=f'{a:%Y-%m-%d}', end=f'{b:%Y-%m-%d}', transfers=ids)
            if k < len(eras) - 1:
                st = s / f'state_save{nxt:%Y%m%d}.txt'
                shutil.copy2(d / st.name, st)
                if ice is not None:
                    if v_year is None:
                        v_year = ice.initial_volume(s)
                    if (nxt.month, nxt.day) == (ice.month, ice.day):
                        v_end = read_glacvol(st)
                        v_new = ice.redistribute(v_year, v_end)
                        write_glacvol(st, v_new)
                        rec['ice_moved_km3'] = float(np.abs(v_new - v_end).sum() / 2e9)
                        rec['ice_change_km3'] = float((v_end.sum() - v_year.sum()) / 1e9)
                        v_year = v_new
            rec['seconds'] = time.time() - t
            log.append(rec)
            logger.info('era %s..%s (%d transfers) %.0f s', a.date(), b.date(), len(ids), log[-1]['seconds'])
    finally:
        for f, data in saved.items():
            (s / f).write_bytes(data)
    _concat_eras(out, len(eras))
    return log


def _split_annual(eras: list, month: int, day: int) -> list:
    """Split each era at every (month, day) inside it; segments keep the era's BranchData."""
    out = []
    for a, b, branch, ids in eras:
        cuts = [c for c in pd.date_range(a, b, freq='D') if (c.month, c.day) == (month, day) and c > a]
        bounds = [a] + cuts + [b + pd.Timedelta(days=1)]
        out += [(x, y - pd.Timedelta(days=1), branch, ids) for x, y in zip(bounds[:-1], bounds[1:])]
    return out


def _concat_eras(out: Path, n: int, cleanup: bool = True) -> None:
    """Join the eras' time*.txt files (same subids, consecutive dates) into ``out``.

    With ``cleanup`` each era's copy is deleted once the joined file holds as many data rows as the eras together
    (logs and state files are kept)."""
    first = sorted(out.glob('era*_*'))[0]
    for f in first.glob('time*.txt'):
        parts = sorted(out.glob(f'era*_*/{f.name}'))
        if len(parts) != n:
            raise ValidationError(f'{f.name}: {len(parts)} of {n} eras have it')
        rows = 0
        with open(out / f.name, 'w') as dst:
            for i, p in enumerate(parts):
                with open(p) as src:
                    head = [src.readline(), src.readline()]
                    if i == 0:
                        dst.writelines(head)
                    for line in src:
                        dst.write(line)
                        rows += 1
        if cleanup:
            with open(out / f.name) as fh:
                joined = sum(1 for _ in fh) - 2
            if joined != rows:
                raise ValidationError(f'{f.name}: joined {joined} rows, eras {rows}')
            for p in parts:
                p.unlink()


def _hype(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep='\t', skiprows=1, index_col=0, parse_dates=True)
    d.columns = [int(float(c)) for c in d.columns]
    return d


def water_balance(settings: Path, p1: Path, p2: Path, sim: Path, gw: pd.DataFrame | None = None) -> dict:
    """National balance in mm/yr over the run: what MODFLOW received, stored and returned, and sea outflow.

    ``gw`` is the daily discharge per unit handed to HYPE (m3/s); without it (the head file may be archived) the
    returned discharge is taken from the drain budget, which it equals."""
    gd = pd.read_csv(Path(settings) / 'GeoData.txt', sep='\t').set_index('subid')
    area = gd.area.sum()
    yrs = None
    b = pd.read_csv(sim / 'gwf.bud.csv')
    b = b[b['time'] > 1.0 + 1e-9]
    yrs = len(b) / 365.25

    def mm(m3):
        return float(m3) / area * 1000 / yrs
    term = gd.index[gd.maindown == 0]
    q1, q2 = _hype(p1 / 'timeCOUT.txt'), _hype(p2 / 'timeCOUT.txt')
    out = dict(
        years=yrs,
        recharge=mm(b['RCH(RCH)_IN'].sum()),
        gw_to_rivers=mm(b['DRN(RIV)_OUT'].sum() - b['DRN(RIV)_IN'].sum()),
        gw_seepage=mm(b['DRN(SEEP)_OUT'].sum() - b['DRN(SEEP)_IN'].sum()),
        gw_to_sea=mm(b['GHB(SEA)_OUT'].sum() - b['GHB(SEA)_IN'].sum()),
        gw_storage_gain=mm(sum(b[c].sum() for c in b if c.startswith("STO") and c.endswith("_OUT"))
                             - sum(b[c].sum() for c in b if c.startswith("STO") and c.endswith("_IN"))),
        sea_outflow_pass1=mm(q1[term].sum().sum() * 86400),
        sea_outflow_pass2=mm(q2[term].sum().sum() * 86400),
    )
    out['discharge_returned'] = (mm(gw.sum().sum() * 86400) if gw is not None
                                 else out['gw_to_rivers'] + out['gw_seepage'])
    out['pass2_minus_pass1'] = out['sea_outflow_pass2'] - out['sea_outflow_pass1']
    return out
