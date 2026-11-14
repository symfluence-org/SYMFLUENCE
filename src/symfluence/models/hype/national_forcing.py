# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE daily forcing for a whole-country fabric from a daily CARRA land-cell archive.

The archive is one ``YYYYMM.npz`` per month (``landslikan.carra daily``): ``days`` and, per variable, a
[n_days, n_cells] array on the flat CARRA cell indices ``cells`` (row-major on the 221 x 301 grid). A model day is
a true UTC day (early-stamped files before 2024-07 are handled by the archive). Units are mapped with the fabric's
``forcing_carra`` area weights; temperature is moved from CARRA orography to the unit's mean DEM elevation with a
fixed lapse rate, and HYPE's own class elevations (``dhslc``) take it from there.

Written (tab separated, one column per subid): Pobs, Tobs, TMINobs, TMAXobs, SWobs (MJ m-2 d-1), RHobs (0-1),
Uobs (m/s at 2 m, FAO-56 eq. 47 from 10 m), SFobs (snowfall fraction: CARRA's own phase split, wet-bulb where a
day has no phase data).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

CARRA_NX = 301
LAPSE_K_PER_M = -0.0065
WIND_10_TO_2 = 4.87 / np.log(67.8 * 10.0 - 5.42)
W_TO_MJ_DAY = 0.0864
OBS = {   # file: (archive variable, transform, format)
    'Pobs': ('p', lambda x: np.clip(x, 0, None), '%.2f'),
    'Tobs': ('t', None, '%.2f'),
    'TMINobs': ('tmin', None, '%.2f'),
    'TMAXobs': ('tmax', None, '%.2f'),
    'SWobs': ('sw', lambda x: np.clip(x, 0, None) * W_TO_MJ_DAY, '%.3f'),
    'RHobs': ('rh', lambda x: np.clip(x / 100.0, 0.01, 1.0), '%.3f'),
    'Uobs': ('u10', lambda x: np.clip(x, 0, None) * WIND_10_TO_2, '%.2f'),
    'SFobs': ('p_solid_frac', lambda x: np.clip(x, 0, 1), '%.3f'),
}
T_FILES = ('Tobs', 'TMINobs', 'TMAXobs')


def weight_matrix(weights: pd.DataFrame, subids: np.ndarray, cells: np.ndarray):
    """Sparse [n_subid, n_cells] matrix from forcing_carra rows (unit_id, carra_j, carra_i, weight)."""
    from scipy import sparse
    flat = weights.carra_j.to_numpy(int) * CARRA_NX + weights.carra_i.to_numpy(int)
    col_of = pd.Series(np.arange(len(cells)), index=cells)
    missing = ~np.isin(flat, cells)
    if missing.any():
        raise ValidationError(f'{missing.sum()} forcing weights point at cells missing from the archive')
    row_of = pd.Series(np.arange(len(subids)), index=subids)
    if not weights.unit_id.isin(row_of.index).all():
        raise ValidationError('forcing weights reference units that are not subbasins')
    W = sparse.csr_matrix((weights.weight.to_numpy(float), (row_of[weights.unit_id].to_numpy(),
                                                             col_of[flat].to_numpy())), shape=(len(subids), len(cells)))
    s = np.asarray(W.sum(axis=1)).ravel()
    if np.abs(s - 1).max() > 1e-6:
        raise ValidationError(f'forcing weights do not sum to 1 per unit (max error {np.abs(s - 1).max():.2e})')
    return W


def write_forcing(daily_dir: Path, weights: pd.DataFrame, subids, out_dir: Path, start: str, end: str,
                  dz: Optional[pd.Series] = None) -> pd.DatetimeIndex:
    """Write the HYPE obs files for [start, end] (inclusive). ``dz`` = unit elevation − CARRA orography (m)."""
    daily_dir, out_dir = Path(daily_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subids = np.asarray(subids, int)
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq='M')
    files = [daily_dir / f'{m.year}{m.month:02d}.npz' for m in months]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise ValidationError(f'Daily forcing archive lacks {len(missing)} months: {missing[:6]}')
    t_shift = (LAPSE_K_PER_M * dz.reindex(subids).fillna(0.0).to_numpy()) if dz is not None else 0.0
    handles = {k: open(out_dir / f'{k}.txt', 'w') for k in OBS}
    header = 'DATE\t' + '\t'.join(str(s) for s in subids) + '\n'
    for h in handles.values():
        h.write(header)
    W, cells_ref, dates = None, None, []
    t0, t1 = np.datetime64(pd.Timestamp(start).date()), np.datetime64(pd.Timestamp(end).date())
    try:
        for f in files:
            z = np.load(f)
            if W is None or not np.array_equal(z['cells'], cells_ref):
                cells_ref = z['cells']
                W = weight_matrix(weights, subids, cells_ref)
            days = z['days'].astype('datetime64[D]')
            keep = (days >= t0) & (days <= t1)
            if not keep.any():
                continue
            labels = pd.DatetimeIndex(days[keep]).strftime('%Y-%m-%d').to_numpy()
            for k, (var, fn, fmt) in OBS.items():
                x = np.asarray(z[var][keep], float)
                bad = ~np.isfinite(x)
                if bad.any():   # a cell without data on a day: fill from the day's mean over cells
                    x = np.where(bad, np.nanmean(x, axis=1, keepdims=True), x)
                u = (W @ x.T).T
                if k in T_FILES:
                    u = u + t_shift
                if fn is not None:
                    u = fn(u)
                rowfmt = '\t'.join([fmt] * u.shape[1])
                handles[k].write(''.join(f'{d}\t{rowfmt % tuple(r)}\n' for d, r in zip(labels, u)))
            dates.extend(days[keep])
    finally:
        for h in handles.values():
            h.close()
    idx = pd.DatetimeIndex(dates)
    if len(idx) and (np.diff(idx.values).astype('timedelta64[D]') != np.timedelta64(1, 'D')).any():
        raise ValidationError('Daily forcing is not continuous')
    logger.info('HYPE forcing: %d days x %d subbasins -> %s', len(idx), len(subids), out_dir)
    return idx
