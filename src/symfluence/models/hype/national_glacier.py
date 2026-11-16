# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Ice flow between HYPE subbasins for the national model (docs/adr/0012-national-hype-mf6.md).

HYPE keeps one glacier per subbasin and no ice moves between subbasins. On a fabric whose units split ice caps
along subglacial divides, accumulation-zone units then gain ice without limit while ablation-zone units melt out
(1991-2025 pass 1: 118 units +421 km3, 621 units -321 km3, 243 melted out). Real outlet glaciers are fed by ice
flow from above.

Once a year (default 1 October, the end of the melt season) the volume change of each ice basin over the past
year is redistributed over its units with the delta-h parameterisation of Huss et al. (2010, HESS 14, 815-829):
change is largest at the terminus and near zero at the top, and the basin total is conserved. An ice basin is a
set of glacier subbasins draining to the same last glacier subbasin along maindown.

The state file layout (HYPE 5.36, statetype_module): the first section holds frozen states as one flat array in
lines of 100: snow, snowage, snowdepth, snowcov, snowmax (each n x nc, subbasin-major), csnow (n x nc x ns), then
glacvol (n, m3) in GeoData order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError

# Huss et al. (2010) delta-h coefficients (a, b, c, gamma) by glacier size
HUSS = {'large': (-0.02, 0.12, 0.00, 6), 'medium': (-0.05, 0.19, 0.01, 4), 'small': (-0.30, 0.60, 0.09, 2)}
PER_LINE = 100


def _glacvol_slice(state_lines: list, ns: int = 0) -> slice:
    h = state_lines[0].split()
    n, nc = int(h[10]), int(h[11])
    first = 5 * n * nc + n * nc * ns
    return slice(first, first + n)


def read_glacvol(state: Path, ns: int = 0) -> np.ndarray:
    """Glacier volume (m3) per subbasin in GeoData order from a HYPE state file."""
    lines = Path(state).read_text().split('\n')
    sl = _glacvol_slice(lines, ns)
    l0, l1 = 1 + sl.start // PER_LINE, 1 + (sl.stop - 1) // PER_LINE
    tok = ' '.join(lines[l0:l1 + 1]).split()
    off = sl.start - (l0 - 1) * PER_LINE
    return np.array(tok[off:off + (sl.stop - sl.start)], float)


def write_glacvol(state: Path, vol: np.ndarray, ns: int = 0) -> None:
    """Replace the glacier volumes in a HYPE state file, keeping every other value and the layout."""
    state = Path(state)
    text = state.read_text()
    nl = '\r\n' if '\r\n' in text else '\n'
    lines = text.split(nl)
    sl = _glacvol_slice(lines, ns)
    if len(vol) != sl.stop - sl.start:
        raise ValidationError(f'{len(vol)} volumes for {sl.stop - sl.start} subbasins')
    for k, v in zip(range(sl.start, sl.stop), vol):
        li, pos = 1 + k // PER_LINE, k % PER_LINE
        parts = lines[li].split('\t')
        nvals = len(parts) - (parts[-1] == '')            # lines end with a tab
        if nvals != PER_LINE and li != 1 + (sl.stop - 1) // PER_LINE:
            raise ValidationError(f'unexpected state line {li}: {nvals} values')
        parts[pos] = f'{v:15.8E}'
        lines[li] = '\t'.join(parts)
    state.write_text(nl.join(lines))


def ice_basins(geodata: pd.DataFrame, is_glacier: pd.Series) -> pd.Series:
    """subid -> root subid: follow maindown while the next subbasin also has glacier."""
    down = geodata.set_index('subid').maindown.astype(int)
    glac = set(is_glacier[is_glacier].index.astype(int))
    root = {}
    for s in glac:
        path, cur = [], s
        while True:
            if cur in root:
                r = root[cur]
                break
            path.append(cur)
            nxt = int(down.get(cur, 0))
            if nxt not in glac or nxt in path:
                r = cur
                break
            cur = nxt
        for p in path:
            root[p] = r
    return pd.Series(root, name='basin').sort_index()


@dataclass
class IceFlux:
    """Annual delta-h redistribution of glacier volume within ice basins."""
    subid: np.ndarray           # GeoData order
    basin: np.ndarray           # root subid per subbasin, 0 = no glacier
    z: np.ndarray               # glacier class elevation (m)
    area: np.ndarray            # glacier class area (m2)
    month: int = 10             # redistribute on the 1st of this month
    day: int = 1

    @classmethod
    def from_settings(cls, settings: Path, month: int = 10) -> 'IceFlux':
        s = Path(settings)
        gd = pd.read_csv(s / 'GeoData.txt', sep='\t')
        gc = pd.read_csv(s / 'GeoClass.txt', sep='\t', comment='!', header=None)
        gslc = [int(r[0]) for _, r in gc.iterrows() if int(r[7]) == 3]
        if len(gslc) != 1:
            raise ValidationError(f'expected one glacier SLC (special class 3), found {gslc}')
        k = gslc[0]
        frac = gd[f'slc_{k}'].to_numpy(float)
        area = frac * gd.area.to_numpy(float)
        z = gd.elev_mean.to_numpy(float) + gd[f'dhslc_{k}'].to_numpy(float)
        isg = pd.Series(area > 0, index=gd.subid.astype(int))
        roots = ice_basins(gd, isg)
        basin = roots.reindex(gd.subid.astype(int)).fillna(0).to_numpy(int)
        return cls(gd.subid.to_numpy(int), basin, z, area, month)

    def initial_volume(self, settings: Path) -> np.ndarray:
        """Volume HYPE starts from with glacierini 1: exp(logvolcor) * glacvcoef1 * A^glacvexp1 (m3, A in m2)."""
        s = Path(settings)
        par = {}
        for line in (s / 'par.txt').read_text().splitlines():
            f = line.split()
            if len(f) >= 2:
                par[f[0]] = f[1]
        gl = pd.read_csv(s / 'GlacierData.txt', sep='\t').set_index('subid')
        corr = gl.logvolcor.reindex(self.subid).fillna(0.0).to_numpy(float)
        v = np.exp(corr) * float(par['glacvcoef1']) * self.area ** float(par['glacvexp1'])
        return np.where(self.area > 0, v, 0.0)

    def redistribute(self, v_start: np.ndarray, v_end: np.ndarray) -> np.ndarray:
        """New volumes: each basin's change v_end - v_start spread by delta-h over its units, total conserved."""
        out = v_end.astype(float).copy()
        for b in np.unique(self.basin[self.basin > 0]):
            m = np.flatnonzero(self.basin == b)
            if len(m) < 2:
                continue
            dv = v_end[m].sum() - v_start[m].sum()
            out[m] = _delta_h(v_start[m], dv, self.z[m], self.area[m])
        return out

    def summary(self) -> dict:
        b = self.basin[self.basin > 0]
        sizes = pd.Series(b).value_counts()
        return dict(glacier_subbasins=int(len(b)), basins=int(len(sizes)), multi_unit_basins=int((sizes > 1).sum()),
                    largest_basin_units=int(sizes.max()) if len(sizes) else 0)


def _delta_h(v0: np.ndarray, dv: float, z: np.ndarray, area: np.ndarray, iters: int = 50) -> np.ndarray:
    """Huss delta-h: v = v0 + fs * dh(hr) * A, units melting out are removed and the rest rescaled."""
    tot_area = area.sum() / 1e6
    a, b, c, g = HUSS['large' if tot_area > 20 else 'medium' if tot_area > 5 else 'small']
    v = v0.astype(float).copy()
    alive = (v0 > 0) | (dv > 0)
    remaining = dv
    for _ in range(iters):
        if abs(remaining) < 1e-6 or not alive.any():
            break
        zz = z[alive]
        span = zz.max() - zz.min()
        hr = (zz.max() - zz) / span if span > 0 else np.ones_like(zz)
        dh = (hr + a) ** g + b * (hr + a) + c
        dh = np.clip(dh, 0.0, None)
        w = dh * area[alive]
        if w.sum() <= 0:
            w = area[alive].copy()
        fs = remaining / w.sum()
        trial = v[alive] + fs * w
        neg = trial < 0
        remaining = float(trial[neg].sum())      # still to remove from units that melted out
        trial[neg] = 0.0
        v[alive] = trial
        idx = np.flatnonzero(alive)
        alive[idx[neg]] = False
    return v
