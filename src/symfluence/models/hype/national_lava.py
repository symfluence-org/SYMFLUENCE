"""A Holocene-lava soil type for the national HYPE model (ADR 0014).

HYPE's soil types come from pedology, which does not separate young lava: Ódáðahraun, Hallmundarhraun and the
Hekla/Veiðivötn lavas take most rain and melt straight down to the aquifer, while the model ran it off as fast flow
from the soil layers. Diagnosed on Svartá, Laxá, Hvítá at Kljáfoss and Ytri-Rangá: the fast flow alone varied 2–6 times
more than the whole observed series, a ceiling no groundwater parameter can lift.

`add_lava_soil` splits every ordinary land class (special class 0) of a subbasin by its Holocene-lava fraction into
the original class and a lava twin with soil type `soil`, same land use and layers. Soil-dependent par.txt rows get a
column for the new soil type, copied from soil type 1, so the split model reproduces the unsplit one until the lava
parameters move (LAVA_PARAMS act on that column only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# par.txt rows indexed by soil type
SOIL_ROWS = ('bcosby', 'bfroznsoil', 'logsatmp', 'mperc1', 'mperc2', 'rrcs1', 'rrcs2', 'sfrost', 'wcep', 'wcfc',
             'wcwp')


def _geoclass(path: Path) -> tuple[list, pd.DataFrame]:
    lines = Path(path).read_text().splitlines()
    head = [ln for ln in lines if ln.startswith('!')]
    rows = [ln.split('\t') for ln in lines if ln.strip() and not ln.startswith('!')]
    return head, pd.DataFrame(rows)


def add_lava_soil(src: Path, dst: Path, lava_frac: pd.Series, soil: int = 4, min_frac: float = 0.005) -> dict:
    """Write GeoClass.txt, GeoData.txt and par.txt with lava twins from `src` into `dst` (other files untouched).
    `lava_frac`: Holocene-lava fraction per subid (0..1). Returns {land class: lava class} and the lava area."""
    src, dst = Path(src), Path(dst)
    head, gc = _geoclass(src / 'GeoClass.txt')
    slc, lu, st, special = (gc[i].astype(int) for i in (0, 1, 2, 7))
    if (st == soil).any():
        raise ValueError(f'soil type {soil} already in {src / "GeoClass.txt"}')
    ordinary = gc[special == 0]
    new_rows, twin = [], {}
    nxt = int(slc.max()) + 1
    for land in sorted(ordinary[1].astype(int).unique()):
        r = ordinary[ordinary[1].astype(int) == land].iloc[0].copy()
        r[0], r[2] = str(nxt), str(soil)
        new_rows.append(r)
        for k in ordinary[ordinary[1].astype(int) == land][0].astype(int):
            twin[int(k)] = nxt
        nxt += 1
    gc2 = pd.concat([gc, pd.DataFrame(new_rows)], ignore_index=True)
    (dst / 'GeoClass.txt').write_text('\n'.join(head + ['\t'.join(map(str, r)) for r in gc2.to_numpy()]) + '\n')

    gd = pd.read_csv(src / 'GeoData.txt', sep='\t')
    f = lava_frac.reindex(gd.subid.astype(int)).fillna(0.0).clip(0, 1).to_numpy()
    f = np.where(f < min_frac, 0.0, f)
    cols = list(gd.columns)
    lava_area = 0.0
    for t in sorted(set(twin.values())):
        srcs = [k for k, v in twin.items() if v == t]
        part = np.zeros(len(gd))
        dh = np.zeros(len(gd))
        for k in srcs:
            moved = gd[f'slc_{k}'].to_numpy(float) * f
            part += moved
            if f'dhslc_{k}' in gd:
                dh += moved * gd[f'dhslc_{k}'].to_numpy(float)
            gd[f'slc_{k}'] = gd[f'slc_{k}'].to_numpy(float) - moved
        gd[f'slc_{t}'] = part
        if any(f'dhslc_{k}' in gd for k in srcs):
            gd[f'dhslc_{t}'] = np.where(part > 0, dh / np.where(part > 0, part, 1), 0.0)
        lava_area += float((part * gd.area).sum())
    # keep slc_ then dhslc_ blocks in class order, other columns first as they were
    base = [c for c in cols if not c.startswith(('slc_', 'dhslc_'))]
    n = int(gc2[0].astype(int).max())
    order = base + [f'slc_{i}' for i in range(1, n + 1) if f'slc_{i}' in gd] + \
        [f'dhslc_{i}' for i in range(1, n + 1) if f'dhslc_{i}' in gd]
    s = gd[[c for c in order if c.startswith('slc_')]].sum(axis=1)
    if not np.allclose(s, pd.read_csv(src / 'GeoData.txt', sep='\t')[[c for c in cols if c.startswith('slc_')]].sum(axis=1), atol=1e-9):
        raise AssertionError('class fractions not conserved')
    gd[order].to_csv(dst / 'GeoData.txt', sep='\t', index=False)

    rows = []
    for ln in (src / 'par.txt').read_text().splitlines():
        tok = ln.split('\t')
        if tok and tok[0].split() and tok[0].split()[0] in SOIL_ROWS:
            vals = [t for t in tok[1:] if _isnum(t)]
            rest = [t for t in tok[1:] if not _isnum(t)]
            while len(vals) < soil - 1:
                vals.append(vals[-1])
            ln = '\t'.join([tok[0]] + vals[:soil - 1] + [vals[0]] + rest)
        rows.append(ln)
    (dst / 'par.txt').write_text('\n'.join(rows) + '\n')
    return {'twins': twin, 'lava_km2': lava_area / 1e6}


def _isnum(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False
