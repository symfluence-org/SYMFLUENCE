# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Ice flow between HYPE subbasins: ice basins, delta-h redistribution and state-file editing."""
import numpy as np
import pandas as pd

from symfluence.models.hype.national_glacier import IceFlux, ice_basins, read_glacvol, write_glacvol
from symfluence.models.modflow.national_coupling import _split_annual


def test_ice_basins_follow_maindown_while_glacier():
    # 1 -> 2 -> 3 -> 4 (no glacier) -> 0; 5 -> 0 alone
    gd = pd.DataFrame({'subid': [1, 2, 3, 4, 5], 'maindown': [2, 3, 4, 0, 0]})
    isg = pd.Series([True, True, True, False, True], index=[1, 2, 3, 4, 5])
    b = ice_basins(gd, isg)
    assert b.to_dict() == {1: 3, 2: 3, 3: 3, 5: 5}


def test_delta_h_conserves_and_moves_ice_down():
    f = IceFlux(subid=np.array([1, 2, 3]), basin=np.array([3, 3, 3]), z=np.array([1800., 1200., 600.]),
                area=np.array([30e6, 30e6, 30e6]))
    v0 = np.array([5e9, 5e9, 5e9])
    v1 = np.array([6e9, 5e9, 3e9])            # top gains, terminus loses: net -1e9
    v = f.redistribute(v0, v1)
    assert np.isclose(v.sum(), v1.sum())
    assert v[0] < v1[0] and v[2] > v1[2]      # the surplus moved down
    assert (v >= 0).all()


def test_delta_h_melts_out_terminus_and_keeps_total():
    f = IceFlux(subid=np.array([1, 2]), basin=np.array([2, 2]), z=np.array([1500., 500.]),
                area=np.array([20e6, 20e6]))
    v = f.redistribute(np.array([4e9, 0.5e9]), np.array([2e9, 0.5e9]))      # basin loses 2e9
    assert np.isclose(v.sum(), 2.5e9) and (v >= 0).all()
    assert v[1] == 0.0                         # terminus melted out, the rest came from above


def test_state_glacvol_roundtrip(tmp_path):
    n, nc = 3, 2
    head = ' '.join(['0'] * 10 + [str(n), str(nc)] + ['0'] * 5)
    vals = [f'{x:15.8E}' for x in np.arange(5 * n * nc + n + 7, dtype=float)]
    lines = [head] + ['\t'.join(vals[i:i + 100]) + '\t' for i in range(0, len(vals), 100)] + ['tail']
    p = tmp_path / 'state_save20000101.txt'
    p.write_text('\n'.join(lines))
    assert np.allclose(read_glacvol(p), np.arange(5 * n * nc, 5 * n * nc + n))
    write_glacvol(p, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(read_glacvol(p), [1.5, 2.5, 3.5])
    assert p.read_text().split('\n')[-1] == 'tail'


def test_split_annual_cuts_on_october_first():
    eras = [(pd.Timestamp('1991-01-01'), pd.Timestamp('1993-12-31'), None, ['A'])]
    seg = _split_annual(eras, 10, 1)
    assert [(a.date().isoformat(), b.date().isoformat()) for a, b, _, _ in seg] == [
        ('1991-01-01', '1991-09-30'), ('1991-10-01', '1992-09-30'), ('1992-10-01', '1993-09-30'),
        ('1993-10-01', '1993-12-31')]
