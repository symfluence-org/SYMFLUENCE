# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""National fabric -> HYPE inputs: routing order with branches and SLC construction."""
import numpy as np
import pandas as pd
import pytest

from symfluence.core.exceptions import ValidationError
from symfluence.models.hype.national_fabric import (
    SPECIAL_GLACIER, SPECIAL_ILAKE, SPECIAL_OLAKE, branch_eras, build_slc, topological_order)


def test_order_puts_sources_before_targets_including_branches():
    # 1 -> 3, 2 -> 3, 3 -> 0 (sea); branch 4 -> 2 (another tree)
    ids = np.array([3, 2, 1, 4])
    edges = [(1, 3), (2, 3), (4, 2)]
    order = topological_order(ids, edges)
    pos = {u: i for i, u in enumerate(order)}
    assert all(pos[a] < pos[b] for a, b in edges)
    assert sorted(order) == [1, 2, 3, 4]


def test_order_rejects_cycles():
    with pytest.raises(ValidationError):
        topological_order(np.array([1, 2]), [(1, 2), (2, 1)])


def test_slc_classes_and_fractions():
    cls = pd.DataFrame({
        'id':        [1,   1,   1,   1,   2,   2],
        'land':      [1,   2,   5,   9,   5,   2],     # glacier, lake, heath, open water | heath, lake
        'soil':      [1,   1,   2,   1,   2,   3],
        'band':      [5,   2,   2,   2,   1,   1],
        'area_km2':  [2.0, 1.0, 6.0, 1.0, 3.0, 1.0],
        'elev_mean': [1000., 400., 450., 410., 250., 240.],
    })
    slc, frac = build_slc(cls, outlet_lake_units={1})
    kinds = dict(zip(slc.kind, slc.special))
    assert kinds == {'glacier': SPECIAL_GLACIER, 'olake': SPECIAL_OLAKE, 'ilake': SPECIAL_ILAKE, 'land': 0}
    f = frac.pivot(index='id', columns='slc', values='frac').fillna(0)
    np.testing.assert_allclose(f.sum(axis=1), 1.0)
    slc_of = slc.set_index('kind').slc
    # unit 1: its lake is the outlet lake; other open water is an internal lake
    assert f.loc[1, slc_of['olake']] == pytest.approx(0.1)
    assert f.loc[1, slc_of['ilake']] == pytest.approx(0.1)
    # unit 2 has no outlet lake: its lake becomes an internal lake
    assert f.loc[2, slc_of['ilake']] == pytest.approx(0.25)
    assert f.loc[2].get(slc_of['olake'], 0) == 0


def test_branch_eras_follow_commissioning_years():
    tr = pd.DataFrame({
        'transfer_id':    ['B-9', 'B-10', 'K-1', 'S-1'],
        'type':           ['headrace', 'headrace', 'headrace', 'natural_bifurcation'],
        'from_unit':      [10, 10, 20, 30],
        'to_unit':        [11, 12, 21, 31],
        'fraction':       [np.nan, np.nan, np.nan, np.nan],
        'capacity_m3s':   [100.0, 50.0, np.nan, np.nan],
        'q_prior_m3s':    [80.0, 80.0, 40.0, 5.0],
        'commissioned':   [1969, 2018, 2007, np.nan],      # natural: always on
        'decommissioned': [np.nan, np.nan, np.nan, np.nan],
    })
    eras = branch_eras(tr, '1991-01-01', '2025-12-31')
    assert [(a.year, b.date().isoformat()) for a, b, _, _ in eras] == [
        (1991, '2006-12-31'), (2007, '2017-12-31'), (2018, '2025-12-31')]
    assert eras[0][3] == ['B-9', 'S-1']
    assert eras[1][3] == ['B-9', 'K-1', 'S-1']
    k1 = eras[1][2].set_index('sourceid').loc[20]
    assert k1.maxqbranch == pytest.approx(60.0)             # default 1.5 x prior mean
    # B-10 joins B-9's source in 2018: HYPE allows one branch per source, the lower number is kept
    assert eras[2][2].set_index('sourceid').loc[10].branchid == 11
    assert eras[0][2].set_index('sourceid').loc[30].mainpart == pytest.approx(0.5)
