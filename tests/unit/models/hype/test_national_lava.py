"""Lava soil type, soil-column parameters and the MODFLOW surrogate of the national calibration (ADR 0014)."""
import numpy as np
import pandas as pd
import pytest

from symfluence.models.hype import national_calibration as nc
from symfluence.models.hype import national_lava as nl

GEOCLASS = """!! SLC\tLULC\tSOIL\tc1\tc2\trot\tveg\tspecial\ttile\tstream\tn\td1\td2\td3
1\t15\t1\t0\t0\t0\t1\t3\t0\t2.3\t3\t0.1\t0.5\t2.3
2\t16\t1\t0\t0\t0\t1\t0\t0\t2.3\t3\t0.1\t0.5\t2.3
3\t16\t2\t0\t0\t0\t1\t0\t0\t2.3\t3\t0.1\t0.5\t2.3
4\t7\t1\t0\t0\t0\t1\t0\t0\t2.3\t3\t0.1\t0.5\t2.3
"""


@pytest.fixture
def settings(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'GeoClass.txt').write_text(GEOCLASS)
    pd.DataFrame({'subid': [1, 2], 'maindown': [2, 0], 'area': [1e6, 2e6],
                  'slc_1': [0.2, 0.0], 'slc_2': [0.4, 0.5], 'slc_3': [0.2, 0.25], 'slc_4': [0.2, 0.25],
                  'dhslc_1': [10.0, 0.0], 'dhslc_2': [5.0, -5.0], 'dhslc_3': [15.0, 5.0], 'dhslc_4': [0.0, 0.0]}
                 ).to_csv(src / 'GeoData.txt', sep='\t', index=False)
    (src / 'par.txt').write_text('mperc1\t10\t10\t10\t!! perc\nrrcs1\t0.3\t0.3\t0.3\ncmlt\t1\t2\t3\t4\t5\n')
    dst = tmp_path / 'dst'
    dst.mkdir()
    return src, dst


def test_add_lava_soil_splits_ordinary_classes(settings):
    src, dst = settings
    r = nl.add_lava_soil(src, dst, pd.Series({1: 0.5, 2: 0.0}))
    assert r['twins'] == {4: 5, 2: 6, 3: 6}               # land use 7 -> 5, land use 16 (two soils) -> 6
    gd = pd.read_csv(dst / 'GeoData.txt', sep='\t').set_index('subid')
    assert gd.loc[1, 'slc_1'] == 0.2                       # glacier untouched
    assert gd.loc[1, ['slc_2', 'slc_3', 'slc_6']].tolist() == pytest.approx([0.2, 0.1, 0.3])
    assert gd.loc[1, 'dhslc_6'] == pytest.approx((0.2 * 5 + 0.1 * 15) / 0.3)
    assert gd.loc[2, 'slc_6'] == 0.0
    assert gd.filter(like='slc_').drop(columns=gd.filter(like='dhslc_').columns).sum(axis=1).tolist() == \
        pytest.approx([1.0, 1.0])
    gc = [ln.split('\t') for ln in (dst / 'GeoClass.txt').read_text().splitlines() if not ln.startswith('!')]
    assert [(r[0], r[1], r[2]) for r in gc[-2:]] == [('5', '7', '4'), ('6', '16', '4')]
    par = (dst / 'par.txt').read_text().splitlines()
    assert par[0] == 'mperc1\t10\t10\t10\t10\t!! perc'      # soil row gets a 4th column copied from soil 1
    assert par[2] == 'cmlt\t1\t2\t3\t4\t5'                  # land-use row untouched


def test_param_col_composes_after_whole_row(tmp_path):
    lines = ['mperc1\t10\t10\t10\t10']
    params = [nc.Param('mperc1', 0.2, 3.0), nc.Param('mperc1', 1.0, 20.0, col=3)]
    nc.write_par(lines, params, np.array([2.0, 5.0]), tmp_path / 'par.txt')
    vals = [float(v) for v in (tmp_path / 'par.txt').read_text().split('\t')[1:]]
    assert vals == [20.0, 20.0, 20.0, 100.0]


def test_gw_return_routes_upstream_difference(tmp_path):
    days = pd.date_range('2000-01-01', periods=400, freq='D')
    ref = pd.DataFrame(86400.0, index=days, columns=[1, 2, 3])        # 1 m3/s each
    np.savez(tmp_path / 'ref.npz', dates=days.values.astype('datetime64[D]'), subids=np.array([1, 2, 3]),
             values=ref.to_numpy(np.float32))
    aqin = ref.copy()
    aqin[1] += 86400.0                                                # +1 m3/s recharge in unit 1
    geo = pd.DataFrame({'subid': [1, 2, 3], 'maindown': [2, 0, 0]})
    r = nc.gw_return(aqin, tmp_path / 'ref.npz', geo, [2, 3], tau_days=30.0)
    assert r[3].abs().max() == 0.0                                    # not downstream of unit 1
    assert r[2].iloc[29] == pytest.approx(1 - np.exp(-1), rel=0.02)   # reservoir: 63 % after tau
    assert r[2].iloc[-1] == pytest.approx(1.0, rel=1e-3)
