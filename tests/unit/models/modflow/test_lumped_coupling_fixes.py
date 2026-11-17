"""Lumped SUMMA-MODFLOW coupling: basin-zone HRU weighting, stress-period recharge, NSTP-aware extraction."""
from __future__ import annotations

import struct

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from symfluence.models.modflow.basin_model import write_recharge_timeseries
from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler, read_tdis
from symfluence.models.modflow.extractor import MODFLOWResultExtractor
from symfluence.models.modflow.land_adapters import get_land_adapter

# ---------------------------------------------------------------------------
# 1. SUMMA adapter: single basin zone takes the area-weighted mean of all HRUs
# ---------------------------------------------------------------------------


def _summa_case(d, hru_ids, areas, drainage, runoff):
    t = pd.date_range('2001-01-01', periods=3, freq='D')
    n = len(hru_ids)
    xr.Dataset({'scalarSoilDrainage': (('time', 'hru'), np.tile(drainage, (3, 1))),
                'scalarSurfaceRunoff': (('time', 'hru'), np.tile(runoff, (3, 1))),
                'hruId': (('hru',), list(hru_ids))},
               coords={'time': t}).to_netcdf(d / 'run_output_day.nc')
    settings = d / 'settings'
    settings.mkdir()
    xr.Dataset({'HRUarea': (('hru',), np.asarray(areas, float)), 'hruId': (('hru',), list(hru_ids))}
               ).to_netcdf(settings / 'attributes.nc')
    (settings / 'fileManager.txt').write_text(
        f"settingsPath    '{settings}/'\nattributeFile   'attributes.nc'\n")
    gpd.GeoDataFrame({'id': [1]}, geometry=[box(500000, 7100000, 500000 + 3000, 7101000)], crs=32627
                     ).to_file(d / 'basin.shp')
    assert n == len(areas)
    return settings, get_land_adapter('SUMMA', {'basin_shapefile': d / 'basin.shp'})


def test_summa_basin_zone_area_weights_all_hrus(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    settings, a = _summa_case(out, [1, 2, 3], [1.0e6, 1.0e6, 2.0e6],
                              np.array([1e-9, 1e-7, 1e-7]), np.array([0.0, 1e-8, 2e-8]))
    rec = a.read_recharge(out, settings)
    assert list(rec.columns) == [1]
    expected = (1e-9 + 1e-7 + 2 * 1e-7) / 4 * 86400.0
    np.testing.assert_allclose(rec[1].to_numpy(), expected, rtol=1e-12)
    write_recharge_timeseries(tmp_path, rec, [1])   # previously only HRU 1's column reached zone 1

    basin_area = float(a.recharge_zones(settings).area_m2.iloc[0])
    q = a.read_fast_flow(out, settings)
    np.testing.assert_allclose(q.to_numpy(), (0 + 1e-8 + 2 * 2e-8) / 4 * basin_area, rtol=1e-9)


def test_summa_basin_zone_single_hru_with_other_id(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    settings, a = _summa_case(out, [7], [5.0e6], np.array([1e-8]), np.array([1e-9]))
    rec = a.read_recharge(out, settings)
    assert list(rec.columns) == [1]
    np.testing.assert_allclose(rec[1].to_numpy(), 1e-8 * 86400.0)
    write_recharge_timeseries(tmp_path, rec, [1])   # previously "Recharge series lack zones [1]"


def test_summa_zone_shapefile_keeps_per_hru_columns(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    settings, _ = _summa_case(out, [1, 2], [1.0, 1.0], np.array([1e-9, 1e-7]), np.array([0.0, 0.0]))
    a = get_land_adapter('SUMMA', {'zone_shapefile': out / 'basin.shp', 'zone_id_field': 'id'})
    assert list(a.read_recharge(out, settings).columns) == [1, 2]


# ---------------------------------------------------------------------------
# 2. Lumped RCH: daily recharge averaged over stress periods
# ---------------------------------------------------------------------------

agg = SUMMAToMODFLOWCoupler.aggregate_to_stress_periods


def test_aggregate_daily_is_identity():
    v = np.array([0.1, 0.2, np.nan, 0.4])
    np.testing.assert_array_equal(agg(v, 1.0), [0.1, 0.2, 0.0, 0.4])
    np.testing.assert_array_equal(agg(v, 1.0, 3), [0.1, 0.2, 0.0])          # truncated to NPER
    np.testing.assert_array_equal(agg(v, 1.0, 6), [0.1, 0.2, 0.0, 0.4, 0.4, 0.4])


def test_aggregate_integer_and_fractional_periods():
    v = np.arange(1.0, 7.0)                       # 1..6
    np.testing.assert_allclose(agg(v, 2.0), [1.5, 3.5, 5.5])
    np.testing.assert_allclose(agg(v, 3.0), [2.0, 5.0])
    np.testing.assert_allclose(agg(v, 4.0), [2.5, 5.5])                      # partial last period
    np.testing.assert_allclose(agg(v, 2.0, 5), [1.5, 3.5, 5.5, 5.5, 5.5])  # periods past the data repeat
    np.testing.assert_allclose(agg(v, 1.5), [(1 + 0.5 * 2) / 1.5, (0.5 * 2 + 3) / 1.5,
                                             (4 + 0.5 * 5) / 1.5, (0.5 * 5 + 6) / 1.5])
    np.testing.assert_allclose(agg(v, 0.5), np.repeat(v, 2))


def _tdis(d, nper, perlen, nstp=1):
    (d / 'gwf.tdis').write_text('BEGIN OPTIONS\n  TIME_UNITS DAYS\nEND OPTIONS\n\nBEGIN DIMENSIONS\n'
                                f'  NPER {nper}\nEND DIMENSIONS\n\nBEGIN PERIODDATA\n'
                                + f'  {perlen} {nstp} 1.0\n' * nper + 'END PERIODDATA\n')


def test_runner_rch_blocks_match_tdis(tmp_path):
    from symfluence.models.modflow.runner import MODFLOWRunner
    _tdis(tmp_path, 3, 2.0)
    assert read_tdis(tmp_path) == (3, 2.0)
    runner = MODFLOWRunner.__new__(MODFLOWRunner)
    runner.settings_dir = tmp_path
    coupler = SUMMAToMODFLOWCoupler({})
    series = pd.Series(np.arange(1.0, 7.0), index=pd.date_range('2001-01-01', periods=6))
    runner.write_lumped_recharge(series, coupler)
    text = (tmp_path / 'gwf.rch').read_text()
    assert text.count('BEGIN PERIOD') == 3
    vals = [float(l.split()[1]) for l in text.splitlines() if l.strip().startswith('CONSTANT')]
    np.testing.assert_allclose(vals, [1.5, 3.5, 5.5])


# ---------------------------------------------------------------------------
# 3. Extractor: records dated from the file's period times, one per period
# ---------------------------------------------------------------------------

def _bud_hdr(f, kstp, kper, text, imeth, delt, pertim, totim):
    f.write(struct.pack('ii', kstp, kper))
    f.write(text.rjust(16).encode())
    f.write(struct.pack('iii', 1, 1, -1))
    f.write(struct.pack('<iddd', imeth, delt, pertim, totim))


def _write_bud(path, perlen, nstp, flows):
    """``flows[kper-1][kstp-1]`` = DRN flow (negative out) of each time step."""
    delt, totim = perlen / nstp, 0.0
    with open(path, 'wb') as f:
        for kper, per in enumerate(flows, 1):
            for kstp, q in enumerate(per, 1):
                totim += delt
                _bud_hdr(f, kstp, kper, 'FLOW-JA-FACE', 1, delt, delt * kstp, totim)
                f.write(struct.pack('d', 0.0))
                _bud_hdr(f, kstp, kper, 'DRN', 6, delt, delt * kstp, totim)
                for s in ('GWF', 'GWF', 'GWF', 'DRN'):
                    f.write(s.ljust(16).encode())
                f.write(struct.pack('ii', 1, 1))
                f.write(struct.pack('ii', 1, 1))
                f.write(struct.pack('d', q))


def _write_hds(path, perlen, nstp, heads):
    delt, totim = perlen / nstp, 0.0
    with open(path, 'wb') as f:
        for kper, per in enumerate(heads, 1):
            for kstp, h in enumerate(per, 1):
                totim += delt
                f.write(struct.pack('iidd', kstp, kper, delt * kstp, totim))
                f.write('HEAD'.rjust(16).encode())
                f.write(struct.pack('iii', 1, 1, 1))
                f.write(struct.pack('d', h))


def test_extractor_default_one_record_per_day(tmp_path):
    _write_bud(tmp_path / 'gwf.bud', 1.0, 1, [[-100.0], [-200.0], [-300.0]])
    s = MODFLOWResultExtractor().extract_variable(tmp_path / 'gwf.bud', 'drain_discharge',
                                                  stress_period_length=1.0, start_date='2000-01-01')
    pd.testing.assert_index_equal(s.index, pd.DatetimeIndex(pd.date_range('2000-01-01', periods=3, freq='D'),
                                                            freq=None))
    np.testing.assert_array_equal(s.to_numpy(), [100.0, 200.0, 300.0])


def test_extractor_nstp_gives_one_value_per_period(tmp_path):
    _write_bud(tmp_path / 'gwf.bud', 1.0, 2, [[-100.0, -300.0], [-200.0, -400.0]])
    _write_hds(tmp_path / 'gwf.hds', 1.0, 2, [[10.0, 11.0], [12.0, 13.0]])
    ex = MODFLOWResultExtractor()
    s = ex.extract_variable(tmp_path / 'gwf.bud', 'drain_discharge', stress_period_length=1.0,
                            start_date='2000-01-01')
    assert list(s.index) == list(pd.date_range('2000-01-01', periods=2, freq='D'))
    np.testing.assert_allclose(s.to_numpy(), [200.0, 300.0])     # time-weighted period means
    h = ex.extract_variable(tmp_path / 'gwf.hds', 'head', start_date='2000-01-01')
    assert list(h.index) == list(pd.date_range('2000-01-01', periods=2, freq='D'))
    np.testing.assert_array_equal(h.to_numpy(), [11.0, 13.0])    # end-of-period head


@pytest.mark.parametrize('perlen', [0.5, 5.0])
def test_extractor_non_daily_periods(tmp_path, perlen):
    _write_bud(tmp_path / 'gwf.bud', perlen, 1, [[-1.0], [-2.0], [-3.0]])
    s = MODFLOWResultExtractor().extract_variable(tmp_path / 'gwf.bud', 'drain_discharge',
                                                  stress_period_length=perlen, start_date='2000-01-01')
    expected = pd.Timestamp('2000-01-01') + pd.to_timedelta(np.arange(3) * perlen, unit='D')
    assert list(s.index) == list(expected)
