"""Real-basin MODFLOW 6 model and land-model recharge adapters on a synthetic basin."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from symfluence.core.exceptions import ConfigurationError, ValidationError
from symfluence.models.modflow.basin_model import (
    BUDGET_CSV,
    GRID_FILE,
    MF6BasinGrid,
    MF6BasinParams,
    apply_basin_parameters,
    read_budget,
    read_groundwater_discharge,
    set_simulation_length,
    write_coupling_record,
    write_recharge_timeseries,
    write_simulation,
    zone_cell_weights,
)
from symfluence.models.modflow.land_adapters import (
    adapter_options,
    available_land_adapters,
    get_land_adapter,
    resolve_land_model,
)

CRS = 32627   # UTM 27N (Iceland), metres
X0, Y0 = 500000.0, 7100000.0


@pytest.fixture
def basin(tmp_path):
    """10 x 6 km basin, two subbasins draining west; DEM rises 100 m/km eastward."""
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    res = 100.0
    cols = np.arange(100)
    dem = np.tile(500.0 + 100.0 * (cols + 0.5) * res / 1000.0, (60, 1)).astype('float32')
    path = tmp_path / 'dem_elv.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=60, width=100, count=1, dtype='float32',
                       crs=f'EPSG:{CRS}', transform=from_origin(X0, Y0, res, res), nodata=-9999) as dst:
        dst.write(dem, 1)
    sub = gpd.GeoDataFrame({'GRU_ID': [1, 2], 'gru_to_seg': [10, 20]},
                           geometry=[box(X0, Y0 - 6000, X0 + 5000, Y0), box(X0 + 5000, Y0 - 6000, X0 + 10000, Y0)],
                           crs=CRS)
    net = gpd.GeoDataFrame({'LINKNO': [10, 20], 'DSLINKNO': [-1, 10], 'Slope': [0.02, 0.02]},
                           geometry=[LineString([(X0 + 5000, Y0 - 3000), (X0 + 50, Y0 - 3000)]),
                                     LineString([(X0 + 9950, Y0 - 3000), (X0 + 5000, Y0 - 3000)])], crs=CRS)
    return sub, net, path


@pytest.fixture
def grid(basin):
    from symfluence.models.gsflow.basin_geometry import build_basin_geometry
    from symfluence.models.modflow.basin_model import grid_from_basin_geometry
    sub, net, dem = basin
    return grid_from_basin_geometry(build_basin_geometry(sub, net, dem, CRS, cell_size=2000.0))


def _zones():
    """Three land-model zones (elevation-band-like strips) that do not follow the grid."""
    return gpd.GeoDataFrame({'zone_id': [11, 12, 13]},
                            geometry=[box(X0, Y0 - 6000, X0 + 3300, Y0), box(X0 + 3300, Y0 - 6000, X0 + 7100, Y0),
                                      box(X0 + 7100, Y0 - 6000, X0 + 10000, Y0)], crs=CRS)


def _mf6():
    exe = shutil.which('mf6')
    data = os.environ.get('SYMFLUENCE_DATA_DIR')
    if not exe and data and Path(data, 'installs/modflow/bin/mf6').exists():
        exe = str(Path(data, 'installs/modflow/bin/mf6'))
    return exe


# ---------------------------------------------------------------- grid & weights
def test_grid_roundtrip(grid, tmp_path):
    grid.to_json(tmp_path / GRID_FILE)
    back = MF6BasinGrid.from_json(tmp_path / GRID_FILE)
    assert (back.active == grid.active).all()
    np.testing.assert_allclose(back.top[grid.active], grid.top[grid.active], atol=1e-3)
    np.testing.assert_allclose(back.basin_frac, grid.basin_frac, atol=1e-6)
    assert back.river.equals(grid.river) or np.allclose(back.river.to_numpy(float), grid.river.to_numpy(float))
    assert grid.basin_frac[grid.active].sum() * grid.cell_area == pytest.approx(60e6, rel=1e-6)


def test_weights_conserve_each_zone_volume(grid):
    zones = _zones()
    area = {11: 20e6, 12: 22.8e6, 13: 17.4e6}   # land-model's own (slightly different) areas
    w = zone_cell_weights(grid, zones, area)
    vol = (w.mult * grid.cell_area).groupby(w.zone_id).sum()
    for z, a in area.items():
        assert vol[z] == pytest.approx(a, rel=1e-9)
    assert set(w.zone_id) == {11, 12, 13}


def test_weights_reject_zone_outside_grid(grid):
    zones = pd.concat([_zones(), gpd.GeoDataFrame({'zone_id': [99]}, geometry=[box(0, 0, 10, 10)], crs=CRS)])
    with pytest.raises(ValidationError, match='99'):
        zone_cell_weights(grid, gpd.GeoDataFrame(zones, crs=CRS))


# ---------------------------------------------------------------- input files
def test_simulation_files(grid, tmp_path):
    w = zone_cell_weights(grid, _zones())
    write_simulation(tmp_path, grid, w, 10, MF6BasinParams(thickness=50.0))
    assert 'NPER 11' in (tmp_path / 'gwf.tdis').read_text()
    rch = (tmp_path / 'gwf.rch').read_text()
    assert f'MAXBOUND {len(w)}' in rch and 'AUXMULTNAME mult' in rch
    assert 'STEADY-STATE' in (tmp_path / 'gwf.sto').read_text()
    # aquifer bottom lies below every streambed in its cell
    bot = np.array([[float(v) for v in line.split()] for line in
                    (tmp_path / 'gwf.dis').read_text().split('BOTM\n    INTERNAL\n')[1].split('  IDOMAIN')[0].strip().splitlines()])
    for r in grid.river.itertuples():
        assert bot[int(r.row) - 1, int(r.col) - 1] <= r.strtop - 50.0 + 1e-3
    set_simulation_length(tmp_path, 3)
    assert 'NPER 4' in (tmp_path / 'gwf.tdis').read_text()


def test_recharge_timeseries(tmp_path):
    days = pd.date_range('2010-01-01', periods=4, freq='D')
    rec = pd.DataFrame({11: [1e-3, 2e-3, 0, 4e-3], 12: [0, 0, 1e-3, 1e-3]}, index=days)
    dates = write_recharge_timeseries(tmp_path, rec, [11, 12])
    assert dates[0] == days[0]
    body = (tmp_path / 'gwf.rch.ts').read_text().split('BEGIN TIMESERIES\n')[1].splitlines()[:-1]
    assert len(body) == 4 + 2                                   # mean, 4 days, end point
    assert [float(v) for v in body[0].split()[1:]] == pytest.approx([1.75e-3, 0.5e-3])
    assert float(body[-1].split()[0]) == 5.0
    with pytest.raises(ValidationError):
        write_recharge_timeseries(tmp_path, rec.drop(index=days[2]), [11, 12])
    with pytest.raises(ValidationError):
        write_recharge_timeseries(tmp_path, rec, [11, 13])


def test_params_update_and_validation(grid, tmp_path):
    p = MF6BasinParams().updated({'K': 2.0, 'river_leakance': 3.0, 'unrelated': 9})
    assert p.k == 2.0 and p.river_leakance == 3.0
    with pytest.raises(ConfigurationError):
        MF6BasinParams().updated({'SY': 0.0})
    assert not apply_basin_parameters(tmp_path, {'K': 1.0}, MF6BasinParams())
    write_simulation(tmp_path, grid, zone_cell_weights(grid, _zones()), 2, MF6BasinParams())
    assert apply_basin_parameters(tmp_path, {'K': 7.5}, MF6BasinParams())
    assert 'CONSTANT 7.5' in (tmp_path / 'gwf.npf').read_text()


def test_read_budget(tmp_path):
    cols = ['time', 'STO-SS(STORAGE)_IN', 'STO-SY(STORAGE)_IN', 'DRN(RIV)_IN', 'DRN(SEEP)_IN', 'RCH(RCH)_IN',
            'STO-SS(STORAGE)_OUT', 'STO-SY(STORAGE)_OUT', 'DRN(RIV)_OUT', 'DRN(SEEP)_OUT', 'RCH(RCH)_OUT',
            'TOTAL_IN', 'TOTAL_OUT', 'PERCENT_DIFFERENCE']
    rows = [[1, 0, 0, 0, 0, 100, 0, 0, 90, 10, 0, 100, 100, 0],
            [2, 0, 5, 0, 0, 80, 0, 0, 75, 10, 0, 85, 85, 0],
            [3, 0, 0, 0, 0, 120, 0, 10, 100, 10, 0, 120, 120, 0]]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / BUDGET_CSV, index=False)
    b = read_budget(tmp_path, '2010-01-01')
    assert list(b.index) == list(pd.date_range('2010-01-01', periods=2))
    assert list(b.discharge) == [85, 110] and list(b.storage_change) == [-5, 10]
    write_coupling_record(tmp_path, 'HYPE', tmp_path, tmp_path, '2010-01-01')
    assert read_groundwater_discharge(tmp_path).iloc[0] == pytest.approx(85 / 86400)


# ---------------------------------------------------------------- adapters
def _hype_settings(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'info.txt').write_text('modeloption deepground\t0\ntimeoutput variable COUT\tEVAP\nresultdir\t/old/\n')
    pd.DataFrame({'subid': [13, 12, 11], 'maindown': [12, 11, 0], 'grwdown': [12, 11, 0],
                  'area': [17.4e6, 22.8e6, 20e6]}).to_csv(d / 'GeoData.txt', sep='\t', index=False)
    (d / 'par.txt').write_text('rcgrw\t0.1\n')
    return d


def _hype_output(d: Path, var: str, data: dict):
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data, index=pd.date_range('2010-01-01', periods=3, freq='D'))
    df.index.name = 'DATE'
    with open(d / f'time{var}.txt', 'w') as f:
        f.write(f'!! variable={var.lower()}\n')
        df.to_csv(f, sep='\t', date_format='%Y-%m-%d')


def test_registry_and_options():
    assert {'HYPE', 'SUMMA'} <= set(available_land_adapters())
    with pytest.raises(ConfigurationError, match='available'):
        get_land_adapter('NOPE')
    opts = adapter_options({'MODFLOW_LAND_ZONE_SHAPEFILE': 'z.shp', 'MODFLOW_LAND_ZONE_ID_FIELD': 'subid'})
    assert opts['zone_shapefile'] == 'z.shp' and opts['zone_id_field'] == 'subid'
    assert resolve_land_model({'MODFLOW_COUPLING_SOURCE': 'hype'}) == 'HYPE'
    assert resolve_land_model({'HYDROLOGICAL_MODEL': 'HYPE', 'MODFLOW_COUPLING_SOURCE': 'SUMMA'}) == 'HYPE'
    assert resolve_land_model({'LAND_SURFACE_MODEL': 'summa', 'HYDROLOGICAL_MODEL': 'HYPE'}) == 'SUMMA'
    assert resolve_land_model({'HYDROLOGICAL_MODEL': 'MODFLOW'}) == 'SUMMA'


def test_hype_adapter(tmp_path):
    from symfluence.models.hype.geodata_manager import GROUNDWATER_OUT_OF_DOMAIN_SUBID as OUT
    s = _hype_settings(tmp_path / 'settings')
    out = tmp_path / 'out'
    _hype_output(out, 'AQIN', {'13': [0, 17400, 0], '12': [22800, 0, 0], '11': [20000, 20000, 0]})
    _hype_output(out, 'COUT', {'13': [1, 1, 1], '12': [2, 2, 2], '11': [3, 4, 5]})
    ad = get_land_adapter('HYPE', {})
    with pytest.raises(ConfigurationError, match='not enabled'):
        ad.read_recharge(out, s)
    ad.enable_recharge_export(s)
    info = (s / 'info.txt').read_text()
    assert 'modeloption deepground\t1' in info and 'AQIN' in info
    assert set(pd.read_csv(s / 'GeoData.txt', sep='\t').grwdown) == {OUT}
    rec = ad.read_recharge(out, s)
    assert rec.loc['2010-01-02', 13] == pytest.approx(1e-3) and rec.loc['2010-01-01', 11] == pytest.approx(1e-3)
    assert list(ad.read_fast_flow(out, s)) == [3, 4, 5]
    ad.set_output_dir(s, tmp_path / 'new')
    assert f"resultdir\t{tmp_path / 'new'}/" in (s / 'info.txt').read_text()


def test_hype_zones_must_match_subids(tmp_path):
    s = _hype_settings(tmp_path / 'settings')
    shp = tmp_path / 'zones.shp'
    _zones().rename(columns={'zone_id': 'HRU_ID'}).to_file(shp)
    z = get_land_adapter('HYPE', {'zone_shapefile': str(shp)}).recharge_zones(s)
    assert dict(zip(z.zone_id, z.area_m2)) == {11: 20e6, 12: 22.8e6, 13: 17.4e6}
    _zones().iloc[:2].rename(columns={'zone_id': 'HRU_ID'}).to_file(shp)
    with pytest.raises(ValidationError):
        get_land_adapter('HYPE', {'zone_shapefile': str(shp)}).recharge_zones(s)


def test_hype_groundwater_export_switch():
    from symfluence.models.hype.geodata_manager import groundwater_export_enabled
    assert groundwater_export_enabled({'GROUNDWATER_MODEL': 'MODFLOW'})
    assert not groundwater_export_enabled({'GROUNDWATER_MODEL': 'MODFLOW', 'HYPE_GROUNDWATER_EXPORT': False})
    assert groundwater_export_enabled({'HYPE_GROUNDWATER_EXPORT': True})
    assert not groundwater_export_enabled({})


# ---------------------------------------------------------------- MODFLOW run
@pytest.mark.skipif(_mf6() is None, reason='mf6 not installed')
def test_mf6_run_conserves_recharge(grid, tmp_path):
    from symfluence.models.modflow.basin_model import check_run
    zones = _zones()
    area = dict(zip(zones.zone_id, zones.to_crs(CRS).area))
    w = zone_cell_weights(grid, zones, area)
    days = pd.date_range('2010-01-01', periods=120, freq='D')
    rng = np.random.default_rng(0)
    rec = pd.DataFrame({z: rng.uniform(0, 4e-3, len(days)) for z in area}, index=days)
    write_simulation(tmp_path, grid, w, len(days), MF6BasinParams(k=10.0, sy=0.05), save_heads='none')
    write_recharge_timeseries(tmp_path, rec, sorted(area))
    r = subprocess.run([_mf6()], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    check_run(tmp_path)
    b = read_budget(tmp_path, days[0])
    expected = (rec * pd.Series(area)).sum(axis=1)
    np.testing.assert_allclose(b.recharge.to_numpy(), expected.to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.recharge - b.discharge - b.storage_change, 0.0, atol=1e-3 * expected.mean())


# ---------------------------------------------------------------- lake, anisotropy, reduced recharge
def _lake():
    """A 3 x 2 km lake at the downstream (west) end, on the river."""
    return gpd.GeoDataFrame(geometry=[box(X0, Y0 - 4000, X0 + 3000, Y0 - 2000)], crs=CRS)


def test_lake_ghb_replaces_drains_under_the_lake(grid, tmp_path):
    grid.set_lake(_lake(), 520.0)
    assert grid.has_lake and grid.lake_frac.sum() * grid.cell_area == pytest.approx(6e6, rel=1e-6)
    grid.to_json(tmp_path / 'g.json')
    back = MF6BasinGrid.from_json(tmp_path / 'g.json')
    np.testing.assert_allclose(back.lake_frac, grid.lake_frac)
    assert back.lake_stage == 520.0
    write_simulation(tmp_path, grid, zone_cell_weights(grid, _zones()), 5, MF6BasinParams(lake_leakance=2.0))
    assert 'GHB6 gwf.lake.ghb LAKE' in (tmp_path / 'gwf.nam').read_text()
    ghb = [line.split() for line in (tmp_path / 'gwf.lake.ghb').read_text().split('BEGIN PERIOD 1\n')[1].splitlines()
           if line.strip() and not line.startswith('END')]
    cond = sum(float(c[4]) for c in ghb)
    assert cond == pytest.approx(2.0 * 6e6, rel=1e-6) and all(float(c[3]) == 520.0 for c in ghb)
    # no river reach in a mostly-lake cell; seepage only on the land part
    riv = (tmp_path / 'gwf.riv.drn').read_text()
    for r, c in zip(*np.nonzero(grid.lake_frac >= 0.5)):
        assert f'  1 {r + 1} {c + 1} ' not in riv
    seep = (tmp_path / 'gwf.seep.drn').read_text().split('BEGIN PERIOD 1\n')[1]
    seep_cond = sum(float(line.split()[4]) for line in seep.splitlines() if line.startswith('  1 '))
    land = (grid.basin_frac - grid.lake_frac)[grid.active].sum() * grid.cell_area
    assert seep_cond == pytest.approx(0.1 * land, rel=1e-6)


def test_no_lake_keeps_previous_packages(grid, tmp_path):
    write_simulation(tmp_path, grid, zone_cell_weights(grid, _zones()), 5, MF6BasinParams())
    assert 'GHB6' not in (tmp_path / 'gwf.nam').read_text()
    assert not (tmp_path / 'gwf.lake.ghb').exists()
    npf = (tmp_path / 'gwf.npf').read_text()
    assert 'K22' not in npf and 'ANGLE1' not in npf


def test_anisotropy_and_parameter_names(grid, tmp_path):
    write_simulation(tmp_path, grid, zone_cell_weights(grid, _zones()), 5, MF6BasinParams())
    assert apply_basin_parameters(tmp_path, {'K': 20.0, 'K_ANISOTROPY': 0.1, 'K_ANGLE': 45.0}, MF6BasinParams())
    npf = (tmp_path / 'gwf.npf').read_text()
    assert 'K22\n    CONSTANT 2\n' in npf and 'ANGLE1\n    CONSTANT 45\n' in npf
    for bad in ({'K_ANISOTROPY': 0.0}, {'ZONE_RECHARGE_FACTOR': 1.5}, {'K_ANGLE': 200.0}):
        with pytest.raises(ConfigurationError):
            MF6BasinParams().updated(bad)


def test_reduced_zone_recharge_multipliers(grid, tmp_path):
    grid.reduced_zones = [13]
    w = zone_cell_weights(grid, _zones())
    write_simulation(tmp_path, grid, w, 5, MF6BasinParams(zone_recharge_factor=0.25))
    rows = [line.split() for line in (tmp_path / 'gwf.rch').read_text().split('BEGIN PERIOD 1\n')[1].splitlines()
            if line.startswith('  1 ')]
    got = {(r[3], int(r[1]), int(r[2])): float(r[4]) for r in rows}
    for x in w.itertuples():
        f = 0.25 if x.zone_id == 13 else 1.0
        assert got[(f'z{x.zone_id}', x.row, x.col)] == pytest.approx(x.mult * f, rel=1e-7)


@pytest.mark.skipif(_mf6() is None, reason='mf6 not installed')
def test_mf6_run_with_lake_anisotropy_and_reduced_recharge_conserves_water(grid, tmp_path):
    from symfluence.models.modflow.basin_model import check_run
    zones = _zones()
    area = dict(zip(zones.zone_id, zones.to_crs(CRS).area))
    grid.set_lake(_lake(), 515.0)
    grid.reduced_zones = [13]
    w = zone_cell_weights(grid, zones, area)
    days = pd.date_range('2010-01-01', periods=90, freq='D')
    rng = np.random.default_rng(1)
    rec = pd.DataFrame({z: rng.uniform(0, 4e-3, len(days)) for z in area}, index=days)
    params = MF6BasinParams(k=10.0, sy=0.05, k_anisotropy=0.2, k_angle=30.0, zone_recharge_factor=0.4)
    write_simulation(tmp_path, grid, w, len(days), params, save_heads='none')
    write_recharge_timeseries(tmp_path, rec, sorted(area))
    r = subprocess.run([_mf6()], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    check_run(tmp_path)
    b = read_budget(tmp_path, days[0])
    land = (rec * pd.Series(area)).sum(axis=1)
    rejected = 0.6 * rec[13] * area[13]
    np.testing.assert_allclose(b.rejected_recharge.to_numpy(), rejected.to_numpy(), rtol=1e-5)
    np.testing.assert_allclose((b.recharge + b.rejected_recharge).to_numpy(), land.to_numpy(), rtol=1e-5)
    assert (b.lake != 0).any()
    # everything the land model passed down leaves as discharge or goes into storage
    np.testing.assert_allclose(land - b.discharge - b.storage_change, 0.0, atol=1e-3 * land.mean())


# ---------------------------------------------------------------- MODPATH travel times
def _mp7():
    exe = shutil.which('mp7')
    m = _mf6()
    if not exe and m and Path(m).with_name('mp7').exists():
        exe = str(Path(m).with_name('mp7'))
    return exe


@pytest.mark.skipif(_mf6() is None or _mp7() is None, reason='mf6 or mp7 not installed')
def test_modpath_travel_times(grid, tmp_path):
    from symfluence.models.modflow.particle_tracking import track_steady_state, weighted_quantiles
    zones = _zones()
    area = dict(zip(zones.zone_id, zones.to_crs(CRS).area))
    grid.set_lake(_lake(), 515.0)
    w = zone_cell_weights(grid, zones, area)
    days = pd.date_range('2010-01-01', periods=10, freq='D')
    rec = pd.DataFrame({z: np.full(len(days), 2e-3) for z in area}, index=days)
    sim = tmp_path / 'sim'
    write_simulation(sim, grid, w, len(days), MF6BasinParams(k=10.0, sy=0.1, k_anisotropy=0.5, k_angle=0.0))
    write_recharge_timeseries(sim, rec, sorted(area))
    tt = track_steady_state(sim, tmp_path / 'mp', _mf6(), _mp7())
    assert len(tt) == int(grid.active.sum())
    assert tt.discharged.all()                             # every particle reached a sink
    assert (tt.travel_days >= 0).all() and tt.travel_days.max() > 0
    assert set(tt.end) <= {'lake', 'river', 'seepage'}
    assert tt.recharge_m3d.sum() == pytest.approx(2e-3 * sum(area.values()), rel=1e-6)
    # flow is westward (DEM falls west): particles end no further east than they start
    assert (tt.x <= tt.x0 + 1e-6).all()
    # halving porosity halves every travel time (MODPATH is linear in porosity)
    tt2 = track_steady_state(sim, tmp_path / 'mp2', _mf6(), _mp7(), porosity=0.05)
    np.testing.assert_allclose(tt2.travel_days, tt.travel_days / 2, rtol=1e-4, atol=1e-6)
    q = weighted_quantiles(tt, by='end')
    assert (q.recharge_m3d.sum() == pytest.approx(tt.recharge_m3d.sum()))
