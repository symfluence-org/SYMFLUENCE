"""Real-basin GSFLOW geometry, inputs and parameter files on a synthetic basin."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from symfluence.core.exceptions import ConfigurationError, ValidationError
from symfluence.models.gsflow.basin_geometry import build_basin_geometry
from symfluence.models.gsflow.basin_inputs import hru_source_ids, statvars, write_modflow, write_parameters
from symfluence.models.gsflow.config_schema import GSFLOWConfig
from symfluence.models.gsflow.parameter_files import apply_gsflow_parameters, replace_block_values

CRS = 32627   # UTM 27N (Iceland), metres


@pytest.fixture
def basin(tmp_path):
    """10 x 6 km basin, two 5 km subbasins draining west; DEM rises 100 m/km eastward."""
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    x0, y0, res = 500000.0, 7100000.0, 100.0
    cols = np.arange(100)
    dem = np.tile(500.0 + 100.0 * (cols + 0.5) * res / 1000.0, (60, 1)).astype('float32')
    path = tmp_path / 'dem_elv.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=60, width=100, count=1, dtype='float32',
                       crs=f'EPSG:{CRS}', transform=from_origin(x0, y0, res, res), nodata=-9999) as dst:
        dst.write(dem, 1)
    sub = gpd.GeoDataFrame({'GRU_ID': [1, 2], 'gru_to_seg': [10, 20]},
                           geometry=[box(x0, y0 - 6000, x0 + 5000, y0), box(x0 + 5000, y0 - 6000, x0 + 10000, y0)],
                           crs=CRS)
    net = gpd.GeoDataFrame({'LINKNO': [10, 20], 'DSLINKNO': [-1, 10], 'Slope': [0.02, 0.02],
                            'DOUTSTART': [5000.0, 10000.0], 'DOUTEND': [0.0, 5000.0]},
                           geometry=[LineString([(x0 + 5000, y0 - 3000), (x0 + 50, y0 - 3000)]),
                                     LineString([(x0 + 9950, y0 - 3000), (x0 + 5000, y0 - 3000)])], crs=CRS)
    ice = gpd.GeoDataFrame({'id': [1]}, geometry=[box(x0 + 8000, y0 - 6000, x0 + 10000, y0)], crs=CRS)
    return sub, net, path, ice


def build(basin, **kw):
    sub, net, dem, ice = basin
    return build_basin_geometry(sub, net, dem, CRS, cell_size=kw.pop('cell_size', 1000.0), band_size=200.0,
                                min_hru_area_km2=kw.pop('min_hru_area_km2', 1.0), glacier_outlines=ice, **kw)


def test_hrus_partition_basin_and_follow_bands(basin):
    g = build(basin)
    h = g.hrus
    assert h.area_m2.sum() == pytest.approx(60e6, rel=1e-6)
    assert h[h.glacier == 1].area_m2.sum() == pytest.approx(12e6, rel=1e-6)
    assert set(h.seg) == {1, 2} and h.hru_id.tolist() == list(range(1, len(h) + 1))
    assert (h.elev.diff().dropna().abs() > 0).any()


def test_gvr_mapping_conserves_hru_and_cell_area(basin):
    g = build(basin)
    gvr = g.gvr
    assert np.allclose(gvr.groupby('hru_id').hru_pct.sum(), 1.0)
    assert gvr.groupby('cell_id').cell_pct.sum().max() <= 1.0 + 1e-9
    assert int(g.active.sum()) == 60 and np.isfinite(g.top[g.active]).all()


def test_sfr_segments_are_upstream_first_and_streambeds_descend(basin):
    g = build(basin)
    segs = g.segments.set_index('linkno')
    assert segs.loc[20, 'seg'] < segs.loc[10, 'seg'] and segs.loc[10, 'outseg'] == 0
    assert segs.loc[20, 'outseg'] == segs.loc[10, 'seg']
    r = g.reaches.sort_values(['seg', 'ireach'])
    z = r.strtop.to_numpy()
    assert (np.diff(z) < 0).all()                      # whole path, upstream segment first
    tops = np.array([g.top[a - 1, b - 1] for a, b in zip(r.row, r.col)])
    assert (z < tops).all()
    assert r.length.sum() == pytest.approx(9900.0, rel=1e-6)


def test_glacier_hrus_chain_to_terminus(basin):
    h = build(basin).hrus
    ice = h[h.glacier == 1].sort_values('elev', ascending=False)
    assert (ice.tohru.iloc[:-1].to_numpy() == ice.hru_id.iloc[1:].to_numpy()).all()
    assert ice.tohru.iloc[-1] == 0 and (h[h.glacier == 0].tohru == 0).all()
    assert (ice.hru_length_km > 0).all()


def test_slivers_merge_within_subbasin(basin):
    coarse = build(basin, min_hru_area_km2=20.0)
    fine = build(basin, min_hru_area_km2=0.0)
    assert coarse.nhru < fine.nhru
    assert coarse.hrus.area_m2.sum() == pytest.approx(fine.hrus.area_m2.sum())


def test_network_must_have_one_outlet(basin):
    sub, net, dem, ice = basin
    net = net.assign(DSLINKNO=[-1, -1])
    with pytest.raises(ValidationError, match='exactly one outlet'):
        build_basin_geometry(sub, net, dem, CRS, cell_size=1000.0)


def test_band_forcing_ids():
    assert hru_source_ids([550, 650, 1350], 200.0, {400: 1, 600: 2, 800: 3}) == [1, 2, 3]
    with pytest.raises(ConfigurationError):
        hru_source_ids([550], 200.0, {})


def test_written_inputs_and_parameter_updates(basin, tmp_path):
    g = build(basin)
    cfg = GSFLOWConfig(gw_grid_mode='basin', glacier_flag=True, forcing_mode='climate_hru', cov_type=0,
                       separate_glacier_depletion_curve=True, cryosphere_constraints=True)
    settings = tmp_path / 'settings'
    (settings / 'modflow').mkdir(parents=True)
    dims = write_parameters(settings / 'params.dat', g, cfg,
                            {'stor_ice': {'dimensions': ['nhru', 'nmonths'], 'values': 29.0}})
    assert dims['nhru'] == g.nhru and dims['nhrucell'] == len(g.gvr) and dims['ndeplval'] == 22
    write_modflow(settings / 'modflow', g, 365, 100.0, 1e-3, 0.3, 'out', ntrail=10, nsets=10)
    sfr = (settings / 'modflow/bow.sfr').read_text()
    assert 'TRANSROUTE' in sfr and f'{len(g.reaches)}  {len(g.segments)}  0  0' in sfr
    assert '  1  1  1  0  0  0  10  10  0  1.0' in (settings / 'modflow/bow.uzf').read_text()
    names = [n for n, _ in statvars(g, True, True)]
    assert 'basin_storage' not in names and names.count('snowcov_area') == g.nhru
    apply_gsflow_parameters(settings, {'rain_cbh_adj': 1.2, 'land_sca_max': 0.7, 'K': 250.0, 'SY': 0.1,
                                       'strhc1': 3.5})
    text = (settings / 'params.dat').read_text()
    block = text.split('####\nrain_cbh_adj\n')[1].split('####')[0].split('\n')
    assert block[0] == '2' and block[5:5 + g.nhru * 12] == ['1.200000'] * (g.nhru * 12)
    upw = (settings / 'modflow/bow.upw').read_text()
    assert upw.count('2.500000e+02') == 2 and '1.000000e-01' in upw
    reach = [line.split() for line in (settings / 'modflow/bow.sfr').read_text().split('\n')
             if len(line.split()) == 14 and line.split()[0] == '1']
    assert reach and all(f[9] == '3.5' for f in reach)
    with pytest.raises(ConfigurationError, match='not in params.dat'):
        apply_gsflow_parameters(settings, {'no_such_parameter': 1.0})


def test_replace_block_values_keeps_multidimensional_layout():
    block = 'rain_cbh_adj\n2\nnhru\nnmonths\n24\n2\n' + '1.0\n' * 24
    out = replace_block_values(block, 'rain_cbh_adj', 0.9).split('\n')
    assert out[:6] == block.split('\n')[:6] and out[6:30] == ['0.900000'] * 24
