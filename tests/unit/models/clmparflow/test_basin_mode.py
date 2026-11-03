# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow-CLM basin mode: pfb/pfidb I/O, grid building, segment inputs and configuration."""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import GeospatialError
from symfluence.models.clmparflow.basin import pfio
from symfluence.models.clmparflow.basin.case import (
    BasinParameters,
    parflow_keys,
    permeability,
    write_clm_files,
    write_forcing,
)
from symfluence.models.clmparflow.basin.chain import glacier_ledger
from symfluence.models.clmparflow.basin.grid import BasinGrid, d4_slopes
from symfluence.models.clmparflow.basin.settings import water_years


def _grid(ny=4, nx=5, ice_cells=((1, 1),)):
    mask = np.ones((ny, nx), bool)
    mask[0, 0] = False
    ice = np.zeros((ny, nx), bool)
    for j, i in ice_cells:
        ice[j, i] = True
    unit = np.where(ice, 1, 0)
    unit[~mask] = -1
    zeros = np.zeros((ny, nx))
    return BasinGrid(mask=mask, z=zeros, zc=zeros, sx=zeros, sy=zeros, unit=unit, ice=ice & mask,
                     icefrac=ice.astype(float), x0=500000., y0=7100000., dx=1000., crs='EPSG:3057',
                     units=['land', 'glacier'], info={})


class TestPfio:
    def test_split_extent_gives_the_first_pieces_one_extra_cell(self):
        assert pfio.split_extent(72, 2) == [(0, 36), (36, 36)]
        assert pfio.split_extent(7, 3) == [(0, 3), (3, 2), (5, 2)]

    @pytest.mark.parametrize('P,Q', [(1, 1), (2, 1), (2, 3)])
    def test_round_trip_and_dist_offsets(self, tmp_path, P, Q):
        a = np.arange(3 * 5 * 7, dtype=float).reshape(3, 5, 7)
        f = tmp_path / 'x.pfb'
        pfio.write_pfb(f, a, dx=10., dy=20., dz=1., P=P, Q=Q)
        np.testing.assert_array_equal(pfio.read_pfb(f), a)
        offsets = [int(x) for x in (tmp_path / 'x.pfb.dist').read_text().split()]
        assert len(offsets) == P * Q and offsets[0] == 0
        raw = f.read_bytes()
        assert struct.unpack('>i', raw[60:64])[0] == P * Q
        for o in offsets[1:]:  # each offset starts a subgrid header (ix, iy, iz, nx, ny, nz, rx, ry, rz)
            ix, iy, iz, sx, sy, sz = struct.unpack('>6i', raw[o:o + 24])
            assert iz == 0 and sz == 3 and 0 <= ix < 7 and 0 <= iy < 5

    def test_pfidb_layout(self, tmp_path):
        pfio.write_pfidb(tmp_path / 'r.pfidb', {'A.B': 1.5, 'Flag': True, 'Names': ''})
        assert (tmp_path / 'r.pfidb').read_text().split('\n') == [
            '3', '3', 'A.B', '3', '1.5', '4', 'Flag', '4', 'True', '5', 'Names', '0', '', '']


class TestSlopes:
    def test_every_active_cell_drains_to_the_outlet(self):
        rng = np.random.default_rng(1)
        z = rng.uniform(500, 600, (6, 8))
        mask = np.ones((6, 8), bool)
        mask[0, :3] = False
        outlet = (5, 7)
        sx, sy, zc = d4_slopes(z, mask, outlet, dx=1000.)
        assert (np.abs(sx[mask]) + np.abs(sy[mask]) > 0).all()
        assert ((sx != 0) & (sy != 0)).sum() == 0  # D4: one direction per cell
        assert (sx[~mask] == 0).all() and (sy[~mask] == 0).all()
        # north-up arrays: Sx > 0 -> west (column - 1); Sy > 0 -> south (row + 1)
        for r, c in zip(*np.nonzero(mask)):
            seen = set()
            while (r, c) != outlet:
                assert (r, c) not in seen
                seen.add((r, c))
                if sx[r, c]:
                    c += -1 if sx[r, c] > 0 else 1
                else:
                    r += 1 if sy[r, c] > 0 else -1
                assert mask[r, c]

    def test_outlet_off_the_edge_is_rejected(self):
        mask = np.ones((5, 5), bool)
        with pytest.raises(GeospatialError, match='edge'):
            d4_slopes(np.zeros((5, 5)), mask, (2, 2), dx=1000.)


class TestCase:
    def test_permeability_layers_and_glacier_bed(self):
        g = _grid()
        p = BasinParameters(soil_k=.1, aquifer_k=1.25, glacier_bed_k=.001)
        K = permeability(g, p)
        assert K.shape == (14, 4, 5)
        assert (K[:4] == 1.25).all()
        assert K[4:, 1, 1].tolist() == [.001] * 10
        assert K[4:, 2, 2].tolist() == [.1] * 10
        assert (permeability(g, BasinParameters())[4:] == .1).all()

    def test_keys_cold_start_and_restart(self):
        g = _grid()
        p = BasinParameters(manning=6e-6)
        cold = parflow_keys(g, p, 'run', 48, P=2)
        assert cold['ICPressure.Type'] == 'HydroStaticPatch'
        assert cold['Geom.domain.ICPressure.Value'] == pytest.approx(70. - 3.)
        assert cold['Patch.top.BCPressure.Type'] == 'OverlandFlow'  # PrintOverlandSum needs this BC type
        assert cold['Mannings.Geom.domain.Value'] == 6e-6 and cold['Process.Topology.P'] == 2
        assert cold['Cell.0.dzScale.Value'] == 28 and cold['Cell.13.dzScale.Value'] == .1
        warm = parflow_keys(g, p, 'run', 48, ic_file='ic_press.pfb')
        assert warm['ICPressure.Type'] == 'PFBFile' and 'Geom.domain.ICPressure.RefPatch' not in warm

    def test_forcing_maps_units_scales_precipitation_and_fills_inactive_cells(self, tmp_path):
        g = _grid()
        times = pd.date_range('2013-10-01', periods=48, freq='h')
        names = {'DSWR': 'sw', 'DLWR': 'lw', 'APCP': 'pr', 'Temp': 'ta', 'UGRD': 'ws', 'Press': 'ps', 'SPFH': 'q'}
        data = {v: (('hru', 'time'), np.vstack([np.full(48, 1.0), np.full(48, 2.0)])) for v in names.values()}
        xr.Dataset(data, coords={'time': times, 'hru': [0, 1]}).assign(
            hruId=('hru', ['land', 'glacier'])).to_netcdf(tmp_path / 'f.nc')
        n = write_forcing(tmp_path / 'forcing', g, BasinParameters(precip_scale=1.15), tmp_path / 'f.nc',
                          '2013-10-01', '2013-10-03', unit_variable='hruId', names=names)
        assert n == 48
        files = sorted((tmp_path / 'forcing').glob('CARRA.APCP.*.pfb'))
        assert [f.name for f in files] == ['CARRA.APCP.000001_to_000024.pfb', 'CARRA.APCP.000025_to_000048.pfb']
        a = pfio.read_pfb(files[0])
        assert a.shape == (24, 4, 5)
        assert a[0, 1, 1] == pytest.approx(2.0 * 1.15, rel=1e-6) and a[0, 2, 2] == pytest.approx(1.15, rel=1e-6)
        assert a[0, 0, 0] == pytest.approx(1.15, rel=1e-6)  # inactive: a land unit's column
        assert (pfio.read_pfb(tmp_path / 'forcing' / 'CARRA.VGRD.000001_to_000024.pfb') == 0).all()

    def test_clm_files(self, tmp_path):
        g = _grid()
        write_clm_files(tmp_path, g, BasinParameters(ice_albedo=(.2, .1)), '2013-10-01', '2014-10-01',
                        restart=True, ranks=2)
        clmin = (tmp_path / 'drv_clmin.dat').read_text()
        for key, value in [('syr', '2013'), ('smo', '10'), ('eyr', '2014'), ('clm_ic', '1'), ('startcode', '2'),
                           ('rstf', 'clm.rst.')]:
            assert any(line.split()[:2] == [key, value] for line in clmin.splitlines()), key
        vegm = (tmp_path / 'drv_vegm.dat').read_text().splitlines()[2:]
        assert len(vegm) == 20
        row = {(int(line[:4]), int(line[4:8])): line.split()[7:] for line in vegm}
        assert row[(2, 2)][14] == '1.0' and row[(3, 3)][15] == '1.0'  # glacier -> class 15, land -> 16
        assert (tmp_path / 'clm_albice.txt').read_text().split() == ['0.200', '0.100']
        assert (tmp_path / 'drv_vegm.dat.1').exists() and (tmp_path / 'qflx_infl').is_dir()


class TestGlacierLedger:
    def test_per_rank_logs_and_exponentless_denormals(self, tmp_path):
        g = _grid(ice_cells=((1, 1), (2, 3)))
        (tmp_path / 'glacier_ice.txt.0').write_text('       2  1.000000E+00  0.000000E+00  1.0E+00\n')
        (tmp_path / 'glacier_ice.txt.1').write_text('       2  3.000000E+00  1.7-315  3.0E+00\n')
        led = glacier_ledger(tmp_path, g)
        assert led['ice_melt_mm'] == pytest.approx(2.0) and led['ice_deficit_mm'] == 0.0
        assert led['log_lines'] == 2


def test_water_years():
    assert water_years('2012-10-01', '2023-09-30 23:00') == list(range(2013, 2024))
    assert water_years('2009-01-01', '2023-09-30') == list(range(2010, 2024))
    assert water_years('2012-10-02', '2014-06-01') == []


def test_basin_config_aliases(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig

    c = SymfluenceConfig(**{
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'), 'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'e', 'EXPERIMENT_TIME_START': '2012-10-01 00:00',
        'EXPERIMENT_TIME_END': '2023-09-30 23:00', 'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs', 'HYDROLOGICAL_MODEL': 'CLMPARFLOW', 'FORCING_DATASET': 'CARRA',
        'CLMPARFLOW_BASIN_MODE': True, 'CLMPARFLOW_BASIN_CRS': 'EPSG:3057', 'CLMPARFLOW_BASIN_ICE_ALBEDO': [.2, .1],
        'CLMPARFLOW_BASIN_GLACIER_BED_K': .001, 'CLMPARFLOW_BASIN_FORCING_VARIABLES': {'APCP': 'PRE'},
        'CLMPARFLOW_BASIN_SPINUP_YEARS': [2010, 2011, 2012],
    })
    from symfluence.models.clmparflow.basin.settings import basin_parameters, chain_settings

    cfg = c.model.clmparflow
    p = basin_parameters(cfg)
    assert cfg.basin_mode and p.ice_albedo == (.2, .1) and p.glacier_bed_k == .001
    s = chain_settings(cfg, '2012-10-01 00:00', '2023-09-30 23:00', 2)
    assert s.spinup_years == [2010, 2011, 2012] and s.production_years == list(range(2013, 2024))
    assert s.P == 2 and s.forcing_names == {'APCP': 'PRE'}


# --- optional structure: anisotropy, K zone, lake, rotated grid ---------------------------------------------

from symfluence.core.exceptions import ConfigurationError  # noqa: E402
from symfluence.models.clmparflow.basin import case as case_mod  # noqa: E402
from symfluence.models.clmparflow.basin.case import mannings, tensor_multipliers, write_static  # noqa: E402
from symfluence.models.clmparflow.basin.grid import build_basin_grid, grid_transform, polygon_cells  # noqa: E402


def _zone_grid():
    g = _grid()
    g.kzone = np.zeros(g.shape, bool)
    g.kzone[2:, 3:] = True
    g.lake = np.zeros(g.shape, bool)
    g.lake[1:3, 2] = True
    return g


class TestOptionalStructure:
    def test_defaults_leave_keys_and_record_unchanged(self):
        g = _zone_grid()  # flags on the grid do nothing without their parameters
        p = BasinParameters(manning=6e-6)
        k = parflow_keys(g, p, 'run', 48)
        assert k['Perm.TensorType'] == 'TensorByGeom' and k['Geom.domain.Perm.TensorValZ'] == 1.
        assert k['Mannings.Type'] == 'Constant' and 'Mannings.FileName' not in k
        assert not any('TensorFile' in key for key in k)
        assert (permeability(g, p) == permeability(_grid(), p)).all()
        assert set(p.record()) == set(BasinParameters().record())
        assert not {'k_tensor', 'k_zone_factor', 'k_zone_soil', 'lake_manning'} & set(p.record())
        assert p.record()['manning'] == 6e-6

    def test_tensor_is_aquifer_only_and_uses_tensor_files(self):
        g = _grid()
        p = BasinParameters(k_tensor=(1.0, 0.25, 0.1))
        kx, ky, kz = tensor_multipliers(g, p)
        assert kx.shape == (14, 4, 5)
        assert (kx == 1).all() and (ky[:4] == .25).all() and (kz[:4] == .1).all()
        assert (ky[4:] == 1).all() and (kz[4:] == 1).all()  # CLM soil layers stay isotropic
        k = parflow_keys(g, p, 'run', 48)
        assert k['Perm.TensorType'] == 'TensorByFile' and k['Geom.Perm.TensorByFileNames'] == 'domain'
        assert [k[f'Geom.domain.Perm.TensorFile{a}'] for a in 'XYZ'] == ['kx.pfb', 'ky.pfb', 'kz.pfb']
        assert 'Geom.Perm.TensorByGeom.Names' not in k and 'Geom.domain.Perm.TensorValX' not in k
        assert p.record()['k_tensor'] == (1.0, 0.25, 0.1)

    def test_k_zone_scales_aquifer_and_optionally_soil(self):
        g = _zone_grid()
        g.ice[2, 3] = True  # a glacier cell inside the zone keeps its bed K
        K = permeability(g, BasinParameters(aquifer_k=2., soil_k=.1, glacier_bed_k=.001, k_zone_factor=.5))
        assert (K[:4, 2:, 3:] == 1.).all() and (K[:4, :2] == 2.).all() and (K[:4, 2:, :3] == 2.).all()
        assert (K[4:, 2, 4] == .1).all()  # soil unchanged by default
        K = permeability(g, BasinParameters(aquifer_k=2., soil_k=.1, glacier_bed_k=.001, k_zone_factor=.5,
                                            k_zone_soil=True))
        assert (K[4:, 2, 4] == .05).all() and (K[4:, 2, 3] == .001).all() and (K[4:, 0, 4] == .1).all()

    def test_k_zone_and_lake_need_grid_cells(self):
        g = _grid()
        with pytest.raises(ConfigurationError, match='K_ZONE'):
            permeability(g, BasinParameters(k_zone_factor=.2))
        with pytest.raises(ConfigurationError, match='LAKE'):
            parflow_keys(g, BasinParameters(lake_manning=1e-4), 'run', 48)

    def test_lake_mannings(self):
        g = _zone_grid()
        p = BasinParameters(manning=6e-6, lake_manning=3e-4)
        n = mannings(g, p)
        assert n.shape == (4, 5) and (n[g.lake] == 3e-4).all() and (n[~g.lake] == 6e-6).all()
        k = parflow_keys(g, p, 'run', 48)
        assert k['Mannings.Type'] == 'PFBFile' and k['Mannings.FileName'] == 'mannings.pfb'
        assert 'Mannings.Geom.domain.Value' not in k

    def test_write_static_writes_option_files(self, tmp_path, monkeypatch):
        def fake_tool(cmd, cwd, **kw):  # pfmask-to-pfsol
            (cwd / 'basin.pfsol').write_text('')
            return type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()

        monkeypatch.setattr(case_mod.subprocess, 'run', fake_tool)
        g = _zone_grid()
        write_static(tmp_path, g, BasinParameters(), tmp_path, P=2)
        assert not {'kx.pfb', 'mannings.pfb'} & {f.name for f in tmp_path.iterdir()}
        p = BasinParameters(k_tensor=(1., 1., .1), lake_manning=1e-4, k_zone_factor=.135)
        write_static(tmp_path, g, p, tmp_path, P=2)
        np.testing.assert_array_equal(pfio.read_pfb(tmp_path / 'kz.pfb'), tensor_multipliers(g, p)[2])
        np.testing.assert_array_equal(pfio.read_pfb(tmp_path / 'mannings.pfb')[0], mannings(g, p))
        np.testing.assert_array_equal(pfio.read_pfb(tmp_path / 'perm.pfb'), permeability(g, p))
        assert len((tmp_path / 'mannings.pfb.dist').read_text().split()) == 2


class TestGridOptions:
    def test_save_load_round_trip_and_default_files(self, tmp_path):
        g = _grid()
        g.save(tmp_path / 'plain')
        assert set(np.load(tmp_path / 'plain' / 'grid.npz').files) == {
            'mask', 'z', 'zc', 'sx', 'sy', 'unit', 'ice', 'icefrac', 'x0', 'y0', 'dx'}
        h = BasinGrid.load(tmp_path / 'plain')
        assert h.rotation == 0 and h.kzone is None and h.lake is None
        g = _zone_grid()
        g.rotation, g.pivot = 45., (501000., 7102000.)
        g.save(tmp_path / 'opt')
        h = BasinGrid.load(tmp_path / 'opt')
        assert h.rotation == 45. and h.pivot == (501000., 7102000.)
        assert (h.kzone == g.kzone).all() and (h.lake == g.lake).all()

    def test_cell_centres_unrotated_and_rotated(self):
        g = _grid()
        x, y = g.cell_centres()
        assert x[0, 0] == 500500. and y[0, 0] == 7100500. and x[2, 3] == 503500. and y[2, 3] == 7102500.
        g.rotation, g.pivot = 90., (500000., 7100000.)  # grid x axis points north
        x, y = g.cell_centres()
        assert x[0, 0] == pytest.approx(499500.) and y[0, 0] == pytest.approx(7100500.)
        assert x[0, 3] == pytest.approx(499500.) and y[0, 3] == pytest.approx(7103500.)  # +x = north
        assert x[2, 0] == pytest.approx(497500.)  # +y (north on the grid) = west

    def test_grid_transform_matches_from_origin_without_rotation(self):
        from rasterio.transform import from_origin

        assert grid_transform(1000., 5000., 333., fine=10) == from_origin(1000., 5000., 33.3, 33.3)
        t = grid_transform(0., 0., 100., rotation=30., pivot=(0., 0.))
        x, y = t * (1, 0)
        assert (x, y) == pytest.approx((100 * np.cos(np.pi / 6), 100 * np.sin(np.pi / 6)))

    def test_polygon_cells_on_a_rotated_grid(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import box

        f = tmp_path / 'zone.gpkg'
        gpd.GeoDataFrame(geometry=[box(0, 0, 2000, 1000)], crs='EPSG:3057').to_file(f)
        t = grid_transform(0., 4000., 1000., fine=10)
        frac = polygon_cells(f, 'EPSG:3057', t, (4, 4))
        assert frac[3, :2].tolist() == [1., 1.] and frac.sum() == pytest.approx(2.)
        # rotated 90 deg about the origin, the grid covers x in [-4000, 0], y in [0, 4000], columns running north
        gpd.GeoDataFrame(geometry=[box(-1000, 0, 0, 2000)], crs='EPSG:3057').to_file(tmp_path / 'east.gpkg')
        t = grid_transform(0., 4000., 1000., rotation=90., pivot=(0., 0.), fine=10)
        frac = polygon_cells(tmp_path / 'east.gpkg', 'EPSG:3057', t, (4, 4))
        assert frac[3, :2].tolist() == [1., 1.] and frac.sum() == pytest.approx(2.)


def _synthetic_basin(tmp_path):
    """A 10 x 6 km basin draining south, a 3 x 2 km lake mid-valley (DEM shore noise), a zone at the outlet."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point, box

    crs = 'EPSG:3057'
    x0, y0 = 400000., 400000.
    gpd.GeoDataFrame(geometry=[box(x0, y0, x0 + 6000, y0 + 10000)], crs=crs).to_file(tmp_path / 'basin.gpkg')
    gpd.GeoDataFrame({'HRU_ID': [1]}, geometry=[box(x0, y0, x0 + 6000, y0 + 10000)], crs=crs).to_file(
        tmp_path / 'units.gpkg')
    gpd.GeoDataFrame(geometry=[Point(x0 + 2500, y0 + 200)], crs=crs).to_file(tmp_path / 'pour.gpkg')
    gpd.GeoDataFrame(geometry=[box(x0 + 1000, y0 + 4000, x0 + 4000, y0 + 6000)], crs=crs).to_file(
        tmp_path / 'lake.gpkg')
    gpd.GeoDataFrame(geometry=[box(x0, y0, x0 + 6000, y0 + 2000)], crs=crs).to_file(tmp_path / 'zone.gpkg')
    res = 100.
    rows, cols = np.mgrid[0:int(12000 / res), 0:int(8000 / res)]
    yy = y0 + 11000 - (rows + .5) * res
    xx = x0 - 1000 + (cols + .5) * res
    z = 100 + 0.02 * (yy - y0) + 0.01 * np.abs(xx - x0 - 2500)
    lake = (xx > x0 + 1000) & (xx < x0 + 4000) & (yy > y0 + 4000) & (yy < y0 + 6000)
    z = np.where(lake, 150. + 5 * (xx > x0 + 3500), z).astype('float32')
    with rasterio.open(tmp_path / 'dem.tif', 'w', driver='GTiff', width=z.shape[1], height=z.shape[0], count=1,
                       dtype='float32', crs=crs, transform=from_origin(x0 - 1000, y0 + 11000, res, res)) as dst:
        dst.write(z, 1)
    return [tmp_path / f for f in ('basin.gpkg', 'units.gpkg')], tmp_path / 'dem.tif', tmp_path / 'pour.gpkg'


class TestBuildWithOptions:
    def test_lake_is_levelled_and_flat_and_zone_flagged(self, tmp_path):
        (outline, units), dem, pour = _synthetic_basin(tmp_path)
        plain = build_basin_grid(outline, units, 'HRU_ID', dem, pour, 'EPSG:3057', 1000.)
        assert plain.lake is None and plain.kzone is None and plain.rotation == 0
        g = build_basin_grid(outline, units, 'HRU_ID', dem, pour, 'EPSG:3057', 1000., lake=tmp_path / 'lake.gpkg',
                             k_zone=tmp_path / 'zone.gpkg')
        assert g.info['lake_cells'] == 6 and g.info['kzone_cells'] == 12
        assert g.info['lake_level'] == pytest.approx(150.)
        assert (g.z[g.lake] == 150.).all()
        slope = np.abs(g.sx) + np.abs(g.sy)
        assert np.allclose(slope[g.lake], 1e-4)  # flat lake: min_slope towards its outlet
        assert (slope[~g.lake & g.mask] > 1e-4).sum() > 0
        assert (g.mask == plain.mask).all() and (g.unit == plain.unit).all()
        _, y = g.cell_centres()
        assert (g.kzone == (g.mask & (y < 402000))).all()  # the two southern (outlet) rows

    def test_rotated_grid_keeps_the_basin(self, tmp_path):
        (outline, units), dem, pour = _synthetic_basin(tmp_path)
        g = build_basin_grid(outline, units, 'HRU_ID', dem, pour, 'EPSG:3057', 1000., rotation=45.,
                             lake=tmp_path / 'lake.gpkg')
        assert g.rotation == 45. and g.info['rotation_deg'] == 45.
        assert abs(g.info['area_km2'] - 60.) < 8  # staircase of a rotated rectangle
        x, y = g.cell_centres()
        assert (x[g.mask].min() > 399000) and (y[g.mask].max() < 410600)
        lx, ly = x[g.lake], y[g.lake]
        assert ((lx > 400800) & (lx < 404200) & (ly > 403800) & (ly < 406200)).all()


def test_option_config_aliases(tmp_path):
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.models.clmparflow.basin.settings import basin_inputs, basin_parameters, grid_dir

    base = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'), 'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'd', 'EXPERIMENT_ID': 'e', 'EXPERIMENT_TIME_START': '2012-10-01 00:00',
        'EXPERIMENT_TIME_END': '2023-09-30 23:00', 'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs', 'HYDROLOGICAL_MODEL': 'CLMPARFLOW', 'FORCING_DATASET': 'CARRA',
        'CLMPARFLOW_BASIN_MODE': True, 'CLMPARFLOW_BASIN_CRS': 'EPSG:3057',
        'CLMPARFLOW_BASIN_DEM': '/x/dem.tif', 'CLMPARFLOW_BASIN_FORCING': '/x/f.nc',
        'CLMPARFLOW_BASIN_OUTLINE': '/x/b.shp', 'CLMPARFLOW_BASIN_UNITS': '/x/u.shp',
        'CLMPARFLOW_BASIN_POUR_POINT': '/x/p.shp',
    }
    cfg = SymfluenceConfig(**base).model.clmparflow
    p = basin_parameters(cfg)
    assert p.k_tensor is None and p.k_zone_factor == 1. and p.lake_manning is None and not p.k_zone_soil
    inp = basin_inputs(cfg, tmp_path, 'd')
    assert inp.rotation == 0. and inp.k_zone is None and inp.lake is None
    assert grid_dir(cfg, tmp_path) == tmp_path / 'settings' / 'CLMPARFLOW' / 'basin'
    cfg = SymfluenceConfig(**base, **{
        'CLMPARFLOW_BASIN_K_TENSOR': [1., .033, .1], 'CLMPARFLOW_BASIN_K_ZONE': '/x/zone.gpkg',
        'CLMPARFLOW_BASIN_K_ZONE_FACTOR': .135, 'CLMPARFLOW_BASIN_K_ZONE_SOIL': True,
        'CLMPARFLOW_BASIN_LAKE': '/x/lake.gpkg', 'CLMPARFLOW_BASIN_LAKE_MANNING': 3e-4,
        'CLMPARFLOW_BASIN_GRID_ROTATION': 45., 'CLMPARFLOW_BASIN_GRID_DIR': str(tmp_path / 'g')}).model.clmparflow
    p = basin_parameters(cfg)
    assert p.k_tensor == (1., .033, .1) and p.k_zone_factor == .135 and p.k_zone_soil and p.lake_manning == 3e-4
    inp = basin_inputs(cfg, tmp_path, 'd')
    assert inp.rotation == 45. and str(inp.k_zone) == '/x/zone.gpkg' and str(inp.lake) == '/x/lake.gpkg'
    assert grid_dir(cfg, tmp_path) == tmp_path / 'g'
    with pytest.raises(ConfigurationError, match='K_TENSOR'):
        basin_parameters(SymfluenceConfig(**base, CLMPARFLOW_BASIN_K_TENSOR=[1., 1.]).model.clmparflow)


def _valley_case(tmp_path, case):
    """8 x 6 cells of 1 km draining south, a 3 x 3 lake mid-valley; 3-day storm in a 15-day run."""
    from symfluence.models.clmparflow.basin.case import run_segment
    from symfluence.models.clmparflow.basin.chain import summarise_segment

    ny, nx = 8, 6
    r, c = np.mgrid[0:ny, 0:nx]  # north-up
    lake = (r >= 3) & (r <= 5) & (c >= 1) & (c <= 3)
    z = np.where(lake, 160., 100. + 30. * (ny - 1 - r) + 5. * np.abs(c - 2))
    mask = np.ones((ny, nx), bool)
    sx, sy, zc = d4_slopes(z, mask, (ny - 1, 2), 1000.)
    if case == 'pit':  # unfilled depression: lake cells slope into the centre cell, which has no slope
        for rr, cc in zip(*np.nonzero(lake)):
            sx[rr, cc] = 1e-4 * np.sign(cc - 2)
            sy[rr, cc] = 0. if cc != 2 else 1e-4 * np.sign(4 - rr)
    f = lambda a: a[::-1].copy()  # noqa: E731
    g = BasinGrid(mask=f(mask), z=f(z), zc=f(zc), sx=f(sx), sy=f(sy), unit=np.zeros((ny, nx), int),
                  ice=np.zeros((ny, nx), bool), icefrac=np.zeros((ny, nx)), x0=380000., y0=400000., dx=1000.,
                  crs='EPSG:3057', units=['land'], info={}, lake=f(lake))
    t = pd.date_range('2010-06-01', periods=24 * 15, freq='h')
    rain = np.where((t >= '2010-06-03') & (t < '2010-06-06'), 10. / 3600, 0.)
    v = dict(SWRadAtm=150., LWRadAtm=300., pptrate=rain, airtemp=283., windspd=3., airpres=1e5, spechum=6e-3)
    xr.Dataset({k: (('time', 'hru'), np.broadcast_to(np.asarray(a, float), t.shape)[:, None]) for k, a in v.items()},
               coords={'time': t, 'hru': [0]}).assign(unit_id=('hru', ['land'])).to_netcdf(tmp_path / f'{case}.nc')
    p = BasinParameters(initial_water_table_depth=.5, manning=6e-6, lake_manning=1e-3 if case == 'lake' else None)
    run = tmp_path / case
    run_segment(run, case, g, p, Path(os.environ['CLMPARFLOW_PATCHED_DIR']), tmp_path / f'{case}.nc',
                '2010-06-01', '2010-06-16')
    q = np.array(list(summarise_segment(run, g)['outlet_m3s'].values()))
    press = sorted(run.glob(f'{case}.out.press.*.pfb'), key=lambda x: int(x.name.split('.')[-2]))
    pond = np.array([(np.maximum(pfio.read_pfb(x)[-1], 0) * 1e6 * g.lake).sum() for x in press])
    return q, pond


@pytest.mark.models
@pytest.mark.skipif(not os.environ.get('CLMPARFLOW_PATCHED_DIR'),
                    reason='set CLMPARFLOW_PATCHED_DIR to a patched ParFlow-CLM install to run ParFlow')
def test_parflow_lake_stores_and_releases_but_a_pit_never_spills(tmp_path):
    q0, pond0 = _valley_case(tmp_path, 'none')
    q1, pond1 = _valley_case(tmp_path, 'lake')
    q2, pond2 = _valley_case(tmp_path, 'pit')
    assert q1.max() < .75 * q0.max()  # attenuated peak
    assert pond1.max() > 5 * pond0.max()  # ponded lake storage
    assert pond1[-1] < .5 * pond1.max() and q1[8:].sum() > 1.5 * q0[8:].sum()  # released during the recession
    assert pond2[-1] >= .99 * pond2.max()  # the kinematic BC never spills a depression
