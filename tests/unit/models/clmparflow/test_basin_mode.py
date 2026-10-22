# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow-CLM basin mode: pfb/pfidb I/O, grid building, segment inputs and configuration."""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
        with pytest.raises(ValueError, match='edge'):
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
