# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The last #339 readers: per-HRU folders, CLM-ParFlow basin mode, benchmarking."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from symfluence.core.exceptions import FileOperationError
from symfluence.core.modeling.forcing_naming import forcing_token, hru_geometry, select_hru_forcing_folder

pytestmark = pytest.mark.unit


def _name(token):
    return Path(f'dom_ERA5_remapped_{token}_2004-01-01-00-00-00.nc')


class TestHruForcingFolder:
    def test_single_set_is_used_as_is(self):
        files = [_name('elevation-h5-aaaaaaaa'), _name('elevation-h5-aaaaaaaa').with_name('x_remapped_elevation-h5-aaaaaaaa_2004-02.nc')]
        assert select_hru_forcing_folder(files) == files

    def test_several_sets_need_the_setting(self):
        files = [_name('elevation-h5-aaaaaaaa'), _name('grus-h1-bbbbbbbb')]
        with pytest.raises(FileOperationError, match='PRMS_HRU_FORCING_DISCRETIZATION'):
            select_hru_forcing_folder(files, None, 'PRMS_HRU_FORCING_DISCRETIZATION')

    def test_setting_picks_the_set(self):
        files = [_name('elevation-h5-aaaaaaaa'), _name('grus-h1-bbbbbbbb')]
        assert select_hru_forcing_folder(files, 'elevation') == [files[0]]

    def test_untokened_beside_tokened_is_ambiguous(self):
        files = [Path('dom_ERA5_remapped_2004-01-01-00-00-00.nc'), _name('grus-h1-bbbbbbbb')]
        with pytest.raises(FileOperationError, match='untokened'):
            select_hru_forcing_folder(files)


def _units(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({'HRU_ID': list(range(1, n + 1))},
                     geometry=[box(i, 0, i + 1, 1) for i in range(n)], crs='EPSG:4326').to_file(path)
    return path


def _forcing(path, n_hru, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({'air_temperature': (('time', 'hru'), np.full((2, n_hru), value))},
               coords={'time': np.array(['2004-01-01T00', '2004-01-01T01'], dtype='datetime64[ns]'),
                       'hruId': ('hru', np.arange(1, n_hru + 1))}).to_netcdf(path)
    return path


class TestClmParflowBasinForcing:
    def _inputs(self, forcing, units):
        from symfluence.models.clmparflow.basin.settings import BasinInputs
        return BasinInputs(outline=Path(), units=units, unit_field='HRU_ID', dem=Path(), pour_point=Path(),
                           crs='EPSG:3057', dx=1000.0, min_slope=0.0, ice_field=None, ice_value=None,
                           forcing=forcing)

    def test_explicit_file_is_unchanged(self, tmp_path):
        from symfluence.models.clmparflow.basin.settings import basin_forcing_files
        f = _forcing(tmp_path / 'hourly_forcing.nc', 2, 1.0)
        assert basin_forcing_files(self._inputs(f, tmp_path / 'units.shp'), 'elevation') == f

    def test_directory_selects_the_units_hru_set(self, tmp_path):
        from symfluence.models.clmparflow.basin.case import open_forcing
        from symfluence.models.clmparflow.basin.settings import basin_forcing_files
        units = _units(tmp_path / 'shp' / 'units.shp', 3)
        store = tmp_path / 'store'
        mine = _forcing(store / _name(forcing_token('elevation', hru_geometry(units, 'HRU_ID'))), 3, 1.0)
        _forcing(store / _name('grus-h1-cccccccc'), 1, 99.0)
        files = basin_forcing_files(self._inputs(store, units), 'elevation')
        assert files == [mine]
        with open_forcing(files) as ds:
            assert float(ds['air_temperature'].max()) == 1.0

    def test_directory_with_only_other_hru_sets_raises(self, tmp_path):
        from symfluence.models.clmparflow.basin.settings import basin_forcing_files
        units = _units(tmp_path / 'shp' / 'units.shp', 3)
        _forcing(tmp_path / 'store' / _name('elevation-h3-dddddddd'), 3, 1.0)
        with pytest.raises(FileOperationError, match='other HRU sets'):
            basin_forcing_files(self._inputs(tmp_path / 'store', units), 'elevation')


def test_benchmark_reads_only_this_runs_forcing(tmp_path):
    import logging

    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.core.mixins.project import resolve_data_subdir
    from symfluence.evaluation.benchmarking import BenchmarkPreprocessor

    cfg = SymfluenceConfig.from_minimal(domain_name='bm', model='SUMMA', time_start='2004-01-01 00:00',
                                        time_end='2004-01-02 00:00', SYMFLUENCE_DATA_DIR=str(tmp_path),
                                        SUB_GRID_DISCRETIZATION='GRUs')
    bp = BenchmarkPreprocessor(cfg, logging.getLogger('t'))
    geometry = hru_geometry(_units(bp._get_catchment_file_path(), 1), 'HRU_ID')
    store = resolve_data_subdir(bp.project_dir, 'forcing') / 'basin_averaged_data'
    for token, n, value in [(forcing_token('GRUs', geometry), 1, 270.0), ('elevation-h4-eeeeeeee', 4, 250.0)]:
        path = store / _name(token)
        path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({'air_temperature': (('time', 'hru'), np.full((2, n), value)),
                    'precipitation_flux': (('time', 'hru'), np.zeros((2, n)))},
                   coords={'time': np.array(['2004-01-01T00', '2004-01-01T01'], dtype='datetime64[ns]')}
                   ).to_netcdf(path)
    df = bp._load_forcing_data()
    assert (df['temperature'] == 270.0).all()
