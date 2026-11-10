# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Geometry-aware remapped-forcing names (issue #339)."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from symfluence.core.exceptions import FileOperationError
from symfluence.core.modeling.forcing_naming import (
    discretization_key_from_name,
    forcing_token,
    hru_geometry,
    resolve_forcing_files,
    select_forcing_files,
    split_token,
)

pytestmark = pytest.mark.unit


def _catchment(path: Path, n: int, width: float = 1.0) -> Path:
    gdf = gpd.GeoDataFrame(
        {'HRU_ID': list(range(1, n + 1))},
        geometry=[box(i * width, 0, (i + 1) * width, 1) for i in range(n)],
        crs='EPSG:4326',
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)
    return path


def _forcing(path: Path, n_hru: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {'pptrate': (('time', 'hru'), np.zeros((2, n_hru)))},
        coords={'time': np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]')},
    ).to_netcdf(path)
    return path


def _name(token: str, month: str = '2020-01-01-00-00-00') -> str:
    return f'dom_ERA5_remapped_{token}_{month}.nc'


class TestHruGeometry:
    def test_same_content_same_digest_after_rewrite(self, tmp_path):
        a = hru_geometry(_catchment(tmp_path / 'a' / 'c.shp', 3), 'HRU_ID')
        b = hru_geometry(_catchment(tmp_path / 'b' / 'c.shp', 3), 'HRU_ID')
        assert a == b
        assert a.count == 3

    def test_geometry_change_changes_digest(self, tmp_path):
        a = hru_geometry(_catchment(tmp_path / 'a' / 'c.shp', 3), 'HRU_ID')
        b = hru_geometry(_catchment(tmp_path / 'b' / 'c.shp', 3, width=2.0), 'HRU_ID')
        assert a.count == b.count and a.digest != b.digest

    def test_cache_invalidated_when_file_changes(self, tmp_path):
        path = tmp_path / 'c.shp'
        first = hru_geometry(_catchment(path, 2), 'HRU_ID')
        second = hru_geometry(_catchment(path, 5), 'HRU_ID')
        assert (first.count, second.count) == (2, 5)


class TestTokens:
    def test_geometry_token_round_trips(self, tmp_path):
        geom = hru_geometry(_catchment(tmp_path / 'c.shp', 12), 'HRU_ID')
        token = forcing_token('elevation,landclass', geom)
        assert token == f'elevation-landclass-h12-{geom.digest[:8]}'
        assert split_token(token) == ('elevation-landclass', 12, geom.digest[:8])
        assert discretization_key_from_name(_name(token)) == token

    def test_method_only_token_without_geometry(self):
        assert forcing_token('GRUs') == 'grus'
        assert split_token('grus') == ('grus', None, None)


class TestSelection:
    def test_picks_this_runs_geometry_among_several(self, tmp_path):
        g1 = hru_geometry(_catchment(tmp_path / 'a' / 'c.shp', 1), 'HRU_ID')
        g2 = hru_geometry(_catchment(tmp_path / 'b' / 'c.shp', 1, width=3.0), 'HRU_ID')
        mine = tmp_path / _name(forcing_token('grus', g1))
        other = tmp_path / _name(forcing_token('grus', g2))
        assert select_forcing_files([mine, other], 'GRUs', g1) == [mine]
        assert select_forcing_files([mine, other], 'GRUs', g2) == [other]

    def test_other_geometry_only_raises(self, tmp_path):
        g1 = hru_geometry(_catchment(tmp_path / 'a' / 'c.shp', 4), 'HRU_ID')
        g2 = hru_geometry(_catchment(tmp_path / 'b' / 'c.shp', 4, width=2.0), 'HRU_ID')
        other = tmp_path / _name(forcing_token('elevation', g2))
        with pytest.raises(FileOperationError, match='other HRU sets'):
            select_forcing_files([other], 'elevation', g1)

    def test_without_geometry_single_variant_is_used(self, tmp_path):
        g = hru_geometry(_catchment(tmp_path / 'c.shp', 2), 'HRU_ID')
        f = tmp_path / _name(forcing_token('elevation', g))
        lumped = tmp_path / _name('grus-h1-00000000')
        assert select_forcing_files([f, lumped], 'elevation') == [f]

    def test_without_geometry_two_variants_is_ambiguous(self):
        files = [Path(_name('grus-h1-aaaaaaaa')), Path(_name('grus-h1-bbbbbbbb'))]
        with pytest.raises(FileOperationError, match='several HRU sets'):
            select_forcing_files(files, 'GRUs')

    def test_geometry_files_win_over_legacy_method_names(self, tmp_path):
        g = hru_geometry(_catchment(tmp_path / 'c.shp', 1), 'HRU_ID')
        new = tmp_path / _name(forcing_token('grus', g))
        legacy = tmp_path / _name('grus')
        assert select_forcing_files([legacy, new], 'GRUs', g) == [new]

    def test_legacy_method_names_checked_against_hru_count(self, tmp_path):
        g = hru_geometry(_catchment(tmp_path / 'c.shp', 3), 'HRU_ID')
        ok = _forcing(tmp_path / 'ok' / _name('elevation'), 3)
        bad = _forcing(tmp_path / 'bad' / _name('elevation'), 5)
        assert select_forcing_files([ok], 'elevation', g) == [ok]
        with pytest.raises(FileOperationError, match='hru=5'):
            select_forcing_files([bad], 'elevation', g)

    def test_untokened_store_passes_through(self):
        files = [Path('dom_ERA5_remapped_2020-01-01-00-00-00.nc')]
        assert select_forcing_files(files, 'GRUs') == files

    def test_resolve_globs_directory_and_hashes_catchment(self, tmp_path):
        shp = _catchment(tmp_path / 'shp' / 'c.shp', 2)
        g = hru_geometry(shp, 'HRU_ID')
        store = tmp_path / 'store'
        mine = _forcing(store / _name(forcing_token('elevation', g)), 2)
        _forcing(store / _name('elevation-h2-ffffffff'), 2)
        assert resolve_forcing_files(store, 'elevation', shp, 'HRU_ID') == [mine]

    def test_resolve_without_catchment_falls_back_to_method(self, tmp_path):
        store = tmp_path / 'store'
        only = _forcing(store / _name('grus-h1-12345678'), 1)
        assert resolve_forcing_files(store, 'GRUs', tmp_path / 'missing.shp') == [only]


class TestWriterNames:
    def _fp(self, tmp_path):
        from symfluence.data.preprocessing.resampling.file_processor import FileProcessor
        cfg = {'DOMAIN_NAME': 'dom', 'FORCING_DATASET': 'ERA5', 'SUB_GRID_DISCRETIZATION': 'GRUs'}
        return FileProcessor(cfg, tmp_path, logging.getLogger('test'))

    def test_writer_uses_geometry_token(self, tmp_path):
        fp = self._fp(tmp_path)
        g = hru_geometry(_catchment(tmp_path / 'c.shp', 1), 'HRU_ID')
        src = Path('domain_dom_ERA5_merged_202001.nc')
        assert fp.determine_output_filename(src).name == 'dom_ERA5_remapped_grus_2020-01-01-00-00-00.nc'
        fp.target_geometry = g
        assert fp.determine_output_filename(src).name == (
            f'dom_ERA5_remapped_grus-h1-{g.short}_2020-01-01-00-00-00.nc')
        assert fp.determine_output_filename(src, token='').name == (
            'dom_ERA5_remapped_2020-01-01-00-00-00.nc')


class TestLegacyAdoption:
    """Existing outputs are renamed in place, not remapped again."""

    def _resampler(self, tmp_path, geometry):
        from types import SimpleNamespace

        from symfluence.data.preprocessing.forcing_resampler import ForcingResampler
        from symfluence.data.preprocessing.resampling.file_processor import FileProcessor
        cfg = {'DOMAIN_NAME': 'dom', 'FORCING_DATASET': 'ERA5', 'SUB_GRID_DISCRETIZATION': 'GRUs'}
        fp = FileProcessor(cfg, tmp_path, logging.getLogger('test'))
        fp.target_geometry = geometry
        fake = SimpleNamespace(file_processor=fp, logger=logging.getLogger('test'),
                               _get_config_value=fp._get_config_value)
        return lambda files, is_current: ForcingResampler._adopt_legacy_outputs(fake, files, is_current)

    def test_current_legacy_files_are_renamed(self, tmp_path):
        g = hru_geometry(_catchment(tmp_path / 'shp' / 'c.shp', 1), 'HRU_ID')
        adopt = self._resampler(tmp_path, g)
        jan, feb = Path('domain_dom_ERA5_merged_202001.nc'), Path('domain_dom_ERA5_merged_202002.nc')
        _forcing(tmp_path / 'dom_ERA5_remapped_grus_2020-01-01-00-00-00.nc', 1)
        _forcing(tmp_path / 'dom_ERA5_remapped_2020-02-01-00-00-00.nc', 1)
        assert adopt([jan, feb], lambda old: True) == 2
        assert sorted(p.name for p in tmp_path.glob('*.nc')) == [
            f'dom_ERA5_remapped_grus-h1-{g.short}_2020-01-01-00-00-00.nc',
            f'dom_ERA5_remapped_grus-h1-{g.short}_2020-02-01-00-00-00.nc',
        ]

    def test_stale_legacy_files_are_left_alone(self, tmp_path):
        g = hru_geometry(_catchment(tmp_path / 'shp' / 'c.shp', 1), 'HRU_ID')
        adopt = self._resampler(tmp_path, g)
        legacy = _forcing(tmp_path / 'dom_ERA5_remapped_grus_2020-01-01-00-00-00.nc', 1)
        assert adopt([Path('domain_dom_ERA5_merged_202001.nc')], lambda old: False) == 0
        assert legacy.exists()
