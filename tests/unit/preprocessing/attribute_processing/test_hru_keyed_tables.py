# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-HRU attribute tables are keyed and joined by HRU id, not row position."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip('geopandas')
netCDF4 = pytest.importorskip('netCDF4')
xr = pytest.importorskip('xarray')

from shapely.geometry import box

from symfluence.core.exceptions import ValidationError
from symfluence.data.model_ready.attributes_builder import AttributesNetCDFBuilder
from symfluence.data.model_ready.hru_tables import align_table_to_hru_ids, normalize_hru_id, order_hru_ids
from symfluence.data.preprocessing.attribute_processors.base import BaseAttributeProcessor
from symfluence.models.summa.calibration.summa_regionalization import load_hru_attributes

LOG = logging.getLogger('test_hru_keyed_tables')
N_HRU = 12


class _Proc(BaseAttributeProcessor):
    """Processor with the heavy __init__ bypassed."""

    def __init__(self, hru_ids=None):
        self.logger = LOG
        self._ids = hru_ids

    def _get_hru_ids(self):
        return self._ids


def _write_processor_csv(out_dir: Path, filename: str, ids, column: str = 'soil.ksat') -> Path:
    # value == 100 * id so a misjoin is detectable
    results = {f'HRU_{i}_{column}': 100.0 * i for i in ids}
    path = _Proc(list(ids))._write_results_csv(results, out_dir, filename)
    assert path is not None
    return path


def _summa_attributes(path: Path, hru_ids) -> None:
    ds = xr.Dataset(
        {
            'hruId': ('hru', np.asarray(hru_ids, dtype='int64')),
            'elevation': ('hru', np.asarray(hru_ids, dtype='float64')),
            'HRUarea': ('hru', np.ones(len(hru_ids))),
            'latitude': ('hru', np.full(len(hru_ids), 64.0)),
            'longitude': ('hru', np.full(len(hru_ids), -19.0)),
        }
    )
    ds.to_netcdf(path)


def test_writer_adds_id_column_in_numeric_order(tmp_path):
    path = _write_processor_csv(tmp_path, 'soil_extended_attributes.csv', range(1, N_HRU + 1))
    df = pd.read_csv(path)
    assert df.columns[0] == 'hru_id'
    assert df['hru_id'].tolist() == list(range(1, N_HRU + 1))
    assert df['soil.ksat'].tolist() == [100.0 * i for i in range(1, N_HRU + 1)]


def test_writer_non_numeric_ids_follow_catchment_order(tmp_path):
    ids = ['b', 'a', 'c']
    results = {f'HRU_{i}_x': float(k) for k, i in enumerate(ids)}
    path = _Proc(ids)._write_results_csv(results, tmp_path, 't.csv')
    df = pd.read_csv(path)
    assert df['hru_id'].tolist() == ids


def test_order_and_normalize_helpers():
    assert normalize_hru_id(12.0) == '12'
    assert normalize_hru_id(' 12 ') == '12'
    assert order_hru_ids(['10', '2', '1']) == ['1', '2', '10']


def test_regionalization_joins_profile_csv_on_hru_id(tmp_path):
    # attributes.nc HRU order is NOT sorted
    nc_ids = [3, 12, 1, 10, 2, 11, 4, 9, 5, 8, 6, 7]
    attr_dir = tmp_path / 'attributes'
    nc_path = tmp_path / 'attributes.nc'
    _summa_attributes(nc_path, nc_ids)
    _write_processor_csv(attr_dir / 'soilclass', 'soil_extended_attributes.csv', range(1, N_HRU + 1))

    # csv_path only anchors the attribute base directory (csv_path.parent.parent)
    df = load_hru_attributes(nc_path, csv_path=attr_dir / 'tf' / 'missing.csv', logger=LOG)
    assert df['soil.ksat'].tolist() == [100.0 * i for i in nc_ids]


def test_regionalization_rejects_legacy_csv_without_ids(tmp_path):
    attr_dir = tmp_path / 'attributes'
    nc_path = tmp_path / 'attributes.nc'
    _summa_attributes(nc_path, list(range(1, N_HRU + 1)))
    legacy = attr_dir / 'soilclass' / 'soil_extended_attributes.csv'
    legacy.parent.mkdir(parents=True)
    # the old writer: string-sorted rows, no id column
    order = sorted(str(i) for i in range(1, N_HRU + 1))
    pd.DataFrame({'soil.ksat': [100.0 * int(i) for i in order]}).to_csv(legacy, index=False)

    with pytest.raises(ValidationError, match='regenerate'):
        load_hru_attributes(nc_path, csv_path=attr_dir / 'tf' / 'missing.csv', logger=LOG)


def test_regionalization_small_legacy_csv_is_mapped_by_text_order(tmp_path):
    nc_ids = [3, 1, 2]
    attr_dir = tmp_path / 'attributes'
    nc_path = tmp_path / 'attributes.nc'
    _summa_attributes(nc_path, nc_ids)
    legacy = attr_dir / 'soilclass' / 'soil_extended_attributes.csv'
    legacy.parent.mkdir(parents=True)
    pd.DataFrame({'soil.ksat': [100.0, 200.0, 300.0]}).to_csv(legacy, index=False)

    df = load_hru_attributes(nc_path, csv_path=attr_dir / 'tf' / 'missing.csv', logger=LOG)
    assert df['soil.ksat'].tolist() == [300.0, 100.0, 200.0]


def test_align_reports_missing_and_foreign_ids():
    table = pd.DataFrame({'hru_id': [1, 2], 'v': [10.0, 20.0]})
    out = align_table_to_hru_ids(table, [2, 3, 1], 't')
    assert out is not None
    assert out['v'].tolist()[0] == 20.0 and np.isnan(out['v'].tolist()[1]) and out['v'].tolist()[2] == 10.0
    assert align_table_to_hru_ids(table, [7, 8], 't') is None
    with pytest.raises(ValidationError, match='duplicate'):
        align_table_to_hru_ids(pd.DataFrame({'hru_id': [1, 1], 'v': [1.0, 2.0]}), [1], 't')


def _builder_project(tmp_path: Path, hru_ids) -> AttributesNetCDFBuilder:
    shp = tmp_path / 'shapefiles' / 'catchment' / 'dom_HRUs_elevation.shp'
    shp.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {'HRU_ID': list(hru_ids), 'geometry': [box(i, 0, i + 1, 1) for i in range(len(hru_ids))]},
        crs='EPSG:4326',
    ).to_file(shp)
    return AttributesNetCDFBuilder(project_dir=tmp_path, domain_name='dom')


def test_builder_csv_groups_carry_hru_ids_in_catchment_order(tmp_path):
    shp_ids = [3, 12, 1, 10, 2, 11, 4, 9, 5, 8, 6, 7]
    builder = _builder_project(tmp_path, shp_ids)
    attrs = tmp_path / 'data' / 'attributes'
    _write_processor_csv(attrs / 'soilclass', 'soil_extended_attributes.csv', range(1, N_HRU + 1))
    _write_processor_csv(attrs / 'climate', 'worldclim_attributes.csv', range(1, N_HRU + 1), 'climate.prec')
    _write_processor_csv(attrs / 'geology', 'glhymps_attributes.csv', range(1, N_HRU + 1), 'geology.perm')

    out = builder.build()
    assert out is not None
    with netCDF4.Dataset(str(out)) as root:
        for group, var in [('soil_extended', 'soil.ksat'), ('climate', 'climate.prec'),
                           ('hydrogeology', 'geology.perm')]:
            grp = root.groups[group]
            ids = [str(v) for v in grp.variables['hru_id'][:]]
            assert ids == [str(i) for i in shp_ids]
            values = np.asarray(grp.variables[var][:])
            np.testing.assert_allclose(values, [100.0 * i for i in shp_ids])


def test_builder_joins_two_csvs_of_one_group_on_id(tmp_path):
    builder = _builder_project(tmp_path, [1, 2, 3])
    soil_dir = tmp_path / 'data' / 'attributes' / 'soilclass'
    soil_dir.mkdir(parents=True)
    pd.DataFrame({'hru_id': [1, 2, 3], 'a': [1.0, 2.0, 3.0]}).to_csv(soil_dir / 'a_attributes.csv', index=False)
    pd.DataFrame({'hru_id': [3, 1, 2], 'b': [30.0, 10.0, 20.0]}).to_csv(soil_dir / 'b_attributes.csv', index=False)
    out = builder.build()
    with netCDF4.Dataset(str(out)) as root:
        grp = root.groups['soil_extended']
        np.testing.assert_allclose(grp.variables['a'][:], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(grp.variables['b'][:], [10.0, 20.0, 30.0])


def test_builder_rejects_legacy_processor_csv(tmp_path):
    builder = _builder_project(tmp_path, list(range(1, N_HRU + 1)))
    climate = tmp_path / 'data' / 'attributes' / 'climate'
    climate.mkdir(parents=True)
    pd.DataFrame({'climate.prec': np.arange(N_HRU, dtype=float)}).to_csv(
        climate / 'worldclim_attributes.csv', index=False)
    with pytest.raises(ValidationError, match='regenerate'):
        builder.build()


# --- catchment shapefile resolution (combined discretization) ----------------

def _resolve(tmp_path: Path, discretization: str) -> Path:
    proc = _Proc()
    proc.domain_name = 'dom'
    proc.project_dir = tmp_path
    values = {
        'CATCHMENT_PATH': 'default', 'CATCHMENT_SHP_NAME': 'default',
        'SUB_GRID_DISCRETIZATION': discretization,
        'DOMAIN_DEFINITION_METHOD': 'semidistributed', 'EXPERIMENT_ID': 'run_1',
    }
    proc._get_config_value = lambda typed, default=None, dict_key=None: values.get(dict_key, default)
    return proc._get_catchment_path()


def test_combined_discretization_resolves_underscore_name(tmp_path):
    d = tmp_path / 'shapefiles' / 'catchment' / 'semidistributed' / 'run_1'
    d.mkdir(parents=True)
    (d / 'dom_HRUs_GRUs.shp').touch()
    wanted = d / 'dom_HRUs_elevation_landclass.shp'
    wanted.touch()
    assert _resolve(tmp_path, 'elevation,landclass') == wanted


def test_missing_catchment_raises_instead_of_using_another_hru_file(tmp_path):
    d = tmp_path / 'shapefiles' / 'catchment' / 'semidistributed' / 'run_1'
    d.mkdir(parents=True)
    (d / 'dom_HRUs_GRUs.shp').touch()
    with pytest.raises(FileNotFoundError, match='dom_HRUs_elevation_landclass.shp'):
        _resolve(tmp_path, 'elevation,landclass')
