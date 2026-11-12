# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mHM lumped grid area, PET, basin-mean forcing and setup defaults.

- mRM discharge is runoff x grid area. The lumped grid was two fixed 0.35 deg
  cells (~1914 km2 at 51N) unless MHM_MATCH_LUMPED_AREA was switched on, so
  discharge was scaled by grid area / catchment area. The match is now the
  default.
- Hamon PET multiplied Lu et al.'s coefficient by (N/12)^2.
- Multi-HRU forcing was collapsed with an unweighted mean.
- A polygon without ``elev_mean`` silently became 1000 m.
- MHM_TEMP_LAPSE_RATE is dT/dz (negative = cooling), opposite to LAPSE_RATE.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from symfluence.core.exceptions import ConfigurationError, GeospatialError
from symfluence.core.mixins.config import SCHEMA_DEFAULT, fallback_default
from symfluence.core.modeling.mixins.pet_calculator import day_length_hours, hamon_pet
from symfluence.models.mhm.config_schema import MHMConfig
from symfluence.models.mhm.preprocessor import MHMPreProcessor


class _Pre(MHMPreProcessor):
    """Bypass __init__; exercise single methods."""

    def __init__(self, polygon=None, *, match=True, **cfg):
        self._polygon = polygon
        self._cfg = cfg
        self.config = SimpleNamespace(model=SimpleNamespace(mhm=SimpleNamespace(match_lumped_area=match)))

    def _get_config_value(self, accessor, default=SCHEMA_DEFAULT, dict_key=None):
        return self._cfg[dict_key] if dict_key in self._cfg else fallback_default(default, dict_key)

    def _catchment_polygon_path(self):
        return self._polygon


def _polygon(tmp_path, boxes, **columns):
    path = tmp_path / 'catchment.gpkg'
    gpd.GeoDataFrame(columns, geometry=[box(*b) for b in boxes], crs='EPSG:4326').to_file(path, driver='GPKG')
    return path


# --- lumped grid area -------------------------------------------------------

def test_match_lumped_area_is_the_default():
    assert MHMConfig().match_lumped_area is True
    assert MHMConfig(MHM_MATCH_LUMPED_AREA=False).match_lumped_area is False


@pytest.mark.parametrize('lat', [-80.0, -45.0, 0.0, 51.0, 64.0, 80.0])
@pytest.mark.parametrize('area_km2', [0.01, 1.0, 100.0, 3480.0, 100000.0])
def test_solved_cells_reproduce_the_catchment_area(lat, area_km2):
    cell = MHMPreProcessor._solve_lumped_cellsize(area_km2 * 1e6, lat)
    assert MHMPreProcessor._lumped_grid_area_m2(cell, lat) == pytest.approx(area_km2 * 1e6, rel=1e-6)
    assert abs(lat - cell) < 89.0


def test_unrepresentable_area_is_refused():
    with pytest.raises(ConfigurationError, match='lumped grid'):
        MHMPreProcessor._solve_lumped_cellsize(5e13, 64.0)   # 50 million km2
    with pytest.raises(ConfigurationError):
        MHMPreProcessor._solve_lumped_cellsize(0.0, 64.0)


def test_default_sizes_grid_to_catchment(tmp_path):
    pre = _Pre(_polygon(tmp_path, [(-19.5, 64.0, -19.0, 64.3)], elev_mean=[800.0]))
    area = pre._get_catchment_properties()['area_m2']
    pre._set_lumped_cellsize()
    lat = pre._get_catchment_properties()['lat']
    assert pre.LUMPED_CELLSIZE != 0.35
    assert pre._lumped_grid_area_m2(pre.LUMPED_CELLSIZE, lat) == pytest.approx(area, rel=1e-6)


def test_legacy_cells_warn_with_the_scale_factor(tmp_path, caplog):
    pre = _Pre(_polygon(tmp_path, [(-19.5, 64.0, -19.0, 64.3)], elev_mean=[800.0]), match=False)
    props = pre._get_catchment_properties()
    with caplog.at_level(logging.WARNING, logger='symfluence.models.mhm.preprocessor'):
        pre._set_lumped_cellsize()
    assert pre.LUMPED_CELLSIZE == 0.35
    factor = pre._lumped_grid_area_m2(0.35, props['lat']) / props['area_m2']
    assert factor > 1.5
    assert 'MHM_MATCH_LUMPED_AREA' in caplog.text
    assert f'{factor:.3f}' in caplog.text


# --- PET --------------------------------------------------------------------

def test_hamon_is_the_shared_lu_form_not_day_length_squared():
    pre = _Pre(MHM_PET_METHOD='hamon')
    assert float(pre._compute_pet(15.0, 80, 45.0)) == pytest.approx(2.12, rel=0.01)
    n = float(day_length_hours(64.0, 172))
    assert n > 20.0
    pet = float(pre._compute_pet(15.0, 172, 64.0))
    assert pet == pytest.approx(float(hamon_pet(15.0, n)), rel=1e-12)
    # The old copy returned lu * (N/12): 6.10 instead of 3.6 mm/day here.
    assert pet < 4.0


def test_oudin_numbers_are_unchanged():
    pre = _Pre(MHM_PET_METHOD='oudin')
    # Exact values of the implementation on develop before the Hamon fix.
    assert float(pre._compute_pet(15.0, 80, 45.0)) == pytest.approx(2.1442501989198646, rel=1e-12)
    assert float(pre._compute_pet(15.0, 172, 64.0)) == pytest.approx(3.377255180874739, rel=1e-12)
    assert float(pre._compute_pet(-10.0, 172, 64.0)) == 0.0


# --- basin-mean forcing -----------------------------------------------------

def _forcing(values, hru_ids=None):
    n, k = values.shape
    ds = xr.Dataset({'air_temperature': (('time', 'hru'), values)},
                    coords={'time': pd.date_range('2020-01-01', periods=n), 'hru': np.arange(k)})
    if hru_ids is not None:
        ds['hruId'] = (('hru',), np.asarray(hru_ids))
    return ds


def test_multi_hru_forcing_is_area_weighted_by_hru_id(tmp_path):
    # HRU 7 is three times the size of HRU 3; the forcing lists them 3, 7.
    polygon = _polygon(tmp_path, [(-19.0, 64.0, -18.7, 64.1), (-18.7, 64.0, -18.6, 64.1)],
                       HRU_ID=[7, 3])
    pre = _Pre(polygon)
    values = np.tile([[0.0, 4.0]], (5, 1))
    series = pre._basin_mean_series(_forcing(values, hru_ids=[3, 7]), 'air_temperature')
    np.testing.assert_allclose(series, 3.0, rtol=1e-3)   # unweighted mean would be 2.0


def test_multi_hru_forcing_without_ids_uses_file_order(tmp_path):
    polygon = _polygon(tmp_path, [(-19.0, 64.0, -18.7, 64.1), (-18.7, 64.0, -18.6, 64.1)],
                       HRU_ID=[1, 2])
    series = _Pre(polygon)._basin_mean_series(_forcing(np.tile([[4.0, 0.0]], (5, 1))), 'air_temperature')
    np.testing.assert_allclose(series, 3.0, rtol=1e-3)


def test_unmatched_hrus_are_refused(tmp_path):
    polygon = _polygon(tmp_path, [(-19.0, 64.0, -18.7, 64.1)], HRU_ID=[1])
    with pytest.raises(GeospatialError, match='not in'):
        _Pre(polygon)._basin_mean_series(_forcing(np.ones((3, 2)), hru_ids=[1, 9]), 'air_temperature')
    with pytest.raises(GeospatialError, match='2 HRUs'):
        _Pre(polygon)._basin_mean_series(_forcing(np.ones((3, 2))), 'air_temperature')


def test_single_hru_and_plain_series_pass_through():
    pre = _Pre()
    one = _forcing(np.arange(4.0).reshape(4, 1))
    np.testing.assert_array_equal(pre._basin_mean_series(one, 'air_temperature'), np.arange(4.0))
    plain = xr.Dataset({'air_temperature': ('time', np.arange(4.0))},
                       coords={'time': pd.date_range('2020-01-01', periods=4)})
    np.testing.assert_array_equal(pre._basin_mean_series(plain, 'air_temperature'), np.arange(4.0))


# --- elevation --------------------------------------------------------------

def test_missing_elevation_is_not_replaced_by_1000_m(tmp_path):
    pre = _Pre(_polygon(tmp_path, [(-19.5, 64.0, -19.0, 64.3)], HRU_ID=[1]))
    assert pre._get_catchment_properties()['elev'] is None
    pre.distributed_morph = False
    pre.morph_dir = tmp_path
    pre._get_simulation_dates = lambda: (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    with pytest.raises(GeospatialError, match='elev_mean'):
        pre._generate_morph_files()
    assert not (tmp_path / 'dem.asc').exists()


def test_elevation_is_read_from_the_polygon(tmp_path):
    pre = _Pre(_polygon(tmp_path, [(-19.5, 64.0, -19.0, 64.3)], elev_mean=[812.5]))
    assert pre._get_catchment_properties()['elev'] == 812.5


# --- lapse-rate sign convention ---------------------------------------------

def _grid():
    return {'nrow': 1, 'ncol': 2, 'mask': np.array([[True, True]]), 'dem': np.array([[500.0, 1500.0]])}


def test_negative_lapse_rate_cools_high_cells_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger='symfluence.models.mhm.preprocessor'):
        out = _Pre(MHM_TEMP_LAPSE_RATE=-6.5)._lapse_temperature(np.array([5.0]), _grid())
    np.testing.assert_allclose(out[0, 0], [8.25, 1.75])
    assert 'MHM_TEMP_LAPSE_RATE' not in caplog.text


def test_positive_lapse_rate_keeps_its_meaning_but_warns(caplog):
    with caplog.at_level(logging.WARNING, logger='symfluence.models.mhm.preprocessor'):
        out = _Pre(MHM_TEMP_LAPSE_RATE=6.5)._lapse_temperature(np.array([5.0]), _grid())
    np.testing.assert_allclose(out[0, 0], [1.75, 8.25])   # unchanged behaviour
    assert 'WARMER' in caplog.text
    assert 'negative' in MHMConfig.model_fields['temp_lapse_rate'].description.lower()
