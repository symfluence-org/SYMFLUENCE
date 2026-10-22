# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regressions for monthly grids that shift without changing shape."""
from __future__ import annotations

import json
import logging
from unittest.mock import Mock

import numpy as np
import pytest
import shapely
import xarray as xr

from symfluence.data.preprocessing.resampling.grid_geometry import projected_cell_polygons
from symfluence.data.preprocessing.resampling.grid_identity import (
    ATTRIBUTE,
    grid_signature,
    manifest_path,
    output_matches,
    target_signature,
)
from symfluence.data.preprocessing.resampling.weight_applier import RemappingWeightApplier

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def write_grid(path, offset=0, reverse=False, projection='original'):
    lat = np.array([64., 65.])
    if reverse:
        lat = lat[::-1]
    ds = xr.Dataset(coords={'latitude': lat, 'longitude': np.array([-19., -18.])+offset},
                    attrs={'MAP_PROJ4_STR': projection})
    ds.to_netcdf(path)


def test_grid_changes_include_shift_order_projection_not_weather(tmp_path):
    keys = []
    for i, opts in enumerate([{}, {'offset': .01}, {'reverse': True}, {'projection': 'changed'}]):
        p = tmp_path/f'{i}.nc'
        write_grid(p, **opts)
        keys.append(grid_signature(p, ('latitude', 'longitude')))
    assert len(set(keys)) == 4
    write_grid(tmp_path/'same.nc')
    assert grid_signature(tmp_path/'same.nc', ('latitude', 'longitude')) == keys[0]


def test_wrong_grid_rejected_even_when_output_exists(tmp_path):
    source = tmp_path/'source.nc'
    write_grid(source)
    weights = tmp_path/'weights.csv'
    expected = grid_signature(source, ('latitude', 'longitude'))
    manifest_path(weights).write_text(json.dumps(dict(grid_signature=expected,
        signature='expected', coordinate_names=['latitude', 'longitude'])))
    write_grid(source, offset=.1)
    output = tmp_path/'existing.nc'
    output.write_bytes(b'previous validated output'*100)
    applier = RemappingWeightApplier({'DOMAIN_NAME': 'test'}, tmp_path, tmp_path, Mock(), logging.getLogger('test'))
    applier._configure_easymore = Mock(side_effect=AssertionError('must reject before EASYMORE'))
    assert not applier.apply_weights(source, weights, output)
    assert output.read_bytes() == b'previous validated output'*100
    applier._configure_easymore.assert_not_called()


def test_missing_or_stale_output_provenance_is_not_reused(tmp_path):
    p = tmp_path/'result.nc'
    xr.Dataset({'x': ('hru', [1.])}).to_netcdf(p)
    assert not output_matches(p, 'new')
    xr.Dataset({'x': ('hru', [1.])}, attrs={ATTRIBUTE: 'old'}).to_netcdf(p)
    assert not output_matches(p, 'new')
    assert output_matches(p, 'old')
    assert not output_matches(tmp_path/'missing.nc', 'old')


def test_target_attributes_invalidate_weights(tmp_path):
    p = tmp_path/'target.shp'
    p.write_bytes(b'geometry')
    p.with_suffix('.dbf').write_bytes(b'IDs and elevations')
    before = target_signature(p)
    p.with_suffix('.dbf').write_bytes(b'different IDs')
    assert before != target_signature(p)


def test_rotated_cells_have_shared_edges_and_conserve_area():
    x, y = np.meshgrid(np.arange(4)*2500., np.arange(3)*2500.)
    cells = projected_cell_polygons(.8*x-.6*y, .6*x+.8*y)
    np.testing.assert_allclose(shapely.area(cells), 2500**2)
    union = shapely.union_all(cells)
    assert union.area == pytest.approx(sum(shapely.area(cells)))
    assert union.geom_type == 'Polygon'
    with pytest.raises(ValueError):
        projected_cell_polygons(x[:1], y[:1])


def test_failed_replacement_cannot_succeed_using_old_output(tmp_path):
    temp = tmp_path/'scratch'
    temp.mkdir()
    output = tmp_path/'old.nc'
    output.write_bytes(b'keep this original')
    applier = RemappingWeightApplier({'DOMAIN_NAME': 'test'}, tmp_path, tmp_path, Mock())
    assert not applier._move_output_file(temp, output, tmp_path/'input.nc', '')
    assert output.read_bytes() == b'keep this original'
