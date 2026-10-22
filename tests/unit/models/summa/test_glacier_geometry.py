"""Multi-GRU input integrity and missing-thickness rejection."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.core.exceptions import ConfigurationError
from symfluence.models.summa.glacier_geometry import prepare_glacier_geometry

pytestmark = [pytest.mark.unit, pytest.mark.quick]


@pytest.fixture
def geometry_inputs(tmp_path):
    attrs = xr.Dataset({
        'hruId': ('hru', [11, 23]), 'gruId': ('gru', [6, 34]),
        'hru2gruId': ('hru', [6, 34]), 'HRUarea': ('hru', [40000., 40000.]),
        'elevation': ('hru', [1000., 1100.]),
    })
    attrs.to_netcdf(tmp_path/'attributes.nc')
    base = xr.Dataset({'hruId': ('hru', [11, 23])})
    for name, value in [('nSnow', 0), ('nSoil', 3), ('scalarAquiferStorage', 1.)]:
        base[name] = (('scalarv', 'hru'), np.full((1, 2), value))
    for name, value in [('mLayerTemp', 280.), ('mLayerVolFracLiq', .2), ('mLayerVolFracIce', 0.)]:
        base[name] = (('midToto', 'hru'), np.full((3, 2), value))
    base['mLayerMatricHead'] = (('midToto', 'hru'), np.full((3, 2), -1.))
    base['iLayerHeight'] = (('ifcToto', 'hru'), np.tile([0., .1, .5, 2.], (2, 1)).T)
    base.to_netcdf(tmp_path/'cold.nc')
    hrus = gpd.GeoDataFrame({'HRU_ID': [11, 23]}, geometry=[box(500000, 7100000, 500200, 7100200),
        box(500400, 7100000, 500600, 7100200)], crs='EPSG:32627')
    hrus.to_file(tmp_path/'hrus.gpkg')
    ice = gpd.GeoDataFrame(geometry=[box(500000, 7100000, 500100, 7100200),
                                   box(500400, 7100000, 500500, 7100200)], crs=hrus.crs)
    ice.to_file(tmp_path/'ice.gpkg')
    for name, value in [('dem.tif', 1000.), ('ice_thickness.tif', 80.)]:
        with rasterio.open(tmp_path/name, 'w', driver='GTiff', width=6, height=2,
                           count=1, dtype='float32', crs=hrus.crs,
                           transform=from_origin(500000, 7100200, 100, 100)) as dst:
            dst.write(np.full((2, 6), value, dtype='float32'), 1)
    return (tmp_path/'attributes.nc', tmp_path/'cold.nc', tmp_path/'hrus.gpkg',
            tmp_path/'dem.tif', tmp_path/'ice.gpkg', tmp_path)


def test_two_grus_keep_ids_area_layers_and_independent_storage(geometry_inputs):
    report = prepare_glacier_geometry(*geometry_inputs)
    assert report['glacier_grus'] == 2
    with xr.open_dataset(geometry_inputs[0]) as attrs, xr.open_dataset(geometry_inputs[1]) as cold:
        np.testing.assert_equal(attrs.gridId.values, [[6, 34]])
        np.testing.assert_allclose(cold.DOMarea.sum('dom').values, [[40000., 40000.]])
        np.testing.assert_equal(cold.nSoil.values[0, :, 0], [3, 3])
        assert cold.mLayerMatricHead.dims == ('midSoil', 'hru', 'dom')
        assert cold.sizes['midSoil'] == 3
        np.testing.assert_equal(cold.nGlce.values[0, :, 1], [5, 5])
        np.testing.assert_allclose(cold.mLayerVolFracIce.values[:5, :, 1]
                                   + cold.mLayerVolFracLiq.values[:5, :, 1], 1.)
        assert (cold.basin__GlacierStorage.values > 0).all()
        np.testing.assert_allclose(cold.basin__GlacierStorage.values[0],
                                   cold.basin__GlacierStorage.values[1])
        np.testing.assert_allclose(cold.surface_elev.values - attrs.bed_elev.values,
                                   attrs.glacierMask.values * 80.)


def test_missing_thickness_fails_instead_of_inventing_ice(geometry_inputs):
    (geometry_inputs[-1]/'ice_thickness.tif').unlink()
    with pytest.raises(ConfigurationError, match='No thickness'):
        prepare_glacier_geometry(*geometry_inputs)


def test_target_area_drops_thinnest_cells_first(geometry_inputs):
    # Ice cells: columns 0 (HRU 11) and 4 (HRU 23), two rows each; make HRU 23's ice thinner.
    thickness = np.full((2, 6), 80., dtype='float32')
    thickness[:, 4] = [10., 20.]
    with rasterio.open(geometry_inputs[-1]/'ice_thickness.tif', 'r+') as dst:
        dst.write(thickness, 1)
    report = prepare_glacier_geometry(*geometry_inputs, target_area_km2=0.03)
    assert report['glacier_area_km2'] == pytest.approx(0.03)
    assert report['trimmed_area_km2'] == pytest.approx(0.01)
    with xr.open_dataset(geometry_inputs[1]) as cold, xr.open_dataset(geometry_inputs[0]) as attrs:
        np.testing.assert_allclose(cold.DOMarea.values[0, :, 1], [20000., 10000.])
        np.testing.assert_allclose(cold.DOMarea.sum('dom').values, [[40000., 40000.]])
        assert attrs.glacierMask.values.sum() == 3
        # Remaining thin cell is the 20 m one: 80+80+20 m of ice at 100 m cells.
        assert float(cold.basin__GlacierStorage.sum()) == pytest.approx(180*1e4*917/1e12, rel=1e-3)
