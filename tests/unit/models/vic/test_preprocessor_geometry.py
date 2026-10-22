"""VIC must never silently use another watershed's geometry."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from symfluence.models.vic.preprocessor import VICPreProcessor


def test_geometry_uses_existing_basin_in_projected_crs(tmp_path):
    path = tmp_path / 'basin.gpkg'
    gpd.GeoDataFrame({'elev_mean': [750.]}, geometry=[box(-19.1,64.0,-19.0,64.1)],
                     crs=4326).to_crs(3057).to_file(path)
    pre = object.__new__(VICPreProcessor)
    pre._find_basin_shapefile = lambda required: path
    props = pre._get_catchment_properties()
    assert 64 < props['lat'] < 64.1
    assert -19.1 < props['lon'] < -19
    assert 4e7 < props['area_m2'] < 7e7
    assert props['elev'] == pytest.approx(750)


def test_missing_geometry_raises_instead_of_using_banff():
    pre = object.__new__(VICPreProcessor)
    def missing(required):
        assert required
        raise FileNotFoundError('No catchment')
    pre._find_basin_shapefile = missing
    with pytest.raises(FileNotFoundError):
        pre._get_catchment_properties()
