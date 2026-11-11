"""NGEN config generator: catchment area units and land-surface defaults."""
from __future__ import annotations

import logging
import re

import geopandas as gpd
import pytest
from shapely.geometry import box

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.ngen.config_generator import NgenConfigGenerator


def _generator(tmp_path, ngen_config):
    for sub in ('CFE', 'PET', 'NOAH'):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return NgenConfigGenerator(ngen_config, logging.getLogger('test_ngen_area'), tmp_path)


def _row(side_m=1000.0, **columns):
    """One square catchment (side_m × side_m) in a projected CRS."""
    gdf = gpd.GeoDataFrame(
        {k: [v] for k, v in columns.items()},
        geometry=[box(500000, 7100000, 500000 + side_m, 7100000 + side_m)],
        crs='EPSG:32627',
    )
    return gdf, gdf.iloc[0]


def _area(gen, gdf, row):
    gen.catchment_crs = gdf.crs
    return gen._catchment_area_km2('1', row)


@pytest.mark.parametrize('column', ['HRU_area', 'GRU_area'])
@pytest.mark.parametrize('area_m2', [800000.0, 5.0e6, 1.0e4])
def test_square_metre_columns_always_converted(tmp_path, ngen_config, column, area_m2):
    """A 0.8 km² catchment (800000 m²) is not 800000 km²."""
    gen = _generator(tmp_path, ngen_config)
    assert _area(gen, *_row(**{column: area_m2})) == pytest.approx(area_m2 / 1e6)


def test_cfe_config_gets_small_catchment_area_in_km2(tmp_path, ngen_config):
    gen = _generator(tmp_path, ngen_config)
    gdf, row = _row(HRU_area=800000.0)
    gen.catchment_crs = gdf.crs
    text = gen.generate_cfe_config('1', row).read_text()
    area = float(re.search(r'catchment_area_km2=([0-9.eE+-]+)\[km2\]', text).group(1))
    assert area == pytest.approx(0.8)


def test_km2_columns_used_as_is(tmp_path, ngen_config):
    gen = _generator(tmp_path, ngen_config)
    assert _area(gen, *_row(areasqkm=0.8, HRU_area=800000.0)) == pytest.approx(0.8)


@pytest.mark.parametrize('value', [1.0e6, 1.0])
def test_area_column_unit_resolved_against_geometry(tmp_path, ngen_config, value):
    """AREA of a 1 km² square is 1 km² whether stored as 1e6 (m²) or 1 (km²)."""
    gen = _generator(tmp_path, ngen_config)
    assert _area(gen, *_row(AREA=value)) == pytest.approx(1.0)


def test_area_column_disagreeing_with_geometry_raises(tmp_path, ngen_config):
    gen = _generator(tmp_path, ngen_config)
    with pytest.raises(ModelExecutionError, match='neither as m² nor as km²'):
        _area(gen, *_row(AREA=500.0))


def test_area_column_without_geometry(tmp_path, ngen_config):
    gen = _generator(tmp_path, ngen_config)
    gdf, row = _row(AREA=5.0e7)
    assert gen._catchment_area_km2('1', row.drop('geometry')) == pytest.approx(50.0)
    gdf, row = _row(AREA=800000.0)
    with pytest.raises(ModelExecutionError, match='has no unit'):
        gen._catchment_area_km2('1', row.drop('geometry'))


def test_geometry_used_without_area_columns(tmp_path, ngen_config):
    gen = _generator(tmp_path, ngen_config)
    assert _area(gen, *_row(side_m=2000.0)) == pytest.approx(4.0)


def test_pet_vegetation_default_does_not_depend_on_elevation(tmp_path, ngen_config, caplog):
    gen = _generator(tmp_path, ngen_config)
    texts = []
    with caplog.at_level(logging.WARNING, logger='test_ngen_area'):
        for cat_id, elev in (('1', 200.0), ('2', 1800.0)):
            gdf, row = _row(elev_mean=elev)
            gen.catchment_crs = gdf.crs
            texts.append(gen.generate_pet_config(cat_id, row).read_text())
    for text in texts:
        assert 'vegetation_height_m=0.12' in text
    # One warning for the whole run, not one per catchment.
    assert sum('reference-grass' in r.getMessage() for r in caplog.records) == 1


def test_noah_uniform_class_defaults_warn_once(tmp_path, ngen_config, caplog):
    gen = _generator(tmp_path, ngen_config)
    (tmp_path / 'NOAH' / 'parameters').mkdir(parents=True, exist_ok=True)
    with caplog.at_level(logging.WARNING, logger='test_ngen_area'):
        for cat_id in ('1', '2'):
            gdf, row = _row()
            gen.catchment_crs = gdf.crs
            gen.generate_noah_config(cat_id, row)
    assert sum('isltyp=3' in r.getMessage() for r in caplog.records) == 1
