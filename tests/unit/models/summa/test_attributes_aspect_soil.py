"""SUMMA attributes: downslope aspect, soil fallback class, missing elevation.

SUMMA's ``aspect`` is the azimuth the slope faces, in degrees east of north
(``clrsky_rad``: 0 = north-facing, 90 = east-facing). The attributes manager
used to write the bearing of the uphill direction (180 degrees off) and to
average degrees arithmetically across the 0/360 wrap.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

nc4 = pytest.importorskip("netCDF4")
gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("rasterstats")

from rasterio.transform import from_origin
from shapely.geometry import box

import symfluence.models.summa as summa_package
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.mixins.config import SCHEMA_DEFAULT, fallback_default
from symfluence.models.summa.attributes_manager import (
    FLAT_ASPECT_DEG,
    SummaAttributesManager,
    aspect_from_components,
    downslope_components,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]

BASE_SETTINGS = Path(summa_package.__file__).parent / 'base_settings'
NY = NX = 50
ROWS = np.arange(NY)[:, None] * np.ones((1, NX))   # increases southward
COLS = np.ones((NY, 1)) * np.arange(NX)[None, :]   # increases eastward


def _manager(tmp_path: Path) -> SummaAttributesManager:
    mgr = SummaAttributesManager.__new__(SummaAttributesManager)
    mgr.logger = logging.getLogger('test_summa_attributes')
    mgr.project_dir = tmp_path
    mgr.setup_dir = tmp_path / 'settings'
    mgr.setup_dir.mkdir(exist_ok=True)
    mgr._get_config_value = lambda accessor=None, default=SCHEMA_DEFAULT, dict_key=None, **kwargs: (
        fallback_default(default, dict_key) or 'HRU_ID')
    mgr._get_default_path = lambda key, default: tmp_path / default
    return mgr


def _aspect(tmp_path: Path, dem: np.ndarray, crs: str = 'EPSG:32627', nodata=None) -> float:
    if crs == 'EPSG:4326':
        transform, hru = from_origin(-20.0, 65.0, 0.001, 0.001), box(-19.99, 64.96, -19.96, 64.99)
    else:
        transform, hru = from_origin(500000, 7100000, 100, 100), box(500500, 7095500, 504500, 7099500)
    mgr = _manager(tmp_path)
    mgr.dem_path = tmp_path / 'dem.tif'
    with rasterio.open(mgr.dem_path, 'w', driver='GTiff', height=NY, width=NX, count=1, dtype='float32',
                       crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(dem.astype('float32'), 1)
    shp = gpd.GeoDataFrame({'HRU_ID': [1]}, geometry=[hru], crs=crs)
    return mgr._calculate_aspect_from_dem(shp)[1]


def _circular_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# aspect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dem, facing', [
    ((NY - ROWS) * 10.0, 180.0),          # rises to the north -> faces south
    (ROWS * 10.0, 0.0),                   # rises to the south -> faces north
    (COLS * 10.0, 270.0),                 # rises to the east  -> faces west
    ((NX - COLS) * 10.0, 90.0),           # rises to the west  -> faces east
    ((COLS + ROWS) * 10.0, 315.0),        # rises to the south-east -> faces north-west
])
def test_aspect_is_the_direction_the_slope_faces(tmp_path, dem, facing):
    assert _circular_distance(_aspect(tmp_path, dem), facing) < 1e-6


def test_aspect_in_geographic_crs(tmp_path):
    assert _circular_distance(_aspect(tmp_path, (NY - ROWS) * 10.0, crs='EPSG:4326'), 180.0) < 1e-6


def test_north_facing_slope_is_not_averaged_across_the_wrap(tmp_path):
    """Cell aspects straddle 0/360; their arithmetic mean would be ~180 (south)."""
    rng = np.random.default_rng(0)
    aspect = _aspect(tmp_path, ROWS * 10.0 + rng.normal(0, 3, (NY, NX)))
    assert _circular_distance(aspect, 0.0) < 5.0


def test_flat_cells_do_not_pull_the_mean(tmp_path):
    """Half the HRU is flat; the aspect is that of the sloping half."""
    dem = np.where(COLS < NX / 2, 0.0, (COLS - NX / 2) * 10.0)   # east half rises eastward
    assert _circular_distance(_aspect(tmp_path, dem), 270.0) < 1e-6


def test_flat_hru_gets_the_defined_flat_value(tmp_path):
    assert _aspect(tmp_path, np.full((NY, NX), 500.0)) == FLAT_ASPECT_DEG


def test_nodata_cells_do_not_contaminate_the_aspect(tmp_path):
    dem = (NY - ROWS) * 10.0
    dem[20:25, 20:25] = -9999.0
    assert _circular_distance(_aspect(tmp_path, dem, nodata=-9999.0), 180.0) < 1e-6


def test_downslope_components_and_bearing():
    east, north = downslope_components((NY - ROWS) * 10.0, 100.0, 100.0)
    np.testing.assert_allclose(east, 0.0, atol=1e-12)
    np.testing.assert_allclose(north, -0.1)          # falls southward at 10 m per 100 m
    assert aspect_from_components(0.0, -0.1) == pytest.approx(180.0)
    assert aspect_from_components(0.1, 0.0) == pytest.approx(90.0)
    assert aspect_from_components(0.0, 0.0) is None
    # south-up raster: the same array now rises to the south
    _, north_flipped = downslope_components((NY - ROWS) * 10.0, 100.0, 100.0, rows_southward=False)
    np.testing.assert_allclose(north_flipped, 0.1)


# ---------------------------------------------------------------------------
# soil fallback class
# ---------------------------------------------------------------------------

def _soil_row(table_file: Path, table: str, index: int) -> str:
    lines = table_file.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == table)
    return next(row for row in lines[start + 2:] if row.replace(',', ' ').split()[0] == str(index))


def test_fallback_soil_class_is_loam_in_the_shipped_table(tmp_path):
    mgr = _manager(tmp_path)
    for name in ('modelDecisions.txt', 'TBL_SOILPARM.TBL'):
        shutil.copy(BASE_SETTINGS / name, mgr.setup_dir / name)

    loam, table = mgr._loam_soil_class()

    assert (loam, table) == (3, 'ROSETTA')
    assert "'LOAM'" in _soil_row(mgr.setup_dir / 'TBL_SOILPARM.TBL', table, loam)
    # the previous hard-coded fallback
    assert "'SANDY CLAY'" in _soil_row(mgr.setup_dir / 'TBL_SOILPARM.TBL', table, 6)


@pytest.mark.parametrize('table, expected', [('STAS', 6), ('STAS-RUC', 6), ('ROSETTA', 3)])
def test_fallback_soil_class_follows_the_configured_table(tmp_path, table, expected):
    mgr = _manager(tmp_path)
    shutil.copy(BASE_SETTINGS / 'TBL_SOILPARM.TBL', mgr.setup_dir / 'TBL_SOILPARM.TBL')
    (mgr.setup_dir / 'modelDecisions.txt').write_text(f"soilCatTbl                      {table}         ! (03)\n")

    loam, found = mgr._loam_soil_class()

    assert (loam, found) == (expected, table)
    assert "'LOAM'" in _soil_row(mgr.setup_dir / 'TBL_SOILPARM.TBL', table, loam)


def test_fallback_soil_class_without_settings_files(tmp_path):
    assert _manager(tmp_path)._loam_soil_class() == (3, 'ROSETTA')


def _attributes(path: Path, hru_ids) -> Path:
    with nc4.Dataset(path, 'w') as att:
        att.createDimension('hru', len(hru_ids))
        att.createVariable('hruId', 'i4', 'hru')[:] = hru_ids
        att.createVariable('hru2gruId', 'i4', 'hru')[:] = hru_ids
        att.createVariable('soilTypeIndex', 'i4', 'hru')[:] = -999
        att.createVariable('elevation', 'f8', 'hru')[:] = -999
        att.createVariable('downHRUindex', 'i4', 'hru')[:] = 0
    return path


def test_missing_soil_class_falls_back_to_loam_with_a_counted_warning(tmp_path, caplog):
    mgr = _manager(tmp_path)
    for name in ('modelDecisions.txt', 'TBL_SOILPARM.TBL'):
        shutil.copy(BASE_SETTINGS / name, mgr.setup_dir / name)
    soil_dir = tmp_path / 'shapefiles' / 'catchment_intersection' / 'with_soilgrids'
    soil_dir.mkdir(parents=True)
    gpd.GeoDataFrame({'HRU_ID': [1], 'USGS_0': [0.1], 'USGS_5': [0.9]},
                     geometry=[box(0, 0, 1, 1)], crs='EPSG:4326').to_file(soil_dir / 'catchment_with_soilclass.shp')
    attributes = _attributes(tmp_path / 'attributes.nc', [1, 2, 3])

    with caplog.at_level(logging.WARNING, logger='test_summa_attributes'):
        mgr.insert_soil_class(attributes)

    with nc4.Dataset(attributes) as att:
        assert list(att['soilTypeIndex'][:]) == [5, 3, 3]
    assert any('2 of 3 HRUs' in record.getMessage() and 'LOAM' in record.getMessage()
               for record in caplog.records)


# ---------------------------------------------------------------------------
# elevation
# ---------------------------------------------------------------------------

def _dem_intersection(tmp_path: Path, hru_ids, elevations) -> None:
    dem_dir = tmp_path / 'shapefiles' / 'catchment_intersection' / 'with_dem'
    dem_dir.mkdir(parents=True)
    gpd.GeoDataFrame({'HRU_ID': hru_ids, 'elev_mean': elevations},
                     geometry=[box(i, 0, i + 1, 1) for i in range(len(hru_ids))],
                     crs='EPSG:4326').to_file(dem_dir / 'catchment_with_dem.shp')


def test_missing_elevation_raises_instead_of_leaving_the_placeholder(tmp_path):
    mgr = _manager(tmp_path)
    _dem_intersection(tmp_path, [1, 2], [900.0, 950.0])
    attributes = _attributes(tmp_path / 'attributes.nc', [1, 2, 3])

    with pytest.raises(ModelExecutionError, match='No elevation found for 1 of 3 HRUs'):
        mgr.insert_elevation(attributes)


def test_elevation_is_written_when_every_hru_has_one(tmp_path):
    mgr = _manager(tmp_path)
    _dem_intersection(tmp_path, [1, 2, 3], [900.0, 950.0, 1000.0])
    attributes = _attributes(tmp_path / 'attributes.nc', [1, 2, 3])

    mgr.insert_elevation(attributes)

    with nc4.Dataset(attributes) as att:
        np.testing.assert_allclose(att['elevation'][:], [900.0, 950.0, 1000.0])
