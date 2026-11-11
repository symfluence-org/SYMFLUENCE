# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GeospatialStatistics: sub-pixel HRUs get data; cached outputs are validated."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

gpd = pytest.importorskip('geopandas')
rasterio = pytest.importorskip('rasterio')
pytest.importorskip('rasterstats')

from rasterio.transform import from_origin
from shapely.geometry import box

from symfluence.data.preprocessing.geospatial_statistics import GeospatialStatistics

X0, Y0 = 500000.0, 7100000.0
CRS = 'EPSG:32627'


class _Stats(GeospatialStatistics):
    """GeospatialStatistics with config/path plumbing replaced by plain attributes."""

    config = None  # type: ignore[assignment]
    project_attributes_dir = None  # type: ignore[assignment]

    def __init__(self, tmp_path: Path):  # noqa: D107 - deliberately skips the heavy base init
        self.logger = MagicMock()
        self.config = SimpleNamespace(
            domain=SimpleNamespace(name='dom', land_class_name='default'),
            paths=SimpleNamespace(
                intersect_land_name='default', intersect_soil_name='default',
                intersect_dem_name='default', soil_class_name='default',
                catchment_hruid='HRU_ID', catchment_gruid='GRU_ID',
            ),
            data=SimpleNamespace(),
        )
        self.paths = MagicMock()
        self.paths.resolve.side_effect = lambda key, default: tmp_path / default
        self.catchment_path = tmp_path / 'catchment'
        self.catchment_name = 'dom_HRUs_elevation.shp'
        self.dem_path = tmp_path / 'dem.tif'
        self.soil_path = tmp_path / 'soil'
        self.land_path = tmp_path / 'land'
        self.project_attributes_dir = tmp_path / 'attributes'
        self._catchment_signature_cache = None
        for d in (self.catchment_path, self.soil_path, self.land_path):
            d.mkdir(parents=True, exist_ok=True)

    def _get_config_value(self, typed, default=None, **_):
        try:
            value = typed()
        except AttributeError:
            return default
        return default if value is None else value


def _stats(tmp_path: Path) -> GeospatialStatistics:
    return _Stats(tmp_path)


def _raster(path, data, dtype, nodata):
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1,
                       dtype=dtype, crs=CRS, transform=from_origin(X0, Y0, 100.0, 100.0),
                       nodata=nodata) as dst:
        dst.write(data.astype(dtype), 1)


def _catchment(gs, hru_ids=(1, 2, 3)):
    gdf = gpd.GeoDataFrame(
        {'HRU_ID': list(hru_ids), 'GRU_ID': [1, 1, 1]},
        geometry=[
            box(X0, Y0 - 2000, X0 + 1000, Y0),                         # class 1 half
            box(X0 + 1000, Y0 - 2000, X0 + 2000, Y0 - 100),            # class 2 half
            box(X0 + 1205, Y0 - 1020 + 1000, X0 + 1225, Y0),           # sliver: no pixel centre
        ],
        crs=CRS,
    )
    gdf.to_file(gs.catchment_path / gs.catchment_name)
    return gdf


def test_land_fractions_of_sub_pixel_hru_are_not_all_zero(tmp_path):
    gs = _stats(tmp_path)
    land = np.ones((20, 20))
    land[:, 10:] = 2
    _raster(gs.land_path / 'domain_dom_land_classes.tif', land, 'uint8', 255)
    _catchment(gs)

    gs.calculate_land_stats()

    out = gpd.read_file(tmp_path / 'shapefiles/catchment_intersection/with_landclass/catchment_with_landclass.shp')
    out = out.set_index('HRU_ID')
    assert out.loc[1, 'IGBP_1'] == pytest.approx(1.0)
    assert out.loc[3, 'IGBP_2'] == pytest.approx(1.0)       # was 0 everywhere -> argmax picked class 1
    np.testing.assert_allclose(out[['IGBP_1', 'IGBP_2']].sum(axis=1).values, 1.0)


def test_elevation_of_sub_pixel_hru_is_computed(tmp_path):
    gs = _stats(tmp_path)
    elev = np.repeat((100.0 + 10.0 * np.arange(20))[None, :], 20, axis=0)
    _raster(gs.dem_path, elev, 'float32', -9999.0)
    _catchment(gs)

    gs.calculate_elevation_stats()

    out = gpd.read_file(tmp_path / 'shapefiles/catchment_intersection/with_dem/catchment_with_dem.shp')
    out = out.set_index('HRU_ID')
    assert out.loc[3, 'elev_mean'] == pytest.approx(220.0)
    assert out['elev_mean'].notna().all()


def test_cached_output_of_another_discretization_is_recomputed(tmp_path):
    gs = _stats(tmp_path)
    elev = np.repeat((100.0 + 10.0 * np.arange(20))[None, :], 20, axis=0)
    _raster(gs.dem_path, elev, 'float32', -9999.0)
    _catchment(gs)
    gs.calculate_elevation_stats()
    output = tmp_path / 'shapefiles/catchment_intersection/with_dem/catchment_with_dem.shp'
    assert gs._output_has_expected_columns(output, required_columns=('elev_mean',))

    # A different discretization of the same domain: other polygons, same ids
    gs2 = _stats(tmp_path)
    gpd.GeoDataFrame(
        {'HRU_ID': [1, 2, 3], 'GRU_ID': [1, 1, 1]},
        geometry=[box(X0 + 1500, Y0 - 2000, X0 + 2000, Y0),
                  box(X0, Y0 - 2000, X0 + 1500, Y0 - 1000),
                  box(X0, Y0 - 1000, X0 + 1500, Y0)],
        crs=CRS,
    ).to_file(gs2.catchment_path / gs2.catchment_name)
    assert not gs2._output_has_expected_columns(output, required_columns=('elev_mean',))
    gs2.calculate_elevation_stats()
    out = gpd.read_file(output).set_index('HRU_ID')
    assert out.loc[1, 'elev_mean'] == pytest.approx(100.0 + 10.0 * 17.0)   # columns 15..19

    # Renumbered ids are detected too
    gs3 = _stats(tmp_path)
    renumbered = gpd.read_file(gs2.catchment_path / gs2.catchment_name)
    renumbered['HRU_ID'] = [11, 12, 13]
    renumbered.to_file(gs3.catchment_path / gs3.catchment_name)
    assert not gs3._output_has_expected_columns(output, required_columns=('elev_mean',))


def test_checkpoints_of_another_catchment_layer_are_discarded(tmp_path):
    gs = _stats(tmp_path)
    _raster(gs.dem_path, np.full((20, 20), 500.0), 'float32', -9999.0)
    gdf = _catchment(gs)
    checkpoint_dir = tmp_path / 'checkpoints'

    gs._validate_checkpoints(checkpoint_dir, gdf, 'chunks:2')
    np.save(checkpoint_dir / 'chunk_0.npy', np.array([1.0, 2.0], dtype=np.float32))
    gs._validate_checkpoints(checkpoint_dir, gdf, 'chunks:2')
    assert (checkpoint_dir / 'chunk_0.npy').exists()            # same layer: kept

    other = gdf.copy()
    other['HRU_ID'] = [7, 8, 9]
    gs._validate_checkpoints(checkpoint_dir, other, 'chunks:2')
    assert not (checkpoint_dir / 'chunk_0.npy').exists()        # other layer: discarded
