# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HYPE GeoData manager reads soil/land/elevation stats from the MRDS.

``_load_gis_stats`` now prefers the model-ready attributes store, reconstructing
the per-basin soil (USGS_*), landcover (IGBP_*) and elevation frames that the SLC
pipeline consumes, and falling back to gistool CSVs / intersection shapefiles.
The store values originate from the same intersection shapefiles, so SLC output
is unchanged.
"""
from __future__ import annotations

import logging
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

netCDF4 = pytest.importorskip("netCDF4")  # noqa: N816
gpd = pytest.importorskip("geopandas")
pytest.importorskip("xarray")

from shapely.geometry import box

from symfluence.data.model_ready.attributes_builder import AttributesNetCDFBuilder
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager

HRU_IDS = [1, 2, 3]


def _catchment_shp(tmp_path):
    path = tmp_path / "shapefiles" / "catchment" / "test_HRUs_lumped.shp"
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({
        "HRU_ID": HRU_IDS,
        "HRU_area": np.full(3, 2.0e6),
        "geometry": [box(i, 50, i + 1, 51) for i in range(3)],
    }, crs="EPSG:4326").to_file(path)


def _soil_shp(tmp_path):
    path = (tmp_path / "shapefiles" / "catchment_intersection"
            / "with_soilgrids" / "catchment_with_soilclass.shp")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"HRU_ID": HRU_IDS,
            "USGS_0": [0.1, 0.1, 0.1], "USGS_1": [0.7, 0.0, 0.0],
            "USGS_2": [0.2, 0.8, 0.1], "USGS_3": [0.0, 0.1, 0.8]}
    data["geometry"] = [box(i, 50, i + 1, 51) for i in range(3)]
    gpd.GeoDataFrame(data, crs="EPSG:4326").to_file(path)


def _land_shp(tmp_path):
    path = (tmp_path / "shapefiles" / "catchment_intersection"
            / "with_landclass" / "catchment_with_landclass.shp")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"HRU_ID": HRU_IDS}
    for i in range(1, 6):
        data[f"IGBP_{i}"] = [0.0, 0.0, 0.0]
    data["IGBP_2"] = [0.9, 0.1, 0.1]
    data["IGBP_4"] = [0.1, 0.9, 0.9]
    data["geometry"] = [box(i, 50, i + 1, 51) for i in range(3)]
    gpd.GeoDataFrame(data, crs="EPSG:4326").to_file(path)


def _dem_shp(tmp_path):
    path = (tmp_path / "shapefiles" / "catchment_intersection"
            / "with_dem" / "catchment_with_dem.shp")
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({
        "HRU_ID": HRU_IDS, "elev_mean": [800.0, 1200.0, 2000.0],
        "geometry": [box(i, 50, i + 1, 51) for i in range(3)],
    }, crs="EPSG:4326").to_file(path)


def _manager(tmp_path):
    mgr = HYPEGeoDataManager.__new__(HYPEGeoDataManager)
    mgr.config = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path.parent),
        "DOMAIN_NAME": "test",
    }
    mgr.logger = logging.getLogger("test_hype_geodata")
    return mgr


def _build(tmp_path):
    # project_dir = <data_dir>/domain_test ; store lives under it.
    assert AttributesNetCDFBuilder(project_dir=tmp_path, domain_name="test").build() is not None


def test_load_gis_stats_from_store(tmp_path):
    project_dir = tmp_path / "domain_test"
    _catchment_shp(project_dir)
    _soil_shp(project_dir)
    _land_shp(project_dir)
    _dem_shp(project_dir)
    _build(project_dir)

    soil, land, elev = _manager(project_dir)._load_gis_stats_from_store("HRU_ID")

    assert list(soil.index) == HRU_IDS
    assert "USGS_1" in soil.columns and soil.loc[1, "USGS_1"] == pytest.approx(0.7)
    assert "IGBP_2" in land.columns and land.loc[1, "IGBP_2"] == pytest.approx(0.9)
    assert elev.loc[3, "elev_mean"] == pytest.approx(2000.0)


def test_store_frames_drive_slc_processing(tmp_path):
    project_dir = tmp_path / "domain_test"
    _catchment_shp(project_dir)
    _soil_shp(project_dir)
    _land_shp(project_dir)
    _dem_shp(project_dir)
    _build(project_dir)

    mgr = _manager(project_dir)
    soil, land, elev = mgr._load_gis_stats_from_store("HRU_ID")
    base_df = pd.DataFrame({"subid": HRU_IDS})

    slc_df, _ = mgr._process_slc(base_df, land, soil, threshold=0.05)

    # Dominant soil: USGS_1 (hru1), USGS_2 (hru2), USGS_3 (hru3).
    # Active land (>0.05): IGBP_2 and/or IGBP_4. SLC = (landcover, soil) combos.
    combos = set(zip(slc_df["landcover"], slc_df["soil"]))
    assert (2, 1) in combos   # hru1: IGBP_2 x USGS_1
    assert (4, 2) in combos   # hru2: IGBP_4 x USGS_2
    assert (4, 3) in combos   # hru3: IGBP_4 x USGS_3
    assert all(slc_df["soil"] >= 1)  # HYPE requires soil >= 1


def test_returns_none_when_store_absent(tmp_path):
    assert _manager(tmp_path / "domain_test")._load_gis_stats_from_store("HRU_ID") is None


def _banded_domain(project_dir):
    """Three elevation-band HRUs of one parent basin; the top band is all glacier (IGBP_15)."""
    inter = project_dir / "shapefiles" / "catchment_intersection"
    common = {"GRU_ID": [1, 1, 1], "HRU_ID": HRU_IDS, "HRU_area": [3.0e6, 2.0e6, 1.0e6],
              "geometry": [box(i, 50, i + 1, 51) for i in range(3)]}
    layers = {
        project_dir / "shapefiles" / "catchment" / "test_HRUs_lumped.shp": {},
        inter / "with_soilgrids" / "catchment_with_soilclass.shp": {"USGS_1": [1.0, 1.0, 0.0], "USGS_2": [0.0, 0.0, 1.0]},
        inter / "with_landclass" / "catchment_with_landclass.shp": {
            "IGBP_15": [0.0, 0.2, 1.0], "IGBP_16": [0.9, 0.8, 0.0], "IGBP_10": [0.1, 0.0, 0.0]},
        inter / "with_dem" / "catchment_with_dem.shp": {"elev_mean": [600.0, 900.0, 1300.0]},
    }
    for path, extra in layers.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({**common, **extra}, crs="EPSG:4326").to_file(path)
    return inter / "with_soilgrids"


def test_shapefile_fallback_keys_band_rows_by_hru(tmp_path):
    # Band rows share the parent id; reading them per parent row used to drop all
    # but the lowest band, and with HYPE_GLACIER_FRACTION the all-ice band raised
    # "No non-glacier land cover available for fraction override".
    project_dir = tmp_path / "domain_test"
    intersect = _banded_domain(project_dir)
    mgr = _manager(project_dir)
    soil, land, elev = mgr._load_gis_stats(tmp_path / "no_gistool", intersect, "GRU_ID")
    for frame in (soil, land, elev):
        assert list(frame.index) == HRU_IDS

    bands = {1: [dict(hru_id=h, area_frac=a) for h, a in zip(HRU_IDS, [3.0, 2.0, 1.0])]}
    parent = HYPEGeoDataManager._aggregate_band_stats(land, bands)
    assert list(parent.index) == [1]
    assert parent.loc[1, "IGBP_15"] == pytest.approx((0.4 + 1.0) / 6)
    assert parent.loc[1, "IGBP_16"] == pytest.approx((2.7 + 1.6) / 6)
    assert parent.loc[1, ["IGBP_10", "IGBP_16"]].sum() > 0  # glacier override has land to rescale


def test_shapefile_fallback_matches_attributes_store(tmp_path):
    project_dir = tmp_path / "domain_test"
    intersect = _banded_domain(project_dir)
    mgr = _manager(project_dir)
    fallback = mgr._load_gis_stats(tmp_path / "no_gistool", intersect, "GRU_ID")
    _build(project_dir)
    store = mgr._load_gis_stats_from_store("HRU_ID")
    assert store is not None
    bands = {1: [dict(hru_id=h, area_frac=a) for h, a in zip(HRU_IDS, [3.0, 2.0, 1.0])]}
    for a, b in zip(fallback, store):
        a, b = (HYPEGeoDataManager._aggregate_band_stats(f, bands) for f in (a, b))
        cols = [c for c in b.columns if c in a.columns and c.startswith(("IGBP_", "USGS_", "elev_mean"))]
        assert cols
        np.testing.assert_allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), atol=1e-6)
