# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SUMMA attributes manager reads soil/land class fractions from the MRDS.

Covers the per-HRU class-fraction reconstruction used by insert_soil_class /
insert_land_class. The store's /soil/ and /landcover/ fraction matrices come
from the same intersection shapefiles, so the dominant-class selection is
unchanged — this verifies the mapping is keyed and ordered correctly.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("xarray")

from shapely.geometry import box

from symfluence.data.model_ready.attributes_builder import AttributesNetCDFBuilder
from symfluence.models.summa.attributes_manager import SummaAttributesManager

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
    # USGS_0..4 fractions; per HRU the dominant non-zero class differs.
    data = {"HRU_ID": HRU_IDS}
    fracs = {
        "USGS_0": [0.1, 0.1, 0.1],
        "USGS_1": [0.7, 0.0, 0.0],
        "USGS_2": [0.2, 0.8, 0.1],
        "USGS_3": [0.0, 0.1, 0.8],
        "USGS_4": [0.0, 0.0, 0.0],
    }
    data.update(fracs)
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
    data["IGBP_4"] = [0.1, 0.9, 0.2]
    data["geometry"] = [box(i, 50, i + 1, 51) for i in range(3)]
    gpd.GeoDataFrame(data, crs="EPSG:4326").to_file(path)


def _manager(tmp_path):
    # Bypass the heavy constructor — the reader helper only needs project_dir/logger.
    mgr = SummaAttributesManager.__new__(SummaAttributesManager)
    mgr.project_dir = tmp_path
    mgr.logger = logging.getLogger("test_summa_attrs")
    return mgr


def test_soil_fractions_from_store(tmp_path):
    _catchment_shp(tmp_path)
    _soil_shp(tmp_path)
    assert AttributesNetCDFBuilder(project_dir=tmp_path, domain_name="test").build() is not None

    frac = _manager(tmp_path)._load_store_class_fractions(
        "soil", "soil_fraction", "soil_class_name")

    assert frac is not None
    assert set(frac) == {1, 2, 3}
    # HRU 1 fractions match the shapefile, keyed by USGS index.
    assert frac[1][1] == pytest.approx(0.7)
    assert frac[1][2] == pytest.approx(0.2)
    # Dominant class (USGS_0 set to -1 in the caller) is USGS_1 for HRU 1,
    # USGS_2 for HRU 2, USGS_3 for HRU 3 — i.e. argmax of the non-zero classes.
    def dominant(h):
        hist = [frac[h].get(j, 0.0) for j in range(5)]
        hist[0] = -1
        return int(np.argmax(hist))
    assert dominant(1) == 1
    assert dominant(2) == 2
    assert dominant(3) == 3


def test_land_fractions_from_store(tmp_path):
    _catchment_shp(tmp_path)
    _land_shp(tmp_path)
    assert AttributesNetCDFBuilder(project_dir=tmp_path, domain_name="test").build() is not None

    frac = _manager(tmp_path)._load_store_class_fractions(
        "landcover", "land_fraction", "land_class_name")

    assert frac is not None
    assert set(frac) == {1, 2, 3}
    # vegTypeIndex = argmax(IGBP_1..17) + 1 ; HRU 1 dominant IGBP_2, HRU 2 IGBP_4.
    def veg(h):
        hist = [frac[h].get(j, 0.0) for j in range(1, 18)]
        return int(np.argmax(hist)) + 1
    assert veg(1) == 2
    assert veg(2) == 4


def test_returns_none_when_store_absent(tmp_path):
    assert _manager(tmp_path)._load_store_class_fractions(
        "soil", "soil_fraction", "soil_class_name") is None
