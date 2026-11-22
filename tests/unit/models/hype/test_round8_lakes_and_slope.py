# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Round-8: water land cover as HYPE internal lakes; slope_mean as land slope in %."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.geodata_manager import ILAKE_CLASS_MODEL, HYPEGeoDataManager

pytestmark = [pytest.mark.unit, pytest.mark.quick]
LOG = logging.getLogger("hype-r8")


def _slc_frames():
    # SLC 1 forest, SLC 2 water/soil 1, SLC 3 water/soil 2 -> both water SLCs merge
    slc = pd.DataFrame({"landcover": [1, 17, 17], "soil": [1, 1, 2], "SLC": [1, 2, 3]})
    base = pd.DataFrame({"subid": [1, 2], "SLC_1": [0.7, 1.0], "SLC_2": [0.2, 0.0], "SLC_3": [0.1, 0.0]})
    return base, slc


def test_water_becomes_single_internal_lake_class(tmp_path):
    mgr = HYPEGeoDataManager({}, LOG, tmp_path, {})
    base, slc = _slc_frames()
    out, slc_out = mgr._apply_internal_lakes(base, slc)

    assert list(out["SLC_2"]) == pytest.approx([0.3, 0.0])
    assert list(out["SLC_3"]) == [0.0, 0.0]
    assert out.filter(like="SLC_").sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    specials = dict(zip(slc_out["SLC"], slc_out["special"]))
    assert specials[2] == ILAKE_CLASS_MODEL
    assert pd.isna(specials[1]) and pd.isna(specials[3])

    mgr._write_geoclass(slc_out)
    geoclass = pd.read_csv(tmp_path / "GeoClass.txt", sep="\t", skiprows=1, header=None)
    assert geoclass.iloc[:, 7].tolist() == [0, 2, 0]   # special class code
    assert geoclass.iloc[:, 6].tolist() == [1, 3, 3]   # vegetation type: water = 3


def test_internal_lakes_can_be_switched_off(tmp_path):
    mgr = HYPEGeoDataManager({"HYPE_WATER_AS_ILAKE": False}, LOG, tmp_path, {})
    base, slc = _slc_frames()
    out, slc_out = mgr._apply_internal_lakes(base, slc)
    assert out.equals(base) and "special" not in slc_out


def test_par_file_gets_ilake_parameters_and_water_evaporates(tmp_path):
    mgr = HYPEGeoDataManager({}, LOG, tmp_path, {})
    base, slc = _slc_frames()
    _, slc_out = mgr._apply_internal_lakes(base, slc)
    mgr._write_geoclass(slc_out)

    HYPEConfigManager({}, LOG, tmp_path).write_par_file(land_uses=np.array([1, 17]))
    par = (tmp_path / "par.txt").read_text()
    rows = {line.split()[0]: line.split()[1:] for line in par.splitlines()
            if line.strip() and not line.startswith("!!")}
    assert float(rows["gldepi"][0]) > 0          # HYPE rejects an ilake with area but no depth
    assert float(rows["gratk"][0]) > 0 and float(rows["gratp"][0]) > 0
    assert "gicatch" in rows
    assert "gldepo" not in rows                  # no outlet lakes configured
    # Land use 17 (column 17) no longer zeroes lake evaporation or snow melt
    assert float(rows["cevp"][16]) > 0
    assert float(rows["cmlt"][16]) > 0
    assert float(rows["rrcs3"][0]) == pytest.approx(0.0002)


def test_par_file_without_lakes_has_no_lake_block(tmp_path):
    mgr = HYPEGeoDataManager({"HYPE_WATER_AS_ILAKE": False}, LOG, tmp_path, {})
    base, slc = _slc_frames()
    _, slc_out = mgr._apply_internal_lakes(base, slc)
    mgr._write_geoclass(slc_out)
    HYPEConfigManager({}, LOG, tmp_path).write_par_file(land_uses=np.array([1, 17]))
    par = (tmp_path / "par.txt").read_text()
    assert "gldepi" not in par and "gratk" not in par


def test_slope_mean_is_land_slope_percent_from_dem_raster(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    project = tmp_path / "domain_x"
    out_dir = project / "settings" / "HYPE"
    slope_dir = project / "attributes" / "elevation" / "slope"
    out_dir.mkdir(parents=True)
    slope_dir.mkdir(parents=True)
    # West half 45 degrees (100 %), east half ~5.71 degrees (10 %)
    deg = np.zeros((10, 20), dtype="float32")
    deg[:, :10] = 45.0
    deg[:, 10:] = np.degrees(np.arctan(0.1))
    with rasterio.open(slope_dir / "dem_slope.tif", "w", driver="GTiff", height=10, width=20, count=1,
                       dtype="float32", crs="EPSG:32633", transform=from_origin(0, 1000, 50, 50)) as dst:
        dst.write(deg, 1)

    cat = gpd.GeoDataFrame({"GRU_ID": [1, 2, 3]},
                           geometry=[box(0, 500, 500, 1000), box(500, 500, 1000, 1000),
                                     box(5000, 5000, 6000, 6000)], crs="EPSG:32633")
    base = pd.DataFrame({"subid": [1, 2, 3], "slope_mean": [0.3, 0.3, 0.3]})
    HYPEGeoDataManager({}, LOG, out_dir, {})._apply_land_slope(base, cat, "GRU_ID")
    assert base["slope_mean"].tolist() == pytest.approx([100.0, 10.0, 0.3], rel=1e-4)


def test_slope_mean_falls_back_to_river_slope_in_percent(tmp_path, caplog):
    cat = gpd.GeoDataFrame({"GRU_ID": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    base = pd.DataFrame({"subid": [1], "slope_mean": [0.5]})
    with caplog.at_level(logging.WARNING, logger="hype-r8"):
        HYPEGeoDataManager({}, LOG, tmp_path / "settings" / "HYPE", {})._apply_land_slope(base, cat, "GRU_ID")
    assert base["slope_mean"].tolist() == [0.5]
    assert "river slope x 100" in caplog.text
