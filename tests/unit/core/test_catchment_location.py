"""Catchment location helpers must use the real catchment, never a default site.

Several preprocessors (WRF-Hydro, CLM-ParFlow, PCR-GLOBWB, CWatM, LISFLOOD,
PIHM, CLM) used to fall back -- or unconditionally default -- to Bow at Banff
(51.17N, -115.57E, 2210 km2), silently building models in the wrong place.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

from symfluence.core.geometry_utils import calculate_catchment_centroid
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.clm.domain_generator import CLMDomainGenerator

pytestmark = pytest.mark.unit

# Two adjacent 0.2 x 0.1 degree HRUs in south Iceland (well away from Banff).
_HRUS = [box(-19.4, 64.2, -19.2, 64.3), box(-19.2, 64.2, -19.0, 64.3)]


def _two_hru_gdf(**columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns, geometry=_HRUS, crs="EPSG:4326")


def _fake_preprocessor(path):
    return SimpleNamespace(get_catchment_path=lambda: path, logger=logging.getLogger("test"))


def test_centroid_is_whole_catchment_not_first_hru():
    lon, lat = calculate_catchment_centroid(_two_hru_gdf())
    # The first HRU's centroid would be at -19.3; the catchment's is at -19.2.
    assert lon == pytest.approx(-19.2, abs=1e-3)
    assert lat == pytest.approx(64.25, abs=1e-3)


def test_get_catchment_location_reads_shapefile(tmp_path):
    shp = tmp_path / "catchment.shp"
    _two_hru_gdf(elev_mean=[500.0, 700.0]).to_file(shp)

    loc = BaseModelPreProcessor.get_catchment_location(_fake_preprocessor(shp))

    assert loc["lon"] == pytest.approx(-19.2, abs=1e-3)
    assert loc["lat"] == pytest.approx(64.25, abs=1e-3)
    # 0.4 deg x 0.1 deg at 64.25N ~ 19.4 km x 11.1 km ~ 215 km2
    assert loc["area_m2"] / 1e6 == pytest.approx(215, rel=0.03)
    # Equal-area HRUs -> area-weighted mean is the plain mean.
    assert loc["elev"] == pytest.approx(600.0, rel=1e-3)


def test_get_catchment_location_without_elev_column(tmp_path):
    shp = tmp_path / "catchment.shp"
    _two_hru_gdf(hru_id=[1, 2]).to_file(shp)
    assert BaseModelPreProcessor.get_catchment_location(_fake_preprocessor(shp))["elev"] is None


def test_get_catchment_location_raises_instead_of_defaulting(tmp_path):
    with pytest.raises(FileNotFoundError, match="Catchment shapefile not found"):
        BaseModelPreProcessor.get_catchment_location(_fake_preprocessor(tmp_path / "missing.shp"))


def test_clm_area_is_projected_and_summed_over_features():
    # Web Mercator would inflate this ~5x at 64N; first-feature-only would halve it.
    area_km2 = CLMDomainGenerator._get_catchment_area_from_gdf(_two_hru_gdf())
    assert area_km2 == pytest.approx(215, rel=0.03)

    with_attr = _two_hru_gdf(Area_km2=[100.0, 115.0])
    assert CLMDomainGenerator._get_catchment_area_from_gdf(with_attr) == pytest.approx(215.0)
