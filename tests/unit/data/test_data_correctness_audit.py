# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for the data-correctness audit fixes (fix/data-correctness)."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Geod
from shapely.geometry import box

pytestmark = [pytest.mark.unit]

_LOG = logging.getLogger("test_data_correctness_audit")


def _geodesic_area(geom) -> float:
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geom)
    return abs(area)


# ---------------------------------------------------------------------------
# 1. GRU areas: no more EPSG:3763 (Portugal TM06)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lon,lat", [(-19.0, 64.0), (-115.6, 51.2), (-105.0, 39.0), (85.3, 27.7)])
def test_subsetter_gru_area_matches_geodesic_area(lon, lat):
    from symfluence.geospatial.geofabric.delineators.subsetter import GeofabricSubsetter

    cells = [box(lon, lat, lon + 0.5, lat + 0.5), box(lon + 0.5, lat, lon + 1.0, lat + 0.5)]
    basins = gpd.GeoDataFrame({"COMID": [1, 2]}, geometry=cells, crs="EPSG:4326")
    rivers = gpd.GeoDataFrame(
        {"COMID": [1, 2], "NextDownID": [2, 0], "slope": [0.01, 0.01]},
        geometry=gpd.GeoSeries.from_wkt(
            [f"LINESTRING ({lon} {lat}, {lon + 0.5} {lat + 0.25})",
             f"LINESTRING ({lon + 0.5} {lat + 0.25}, {lon + 1.0} {lat + 0.5})"]),
        crs="EPSG:4326",
    )
    sub = object.__new__(GeofabricSubsetter)
    sub._add_symfluence_columns(basins, rivers, "MERIT")

    for geom, area in zip(cells, basins["GRU_area"]):
        assert area == pytest.approx(_geodesic_area(geom), rel=2e-3)
    geod = Geod(ellps="WGS84")
    for geom, length in zip(rivers.geometry, rivers["Length"]):
        assert length == pytest.approx(geod.geometry_length(geom), rel=5e-3)


def test_lumped_aggregate_area_is_equal_area(tmp_path):
    from symfluence.geospatial.geofabric.delineators.subsetter import GeofabricSubsetter

    cells = [box(-105.0, 39.0, -104.5, 39.5), box(-104.5, 39.0, -104.0, 39.5)]
    basins = gpd.GeoDataFrame({"COMID": [1, 2]}, geometry=cells, crs="EPSG:4326")
    sub = object.__new__(GeofabricSubsetter)
    sub.logger = _LOG
    out = sub.aggregate_to_lumped(basins, tmp_path / "orig.shp")
    expected = sum(_geodesic_area(g) for g in cells)
    assert out["GRU_area"].iloc[0] == pytest.approx(expected, rel=2e-3)


# ---------------------------------------------------------------------------
# 2. _find_col honours candidate priority
# ---------------------------------------------------------------------------

def _find_col(columns, candidates):
    from symfluence.data.observation.base import BaseObservationHandler
    return BaseObservationHandler._find_col(None, columns, candidates)


def test_find_col_candidate_priority_beats_file_order():
    cols = ["date", "discharge_spec", "discharge_vol"]
    assert _find_col(cols, ["discharge_vol", "discharge_spec", "discharge"]) == "discharge_vol"


def test_find_col_exact_match_beats_substring():
    assert _find_col(["update_date", "date", "q"], ["date"]) == "date"
    assert _find_col(["Date", "x"], ["date"]) == "Date"


def test_find_col_substring_fallback_in_candidate_order():
    cols = ["agency", "14n_00060_00003", "14n_00060_00003_cd"]
    assert _find_col(cols, ["00060", "discharge"]) == "14n_00060_00003"
    assert _find_col(["a", "b"], ["discharge"]) is None


def test_camels_gb_refuses_specific_discharge(tmp_path):
    from symfluence.core.exceptions import DataAcquisitionError
    from symfluence.data.observation.handlers import camels

    gb_cls = next(c for n, c in vars(camels).items()
                  if isinstance(c, type) and n.upper().startswith("CAMELSGB"))
    f = tmp_path / "gb.csv"
    pd.DataFrame({"date": ["2000-01-01", "2000-01-02"],
                  "discharge_spec": [1.0, 2.0]}).to_csv(f, index=False)
    h = object.__new__(gb_cls)
    with pytest.raises(DataAcquisitionError, match="discharge_spec"):
        h._parse_streamflow_file(f)

    pd.DataFrame({"date": ["2000-01-01"], "discharge_spec": [1.0],
                  "discharge_vol": [5.0]}).to_csv(f, index=False)
    assert h._parse_streamflow_file(f)["discharge_cms"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# 3. Remap output filename date extraction with digit-bearing domain names
# ---------------------------------------------------------------------------

def _fp(domain, tmp_path, dataset="ERA5"):
    from symfluence.data.preprocessing.resampling.file_processor import FileProcessor
    return FileProcessor(
        config={"DOMAIN_NAME": domain, "FORCING_DATASET": dataset,
                "SUB_GRID_DISCRETIZATION": "lumped"},
        output_dir=tmp_path, logger=_LOG)


@pytest.mark.parametrize("domain", ["usgs_13201010", "basin_2010", "wsc_05BB001", "lamah_200512"])
def test_monthly_files_map_to_distinct_outputs_for_digit_domains(domain, tmp_path):
    fp = _fp(domain, tmp_path)
    names = {fp.determine_output_filename(Path(f"domain_{domain}_ERA5_merged_2008{m:02d}.nc")).name
             for m in range(1, 13)}
    assert len(names) == 12
    jan = fp.determine_output_filename(Path(f"domain_{domain}_ERA5_merged_200801.nc")).name
    assert jan.endswith("_2008-01-01-00-00-00.nc")


def test_date_tag_variants(tmp_path):
    fp = _fp("usgs_13201010", tmp_path, dataset="CARRA1")
    assert fp.determine_output_filename(
        Path("usgs_13201010_CARRA1_20150301.nc")).name.endswith("_2015-03-01-00-00-00.nc")
    assert fp.determine_output_filename(
        Path("domain_usgs_13201010_CARRA1_2015-03-01-06-00-00.nc")).name.endswith(
            "_2015-03-01-06-00-00.nc")


# ---------------------------------------------------------------------------
# 5. CaSR units from the units attribute
# ---------------------------------------------------------------------------

def _casr():
    from symfluence.data.preprocessing.dataset_handlers.casr_utils import CASRHandler
    return CASRHandler(config={"DOMAIN_NAME": "d"}, logger=_LOG, project_dir=Path("."))


def test_casr_v31_small_precip_and_wind_converted_by_units():
    ds = xr.Dataset({
        "precipitation_flux": ("time", np.array([0.0, 0.002, 0.004]), {"units": "m"}),
        "wind_speed": ("time", np.array([2.0, 10.0, 20.0]), {"units": "kts"}),
        "air_temperature": ("time", np.array([-5.0, 0.0, 120.0]), {"units": "deg_C"}),
        "surface_air_pressure": ("time", np.array([900.0, 1000.0, 1010.0]), {"units": "mb"}),
    })
    out = _casr().process_dataset(ds)
    np.testing.assert_allclose(out["precipitation_flux"].values, [0.0, 2.0 / 3600, 4.0 / 3600])
    np.testing.assert_allclose(out["wind_speed"].values, np.array([2.0, 10.0, 20.0]) * 0.514444)
    np.testing.assert_allclose(out["air_temperature"].values[0], 268.15)
    np.testing.assert_allclose(out["surface_air_pressure"].values[0], 90000.0)


def test_casr_v32_si_left_alone():
    ds = xr.Dataset({
        "precipitation_flux": ("time", np.array([0.0, 0.2]), {"units": "kg m-2 s-1"}),
        "wind_speed": ("time", np.array([60.0, 3.0]), {"units": "m s-1"}),
    })
    out = _casr().process_dataset(ds)
    np.testing.assert_allclose(out["precipitation_flux"].values, [0.0, 0.2])
    np.testing.assert_allclose(out["wind_speed"].values, [60.0, 3.0])


def test_casr_missing_units_falls_back_with_warning(caplog):
    ds = xr.Dataset({"wind_speed": ("time", np.array([80.0, 3.0]))})
    with caplog.at_level(logging.WARNING):
        out = _casr().process_dataset(ds)
    np.testing.assert_allclose(out["wind_speed"].values, np.array([80.0, 3.0]) * 0.514444)
    assert "no recognised units" in caplog.text


# ---------------------------------------------------------------------------
# 6. AORC / NWM3 hourly accumulations divided by 3600 s, not the model step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,cls,src_var", [
    ("aorc_utils", "AORCHandler", "APCP_surface"),
    ("nwm3_retrospective_utils", "NWM3RetrospectiveHandler", "RAINRATE"),
])
def test_hourly_accumulation_uses_native_period(module, cls, src_var):
    import importlib
    mod = importlib.import_module(f"symfluence.data.preprocessing.dataset_handlers.{module}")
    handler = getattr(mod, cls)(config={"DOMAIN_NAME": "d"}, logger=_LOG,
                                project_dir=Path("."), forcing_timestep_seconds=10800)
    ds = xr.Dataset({src_var: ("time", np.array([3.6, 7.2]), {"units": "kg/m^2"})},
                    coords={"time": pd.date_range("2010-01-01", periods=2, freq="h")})
    out = handler.process_dataset(ds)
    np.testing.assert_allclose(out["precipitation_flux"].values, [0.001, 0.002])


def test_aorc_rate_units_not_mistaken_for_accumulation():
    from symfluence.data.preprocessing.dataset_handlers.aorc_utils import AORCHandler
    handler = AORCHandler(config={"DOMAIN_NAME": "d"}, logger=_LOG, project_dir=Path("."))
    ds = xr.Dataset({"APCP_surface": ("time", np.array([0.001]), {"units": "kg m-2 s-1"})},
                    coords={"time": pd.date_range("2010-01-01", periods=1, freq="h")})
    np.testing.assert_allclose(handler.process_dataset(ds)["precipitation_flux"].values, [0.001])


# ---------------------------------------------------------------------------
# 7. HRRR: no silent zero precipitation
# ---------------------------------------------------------------------------

def _hrrr_ds():
    return xr.Dataset({"air_temperature": (("time", "y"), np.full((2, 2), 280.0))},
                      coords={"time": pd.date_range("2020-01-01", periods=2, freq="h")})


def test_hrrr_without_precip_raises():
    from symfluence.core.exceptions import DataAcquisitionError
    from symfluence.data.preprocessing.dataset_handlers.hrrr_utils import HRRRHandler
    h = HRRRHandler(config={"DOMAIN_NAME": "d"}, logger=_LOG, project_dir=Path("."))
    with pytest.raises(DataAcquisitionError, match="HRRR_ALLOW_ZERO_PRECIP"):
        h.process_dataset(_hrrr_ds())


def test_hrrr_zero_precip_requires_explicit_opt_in():
    from symfluence.data.preprocessing.dataset_handlers.hrrr_utils import HRRRHandler
    h = HRRRHandler(config={"DOMAIN_NAME": "d", "HRRR_ALLOW_ZERO_PRECIP": True},
                    logger=_LOG, project_dir=Path("."))
    out = h.process_dataset(_hrrr_ds())
    assert float(out["precipitation_flux"].max()) == 0.0


# ---------------------------------------------------------------------------
# 8. Daily streamflow onto an hourly grid: step within day, no interpolation
# ---------------------------------------------------------------------------

class _FreqStub:
    def __init__(self, freq):
        self._freq = freq

    def _get_resample_freq(self):
        return self._freq


def _to_target(series, freq):
    from symfluence.data.observation.base import BaseObservationHandler
    return BaseObservationHandler._daily_streamflow_to_target(_FreqStub(freq), series)


def test_daily_to_hourly_preserves_daily_means_and_gaps():
    idx = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"])
    daily = pd.Series([10.0, 20.0, 40.0], index=idx, name="discharge_cms")
    hourly = _to_target(daily, "h")

    assert len(hourly) == 4 * 24
    back = hourly.resample("D").mean()
    assert back.loc["2000-01-01"] == 10.0
    assert back.loc["2000-01-02"] == 20.0
    assert np.isnan(back.loc["2000-01-03"])          # gap stays a gap
    assert hourly.loc["2000-01-03"].isna().all()
    assert (hourly.loc["2000-01-04"] == 40.0).all()  # last day fully covered


def test_daily_to_daily_is_plain_mean():
    idx = pd.date_range("2000-01-01", periods=3, freq="D")
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    pd.testing.assert_series_equal(_to_target(s, "D"), s.resample("D").mean())


def test_no_handler_interpolates_daily_streamflow():
    root = Path(__file__).resolve().parents[3] / "src/symfluence/data/observation/handlers"
    for name in ("wsc.py", "smhi.py", "camels.py", "dga.py", "ana.py"):
        text = (root / name).read_text()
        assert "interpolate(" not in text, name
        assert "_daily_streamflow_to_target" in text, name


# ---------------------------------------------------------------------------
# 9. SWE units are explicit
# ---------------------------------------------------------------------------

def test_swe_mm_column_never_converted_even_when_shallow():
    from symfluence.core.constants import UnitConverter
    s = pd.Series([0.0, 30.0, 120.0])  # shallow CanSWE/NorSWE snowpack in mm
    pd.testing.assert_series_equal(UnitConverter.swe_series_to_mm(s, "swe_mm"), s)


def test_swe_in_column_always_converted():
    from symfluence.core.constants import UnitConverter
    s = pd.Series([0.0, 300.0])  # deep snowpack in inches (above old threshold)
    np.testing.assert_allclose(UnitConverter.swe_series_to_mm(s, "swe_in"), [0.0, 7620.0])
    np.testing.assert_allclose(UnitConverter.swe_series_to_mm(s, "SWE_in"), [0.0, 7620.0])


def test_swe_legacy_column_warns(caplog):
    from symfluence.core.constants import UnitConverter
    with caplog.at_level(logging.WARNING):
        out = UnitConverter.swe_series_to_mm(pd.Series([1.0, 2.0]), "swe", logger=_LOG)
    np.testing.assert_allclose(out, [25.4, 50.8])
    assert "does not state its units" in caplog.text


# ---------------------------------------------------------------------------
# 10. 0-360 source grids and targets straddling the prime meridian
# ---------------------------------------------------------------------------

def _weight_generator():
    from symfluence.data.preprocessing.resampling.weight_generator import RemappingWeightGenerator
    return RemappingWeightGenerator(
        config={"DOMAIN_NAME": "d", "FORCING_SHAPE_LON_NAME": "lon",
                "CATCHMENT_SHP_LON": "center_lon"},
        project_dir=Path("."), dataset_handler=None, logger=_LOG)


def _source_0_360(path):
    lons = [358.0, 359.0, 0.0, 1.0]
    gdf = gpd.GeoDataFrame(
        {"ID": range(4), "lon": [x + 0.5 for x in lons], "lat": [51.5] * 4},
        geometry=[box(x, 51.0, x + 1.0, 52.0) for x in lons], crs="EPSG:4326")
    gdf.to_file(path)
    return path


def test_target_straddling_zero_shifts_source_not_target(tmp_path):
    src = _source_0_360(tmp_path / "src.shp")
    tgt = tmp_path / "tgt.shp"
    gpd.GeoDataFrame({"center_lon": [0.0]}, geometry=[box(-1.5, 51.2, 1.5, 51.8)],
                     crs="EPSG:4326").to_file(tgt)

    new_src, new_tgt, _ = _weight_generator()._align_longitudes(src, tgt, tmp_path)
    assert new_tgt == tgt
    shifted = gpd.read_file(new_src)
    minx, _, maxx, _ = shifted.total_bounds
    assert (minx, maxx) == (-2.0, 2.0)
    # lon attribute (used to map back to the NetCDF grid) is untouched
    assert sorted(shifted["lon"]) == [0.5, 1.5, 358.5, 359.5]
    target = gpd.read_file(new_tgt).geometry.iloc[0]
    assert shifted.intersects(target).sum() == 4


def test_target_entirely_west_is_shifted_into_0_360(tmp_path):
    src = _source_0_360(tmp_path / "src.shp")
    tgt = tmp_path / "tgt.shp"
    gpd.GeoDataFrame({"center_lon": [-1.0]}, geometry=[box(-1.8, 51.2, -0.2, 51.8)],
                     crs="EPSG:4326").to_file(tgt)
    new_src, new_tgt, _ = _weight_generator()._align_longitudes(src, tgt, tmp_path)
    assert new_src == src
    shifted = gpd.read_file(new_tgt)
    assert shifted.total_bounds[0] == pytest.approx(358.2)
    assert shifted["center_lon"].iloc[0] == 359.0


def test_align_longitudes_does_not_swallow_errors(tmp_path):
    with pytest.raises(Exception):
        _weight_generator()._align_longitudes(tmp_path / "missing.shp", tmp_path / "t.shp", tmp_path)


# ---------------------------------------------------------------------------
# 11. Remap cache signature ignores irrelevant config
# ---------------------------------------------------------------------------

def test_remap_signature_ignores_irrelevant_keys():
    from symfluence.data.preprocessing.resampling.grid_identity import remap_config_signature
    base = {"DOMAIN_NAME": "d", "FORCING_DATASET": "ERA5", "CATCHMENT_SHP_HRUID": "HRU_ID",
            "EXPERIMENT_ID": "run_1", "HYDROLOGICAL_MODEL": "SUMMA", "NUM_PROCESSES": 1,
            "EXPERIMENT_TIME_END": "2010-12-31 23:00"}
    sig = remap_config_signature(base, "ERA5")
    for key, value in [("EXPERIMENT_ID", "run_2"), ("HYDROLOGICAL_MODEL", "FUSE"),
                       ("NUM_PROCESSES", 32), ("EXPERIMENT_TIME_END", "2020-12-31 23:00")]:
        assert remap_config_signature({**base, key: value}, "ERA5") == sig, key
    for key, value in [("FORCING_DATASET", "CARRA"), ("CATCHMENT_SHP_HRUID", "GRU_ID"),
                       ("SUB_GRID_DISCRETIZATION", "elevation"), ("ERA5_EXTRA_OPTION", 1)]:
        assert remap_config_signature({**base, key: value}, "ERA5") != sig, key


# ---------------------------------------------------------------------------
# 12. MODIS ET special values; raster valid-pixel mask
# ---------------------------------------------------------------------------

def test_modis_et_masks_special_values(tmp_path):
    from symfluence.data.acquisition.handlers.modis_et import MOD16ETAcquirer

    raw = np.array([[[100, 32761], [32766, 32767]]], dtype=np.int16)
    ds = xr.Dataset({"ET_500m": (("time", "lat", "lon"), raw)},
                    coords={"time": pd.date_range("2010-01-01", periods=1),
                            "lat": [50.0, 50.1], "lon": [-115.0, -114.9]})
    path = tmp_path / "et.nc"
    ds.to_netcdf(path)
    h = object.__new__(MOD16ETAcquirer)
    h.logger = _LOG
    h.config = {"DOMAIN_NAME": "d", "MOD16_CONVERT_TO_DAILY": False, "MOD16_QC_FILTER": False}
    out = tmp_path / "out.nc"
    h.domain_name = "d"
    h._process_appeears_products({"MOD16A2": path}, out, "ET_500m")
    h.domain_name = "d"
    with xr.open_dataset(out) as res:
        assert float(res["ET_basin_mean"].values.ravel()[0]) == pytest.approx(10.0)
        assert int(np.isfinite(res["ET"].values).sum()) == 1


def test_valid_pixel_mask_handles_float32_nan_and_masked():
    from symfluence.geospatial.raster_utils import valid_pixel_mask
    data = np.ma.masked_array(np.array([1.0, np.nan, 3.0, -9999.0], dtype=np.float32),
                              mask=[False, False, True, False])
    np.testing.assert_array_equal(valid_pixel_mask(data, -9999.0), [True, False, False, False])
    np.testing.assert_array_equal(valid_pixel_mask(np.array([1.0, np.nan], dtype=np.float32)),
                                  [True, False])
    np.testing.assert_array_equal(valid_pixel_mask(np.array([1.0, np.nan]), float("nan")),
                                  [True, False])


# ---------------------------------------------------------------------------
# 13. Dead remapping_weights module is gone; canonical exports remain
# ---------------------------------------------------------------------------

def test_remapping_weights_module_removed():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("symfluence.data.preprocessing.remapping_weights")
    from symfluence.data import preprocessing
    from symfluence.data.preprocessing.resampling import RemappingWeightGenerator
    assert preprocessing.RemappingWeightGenerator is RemappingWeightGenerator


def test_multipolygon_hrus_reproject_grus_to_raster_crs(tmp_path):
    """GRUs in EPSG:4326 over a UTM (float32, NaN nodata) raster still get HRUs."""
    import rasterio
    from rasterio.transform import from_origin

    from symfluence.geospatial.discretization.core import DomainDiscretizer

    data = np.full((20, 20), np.nan, dtype=np.float32)
    data[:, :10] = 1.0
    data[:, 10:] = 2.0
    data[0, 0] = np.nan
    raster = tmp_path / "landclass.tif"
    with rasterio.open(raster, "w", driver="GTiff", height=20, width=20, count=1,
                       dtype="float32", crs="EPSG:32611",
                       transform=from_origin(500000, 5700000, 100, 100)) as dst:
        dst.write(data, 1)

    utm = gpd.GeoDataFrame({"GRU_ID": [1]}, geometry=[box(500200, 5698200, 501800, 5699800)],
                           crs="EPSG:32611")
    grus = utm.to_crs("EPSG:4326")
    disc = object.__new__(DomainDiscretizer)
    disc.logger = _LOG
    hrus = disc._create_multipolygon_hrus(grus, raster, None, "landClass")

    assert len(hrus) == 2
    assert hrus.crs == grus.crs
    assert sorted(hrus["landClass"].tolist()) == [1.0, 2.0]
    assert not hrus["avg_landclass"].isna().any()
